//! Per-model admission control (#53).
//!
//! Inference against a loaded model is batch-1: one request runs at a time,
//! serialized by the model's `inference_lock` (single-GPU) / `pool` mutex
//! (TP). Before this, the wait for that lock was an **unbounded FIFO of
//! mutex waiters with no timeout** — a busy model made every new request
//! hang until its client gave up (~300s) with an opaque error.
//!
//! [`AdmissionController`] replaces that implicit unbounded wait with an
//! explicit bounded scheduler: at most `max_in_flight` running (1, batch-1)
//! plus a bounded queue of `max_queue_depth` waiters, each waiting at most
//! `max_wait`. When the queue is full or the wait elapses, the request is
//! rejected *immediately* — an honest, fast, retryable "busy" signal
//! (`429`/`503` + `Retry-After` per #63) instead of a silent stall.
//!
//! The controller is pure async (no CUDA), so the inference paths just call
//! [`AdmissionController::enter`] before taking the inference lock and hold
//! the returned [`AdmissionPermit`] for the request's lifetime. Its counters
//! ([`in_flight`](AdmissionController::in_flight) /
//! [`queue_depth`](AdmissionController::queue_depth)) are lock-free, so
//! `/health` can read live load without contending with inference.

use crate::config::AdmissionConfig;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Why admission was refused. All map to the #63 backpressure envelope
/// (`rate_limit_exceeded` + `Retry-After`); they differ in cause (and HTTP
/// status — load → `503`, per-principal → `429`).
#[derive(Debug, Clone, Copy)]
pub enum AdmissionRejection {
    /// The bounded wait queue was already full (server-side load).
    QueueFull { retry_after_secs: u64 },
    /// A queue slot was taken but the in-flight slot didn't free within
    /// `max_wait` (server-side load).
    Timeout { retry_after_secs: u64 },
    /// This principal already has `max_per_principal` requests in flight or
    /// queued (#54 fair-share) — one principal can't monopolize the model.
    PrincipalCap { retry_after_secs: u64 },
}

impl AdmissionRejection {
    pub fn retry_after_secs(&self) -> u64 {
        match self {
            AdmissionRejection::QueueFull { retry_after_secs }
            | AdmissionRejection::Timeout { retry_after_secs }
            | AdmissionRejection::PrincipalCap { retry_after_secs } => *retry_after_secs,
        }
    }
}

/// Admission accounting, mutated under a brief lock (never held across an
/// await). `pending` is queued + in-flight overall; `per_principal` is the
/// same count keyed by principal for fair-share (#54).
#[derive(Default, Debug)]
struct AdmissionState {
    pending: usize,
    per_principal: HashMap<String, usize>,
}

/// Bounded batch-1 scheduler for one loaded model, with per-principal
/// fair-share.
pub struct AdmissionController {
    /// In-flight slots — `max_in_flight` permits (1 for batch-1).
    slots: Arc<Semaphore>,
    /// Queued + in-flight accounting (overall + per principal).
    state: Arc<Mutex<AdmissionState>>,
    /// `max_in_flight + max_queue_depth` — the overall rejection threshold.
    max_pending: usize,
    /// Max in-flight + queued for any single principal (#54). `0` disables.
    max_per_principal: usize,
    max_in_flight: usize,
    max_wait: Duration,
}

impl AdmissionController {
    pub fn new(cfg: &AdmissionConfig) -> Self {
        // A controller with zero in-flight slots would deadlock; clamp.
        let max_in_flight = cfg.max_in_flight.max(1);
        Self {
            slots: Arc::new(Semaphore::new(max_in_flight)),
            state: Arc::new(Mutex::new(AdmissionState::default())),
            max_pending: max_in_flight + cfg.max_queue_depth,
            max_per_principal: cfg.max_per_principal,
            max_in_flight,
            max_wait: Duration::from_secs(cfg.max_wait_secs),
        }
    }

    /// Admit a request for `principal` (`None` = anonymous, exempt from the
    /// per-principal cap). Reserves a queue slot — fast-rejecting if the
    /// overall queue is full or the principal is over its fair-share cap —
    /// then waits up to `max_wait` for an in-flight slot. The returned permit
    /// must be held for the request's lifetime; dropping it frees the slots.
    pub async fn enter(
        &self,
        principal: Option<&str>,
    ) -> Result<AdmissionPermit, AdmissionRejection> {
        // Decision + reservation under one brief lock so concurrent callers
        // can't both slip past the thresholds. No await is held here.
        {
            let mut st = self.state.lock().expect("admission state poisoned");
            if st.pending >= self.max_pending {
                return Err(AdmissionRejection::QueueFull {
                    retry_after_secs: self.retry_hint(st.pending),
                });
            }
            if let Some(p) = principal
                && self.max_per_principal > 0
                && st.per_principal.get(p).copied().unwrap_or(0) >= self.max_per_principal
            {
                return Err(AdmissionRejection::PrincipalCap {
                    retry_after_secs: self.retry_hint(st.pending),
                });
            }
            st.pending += 1;
            if let Some(p) = principal {
                *st.per_principal.entry(p.to_string()).or_insert(0) += 1;
            }
        }

        match tokio::time::timeout(self.max_wait, Arc::clone(&self.slots).acquire_owned()).await {
            Ok(Ok(permit)) => Ok(AdmissionPermit {
                _permit: permit,
                state: Arc::clone(&self.state),
                principal: principal.map(str::to_string),
            }),
            // Semaphore is never closed; treat a closed/elapsed wait the same.
            Ok(Err(_)) | Err(_) => {
                self.release(principal);
                Err(AdmissionRejection::Timeout {
                    retry_after_secs: self.retry_hint(self.max_pending),
                })
            }
        }
    }

    /// Roll back a reserved-but-not-admitted slot (wait timed out).
    fn release(&self, principal: Option<&str>) {
        let mut st = self.state.lock().expect("admission state poisoned");
        st.pending = st.pending.saturating_sub(1);
        decrement_principal(&mut st.per_principal, principal);
    }

    /// Requests currently running (holding an in-flight slot).
    pub fn in_flight(&self) -> usize {
        self.max_in_flight
            .saturating_sub(self.slots.available_permits())
    }

    /// Requests waiting for an in-flight slot.
    pub fn queue_depth(&self) -> usize {
        let pending = self.state.lock().expect("admission state poisoned").pending;
        pending.saturating_sub(self.in_flight())
    }

    /// Rough `Retry-After`: scale with how backed-up the model is, clamped to
    /// a sane band. Without per-request timing this is a heuristic, but it
    /// gives well-behaved clients (opencode/AI SDK) a sensible backoff.
    fn retry_hint(&self, pending: usize) -> u64 {
        let queued = pending.saturating_sub(self.max_in_flight) as u64;
        ((queued + 1) * 2).clamp(1, 120)
    }
}

/// Decrement (and prune at zero) a principal's outstanding count.
fn decrement_principal(map: &mut HashMap<String, usize>, principal: Option<&str>) {
    if let Some(p) = principal
        && let Some(count) = map.get_mut(p)
    {
        *count -= 1;
        if *count == 0 {
            map.remove(p);
        }
    }
}

/// Held for a request's lifetime; frees the in-flight + queue slot (and the
/// principal's fair-share slot) on drop.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
    state: Arc<Mutex<AdmissionState>>,
    principal: Option<String>,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        let mut st = self.state.lock().expect("admission state poisoned");
        st.pending = st.pending.saturating_sub(1);
        decrement_principal(&mut st.per_principal, self.principal.as_deref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Config with the per-principal cap disabled (0) — most tests exercise
    /// the overall queue with anonymous (`None`) callers.
    fn cfg(max_in_flight: usize, max_queue_depth: usize, max_wait_secs: u64) -> AdmissionConfig {
        AdmissionConfig {
            max_in_flight,
            max_queue_depth,
            max_wait_secs,
            max_per_principal: 0,
        }
    }

    #[tokio::test]
    async fn admits_up_to_in_flight_and_reports_load() {
        let ctrl = AdmissionController::new(&cfg(1, 4, 30));
        assert_eq!(ctrl.in_flight(), 0);
        let p = ctrl.enter(None).await.expect("first admits");
        assert_eq!(ctrl.in_flight(), 1);
        assert_eq!(ctrl.queue_depth(), 0);
        drop(p);
        assert_eq!(ctrl.in_flight(), 0);
    }

    #[tokio::test]
    async fn rejects_when_queue_full() {
        // 1 in-flight + 1 queue slot = capacity 2; the 3rd is refused fast.
        let ctrl = Arc::new(AdmissionController::new(&cfg(1, 1, 30)));
        let _running = ctrl.enter(None).await.expect("admit running");

        // Fill the single queue slot with a waiter that parks on the semaphore.
        let ctrl2 = Arc::clone(&ctrl);
        let waiter = tokio::spawn(async move { ctrl2.enter(None).await.map(|p| drop(p)) });
        // Give the waiter a moment to occupy the queue slot.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 1);

        // Queue full → immediate QueueFull with a Retry-After hint.
        match ctrl.enter(None).await {
            Err(AdmissionRejection::QueueFull { retry_after_secs }) => {
                assert!(retry_after_secs >= 1)
            }
            other => panic!("expected QueueFull, got {other:?}"),
        }

        // Release the runner so the parked waiter can proceed and finish.
        drop(_running);
        waiter.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn rejects_on_wait_timeout() {
        // Zero queue depth + a runner holding the only slot → a second
        // request can't even queue, so it's QueueFull, not Timeout. Use a
        // queue of 1 and a tiny max_wait to exercise the timeout path.
        let ctrl = Arc::new(AdmissionController::new(&cfg(1, 1, 0)));
        let _running = ctrl.enter(None).await.expect("admit running");
        // max_wait 0 → the queued request times out almost immediately.
        match ctrl.enter(None).await {
            Err(AdmissionRejection::Timeout { .. }) => {}
            other => panic!("expected Timeout, got {other:?}"),
        }
        // The timed-out request released its queue slot.
        assert_eq!(ctrl.queue_depth(), 0);
    }

    #[tokio::test]
    async fn per_principal_cap_protects_other_principals() {
        // Generous overall queue, but each principal capped at 1 in-flight+
        // queued. Principal A holds the running slot; A's second request is
        // refused (PrincipalCap) rather than occupying the queue, so B's
        // single request still gets a queue slot and proceeds.
        let cfg = AdmissionConfig {
            max_in_flight: 1,
            max_queue_depth: 8,
            max_wait_secs: 30,
            max_per_principal: 1,
        };
        let ctrl = Arc::new(AdmissionController::new(&cfg));

        let _a1 = ctrl.enter(Some("acct-a/key-a")).await.expect("A admits");

        // A is over its fair-share cap → fast PrincipalCap, no queue slot taken.
        match ctrl.enter(Some("acct-a/key-a")).await {
            Err(AdmissionRejection::PrincipalCap { retry_after_secs }) => {
                assert!(retry_after_secs >= 1)
            }
            other => panic!("expected PrincipalCap, got {other:?}"),
        }

        // B (a different principal) is admitted to the queue and proceeds
        // once A releases — it was never stuck behind A's backlog.
        let ctrl2 = Arc::clone(&ctrl);
        let b = tokio::spawn(async move { ctrl2.enter(Some("acct-b/key-b")).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 1, "B is queued, not rejected");
        drop(_a1);
        b.await.unwrap().expect("B is served after A releases");
    }
}
