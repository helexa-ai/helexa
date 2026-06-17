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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Why admission was refused. Both map to the #63 backpressure envelope
/// (`429`/`503` + `rate_limit_exceeded` + `Retry-After`); they differ only
/// in cause, for logging.
#[derive(Debug, Clone, Copy)]
pub enum AdmissionRejection {
    /// The bounded wait queue was already full.
    QueueFull { retry_after_secs: u64 },
    /// A queue slot was taken but the in-flight slot didn't free within
    /// `max_wait`.
    Timeout { retry_after_secs: u64 },
}

impl AdmissionRejection {
    pub fn retry_after_secs(&self) -> u64 {
        match self {
            AdmissionRejection::QueueFull { retry_after_secs }
            | AdmissionRejection::Timeout { retry_after_secs } => *retry_after_secs,
        }
    }
}

/// Bounded batch-1 scheduler for one loaded model.
pub struct AdmissionController {
    /// In-flight slots — `max_in_flight` permits (1 for batch-1).
    slots: Arc<Semaphore>,
    /// Queued + in-flight count, for fast rejection and load reporting.
    pending: Arc<AtomicUsize>,
    /// `max_in_flight + max_queue_depth` — the rejection threshold.
    max_pending: usize,
    max_in_flight: usize,
    max_wait: Duration,
}

impl AdmissionController {
    pub fn new(cfg: &AdmissionConfig) -> Self {
        // A controller with zero in-flight slots would deadlock; clamp.
        let max_in_flight = cfg.max_in_flight.max(1);
        Self {
            slots: Arc::new(Semaphore::new(max_in_flight)),
            pending: Arc::new(AtomicUsize::new(0)),
            max_pending: max_in_flight + cfg.max_queue_depth,
            max_in_flight,
            max_wait: Duration::from_secs(cfg.max_wait_secs),
        }
    }

    /// Admit a request: reserve a queue slot (fast-rejecting if full), then
    /// wait up to `max_wait` for an in-flight slot. The returned permit must
    /// be held for the request's lifetime; dropping it frees both slots.
    pub async fn enter(&self) -> Result<AdmissionPermit, AdmissionRejection> {
        // Reserve a pending slot up front so concurrent callers can't all
        // slip past the threshold check. Roll back if we're over capacity.
        let prev = self.pending.fetch_add(1, Ordering::AcqRel);
        if prev >= self.max_pending {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            return Err(AdmissionRejection::QueueFull {
                retry_after_secs: self.retry_hint(),
            });
        }

        match tokio::time::timeout(self.max_wait, Arc::clone(&self.slots).acquire_owned()).await {
            Ok(Ok(permit)) => Ok(AdmissionPermit {
                _permit: permit,
                pending: Arc::clone(&self.pending),
            }),
            // Semaphore is never closed; treat a closed/elapsed wait the same.
            Ok(Err(_)) | Err(_) => {
                self.pending.fetch_sub(1, Ordering::AcqRel);
                Err(AdmissionRejection::Timeout {
                    retry_after_secs: self.retry_hint(),
                })
            }
        }
    }

    /// Requests currently running (holding an in-flight slot).
    pub fn in_flight(&self) -> usize {
        self.max_in_flight
            .saturating_sub(self.slots.available_permits())
    }

    /// Requests waiting for an in-flight slot.
    pub fn queue_depth(&self) -> usize {
        self.pending
            .load(Ordering::Acquire)
            .saturating_sub(self.in_flight())
    }

    /// Rough `Retry-After`: scale with how backed-up the model is, clamped to
    /// a sane band. Without per-request timing this is a heuristic, but it
    /// gives well-behaved clients (opencode/AI SDK) a sensible backoff.
    fn retry_hint(&self) -> u64 {
        ((self.queue_depth() as u64 + 1) * 2).clamp(1, 120)
    }
}

/// Held for a request's lifetime; frees the in-flight + queue slot on drop.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
    pending: Arc<AtomicUsize>,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_in_flight: usize, max_queue_depth: usize, max_wait_secs: u64) -> AdmissionConfig {
        AdmissionConfig {
            max_in_flight,
            max_queue_depth,
            max_wait_secs,
        }
    }

    #[tokio::test]
    async fn admits_up_to_in_flight_and_reports_load() {
        let ctrl = AdmissionController::new(&cfg(1, 4, 30));
        assert_eq!(ctrl.in_flight(), 0);
        let p = ctrl.enter().await.expect("first admits");
        assert_eq!(ctrl.in_flight(), 1);
        assert_eq!(ctrl.queue_depth(), 0);
        drop(p);
        assert_eq!(ctrl.in_flight(), 0);
    }

    #[tokio::test]
    async fn rejects_when_queue_full() {
        // 1 in-flight + 1 queue slot = capacity 2; the 3rd is refused fast.
        let ctrl = Arc::new(AdmissionController::new(&cfg(1, 1, 30)));
        let _running = ctrl.enter().await.expect("admit running");

        // Fill the single queue slot with a waiter that parks on the semaphore.
        let ctrl2 = Arc::clone(&ctrl);
        let waiter = tokio::spawn(async move { ctrl2.enter().await.map(|p| drop(p)) });
        // Give the waiter a moment to occupy the queue slot.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 1);

        // Queue full → immediate QueueFull with a Retry-After hint.
        match ctrl.enter().await {
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
        let _running = ctrl.enter().await.expect("admit running");
        // max_wait 0 → the queued request times out almost immediately.
        match ctrl.enter().await {
            Err(AdmissionRejection::Timeout { .. }) => {}
            other => panic!("expected Timeout, got {other:?}"),
        }
        // The timed-out request released its queue slot.
        assert_eq!(ctrl.queue_depth(), 0);
    }
}
