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
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Enough KV budget never freed within `kv_max_wait` (#257). Transient:
    /// the in-flight sequences holding it will finish.
    KvTimeout {
        retry_after_secs: u64,
        required_mb: u64,
        budget_mb: u64,
    },
    /// The request's KV reservation exceeds the model's *entire* KV budget,
    /// so no amount of waiting can admit it (#257). Permanent for this
    /// prompt on this node — the caller must shorten it.
    KvUnservable { required_mb: u64, budget_mb: u64 },
}

impl AdmissionRejection {
    pub fn retry_after_secs(&self) -> u64 {
        match self {
            AdmissionRejection::QueueFull { retry_after_secs }
            | AdmissionRejection::Timeout { retry_after_secs }
            | AdmissionRejection::PrincipalCap { retry_after_secs }
            | AdmissionRejection::KvTimeout {
                retry_after_secs, ..
            } => *retry_after_secs,
            // Waiting cannot help; advertise no retry.
            AdmissionRejection::KvUnservable { .. } => 0,
        }
    }
}

/// Monotonic per-reason rejection tallies (#137), counted since this
/// controller was created (i.e. since the model last loaded). Lock-free so
/// `/health` can read them without contending with the admission path.
#[derive(Default)]
struct RejectionCounters {
    queue_full: AtomicU64,
    timeout: AtomicU64,
    per_principal: AtomicU64,
    kv_timeout: AtomicU64,
    kv_unservable: AtomicU64,
}

/// Snapshot of [`RejectionCounters`] for the `/health` payload — the
/// definitive "this model is shedding load" signal (#137). Cumulative since
/// load; cortex publishes each as a Prometheus counter.
#[derive(Debug, Clone, Copy, Default)]
pub struct RejectionCounts {
    pub queue_full: u64,
    pub timeout: u64,
    pub per_principal: u64,
    /// KV budget never freed in time (#257).
    pub kv_timeout: u64,
    /// Prompt too large for this model's whole KV budget (#257).
    pub kv_unservable: u64,
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
    /// KV budget in MiB, one permit per MiB (#257). Starts empty and is
    /// filled once by [`set_kv_budget_mb`](Self::set_kv_budget_mb) after the
    /// model's weights are resident; `kv_budget_mb == 0` means the gate is
    /// disabled (CPU loads, or an arch with no captured context profile).
    kv_budget: Arc<Semaphore>,
    kv_budget_mb: AtomicU64,
    /// Queued + in-flight accounting (overall + per principal).
    state: Arc<Mutex<AdmissionState>>,
    /// `max_in_flight + max_queue_depth` — the overall rejection threshold.
    max_pending: usize,
    /// Max in-flight + queued for any single principal (#54). `0` disables.
    max_per_principal: usize,
    max_in_flight: usize,
    max_wait: Duration,
    kv_max_wait: Duration,
    rejections: RejectionCounters,
}

impl AdmissionController {
    pub fn new(cfg: &AdmissionConfig) -> Self {
        // A controller with zero in-flight slots would deadlock; clamp.
        let max_in_flight = cfg.max_in_flight.max(1);
        Self {
            slots: Arc::new(Semaphore::new(max_in_flight)),
            kv_budget: Arc::new(Semaphore::new(0)),
            kv_budget_mb: AtomicU64::new(0),
            state: Arc::new(Mutex::new(AdmissionState::default())),
            max_pending: max_in_flight + cfg.max_queue_depth,
            max_per_principal: cfg.max_per_principal,
            max_in_flight,
            max_wait: Duration::from_secs(cfg.max_wait_secs),
            kv_max_wait: Duration::from_secs(cfg.kv_max_wait_secs),
            rejections: RejectionCounters::default(),
        }
    }

    /// Publish this model's KV budget, once, after its weights are resident
    /// and before it serves (#257).
    ///
    /// The budget is measured with **nothing in flight**, which is the whole
    /// point: free VRAM read at any later moment already excludes the KV of
    /// running sequences, so deriving the budget from a live reading would
    /// double-count their reservations and progressively starve the model.
    ///
    /// Idempotent in the sense that matters — calling it twice would add
    /// permits twice, so it refuses after the first call rather than
    /// silently inflating the budget.
    pub fn set_kv_budget_mb(&self, budget_mb: u64) {
        if self.kv_budget_mb.load(Ordering::Acquire) != 0 || budget_mb == 0 {
            return;
        }
        self.kv_budget_mb.store(budget_mb, Ordering::Release);
        self.kv_budget.add_permits(budget_mb as usize);
    }

    /// Stop admitting and wake every waiter, for shutdown (#256).
    ///
    /// Closing the semaphores makes pending `acquire` calls return `Err`
    /// immediately, which `enter_with_kv` already treats as a rejection —
    /// so queued requests fail fast with a retryable signal instead of
    /// waiting out `max_wait` while the process is trying to exit.
    ///
    /// Without this the drain waits for work that will never be admitted:
    /// on 2026-08-15 a queued request sat through the whole shutdown and
    /// systemd SIGKILLed at `TimeoutStopSec`. Requests already running are
    /// untouched — they hold their permits and either finish or die with
    /// the process, which is the drain's business, not admission's.
    pub fn close(&self) {
        self.slots.close();
        self.kv_budget.close();
    }

    /// This model's total KV budget in MiB; `0` when the gate is disabled.
    pub fn kv_budget_mb(&self) -> u64 {
        self.kv_budget_mb.load(Ordering::Acquire)
    }

    /// KV budget currently unreserved, in MiB.
    pub fn kv_available_mb(&self) -> u64 {
        self.kv_budget.available_permits() as u64
    }

    /// Admit a request for `principal` (`None` = anonymous, exempt from the
    /// per-principal cap). Reserves a queue slot — fast-rejecting if the
    /// overall queue is full or the principal is over its fair-share cap —
    /// then waits up to `max_wait` for an in-flight slot. The returned permit
    /// must be held for the request's lifetime; dropping it frees the slots.
    ///
    /// CANCELLATION SAFETY: the semaphore wait below is where a client
    /// disconnect lands — axum drops the request future mid-await. The
    /// reservation therefore lives in a RAII [`PendingReservation`] taken
    /// BEFORE the await: if this future is dropped while queued, the
    /// guard's Drop rolls the counts back. (The original version
    /// incremented raw counters and only decremented on the timeout
    /// branch — every abandoned wait leaked a `pending` + per-principal
    /// slot, ratcheting the model into a permanent instant-429 state
    /// under client retry storms. Observed live 2026-07-02:
    /// `queue_depth: 1` pinned on an idle model.)
    /// Admit with no KV reservation — the gate is skipped entirely. Used by
    /// paths with no context profile to price a sequence with (image
    /// generation, CPU loads).
    pub async fn enter(
        &self,
        principal: Option<&str>,
    ) -> Result<AdmissionPermit, AdmissionRejection> {
        self.enter_with_kv(principal, 0).await
    }

    /// Admit a request that will hold `kv_mb` MiB of KV cache for its
    /// lifetime (#257).
    ///
    /// Two gates, and the **order between them is load-bearing**: the KV
    /// budget is taken first, then the in-flight slot. Taking the slot first
    /// deadlocks — every slot could be held by a request waiting for KV
    /// while the KV is held by requests waiting for a slot, and nothing
    /// would ever run to release either. Acquiring KV first means a waiter
    /// holds nothing anyone else needs, and the requests that do hold both
    /// are, by construction, running.
    ///
    /// tokio's semaphore is FIFO, so a large reservation queues fairly
    /// rather than being starved by a stream of small ones behind it.
    pub async fn enter_with_kv(
        &self,
        principal: Option<&str>,
        kv_mb: u64,
    ) -> Result<AdmissionPermit, AdmissionRejection> {
        // Decision + reservation under one brief lock so concurrent callers
        // can't both slip past the thresholds. No await is held here.
        let reservation = {
            let mut st = self.state.lock().expect("admission state poisoned");
            if st.pending >= self.max_pending {
                self.rejections.queue_full.fetch_add(1, Ordering::Relaxed);
                return Err(AdmissionRejection::QueueFull {
                    retry_after_secs: self.retry_hint(st.pending),
                });
            }
            if let Some(p) = principal
                && self.max_per_principal > 0
                && st.per_principal.get(p).copied().unwrap_or(0) >= self.max_per_principal
            {
                self.rejections
                    .per_principal
                    .fetch_add(1, Ordering::Relaxed);
                return Err(AdmissionRejection::PrincipalCap {
                    retry_after_secs: self.retry_hint(st.pending),
                });
            }
            st.pending += 1;
            if let Some(p) = principal {
                *st.per_principal.entry(p.to_string()).or_insert(0) += 1;
            }
            PendingReservation {
                state: Arc::clone(&self.state),
                principal: principal.map(str::to_string),
            }
        };

        // Gate 1 — KV budget. Skipped when the budget is unset (CPU / no
        // context profile) or the caller priced the request at zero.
        let budget_mb = self.kv_budget_mb.load(Ordering::Acquire);
        let kv_permit = if kv_mb > 0 && budget_mb > 0 {
            if kv_mb > budget_mb {
                // No amount of waiting frees more than the whole budget.
                // Rejecting immediately is the honest answer, and the only
                // one that doesn't hang the caller until its deadline.
                self.rejections
                    .kv_unservable
                    .fetch_add(1, Ordering::Relaxed);
                return Err(AdmissionRejection::KvUnservable {
                    required_mb: kv_mb,
                    budget_mb,
                });
            }
            match tokio::time::timeout(
                self.kv_max_wait,
                Arc::clone(&self.kv_budget).acquire_many_owned(kv_mb as u32),
            )
            .await
            {
                Ok(Ok(p)) => Some(p),
                Ok(Err(_)) | Err(_) => {
                    self.rejections.kv_timeout.fetch_add(1, Ordering::Relaxed);
                    return Err(AdmissionRejection::KvTimeout {
                        retry_after_secs: self.retry_hint(self.max_pending),
                        required_mb: kv_mb,
                        budget_mb,
                    });
                }
            }
        } else {
            None
        };

        // Gate 2 — in-flight slot.
        match tokio::time::timeout(self.max_wait, Arc::clone(&self.slots).acquire_owned()).await {
            Ok(Ok(permit)) => Ok(AdmissionPermit {
                _permit: permit,
                _kv_permit: kv_permit,
                _reservation: reservation,
            }),
            // Semaphore is never closed; treat a closed/elapsed wait the
            // same. `reservation` and the KV permit drop here, rolling back
            // the counts and returning the reservation to the budget.
            Ok(Err(_)) | Err(_) => {
                self.rejections.timeout.fetch_add(1, Ordering::Relaxed);
                Err(AdmissionRejection::Timeout {
                    retry_after_secs: self.retry_hint(self.max_pending),
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
        let pending = self.state.lock().expect("admission state poisoned").pending;
        pending.saturating_sub(self.in_flight())
    }

    /// Configured concurrency ceiling (#137) — the saturation denominator
    /// (`in_flight / max_in_flight`). Reflects the clamped value (min 1).
    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Configured admission queue capacity (#137): waiters allowed beyond the
    /// in-flight slots before the model sheds load. Derived from `max_pending`
    /// (`max_in_flight + max_queue_depth`) so it stays consistent with the
    /// rejection threshold.
    pub fn max_queue_depth(&self) -> usize {
        self.max_pending.saturating_sub(self.max_in_flight)
    }

    /// Cumulative per-reason rejection tally (#137) since this model loaded —
    /// the load-shedding signal. Lock-free.
    pub fn rejections(&self) -> RejectionCounts {
        RejectionCounts {
            queue_full: self.rejections.queue_full.load(Ordering::Relaxed),
            timeout: self.rejections.timeout.load(Ordering::Relaxed),
            per_principal: self.rejections.per_principal.load(Ordering::Relaxed),
            kv_timeout: self.rejections.kv_timeout.load(Ordering::Relaxed),
            kv_unservable: self.rejections.kv_unservable.load(Ordering::Relaxed),
        }
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

/// RAII accounting for one reserved slot (queued or in-flight): decrements
/// `pending` and the principal's fair-share count on drop, whichever way
/// the reservation ends — admitted-and-finished, wait timeout, or the
/// caller's future being dropped mid-queue (client disconnect).
#[derive(Debug)]
struct PendingReservation {
    state: Arc<Mutex<AdmissionState>>,
    principal: Option<String>,
}

impl Drop for PendingReservation {
    fn drop(&mut self) {
        let mut st = self.state.lock().expect("admission state poisoned");
        st.pending = st.pending.saturating_sub(1);
        decrement_principal(&mut st.per_principal, self.principal.as_deref());
    }
}

/// Held for a request's lifetime; frees the in-flight slot (semaphore
/// permit) and the queue + fair-share accounting (reservation) on drop.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
    /// KV budget reservation (#257); `None` when the gate is disabled.
    /// Dropping it returns the MiB to the model's budget, which is what
    /// lets a queued long-context request proceed.
    _kv_permit: Option<OwnedSemaphorePermit>,
    _reservation: PendingReservation,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #256: closing admission must wake queued waiters immediately, so a
    /// drain has a bounded amount of work to wait for.
    ///
    /// Before this, a request queued behind a busy model kept waiting
    /// through the whole shutdown; axum's graceful drain waits for it, and
    /// systemd SIGKILLed at TimeoutStopSec.
    #[tokio::test]
    async fn close_wakes_queued_waiters() {
        let ctrl = Arc::new(AdmissionController::new(&kv_cfg(1, 4, 30)));
        let _running = ctrl.enter(None).await.expect("first admits");

        let ctrl2 = Arc::clone(&ctrl);
        let waiter = tokio::spawn(async move { ctrl2.enter(None).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "precondition: the second is queued");

        ctrl.close();
        let outcome = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("closing must wake the waiter, not leave it queued")
            .expect("join");
        assert!(
            outcome.is_err(),
            "a waiter woken by close must be refused, not admitted into a dying process"
        );
    }

    /// Closing also refuses callers that arrive afterwards — the listener
    /// may still accept for a moment while the drain proceeds.
    #[tokio::test]
    async fn close_refuses_later_arrivals() {
        let ctrl = AdmissionController::new(&kv_cfg(4, 8, 30));
        ctrl.close();
        assert!(
            ctrl.enter(None).await.is_err(),
            "a closed controller must admit nothing"
        );
    }

    /// The KV gate is closed too — a request waiting on budget is just as
    /// stuck as one waiting on a slot.
    #[tokio::test]
    async fn close_wakes_kv_budget_waiters() {
        let ctrl = Arc::new(AdmissionController::new(&kv_cfg(4, 8, 600)));
        ctrl.set_kv_budget_mb(1_000);
        let _held = ctrl.enter_with_kv(None, 800).await.expect("admits");

        let ctrl2 = Arc::clone(&ctrl);
        let waiter = tokio::spawn(async move { ctrl2.enter_with_kv(None, 800).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "precondition: queued on KV budget");

        ctrl.close();
        assert!(
            tokio::time::timeout(Duration::from_secs(2), waiter)
                .await
                .expect("closing must wake a KV waiter too")
                .expect("join")
                .is_err(),
            "a KV waiter woken by close must be refused"
        );
    }

    /// Config with a KV wait long enough that the gate queues rather than
    /// times out, for the tests that assert queueing.
    fn kv_cfg(
        max_in_flight: usize,
        max_queue_depth: usize,
        kv_max_wait_secs: u64,
    ) -> AdmissionConfig {
        AdmissionConfig {
            max_in_flight,
            max_queue_depth,
            max_wait_secs: 30,
            max_per_principal: 0,
            kv_max_wait_secs,
        }
    }

    /// The regression #257 was filed for: a request whose KV does not fit
    /// *yet* must WAIT for a running sequence to return its budget, not be
    /// rejected while the queue sits empty. Before this, the pre-admission
    /// VRAM backstop killed a 30-minute agentic session outright.
    #[tokio::test]
    async fn oversized_request_waits_for_budget_instead_of_being_rejected() {
        let ctrl = Arc::new(AdmissionController::new(&kv_cfg(4, 8, 30)));
        ctrl.set_kv_budget_mb(3_000);

        // Two 1200 MiB sequences resident → 600 MiB left.
        let a = ctrl.enter_with_kv(None, 1_200).await.expect("a admits");
        let _b = ctrl.enter_with_kv(None, 1_200).await.expect("b admits");
        assert_eq!(ctrl.kv_available_mb(), 600);

        // A third 1200 MiB request cannot fit yet.
        let ctrl2 = Arc::clone(&ctrl);
        let waiter = tokio::spawn(async move { ctrl2.enter_with_kv(None, 1_200).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "must queue, not reject");

        // Freeing one admits it.
        drop(a);
        assert!(
            waiter.await.expect("join").is_ok(),
            "queued request must be admitted once budget frees"
        );
    }

    /// A reservation larger than the entire budget can never be satisfied,
    /// so it must fail fast rather than hang until the deadline.
    #[tokio::test]
    async fn request_larger_than_whole_budget_fails_fast() {
        let ctrl = AdmissionController::new(&kv_cfg(4, 8, 600));
        ctrl.set_kv_budget_mb(2_000);
        let started = std::time::Instant::now();
        match ctrl.enter_with_kv(None, 5_000).await {
            Err(AdmissionRejection::KvUnservable {
                required_mb,
                budget_mb,
            }) => {
                assert_eq!((required_mb, budget_mb), (5_000, 2_000));
            }
            other => panic!("expected KvUnservable, got {other:?}"),
        }
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "must not wait out the deadline for something that can never fit"
        );
        assert_eq!(ctrl.rejections().kv_unservable, 1);
    }

    /// Dropping the permit returns the reservation, so the budget does not
    /// leak across requests.
    #[tokio::test]
    async fn budget_is_returned_on_drop() {
        let ctrl = AdmissionController::new(&kv_cfg(4, 8, 30));
        ctrl.set_kv_budget_mb(1_000);
        assert_eq!(ctrl.kv_available_mb(), 1_000);
        {
            let _p = ctrl.enter_with_kv(None, 400).await.expect("admits");
            assert_eq!(ctrl.kv_available_mb(), 600);
        }
        assert_eq!(ctrl.kv_available_mb(), 1_000);
    }

    /// An unset budget (CPU load, or an arch with no context profile) leaves
    /// the gate disabled — every request passes it regardless of size.
    #[tokio::test]
    async fn unset_budget_disables_the_gate() {
        let ctrl = AdmissionController::new(&kv_cfg(2, 4, 30));
        assert_eq!(ctrl.kv_budget_mb(), 0);
        let _a = ctrl.enter_with_kv(None, 99_999).await.expect("admits");
        let _b = ctrl.enter_with_kv(None, 99_999).await.expect("admits");
    }

    /// The budget is published once. A second call must not inflate it —
    /// a re-seeded budget would over-admit and OOM the card.
    #[tokio::test]
    async fn budget_is_published_once() {
        let ctrl = AdmissionController::new(&kv_cfg(2, 4, 30));
        ctrl.set_kv_budget_mb(1_000);
        ctrl.set_kv_budget_mb(1_000);
        assert_eq!(ctrl.kv_budget_mb(), 1_000);
        assert_eq!(ctrl.kv_available_mb(), 1_000);
    }

    /// Waiting for budget that never frees ends in a bounded, retryable
    /// rejection rather than an unbounded hang.
    #[tokio::test]
    async fn kv_wait_times_out() {
        let ctrl = AdmissionController::new(&kv_cfg(4, 8, 0));
        ctrl.set_kv_budget_mb(1_000);
        let _held = ctrl.enter_with_kv(None, 800).await.expect("admits");
        match ctrl.enter_with_kv(None, 800).await {
            Err(AdmissionRejection::KvTimeout {
                required_mb,
                budget_mb,
                ..
            }) => assert_eq!((required_mb, budget_mb), (800, 1_000)),
            other => panic!("expected KvTimeout, got {other:?}"),
        }
        assert_eq!(ctrl.rejections().kv_timeout, 1);
    }

    /// Config with the per-principal cap disabled (0) — most tests exercise
    /// the overall queue with anonymous (`None`) callers.
    fn cfg(max_in_flight: usize, max_queue_depth: usize, max_wait_secs: u64) -> AdmissionConfig {
        AdmissionConfig {
            max_in_flight,
            max_queue_depth,
            max_wait_secs,
            max_per_principal: 0,
            kv_max_wait_secs: 30,
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

        // #137: the refused request is tallied under queue_full, and only
        // there — the admitted ones don't touch the counters.
        let rej = ctrl.rejections();
        assert_eq!(rej.queue_full, 1);
        assert_eq!(rej.timeout, 0);
        assert_eq!(rej.per_principal, 0);
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
            kv_max_wait_secs: 30,
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

    /// Regression for the 2026-07-02 retry-storm incident: a client that
    /// disconnects while QUEUED drops the `enter()` future mid-await.
    /// The reservation must roll back — the original implementation
    /// leaked `pending` + the per-principal count on this path, pinning
    /// the model in a permanent instant-429 state.
    #[tokio::test]
    async fn cancelled_queued_waiter_rolls_back_accounting() {
        let cfg = AdmissionConfig {
            max_in_flight: 1,
            max_queue_depth: 2,
            max_wait_secs: 30,
            // Cap 3 lets the runner + both waiters coexist; if the two
            // cancelled waiters leaked their counts, the principal would
            // sit at 3 == cap and the post-cancel enter below would hit
            // PrincipalCap instead of queueing.
            max_per_principal: 3,
            kv_max_wait_secs: 30,
        };
        let ctrl = Arc::new(AdmissionController::new(&cfg));
        let running = ctrl.enter(Some("acct/key")).await.expect("admit running");

        // Two waiters from the same principal park in the queue…
        let mut waiters = Vec::new();
        for _ in 0..2 {
            let c = Arc::clone(&ctrl);
            waiters.push(tokio::spawn(async move {
                c.enter(Some("acct/key")).await.map(drop)
            }));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 2);

        // …and both clients vanish (abort = the dropped request future).
        for w in &waiters {
            w.abort();
        }
        for w in waiters {
            let _ = w.await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(
            ctrl.queue_depth(),
            0,
            "cancelled waiters must not leak queue slots"
        );

        // The principal's fair-share count must also be clean: with the
        // runner still holding 1 of its cap of 3, a new request from the
        // same principal queues instead of hitting PrincipalCap (which a
        // leak of the two cancelled counts would trigger).
        let c = Arc::clone(&ctrl);
        let retry = tokio::spawn(async move { c.enter(Some("acct/key")).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 1, "post-cancel request queues normally");
        drop(running);
        retry
            .await
            .unwrap()
            .expect("post-cancel request is served — no leaked principal count");
    }
}
