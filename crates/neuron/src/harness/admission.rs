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
//! ## Two classes: identified, and whatever is left (#262)
//!
//! Fair-share ([`AdmissionConfig::max_per_principal`]) bounds what one
//! *identified* caller may hold. A caller with no principal cannot be
//! bounded by it — cortex stamps the `x-helexa-*` headers only after a
//! bearer resolves, and with `require_auth = false` both a missing and an
//! unrecognised key proceed unstamped. So the callers we know least about
//! were the only ones nothing could cap. On 2026-08-16 a credential-less
//! batch classifier at concurrency 4 held every one of beast's 8 seats
//! while an interactive session waited out `max_wait` and was abandoned;
//! `rejected_per_principal: 0` recorded that the cap never fired, because
//! it was never eligible to.
//!
//! Anonymous traffic is therefore served from **capacity left over once
//! identified traffic is satisfied**, enforced at three points:
//!
//! 1. **Priority.** An anonymous request never takes a seat while any
//!    identified request is waiting for one — regardless of who arrived
//!    first. It waits at the class gate below, ahead of every other gate,
//!    so it holds no KV budget and no seat while it yields.
//! 2. **A seat ceiling** (`anon_max_in_flight`). Priority alone is not
//!    enough: a request already running cannot be preempted, so without a
//!    ceiling an anonymous burst arriving during an idle moment still locks
//!    an identified caller out for the full duration of every one of them.
//!    Holding back one seat bounds that wait to a single request.
//! 3. **A queue ceiling** (`anon_max_pending`). The queue needs the same
//!    reservation, or an anonymous flood refuses identified callers at the
//!    door — before priority gets a chance to favour them.
//!
//! Anonymous callers still contend with each other, and under sustained
//! identified load they are refused rather than served slowly: that is the
//! intent, and it is why the refusal is a retryable `503` + `Retry-After`
//! rather than a stall. What the mechanism cannot do is reach back into a
//! request that is already running — hence (2).
//!
//! The controller is pure async (no CUDA), so the inference paths just call
//! [`AdmissionController::enter`] before taking the inference lock and hold
//! the returned [`AdmissionPermit`] for the request's lifetime. Its counters
//! ([`in_flight`](AdmissionController::in_flight) /
//! [`queue_depth`](AdmissionController::queue_depth)) are lock-free, so
//! `/health` can read live load without contending with inference.

use crate::config::AdmissionConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

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
    /// An anonymous request found no capacity left over after identified
    /// traffic (#262): identified callers were waiting for the whole of
    /// `max_wait`, or anonymous traffic was already at its seat/queue
    /// ceiling. Server-side load like [`QueueFull`](Self::QueueFull) — the
    /// same request from an authenticated caller may well be admitted.
    AnonYield { retry_after_secs: u64 },
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
            | AdmissionRejection::AnonYield { retry_after_secs }
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
    anon_yield: AtomicU64,
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
    /// Anonymous requests refused for want of leftover capacity (#262).
    /// Distinguishes "we shed load" from "we shed *unidentified* load" —
    /// a rising count here with a flat `timeout` means the reservation is
    /// doing its job, not that the model is overloaded.
    pub anon_yield: u64,
    /// KV budget never freed in time (#257).
    pub kv_timeout: u64,
    /// Prompt too large for this model's whole KV budget (#257).
    pub kv_unservable: u64,
}

/// Which resource an opportunistic (anonymous) claim failed on (#262).
/// Decides both how long the request keeps waiting and what it is told
/// when it stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Blocked {
    /// Not enough KV budget free right now (#257 territory: minutes).
    Kv,
    /// No seat free right now (seconds).
    Slot,
}

/// Which class a request belongs to for the leftover-capacity rule (#262).
/// Derived purely from whether cortex resolved a principal for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    /// Cortex stamped an account/key — capped by fair-share, first in line.
    Identified,
    /// No principal: no credential, or one that didn't resolve (#49).
    Anonymous,
}

/// Admission accounting, mutated under a brief lock (never held across an
/// await). `pending` is queued + in-flight overall; `per_principal` is the
/// same count keyed by principal for fair-share (#54).
#[derive(Default, Debug)]
struct AdmissionState {
    pending: usize,
    per_principal: HashMap<String, usize>,
    /// Anonymous requests queued + holding a seat (#262).
    anon_pending: usize,
    /// Anonymous requests holding a seat reservation — claimed at the class
    /// gate rather than on admission, so two anonymous callers can't both
    /// pass a ceiling check and then both take the last seat.
    anon_seats: usize,
    /// Identified requests that have reserved but not yet been admitted.
    /// While this is non-zero, anonymous requests wait (#262). It counts
    /// waiting on *any* gate, KV included, so an identified caller queued
    /// behind a long-context sequence still outranks anonymous traffic.
    identified_waiting: usize,
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
    /// Seats anonymous traffic may hold at once (#262); `0` refuses it.
    anon_max_in_flight: usize,
    /// Anonymous queued + seated (#262).
    anon_max_pending: usize,
    /// Woken whenever the class gate's answer might have changed: an
    /// identified request stopped waiting, or an anonymous seat freed.
    gate: Arc<Notify>,
    /// Set by [`close`](Self::close) so anonymous requests parked at the
    /// class gate fail fast on shutdown instead of waiting out `max_wait`
    /// (#256) — closing the semaphores can't reach them, they aren't on one.
    closed: AtomicBool,
    max_wait: Duration,
    kv_max_wait: Duration,
    rejections: RejectionCounters,
}

impl AdmissionController {
    pub fn new(cfg: &AdmissionConfig) -> Self {
        // A controller with zero in-flight slots would deadlock; clamp.
        let max_in_flight = cfg.max_in_flight.max(1);
        let max_pending = max_in_flight + cfg.max_queue_depth;
        // Hold back one seat and one queue place for identified callers
        // (#262). Floored at 1 rather than 0: on a single-seat model the
        // reservation would otherwise refuse anonymous traffic outright,
        // which is a policy decision, not a default — an operator who wants
        // it says `anon_max_in_flight = 0`. Priority still applies there,
        // so anonymous simply loses every race it would have won.
        let anon_max_in_flight = cfg
            .anon_max_in_flight
            .unwrap_or_else(|| max_in_flight.saturating_sub(1).max(1));
        let anon_max_pending = cfg
            .anon_max_pending
            .unwrap_or_else(|| max_pending.saturating_sub(1).max(1));
        Self {
            slots: Arc::new(Semaphore::new(max_in_flight)),
            kv_budget: Arc::new(Semaphore::new(0)),
            kv_budget_mb: AtomicU64::new(0),
            state: Arc::new(Mutex::new(AdmissionState::default())),
            max_pending,
            max_per_principal: cfg.max_per_principal,
            max_in_flight,
            anon_max_in_flight: anon_max_in_flight.min(max_in_flight),
            anon_max_pending: anon_max_pending.min(max_pending),
            gate: Arc::new(Notify::new()),
            closed: AtomicBool::new(false),
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
        // Anonymous requests yielding at the class gate (#262) are on
        // neither semaphore, so closing those cannot reach them. Flag and
        // wake them explicitly or they wait out `max_wait` while the
        // process tries to exit — the exact stall #256 removed.
        self.closed.store(true, Ordering::Release);
        self.gate.notify_waiters();
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
        let class = match principal {
            Some(_) => Class::Identified,
            None => Class::Anonymous,
        };

        // Decision + reservation under one brief lock so concurrent callers
        // can't both slip past the thresholds. No await is held here.
        let mut reservation = {
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
            // Anonymous queue ceiling (#262): refuse at the door rather than
            // let unidentified traffic occupy the places that keep an
            // identified caller from being refused at the door.
            if class == Class::Anonymous
                && (self.anon_max_in_flight == 0 || st.anon_pending >= self.anon_max_pending)
            {
                self.rejections.anon_yield.fetch_add(1, Ordering::Relaxed);
                return Err(AdmissionRejection::AnonYield {
                    retry_after_secs: self.retry_hint(st.pending),
                });
            }
            st.pending += 1;
            if let Some(p) = principal {
                *st.per_principal.entry(p.to_string()).or_insert(0) += 1;
            }
            match class {
                Class::Anonymous => st.anon_pending += 1,
                Class::Identified => st.identified_waiting += 1,
            }
            PendingReservation {
                state: Arc::clone(&self.state),
                gate: Arc::clone(&self.gate),
                principal: principal.map(str::to_string),
                class,
                waiting: true,
                anon_seat: false,
            }
        };

        let budget_mb = self.kv_budget_mb.load(Ordering::Acquire);
        if kv_mb > 0 && budget_mb > 0 && kv_mb > budget_mb {
            // No amount of waiting frees more than the whole budget.
            // Rejecting immediately is the honest answer, and the only one
            // that doesn't hang the caller until its deadline. Checked
            // ahead of both paths so an anonymous request isn't sent to
            // park on a gate that can never open for it.
            self.rejections
                .kv_unservable
                .fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionRejection::KvUnservable {
                required_mb: kv_mb,
                budget_mb,
            });
        }

        // Anonymous requests take a different route through the same two
        // resources (#262): leftover-only, never queued. See
        // `await_leftover_capacity` for why they cannot simply queue behind
        // the gates below.
        if class == Class::Anonymous {
            let (permit, kv_permit) = self
                .await_leftover_capacity(&mut reservation, kv_mb, budget_mb)
                .await?;
            reservation.mark_admitted();
            return Ok(AdmissionPermit {
                _permit: permit,
                _kv_permit: kv_permit,
                _reservation: reservation,
            });
        }

        // Gate 1 — KV budget. Skipped when the budget is unset (CPU / no
        // context profile) or the caller priced the request at zero.
        let kv_permit = if kv_mb > 0 && budget_mb > 0 {
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
            Ok(Ok(permit)) => {
                // Admitted: an identified request leaves the waiting set,
                // which may be what anonymous callers are parked on.
                reservation.mark_admitted();
                Ok(AdmissionPermit {
                    _permit: permit,
                    _kv_permit: kv_permit,
                    _reservation: reservation,
                })
            }
            // A closed semaphore (shutdown, #256) and an elapsed wait are
            // treated the same: both mean "not admitted, try later".
            // `reservation` and the KV permit drop here, rolling back the
            // counts and returning the reservation to the budget.
            Ok(Err(_)) | Err(_) => {
                self.rejections.timeout.fetch_add(1, Ordering::Relaxed);
                Err(AdmissionRejection::Timeout {
                    retry_after_secs: self.retry_hint(self.max_pending),
                })
            }
        }
    }

    /// Admit an anonymous request out of whatever capacity identified
    /// traffic has left (#262), returning its seat and KV permits.
    ///
    /// The shape here is the whole point. An anonymous request **never
    /// queues on the slot semaphore**: it takes a seat only when one is
    /// free at that instant, and otherwise parks on the class gate and
    /// re-tests. Queueing would defeat the rule — the semaphore is FIFO and
    /// knows nothing of classes, so an anonymous waiter that got in line
    /// first would be handed the next seat ahead of an identified caller
    /// that arrived later, which is exactly the inversion this exists to
    /// prevent. Parking instead of queueing means the priority test is
    /// re-evaluated on every wakeup rather than once on the way in.
    ///
    /// It also means a yielding request holds **nothing**: no seat, no KV
    /// budget, no queue position. That is what keeps the leftover rule from
    /// deadlocking against the KV gate — see [`enter_with_kv`](Self::enter_with_kv)
    /// on why identified callers must take KV before their seat. An
    /// anonymous request that took its seat first and then waited for KV
    /// could hold the last seat while an identified caller holding the KV
    /// waited for a seat, and neither would ever proceed.
    ///
    /// CANCELLATION SAFETY: a client disconnect lands on the await below,
    /// dropping `reservation` and rolling back its counts. Nothing is held
    /// across that await, so there is nothing else to release.
    async fn await_leftover_capacity(
        &self,
        reservation: &mut PendingReservation,
        kv_mb: u64,
        budget_mb: u64,
    ) -> Result<(OwnedSemaphorePermit, Option<OwnedSemaphorePermit>), AdmissionRejection> {
        // Two deadlines, because the identified path has two: a seat turns
        // over in seconds, whereas KV budget frees only when a whole
        // long-context sequence finishes, which #257 measured in minutes.
        // Which one applies is decided per iteration by what is actually
        // blocking — otherwise an anonymous long-context request would be
        // refused on the seat timeout while it waits for budget, the exact
        // premature rejection #257 removed.
        let seat_deadline = Instant::now() + self.max_wait;
        let kv_deadline = Instant::now() + self.kv_max_wait;
        // What blocked the last attempt, so the refusal names the true
        // cause. Conflating these would make `anon_yield` tick on ordinary
        // overload and stop meaning anything.
        // Deliberately uninitialised: every path through the loop body
        // either returns or sets both, so a placeholder here could only
        // mask a missed case.
        let mut blocked;
        let mut blocked_by_reservation;
        loop {
            // Register on the notify BEFORE testing the state, or a wakeup
            // that fires between the test and the await is lost and this
            // request sleeps to its deadline while capacity sits free.
            let notified = self.gate.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            {
                let mut st = self.state.lock().expect("admission state poisoned");
                if self.closed.load(Ordering::Acquire) {
                    // Shutting down (#256): fail fast, exactly as a closed
                    // semaphore does for the callers queued on one.
                    self.rejections.timeout.fetch_add(1, Ordering::Relaxed);
                    return Err(AdmissionRejection::Timeout {
                        retry_after_secs: self.retry_hint(self.max_pending),
                    });
                }
                if st.identified_waiting > 0 || st.anon_seats >= self.anon_max_in_flight {
                    // Held back by the rule — but only call it a yield if
                    // there was capacity to be held back FROM. A full model
                    // would have refused this request either way, and that
                    // is ordinary load, not a reservation effect.
                    blocked_by_reservation = self.slots.available_permits() > 0;
                    blocked = Blocked::Slot;
                } else {
                    match self.try_claim(kv_mb, budget_mb) {
                        // Claim the seat under the same lock that cleared
                        // the check: two anonymous callers must not both
                        // see the last seat free and both take it.
                        Ok(permits) => {
                            st.anon_seats += 1;
                            reservation.anon_seat = true;
                            return Ok(permits);
                        }
                        Err(why) => {
                            blocked = why;
                            blocked_by_reservation = false;
                        }
                    }
                }
            }

            // Wait only as long as the thing currently blocking us
            // deserves.
            let deadline = match blocked {
                Blocked::Kv => kv_deadline,
                Blocked::Slot => seat_deadline,
            };
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(self.refuse_anonymous(
                    blocked,
                    blocked_by_reservation,
                    kv_mb,
                    budget_mb,
                ));
            }
        }
    }

    /// Name the reason an anonymous request ran out of patience, so the
    /// counters stay diagnostic and the caller gets the same explanation an
    /// identified caller would have got in the same circumstances.
    fn refuse_anonymous(
        &self,
        blocked: Blocked,
        by_reservation: bool,
        kv_mb: u64,
        budget_mb: u64,
    ) -> AdmissionRejection {
        if by_reservation {
            self.rejections.anon_yield.fetch_add(1, Ordering::Relaxed);
            return AdmissionRejection::AnonYield {
                retry_after_secs: self.retry_hint(self.max_pending),
            };
        }
        match blocked {
            // Budget never freed: report it as #257 does, with the MiB that
            // explain it, rather than as a generic busy signal.
            Blocked::Kv => {
                self.rejections.kv_timeout.fetch_add(1, Ordering::Relaxed);
                AdmissionRejection::KvTimeout {
                    retry_after_secs: self.retry_hint(self.max_pending),
                    required_mb: kv_mb,
                    budget_mb,
                }
            }
            Blocked::Slot => {
                self.rejections.timeout.fetch_add(1, Ordering::Relaxed);
                AdmissionRejection::Timeout {
                    retry_after_secs: self.retry_hint(self.max_pending),
                }
            }
        }
    }

    /// Take a KV reservation and a seat, or neither, without ever waiting.
    ///
    /// Both or nothing: an anonymous request that held KV budget while
    /// waiting for a seat would be blocking the identified traffic it is
    /// supposed to be yielding to. Returning the budget on a failed seat
    /// grab is what the `Blocked::Slot` arm does implicitly — `kv` drops
    /// there.
    ///
    /// Synchronous by design: the caller holds the state lock across this,
    /// which is sound because no `.await` is involved.
    fn try_claim(
        &self,
        kv_mb: u64,
        budget_mb: u64,
    ) -> Result<(OwnedSemaphorePermit, Option<OwnedSemaphorePermit>), Blocked> {
        let kv = if kv_mb > 0 && budget_mb > 0 {
            match Arc::clone(&self.kv_budget).try_acquire_many_owned(kv_mb as u32) {
                Ok(p) => Some(p),
                Err(_) => return Err(Blocked::Kv),
            }
        } else {
            None
        };
        match Arc::clone(&self.slots).try_acquire_owned() {
            Ok(slot) => Ok((slot, kv)),
            Err(_) => Err(Blocked::Slot),
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

    /// Anonymous requests holding a seat reservation (#262).
    ///
    /// Counts from the class gate, so it includes a request that has been
    /// granted leftover capacity but is still acquiring KV budget or the
    /// seat itself — that is the number the ceiling is enforced against.
    /// Against `in_flight()` it answers "how much of this model's load is
    /// unattributable", which is the signal that says whether the
    /// reservation is earning its keep.
    pub fn anon_in_flight(&self) -> usize {
        self.state
            .lock()
            .expect("admission state poisoned")
            .anon_seats
    }

    /// Seats anonymous traffic may hold at once (#262) — the ceiling
    /// `anon_in_flight()` is measured against. `0` means anonymous traffic
    /// is refused outright.
    pub fn anon_max_in_flight(&self) -> usize {
        self.anon_max_in_flight
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
            anon_yield: self.rejections.anon_yield.load(Ordering::Relaxed),
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
    gate: Arc<Notify>,
    principal: Option<String>,
    class: Class,
    /// Still waiting for a seat. Identified requests count towards
    /// `identified_waiting` while this holds (#262).
    waiting: bool,
    /// Holds one of the anonymous seats, claimed at the class gate.
    anon_seat: bool,
}

impl PendingReservation {
    /// The request has its seat: an identified one leaves the waiting set,
    /// which is what anonymous requests are parked on (#262).
    fn mark_admitted(&mut self) {
        if !self.waiting {
            return;
        }
        {
            let mut st = self.state.lock().expect("admission state poisoned");
            self.waiting = false;
            if self.class == Class::Identified {
                st.identified_waiting = st.identified_waiting.saturating_sub(1);
            }
        }
        self.gate.notify_waiters();
    }
}

impl Drop for PendingReservation {
    fn drop(&mut self) {
        {
            let mut st = self.state.lock().expect("admission state poisoned");
            st.pending = st.pending.saturating_sub(1);
            decrement_principal(&mut st.per_principal, self.principal.as_deref());
            match self.class {
                Class::Anonymous => {
                    st.anon_pending = st.anon_pending.saturating_sub(1);
                    if self.anon_seat {
                        st.anon_seats = st.anon_seats.saturating_sub(1);
                    }
                }
                Class::Identified => {
                    if self.waiting {
                        st.identified_waiting = st.identified_waiting.saturating_sub(1);
                    }
                }
            }
        }
        // Whichever count moved, the class gate's answer may have changed:
        // an identified request stopped waiting, or an anonymous seat came
        // free. Waking here is what turns a finished request into someone
        // else's admission.
        self.gate.notify_waiters();
    }
}

/// Held for a request's lifetime; frees the in-flight slot (semaphore
/// permit) and the queue + fair-share accounting (reservation) on drop.
///
/// FIELD ORDER IS LOAD-BEARING. Rust drops fields in declaration order, so
/// the seat and the KV budget are returned *before* `_reservation` drops —
/// and it is that drop which wakes anonymous requests parked on the class
/// gate (#262). Reordering would wake them to look at resources not yet
/// released, and they would go back to sleep until their deadline with the
/// capacity sitting free.
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
            // These tests drive anonymous callers to exercise the KV and
            // queue gates, so the #262 class ceilings are lifted (clamped
            // to the real maxima in the constructor). The class rule has
            // its own tests below.
            anon_max_in_flight: Some(usize::MAX),
            anon_max_pending: Some(usize::MAX),
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
    ///
    /// The #262 class ceilings are lifted for the same reason: these tests
    /// are about the overall queue, and with the derived default an
    /// anonymous caller could not fill it. Tests of the class rule itself
    /// build their own config.
    fn cfg(max_in_flight: usize, max_queue_depth: usize, max_wait_secs: u64) -> AdmissionConfig {
        AdmissionConfig {
            max_in_flight,
            max_queue_depth,
            max_wait_secs,
            max_per_principal: 0,
            anon_max_in_flight: Some(usize::MAX),
            anon_max_pending: Some(usize::MAX),
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
            anon_max_in_flight: None,
            anon_max_pending: None,
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
            anon_max_in_flight: None,
            anon_max_pending: None,
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

    // ---- #262: anonymous is served from leftover capacity ----------------

    /// Config exercising the class rule with explicit ceilings.
    fn anon_cfg(
        max_in_flight: usize,
        max_queue_depth: usize,
        anon_max_in_flight: usize,
        anon_max_pending: usize,
    ) -> AdmissionConfig {
        AdmissionConfig {
            max_in_flight,
            max_queue_depth,
            max_wait_secs: 30,
            max_per_principal: 0,
            anon_max_in_flight: Some(anon_max_in_flight),
            anon_max_pending: Some(anon_max_pending),
            kv_max_wait_secs: 30,
        }
    }

    /// The incident #262 was filed for, in miniature: a credential-less
    /// batch client takes every seat and an interactive authenticated
    /// caller is left waiting out `max_wait`.
    ///
    /// With the seat ceiling, the anonymous client cannot take the last
    /// one, so the identified caller is admitted immediately rather than
    /// queueing behind work that will run for minutes.
    #[tokio::test]
    async fn anonymous_cannot_take_the_last_seat_from_an_identified_caller() {
        let ctrl = Arc::new(AdmissionController::new(&anon_cfg(4, 8, 3, 8)));

        let _anon: Vec<_> = {
            let mut held = Vec::new();
            for _ in 0..3 {
                held.push(ctrl.enter(None).await.expect("anonymous fills its ceiling"));
            }
            held
        };
        assert_eq!(ctrl.in_flight(), 3);
        assert_eq!(ctrl.anon_in_flight(), 3);

        // A fourth anonymous request is at the ceiling — it must not get the
        // free seat, however idle that seat is.
        let c = Arc::clone(&ctrl);
        let extra = tokio::spawn(async move { c.enter(None).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!extra.is_finished(), "anonymous waits at its ceiling");

        // The identified caller takes the reserved seat with no wait.
        let permit =
            tokio::time::timeout(Duration::from_millis(200), ctrl.enter(Some("acct-a/key-a")))
                .await
                .expect("identified caller must not wait behind anonymous traffic")
                .expect("identified caller is admitted");
        assert_eq!(ctrl.in_flight(), 4);
        drop(permit);
        extra.abort();
    }

    /// Priority, not arrival order: an anonymous request that reaches the
    /// class gate first still yields to an identified caller that is
    /// waiting, and only proceeds once that caller has its seat.
    #[tokio::test]
    async fn anonymous_yields_to_a_waiting_identified_caller() {
        // One seat, so both classes must queue behind the running request.
        let ctrl = Arc::new(AdmissionController::new(&anon_cfg(1, 8, 1, 8)));
        let running = ctrl.enter(Some("acct-run/key")).await.expect("admit");

        // Anonymous queues FIRST.
        let c = Arc::clone(&ctrl);
        let anon = tokio::spawn(async move { c.enter(None).await });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Identified arrives SECOND.
        let c = Arc::clone(&ctrl);
        let ident = tokio::spawn(async move { c.enter(Some("acct-b/key-b")).await });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // The seat frees. FIFO would hand it to the anonymous waiter; the
        // class rule hands it to the identified one.
        drop(running);
        let ident_permit = tokio::time::timeout(Duration::from_millis(500), ident)
            .await
            .expect("identified waiter must be served first")
            .unwrap()
            .expect("identified admitted");
        assert!(!anon.is_finished(), "anonymous still yielding");

        // Once the identified caller is done, the anonymous one proceeds —
        // yielding is a deferral, not a refusal, while capacity returns.
        drop(ident_permit);
        tokio::time::timeout(Duration::from_millis(500), anon)
            .await
            .expect("anonymous proceeds once identified traffic is served")
            .unwrap()
            .expect("anonymous admitted");
    }

    /// An anonymous flood must not refuse identified callers at the door:
    /// the queue carries the same reservation as the seats.
    #[tokio::test]
    async fn anonymous_cannot_fill_the_whole_queue() {
        // 1 seat + 3 queue places = 4 pending; anonymous may hold 3.
        let ctrl = Arc::new(AdmissionController::new(&anon_cfg(1, 3, 1, 3)));
        let _running = ctrl.enter(None).await.expect("anonymous runs");

        let mut waiters = Vec::new();
        for _ in 0..2 {
            let c = Arc::clone(&ctrl);
            waiters.push(tokio::spawn(async move { c.enter(None).await.map(drop) }));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 2);

        // Anonymous is now at its pending ceiling: refused without taking
        // the last queue place.
        match ctrl.enter(None).await {
            Err(AdmissionRejection::AnonYield { retry_after_secs }) => {
                assert!(retry_after_secs >= 1)
            }
            other => panic!("expected AnonYield, got {other:?}"),
        }
        assert_eq!(ctrl.rejections().anon_yield, 1);

        // The identified caller still finds a queue place.
        let c = Arc::clone(&ctrl);
        let ident = tokio::spawn(async move { c.enter(Some("acct/key")).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 3, "identified queued, not refused");
        ident.abort();
        for w in waiters.drain(..) {
            w.abort();
        }
    }

    /// Yielding is bounded, and names its cause honestly.
    ///
    /// The setup is the one case where the reservation is unambiguously
    /// what refuses the request: a seat is free, no identified caller
    /// wants it, and the anonymous ceiling is the only thing in the way.
    /// The request waits its `max_wait` and is then refused `AnonYield` —
    /// a retryable signal, not a stall, and counted separately from
    /// ordinary overload.
    #[tokio::test(start_paused = true)]
    async fn anonymous_refused_when_the_ceiling_holds_back_a_free_seat() {
        let cfg = AdmissionConfig {
            max_in_flight: 2,
            max_queue_depth: 4,
            max_wait_secs: 1,
            max_per_principal: 0,
            // One of the two seats is anonymous's; the other is reserved.
            anon_max_in_flight: Some(1),
            anon_max_pending: Some(4),
            kv_max_wait_secs: 30,
        };
        let ctrl = Arc::new(AdmissionController::new(&cfg));

        let _running = ctrl.enter(None).await.expect("anonymous takes its seat");
        assert_eq!(ctrl.in_flight(), 1, "the second seat is free throughout");

        match ctrl.enter(None).await {
            Err(AdmissionRejection::AnonYield { retry_after_secs }) => {
                assert!(retry_after_secs >= 1)
            }
            other => panic!("expected AnonYield, got {other:?}"),
        }
        assert_eq!(ctrl.rejections().anon_yield, 1);
        // Refusing must not leave the yielded request on the books.
        assert_eq!(ctrl.anon_in_flight(), 1);
        assert_eq!(ctrl.queue_depth(), 0);
    }

    /// #257 must keep holding for anonymous callers: a long-context request
    /// waiting for KV budget is waiting for a whole sequence to finish,
    /// which takes minutes, so it gets `kv_max_wait` — not the seat
    /// timeout. Refusing it at `max_wait` would reinstate exactly the
    /// premature rejection #257 removed, for the one class of caller least
    /// able to report it.
    #[tokio::test(start_paused = true)]
    async fn anonymous_waits_kv_max_wait_for_budget_not_the_seat_timeout() {
        let cfg = AdmissionConfig {
            // Two seats, so the seat is never what blocks — only KV is.
            max_in_flight: 2,
            max_queue_depth: 8,
            max_wait_secs: 1,
            max_per_principal: 0,
            anon_max_in_flight: Some(2),
            anon_max_pending: Some(8),
            kv_max_wait_secs: 300,
        };
        let ctrl = Arc::new(AdmissionController::new(&cfg));
        ctrl.set_kv_budget_mb(1_000);

        let running = ctrl
            .enter_with_kv(None, 800)
            .await
            .expect("first anonymous request takes most of the budget");

        let c = Arc::clone(&ctrl);
        let waiting = tokio::spawn(async move { c.enter_with_kv(None, 800).await.map(drop) });

        // Well past the seat timeout: a request blocked on budget must
        // still be waiting, with a seat sitting free beside it.
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(!waiting.is_finished(), "must wait for budget, not the seat");
        assert_eq!(ctrl.rejections().timeout, 0);

        // Budget returns → admitted.
        drop(running);
        tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("admitted once the budget frees")
            .unwrap()
            .expect("admitted");
    }

    /// The counterpart: a genuinely full model refusing an anonymous
    /// caller is ordinary overload, and must be reported as `Timeout`.
    /// Charging it to `anon_yield` would make the reservation look like
    /// the cause of every anonymous refusal and leave the metric unable to
    /// answer the one question it exists for.
    #[tokio::test(start_paused = true)]
    async fn a_full_model_refuses_anonymous_as_ordinary_overload() {
        let cfg = AdmissionConfig {
            max_in_flight: 1,
            max_queue_depth: 4,
            max_wait_secs: 1,
            max_per_principal: 0,
            anon_max_in_flight: Some(1),
            anon_max_pending: Some(4),
            kv_max_wait_secs: 30,
        };
        let ctrl = Arc::new(AdmissionController::new(&cfg));
        let _running = ctrl.enter(Some("acct-a/key")).await.expect("admit");

        match ctrl.enter(None).await {
            Err(AdmissionRejection::Timeout { .. }) => {}
            other => panic!("expected Timeout, got {other:?}"),
        }
        assert_eq!(ctrl.rejections().anon_yield, 0, "not a reservation effect");
        assert_eq!(ctrl.rejections().timeout, 1);
    }

    /// Cancellation safety at the new gate: a client that disconnects while
    /// yielding must roll back its pending count and (if claimed) its seat,
    /// exactly as the queue gate does. A leak here would ratchet the
    /// anonymous ceiling down until no anonymous request could ever be
    /// admitted — the #262 mechanism silently becoming a total block.
    #[tokio::test]
    async fn cancelled_yielding_waiter_rolls_back_accounting() {
        let ctrl = Arc::new(AdmissionController::new(&anon_cfg(1, 4, 1, 4)));
        let running = ctrl.enter(Some("acct/key")).await.expect("admit");

        let c = Arc::clone(&ctrl);
        let abandoned = tokio::spawn(async move { c.enter(None).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 1);

        abandoned.abort();
        let _ = abandoned.await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ctrl.queue_depth(), 0, "yielding waiter left no queue slot");
        assert_eq!(ctrl.anon_in_flight(), 0, "yielding waiter left no seat");

        // Proof the ceiling is intact: a fresh anonymous request is served.
        drop(running);
        tokio::time::timeout(Duration::from_millis(500), ctrl.enter(None))
            .await
            .expect("anonymous admission still works after a cancelled yield")
            .expect("admitted");
    }

    /// `anon_max_in_flight = 0` refuses anonymous traffic outright — and
    /// does so immediately, rather than making it wait out `max_wait` for
    /// capacity that policy will never grant it.
    #[tokio::test]
    async fn zero_anon_ceiling_refuses_immediately() {
        let ctrl = AdmissionController::new(&anon_cfg(4, 8, 0, 8));
        match ctrl.enter(None).await {
            Err(AdmissionRejection::AnonYield { .. }) => {}
            other => panic!("expected AnonYield, got {other:?}"),
        }
        assert_eq!(ctrl.in_flight(), 0);
        // Identified traffic is unaffected.
        ctrl.enter(Some("acct/key"))
            .await
            .expect("identified admits");
    }

    /// #256 interaction: a shutdown must wake requests parked at the class
    /// gate too. They are on no semaphore, so closing those cannot reach
    /// them — without an explicit wake they wait out `max_wait` while the
    /// process tries to exit, which is the stall #256 removed.
    #[tokio::test]
    async fn close_wakes_a_yielding_anonymous_waiter() {
        let ctrl = Arc::new(AdmissionController::new(&anon_cfg(1, 4, 1, 4)));
        let _running = ctrl.enter(Some("acct/key")).await.expect("admit");

        let c = Arc::clone(&ctrl);
        let yielding = tokio::spawn(async move { c.enter(None).await.map(drop) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!yielding.is_finished(), "parked at the class gate");

        ctrl.close();
        assert!(
            tokio::time::timeout(Duration::from_millis(500), yielding)
                .await
                .expect("close must wake a yielding waiter")
                .expect("join")
                .is_err(),
            "a yielding waiter woken by close must be refused"
        );
    }

    /// The derived defaults hold a seat and a queue place back without an
    /// operator having to configure anything — and never reach zero on a
    /// single-seat model, where refusing anonymous traffic outright would
    /// be a policy choice rather than a default.
    #[test]
    fn derived_defaults_reserve_one_of_each() {
        let derived = |max_in_flight, max_queue_depth| {
            AdmissionController::new(&AdmissionConfig {
                max_in_flight,
                max_queue_depth,
                max_wait_secs: 30,
                max_per_principal: 0,
                anon_max_in_flight: None,
                anon_max_pending: None,
                kv_max_wait_secs: 30,
            })
        };

        // beast's shape: 8 seats → anonymous may hold 7.
        assert_eq!(derived(8, 8).anon_max_in_flight(), 7);
        // A single-seat model (benjy today) still serves anonymous callers;
        // priority alone decides who wins there.
        assert_eq!(derived(1, 8).anon_max_in_flight(), 1);
    }
}
