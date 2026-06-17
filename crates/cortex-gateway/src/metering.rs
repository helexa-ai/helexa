//! Per-request token metering (#51).
//!
//! Captures the real `(prompt, completion)` usage of every request and feeds
//! it to two places: the [`EntitlementProvider`] spend ledger (via
//! reserve→settle) and per-principal Prometheus counters. The principal is
//! reconstructed from the internal headers the auth middleware stamped (#49),
//! so this works uniformly across every proxy path without threading the
//! typed principal through each handler.
//!
//! The reserve→settle lifecycle is established here but, in this phase,
//! reserves **zero** tokens — metering only, no enforcement. Budget
//! enforcement (#52) flips the reserved amount to the real
//! `prompt + max_output` and handles the [`BudgetError`] rejection; the
//! settle/release plumbing is identical, so that change is localized.
//!
//! [`ReservationGuard`] makes leaks impossible: settling records actual
//! spend and releases the unused remainder; dropping a guard that was never
//! settled releases the whole reservation. So an early return, error path,
//! or dropped stream can't strand a reservation.

use axum::http::HeaderMap;
use cortex_core::entitlements::{EntitlementProvider, HEADER_ACCOUNT_ID, HEADER_KEY_ID, Principal};
use std::sync::Arc;

/// Invoked exactly once at request completion with best-effort
/// `(prompt_tokens, completion_tokens)`. When no usage could be observed
/// (e.g. a pre-dispatch failure or a dropped stream) it is dropped unused —
/// which releases the held reservation via [`ReservationGuard`]'s `Drop`.
pub type UsageSink = Box<dyn FnOnce(u64, u64) + Send>;

/// Reconstruct the principal from the cortex-stamped internal headers. The
/// auth middleware strips any client copy and stamps the authoritative value,
/// so these headers are trustworthy within cortex. `None` for anonymous
/// (unauthenticated) requests.
pub fn principal_from_headers(headers: &HeaderMap) -> Option<Principal> {
    let account_id = headers.get(HEADER_ACCOUNT_ID)?.to_str().ok()?.to_string();
    let key_id = headers.get(HEADER_KEY_ID)?.to_str().ok()?.to_string();
    Some(Principal { account_id, key_id })
}

/// Emit per-principal spend counters (#51). Labelled by account/key only —
/// both are operator-bounded, so cardinality is controlled.
pub fn record_spend(principal: &Principal, prompt: u64, completion: u64) {
    let labels = [
        ("account", principal.account_id.clone()),
        ("key", principal.key_id.clone()),
    ];
    metrics::counter!("cortex_spend_tokens_total", &labels).increment(prompt + completion);
    metrics::counter!("cortex_spend_prompt_tokens_total", &labels).increment(prompt);
    metrics::counter!("cortex_spend_completion_tokens_total", &labels).increment(completion);
}

/// Holds a budget reservation for the life of a request. [`settle`] records
/// actual spend and releases the remainder; an un-settled guard releases the
/// whole reservation when dropped. Anonymous requests carry an empty guard,
/// where every operation is a no-op.
///
/// [`settle`]: ReservationGuard::settle
pub struct ReservationGuard {
    provider: Arc<dyn EntitlementProvider>,
    reservation: Option<cortex_core::entitlements::Reservation>,
}

impl ReservationGuard {
    /// An empty guard for an anonymous request — no reservation to resolve.
    pub fn anonymous(provider: Arc<dyn EntitlementProvider>) -> Self {
        Self {
            provider,
            reservation: None,
        }
    }

    /// Reserve `max_tokens` for the principal, returning a guard. In this
    /// phase callers pass `0` (metering only); #52 passes the real cap and
    /// surfaces the [`cortex_core::entitlements::BudgetError`] instead.
    pub async fn reserve(
        provider: Arc<dyn EntitlementProvider>,
        principal: &Principal,
        max_tokens: u64,
    ) -> Self {
        let reservation = provider.reserve(principal, max_tokens).await.ok();
        Self {
            provider,
            reservation,
        }
    }

    /// Settle with the tokens actually consumed, disarming the drop-release.
    /// Spawns the (fast, in-process for the local provider) settle so the
    /// caller — which may be a sync stream-completion callback — needn't
    /// await.
    pub fn settle(mut self, actual_tokens: u64) {
        if let Some(reservation) = self.reservation.take() {
            let provider = Arc::clone(&self.provider);
            tokio::spawn(async move {
                provider.settle(reservation, actual_tokens).await;
            });
        }
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            let provider = Arc::clone(&self.provider);
            tokio::spawn(async move {
                provider.release(reservation).await;
            });
        }
    }
}

/// Build the completion sink for an authenticated request: record spend and
/// settle the reservation with the observed total. Dropping it unused (no
/// usage observed) releases the reservation via the guard.
pub fn usage_sink(principal: Principal, guard: ReservationGuard) -> UsageSink {
    Box::new(move |prompt, completion| {
        record_spend(&principal, prompt, completion);
        guard.settle(prompt + completion);
    })
}
