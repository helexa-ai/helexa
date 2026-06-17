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
use cortex_core::entitlements::{
    BudgetError, EntitlementProvider, HEADER_ACCOUNT_ID, HEADER_KEY_ID, Principal,
};
use cortex_core::error_envelope::OpenAiError;
use std::sync::Arc;

/// Fallback output-token budget when neither the request nor the model's
/// advertised limit gives one. Bounds the reservation so a capped key is
/// still gated even on under-specified requests (#52).
pub const FALLBACK_MAX_OUTPUT: u64 = 4096;

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

    /// Wrap an already-acquired reservation.
    fn held(
        provider: Arc<dyn EntitlementProvider>,
        reservation: cortex_core::entitlements::Reservation,
    ) -> Self {
        Self {
            provider,
            reservation: Some(reservation),
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

/// Reserve the request's upper-bound token cost for the principal, refusing
/// *before* dispatch if it would exceed the hard cap (#52). On success
/// returns a guard the caller settles with actual usage; on refusal returns
/// the #63 envelope (`rate_limit_exceeded` + `Retry-After` for a resetting
/// window, `insufficient_quota` for a hard balance — never `402`).
pub async fn reserve_or_reject(
    provider: Arc<dyn EntitlementProvider>,
    principal: &Principal,
    max_tokens: u64,
) -> Result<ReservationGuard, OpenAiError> {
    match provider.reserve(principal, max_tokens).await {
        Ok(reservation) => Ok(ReservationGuard::held(provider, reservation)),
        Err(err) => Err(budget_error_to_envelope(err)),
    }
}

/// Map a [`BudgetError`] to the #63 envelope. The provider chose the window
/// semantics; this only translates them to HTTP.
fn budget_error_to_envelope(err: BudgetError) -> OpenAiError {
    match err {
        BudgetError::RateLimited {
            retry_after_secs, ..
        } => OpenAiError::rate_limit_exceeded(err.to_string(), retry_after_secs),
        BudgetError::InsufficientQuota { .. } => OpenAiError::insufficient_quota(err.to_string()),
    }
}

/// Upper-bound tokens to reserve for a request (#52): an over-estimate of
/// the prompt plus the maximum output. `advertised_output` is the model's
/// `limit.output` (#62), used when the request omits `max_(completion_)tokens`.
/// Over-reserving is safe — settle corrects spend to the actual usage.
pub fn reservation_estimate(body: &[u8], advertised_output: Option<u64>) -> u64 {
    let max_output = requested_max_output(body)
        .or(advertised_output)
        .unwrap_or(FALLBACK_MAX_OUTPUT);
    estimate_prompt_tokens(body).saturating_add(max_output)
}

/// The client's requested output cap, from `max_completion_tokens` (or the
/// legacy `max_tokens`). `None` when unspecified.
fn requested_max_output(body: &[u8]) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("max_completion_tokens")
        .or_else(|| v.get("max_tokens"))
        .and_then(serde_json::Value::as_u64)
}

/// Rough prompt-token estimate at ~4 chars/token over the whole body. cortex
/// has no tokenizer; JSON overhead makes this a conservative over-estimate,
/// and neuron remains the exact context wall (#56/#60). Settle reconciles to
/// the real usage afterward.
fn estimate_prompt_tokens(body: &[u8]) -> u64 {
    (body.len() as u64 / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_max_output_prefers_max_completion_tokens() {
        let body = br#"{"model":"m","max_completion_tokens":256,"max_tokens":99}"#;
        assert_eq!(requested_max_output(body), Some(256));
    }

    #[test]
    fn requested_max_output_falls_back_to_legacy_max_tokens() {
        let body = br#"{"model":"m","max_tokens":128}"#;
        assert_eq!(requested_max_output(body), Some(128));
    }

    #[test]
    fn estimate_uses_requested_output_when_present() {
        // Requested output dominates; prompt estimate is small for a tiny body.
        let body = br#"{"model":"m","max_tokens":1000}"#;
        let est = reservation_estimate(body, Some(8192));
        assert!(est >= 1000 && est < 1100, "est was {est}");
    }

    #[test]
    fn estimate_uses_advertised_output_when_request_omits_it() {
        let body = br#"{"model":"m","messages":[]}"#;
        let est = reservation_estimate(body, Some(8192));
        assert!(est >= 8192, "est was {est}");
    }

    #[test]
    fn estimate_falls_back_when_nothing_advertised() {
        let body = br#"{"model":"m"}"#;
        let est = reservation_estimate(body, None);
        assert!(est >= FALLBACK_MAX_OUTPUT, "est was {est}");
    }
}
