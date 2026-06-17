//! Identity and entitlement primitives for multi-tenant governance (#47).
//!
//! Identity is the shared substrate the whole epic hangs off:
//! `identity (principal) → accounting (spend) → policy → enforcement`. This
//! module defines the seam — the [`EntitlementProvider`] trait and its data
//! types — so the local/static provider (operator-config caps, in
//! cortex-gateway) can land the auth + per-key-cap + amplification fix
//! *before* any upstream clearing house exists. The future helexa-upstream
//! client (#57) is just another impl of this trait.
//!
//! The provider owns three jobs:
//! 1. **resolve** a bearer key to a [`Principal`] (drives auth, #49);
//! 2. **reserve → settle/release** token budget around a request so spend
//!    can never overshoot a hard cap under concurrency (drives budget
//!    enforcement, #52);
//! 3. expose a [`BudgetSnapshot`] for metering/metrics (#51).
//!
//! [`BudgetError`] carries the cap-window semantics so the caller can pick
//! the correct #63 rejection (`rate_limit_exceeded` + `Retry-After` for a
//! resetting window vs `insufficient_quota` for a hard balance) without the
//! provider knowing anything about HTTP.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Internal header carrying the resolved account id from cortex to neuron.
/// neuron trusts these over the WireGuard link (#54); cortex **strips** any
/// client-supplied copy before stamping the authoritative value, so a client
/// can never assert a principal directly.
pub const HEADER_ACCOUNT_ID: &str = "x-helexa-account-id";
/// Internal header carrying the resolved key id from cortex to neuron.
pub const HEADER_KEY_ID: &str = "x-helexa-key-id";

/// Who a request is for. Resolved once at the edge from the bearer key and
/// carried through the request context. `account_id` is the billable owner
/// (spendable at any operator, by decision); `key_id` identifies the
/// specific API key for per-key hard caps and ledger/metrics labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    pub account_id: String,
    pub key_id: String,
}

/// Cap-window semantics for a key's hard cap. Determines which #63 code an
/// over-cap reservation maps to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CapWindow {
    /// Hard balance — the cap never resets. Exhaustion is permanent
    /// (`429 insufficient_quota`, no `Retry-After`).
    #[default]
    Balance,
    /// Rolling window of `seconds` that resets. Exhaustion is transient
    /// (`429 rate_limit_exceeded` + `Retry-After` until reset).
    Rolling { seconds: u64 },
}

/// An outstanding budget reservation. The caller holds this opaque handle
/// between [`EntitlementProvider::reserve`] and exactly one of
/// [`EntitlementProvider::settle`] / [`EntitlementProvider::release`]. Not
/// `Clone` — a reservation is consumed once.
#[derive(Debug)]
pub struct Reservation {
    /// Provider-local handle; opaque to the caller.
    pub id: u64,
    /// The principal this reservation belongs to.
    pub principal: Principal,
    /// Tokens reserved against the cap.
    pub reserved: u64,
}

/// A point-in-time view of a key's budget, for metering and metrics (#51).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetSnapshot {
    /// Hard cap in tokens. `None` means uncapped (e.g. an operator infra
    /// key, #58).
    pub hard_cap: Option<u64>,
    /// Settled spend in the current window.
    pub spent: u64,
    /// Sum of outstanding (un-settled) reservations.
    pub reserved: u64,
}

/// Authentication failure — the bearer key could not be resolved. Maps to
/// `401 invalid_api_key` (#49/#63).
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("invalid or unknown API key")]
    InvalidKey,
}

/// Why a reservation was refused. Carries enough for the caller to build the
/// correct #63 envelope without the provider touching HTTP.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// A resetting window is exhausted → `429 rate_limit_exceeded` +
    /// `Retry-After: retry_after_secs`.
    #[error(
        "rolling-window budget exhausted ({requested} requested, {available} available); \
         resets in {retry_after_secs}s"
    )]
    RateLimited {
        requested: u64,
        available: u64,
        retry_after_secs: u64,
    },
    /// A hard balance is exhausted → `429 insufficient_quota` (no
    /// `Retry-After`; the client surfaces and stops). Never `402`.
    #[error("hard balance exhausted ({requested} requested, {available} available)")]
    InsufficientQuota { requested: u64, available: u64 },
}

/// The seam between cortex's enforcement and whatever decides entitlement —
/// a local/static config provider today (#50), the helexa-upstream client
/// later (#57). All methods are async so the upstream impl can do network
/// I/O; the local impl resolves in-process.
#[async_trait]
pub trait EntitlementProvider: Send + Sync {
    /// Resolve a bearer API key to its principal. `Err(InvalidKey)` for an
    /// unknown/empty key.
    async fn resolve(&self, api_key: &str) -> Result<Principal, AuthError>;

    /// Reserve up to `max_tokens` against the principal's cap. Returns a
    /// handle on success, or a [`BudgetError`] (which the caller maps to a
    /// #63 `429`) if the reservation would exceed the cap. Reserving the
    /// *maximum* a request could consume before dispatch is what prevents
    /// overshoot under concurrency.
    async fn reserve(
        &self,
        principal: &Principal,
        max_tokens: u64,
    ) -> Result<Reservation, BudgetError>;

    /// Settle a reservation with the tokens actually consumed, releasing the
    /// unused remainder back to the cap.
    async fn settle(&self, reservation: Reservation, actual_tokens: u64);

    /// Release a reservation in full — e.g. dispatch failed before any
    /// tokens were consumed.
    async fn release(&self, reservation: Reservation);

    /// Current budget snapshot for a principal, for metering/metrics.
    /// `None` if the provider doesn't track this principal.
    async fn snapshot(&self, principal: &Principal) -> Option<BudgetSnapshot>;
}
