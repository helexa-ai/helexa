//! The local/static [`EntitlementProvider`] (#50).
//!
//! Accounts, keys, and hard caps come from operator config
//! ([`cortex_core::config::EntitlementsConfig`]); reservations and settled
//! spend are tracked in-process. This lands auth + per-key caps + the
//! amplification fix before any upstream clearing house exists; the future
//! helexa-upstream client (#57) implements the same trait.
//!
//! Budget math is serialized under a single [`std::sync::Mutex`] so
//! reserve/settle/release are atomic — a key's `spent + reserved` can never
//! exceed its hard cap even under concurrent requests (the #52 guarantee).
//! The lock is held only for the in-memory arithmetic, never across an
//! await.

use cortex_core::config::{ApiKeyConfig, EntitlementsConfig};
use cortex_core::entitlements::{
    AuthError, BudgetError, BudgetSnapshot, CapWindow, EntitlementProvider, Principal, Reservation,
};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Per-key budget configuration (resolved from [`ApiKeyConfig`]).
struct Budget {
    hard_cap: Option<u64>,
    window: CapWindow,
}

/// Live, mutable accounting for one key over its current window.
#[derive(Default)]
struct Ledger {
    /// Settled spend in the current window.
    spent: u64,
    /// Sum of outstanding (un-settled) reservations.
    reserved: u64,
    /// Start of the current rolling window; `None` until the first reserve.
    /// Unused for [`CapWindow::Balance`].
    window_start: Option<Instant>,
}

pub struct LocalEntitlementProvider {
    /// Bearer token → principal.
    keys: HashMap<String, Principal>,
    /// `key_id` → budget config.
    budgets: HashMap<String, Budget>,
    /// `key_id` → live ledger.
    ledgers: Mutex<HashMap<String, Ledger>>,
    /// Monotonic source of opaque reservation handles.
    next_id: AtomicU64,
}

impl LocalEntitlementProvider {
    /// Build from the `[entitlements]` config. A key without an explicit
    /// `key_id` is tracked at `account_id` granularity (its secret is never
    /// used as a label).
    pub fn from_config(config: &EntitlementsConfig) -> Self {
        let mut keys = HashMap::new();
        let mut budgets = HashMap::new();
        for ApiKeyConfig {
            key,
            account_id,
            key_id,
            hard_cap,
            window,
        } in &config.keys
        {
            let key_id = key_id.clone().unwrap_or_else(|| account_id.clone());
            keys.insert(
                key.clone(),
                Principal {
                    account_id: account_id.clone(),
                    key_id: key_id.clone(),
                },
            );
            budgets.insert(
                key_id,
                Budget {
                    hard_cap: *hard_cap,
                    window: window.clone(),
                },
            );
        }
        Self {
            keys,
            budgets,
            ledgers: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }
}

/// Tokens still available under `cap` given current `spent`/`reserved`.
/// `None` cap = unlimited.
fn available(cap: Option<u64>, spent: u64, reserved: u64) -> Option<u64> {
    cap.map(|c| c.saturating_sub(spent).saturating_sub(reserved))
}

#[async_trait::async_trait]
impl EntitlementProvider for LocalEntitlementProvider {
    async fn resolve(&self, api_key: &str) -> Result<Principal, AuthError> {
        self.keys.get(api_key).cloned().ok_or(AuthError::InvalidKey)
    }

    async fn reserve(
        &self,
        principal: &Principal,
        max_tokens: u64,
    ) -> Result<Reservation, BudgetError> {
        // A principal with no configured budget (or an uncapped one) always
        // reserves; we still track spend for metrics.
        let budget = self.budgets.get(&principal.key_id);
        let (cap, window) = match budget {
            Some(b) => (b.hard_cap, b.window.clone()),
            None => (None, CapWindow::Balance),
        };

        let mut ledgers = self.ledgers.lock().expect("ledger mutex poisoned");
        let ledger = ledgers.entry(principal.key_id.clone()).or_default();

        // Lazily reset a rolling window that has elapsed before checking.
        let mut retry_after_secs = 0;
        if let CapWindow::Rolling { seconds } = window {
            let now = Instant::now();
            match ledger.window_start {
                Some(start) if now.duration_since(start).as_secs() < seconds => {
                    retry_after_secs = seconds - now.duration_since(start).as_secs();
                }
                _ => {
                    // First reserve, or the window has fully elapsed: reset.
                    ledger.spent = 0;
                    ledger.window_start = Some(now);
                    retry_after_secs = seconds;
                }
            }
        }

        if let Some(avail) = available(cap, ledger.spent, ledger.reserved)
            && max_tokens > avail
        {
            return Err(match window {
                CapWindow::Rolling { .. } => BudgetError::RateLimited {
                    requested: max_tokens,
                    available: avail,
                    // At least 1s so clients don't hot-loop on a sub-second
                    // remainder.
                    retry_after_secs: retry_after_secs.max(1),
                },
                CapWindow::Balance => BudgetError::InsufficientQuota {
                    requested: max_tokens,
                    available: avail,
                },
            });
        }

        ledger.reserved += max_tokens;
        Ok(Reservation {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            principal: principal.clone(),
            reserved: max_tokens,
        })
    }

    async fn settle(&self, reservation: Reservation, actual_tokens: u64) {
        let mut ledgers = self.ledgers.lock().expect("ledger mutex poisoned");
        if let Some(ledger) = ledgers.get_mut(&reservation.principal.key_id) {
            ledger.reserved = ledger.reserved.saturating_sub(reservation.reserved);
            ledger.spent += actual_tokens;
        }
    }

    async fn release(&self, reservation: Reservation) {
        let mut ledgers = self.ledgers.lock().expect("ledger mutex poisoned");
        if let Some(ledger) = ledgers.get_mut(&reservation.principal.key_id) {
            ledger.reserved = ledger.reserved.saturating_sub(reservation.reserved);
        }
    }

    async fn snapshot(&self, principal: &Principal) -> Option<BudgetSnapshot> {
        let ledgers = self.ledgers.lock().expect("ledger mutex poisoned");
        let (spent, reserved) = ledgers
            .get(&principal.key_id)
            .map(|l| (l.spent, l.reserved))
            .unwrap_or((0, 0));
        let hard_cap = self.budgets.get(&principal.key_id).and_then(|b| b.hard_cap);
        Some(BudgetSnapshot {
            hard_cap,
            spent,
            reserved,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> LocalEntitlementProvider {
        let config = EntitlementsConfig {
            require_auth: true,
            keys: vec![
                ApiKeyConfig {
                    key: "sk-balance".into(),
                    account_id: "acct-a".into(),
                    key_id: Some("key-balance".into()),
                    hard_cap: Some(1_000),
                    window: CapWindow::Balance,
                },
                ApiKeyConfig {
                    key: "sk-rolling".into(),
                    account_id: "acct-b".into(),
                    key_id: Some("key-rolling".into()),
                    hard_cap: Some(500),
                    window: CapWindow::Rolling { seconds: 3_600 },
                },
                ApiKeyConfig {
                    key: "sk-infra".into(),
                    account_id: "operator".into(),
                    key_id: Some("key-infra".into()),
                    hard_cap: None,
                    window: CapWindow::Balance,
                },
            ],
        };
        LocalEntitlementProvider::from_config(&config)
    }

    #[tokio::test]
    async fn resolves_configured_key_to_principal() {
        let p = provider();
        let principal = p.resolve("sk-balance").await.expect("known key resolves");
        assert_eq!(principal.account_id, "acct-a");
        assert_eq!(principal.key_id, "key-balance");
    }

    #[tokio::test]
    async fn unknown_key_is_invalid() {
        let p = provider();
        assert!(matches!(
            p.resolve("sk-nope").await,
            Err(AuthError::InvalidKey)
        ));
    }

    #[tokio::test]
    async fn reserve_settle_release_round_trip() {
        let p = provider();
        let principal = p.resolve("sk-balance").await.unwrap();

        let r = p.reserve(&principal, 400).await.expect("within cap");
        // Reserved, not yet spent.
        let snap = p.snapshot(&principal).await.unwrap();
        assert_eq!(snap.hard_cap, Some(1_000));
        assert_eq!(snap.reserved, 400);
        assert_eq!(snap.spent, 0);

        // Used fewer tokens than reserved → remainder released, spend exact.
        p.settle(r, 250).await;
        let snap = p.snapshot(&principal).await.unwrap();
        assert_eq!(snap.reserved, 0);
        assert_eq!(snap.spent, 250);

        // A reservation that is released contributes no spend.
        let r2 = p.reserve(&principal, 100).await.unwrap();
        p.release(r2).await;
        let snap = p.snapshot(&principal).await.unwrap();
        assert_eq!(snap.reserved, 0);
        assert_eq!(snap.spent, 250);
    }

    #[tokio::test]
    async fn balance_over_cap_is_insufficient_quota() {
        let p = provider();
        let principal = p.resolve("sk-balance").await.unwrap();
        // Reserve most of the cap, then ask for more than remains.
        let _r = p.reserve(&principal, 900).await.unwrap();
        let err = p.reserve(&principal, 200).await.expect_err("over cap");
        match err {
            BudgetError::InsufficientQuota {
                requested,
                available,
            } => {
                assert_eq!(requested, 200);
                assert_eq!(available, 100);
            }
            other => panic!("expected InsufficientQuota, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rolling_over_cap_is_rate_limited_with_retry_after() {
        let p = provider();
        let principal = p.resolve("sk-rolling").await.unwrap();
        let _r = p.reserve(&principal, 500).await.unwrap();
        let err = p.reserve(&principal, 1).await.expect_err("over cap");
        match err {
            BudgetError::RateLimited {
                retry_after_secs, ..
            } => {
                assert!(retry_after_secs >= 1, "must advertise a retry hint");
                assert!(retry_after_secs <= 3_600);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn uncapped_infra_key_never_refuses() {
        let p = provider();
        let principal = p.resolve("sk-infra").await.unwrap();
        let r = p.reserve(&principal, 10_000_000).await.expect("uncapped");
        p.settle(r, 10_000_000).await;
        let snap = p.snapshot(&principal).await.unwrap();
        assert_eq!(snap.hard_cap, None);
        assert_eq!(snap.spent, 10_000_000);
    }
}
