//! Chained entitlement provider (#57): operator-local keys first, mesh
//! upstream for everything else.
//!
//! `resolve` tries the [`LocalEntitlementProvider`] (operator + infra keys —
//! never a network hop); only a locally-unknown key falls through to
//! [`UpstreamEntitlementProvider`]. Because the local provider treats an
//! unconfigured principal as uncapped, reserve/settle/release/snapshot must
//! **not** blindly hit local — they dispatch to whichever backend resolved
//! that account, remembered in a map keyed by `account_id` (populated at
//! resolve time).

use crate::entitlements_local::LocalEntitlementProvider;
use crate::entitlements_upstream::UpstreamEntitlementProvider;
use async_trait::async_trait;
use cortex_core::entitlements::{
    AuthError, BudgetError, BudgetSnapshot, EntitlementProvider, Principal, Reservation,
};
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Local,
    Upstream,
}

pub struct ChainedEntitlementProvider {
    local: LocalEntitlementProvider,
    upstream: UpstreamEntitlementProvider,
    /// account_id → which backend owns it, learned at resolve time.
    backends: RwLock<HashMap<String, Backend>>,
}

impl ChainedEntitlementProvider {
    pub fn new(local: LocalEntitlementProvider, upstream: UpstreamEntitlementProvider) -> Self {
        Self {
            local,
            upstream,
            backends: RwLock::new(HashMap::new()),
        }
    }

    async fn record(&self, account_id: &str, backend: Backend) {
        self.backends
            .write()
            .await
            .insert(account_id.to_string(), backend);
    }

    /// The backend that owns `account_id`. Defaults to `Upstream` for an
    /// account never resolved this process-lifetime (a resolve always
    /// precedes reserve in a request, so this is just a safe fallback —
    /// upstream fails closed if the account is bogus).
    async fn backend_for(&self, account_id: &str) -> Backend {
        self.backends
            .read()
            .await
            .get(account_id)
            .copied()
            .unwrap_or(Backend::Upstream)
    }
}

#[async_trait]
impl EntitlementProvider for ChainedEntitlementProvider {
    async fn resolve(&self, api_key: &str) -> Result<Principal, AuthError> {
        match self.local.resolve(api_key).await {
            Ok(p) => {
                self.record(&p.account_id, Backend::Local).await;
                Ok(p)
            }
            Err(AuthError::InvalidKey) => {
                let p = self.upstream.resolve(api_key).await?;
                self.record(&p.account_id, Backend::Upstream).await;
                Ok(p)
            }
            Err(e) => Err(e),
        }
    }

    async fn reserve(
        &self,
        principal: &Principal,
        max_tokens: u64,
    ) -> Result<Reservation, BudgetError> {
        match self.backend_for(&principal.account_id).await {
            Backend::Local => self.local.reserve(principal, max_tokens).await,
            Backend::Upstream => self.upstream.reserve(principal, max_tokens).await,
        }
    }

    async fn settle(&self, reservation: Reservation, actual_tokens: u64) {
        match self.backend_for(&reservation.principal.account_id).await {
            Backend::Local => self.local.settle(reservation, actual_tokens).await,
            Backend::Upstream => self.upstream.settle(reservation, actual_tokens).await,
        }
    }

    async fn release(&self, reservation: Reservation) {
        match self.backend_for(&reservation.principal.account_id).await {
            Backend::Local => self.local.release(reservation).await,
            Backend::Upstream => self.upstream.release(reservation).await,
        }
    }

    async fn snapshot(&self, principal: &Principal) -> Option<BudgetSnapshot> {
        match self.backend_for(&principal.account_id).await {
            Backend::Local => self.local.snapshot(principal).await,
            Backend::Upstream => self.upstream.snapshot(principal).await,
        }
    }
}
