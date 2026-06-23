//! helexa-upstream client (#57): an [`EntitlementProvider`] that resolves
//! keys and reserves/settles budget against the mesh authority's
//! `/authz/v1` surface (B2). It is "just another impl of the trait" — cortex
//! enforcement (`auth.rs`, `metering.rs`) is unchanged.
//!
//! **Fail closed.** When upstream is unreachable, `resolve` returns
//! [`AuthError::Unavailable`] (→ `503`, never `401`) and `reserve` refuses
//! with a retryable [`BudgetError::RateLimited`] — a request is never served
//! on an un-authorized key, and a real key is never rejected as invalid
//! during a blip.

use async_trait::async_trait;
use cortex_core::config::UpstreamClientConfig;
use cortex_core::entitlements::{
    AuthError, BudgetError, BudgetSnapshot, EntitlementProvider, Principal, Reservation,
};
use serde::Deserialize;
use std::time::Duration;

/// Retry-After (seconds) advertised when we fail closed on an upstream
/// outage.
const FAIL_CLOSED_RETRY_SECS: u64 = 5;

pub struct UpstreamEntitlementProvider {
    client: reqwest::Client,
    base_url: String,
    bearer: String,
}

#[derive(Deserialize)]
struct PrincipalDto {
    account_id: String,
    key_id: String,
}
#[derive(Deserialize)]
struct SnapshotDto {
    hard_cap: Option<u64>,
    spent: u64,
    reserved: u64,
}
#[derive(Deserialize)]
struct ResolveResp {
    principal: PrincipalDto,
    #[allow(dead_code)]
    snapshot: Option<SnapshotDto>,
}
#[derive(Deserialize)]
struct ReserveResp {
    reservation_id: Option<i64>,
    rejected: Option<Rejection>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Rejection {
    InsufficientQuota {
        requested: u64,
        available: u64,
    },
    RateLimited {
        requested: u64,
        available: u64,
        retry_after_secs: u64,
    },
}

impl UpstreamEntitlementProvider {
    pub fn new(cfg: &UpstreamClientConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("failed to build upstream HTTP client");
        Self {
            client,
            base_url: cfg.url.trim_end_matches('/').to_string(),
            bearer: cfg.bearer.clone(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

#[async_trait]
impl EntitlementProvider for UpstreamEntitlementProvider {
    async fn resolve(&self, api_key: &str) -> Result<Principal, AuthError> {
        let resp = self
            .client
            .post(self.url("/authz/v1/resolve"))
            .bearer_auth(&self.bearer)
            .json(&serde_json::json!({ "api_key": api_key }))
            .send()
            .await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "upstream resolve unreachable; failing closed");
                return Err(AuthError::Unavailable {
                    retry_after_secs: FAIL_CLOSED_RETRY_SECS,
                });
            }
        };
        if resp.status().as_u16() == 401 {
            return Err(AuthError::InvalidKey);
        }
        if !resp.status().is_success() {
            return Err(AuthError::Unavailable {
                retry_after_secs: FAIL_CLOSED_RETRY_SECS,
            });
        }
        match resp.json::<ResolveResp>().await {
            Ok(r) => Ok(Principal {
                account_id: r.principal.account_id,
                key_id: r.principal.key_id,
            }),
            Err(e) => {
                tracing::warn!(error = %e, "upstream resolve: bad body; failing closed");
                Err(AuthError::Unavailable {
                    retry_after_secs: FAIL_CLOSED_RETRY_SECS,
                })
            }
        }
    }

    async fn reserve(
        &self,
        principal: &Principal,
        max_tokens: u64,
    ) -> Result<Reservation, BudgetError> {
        let fail_closed = || BudgetError::RateLimited {
            requested: max_tokens,
            available: 0,
            retry_after_secs: FAIL_CLOSED_RETRY_SECS,
        };
        let resp = self
            .client
            .post(self.url("/authz/v1/reserve"))
            .bearer_auth(&self.bearer)
            .json(&serde_json::json!({
                "account_id": principal.account_id,
                "key_id": principal.key_id,
                "max_tokens": max_tokens,
            }))
            .send()
            .await;
        let resp = match resp {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                tracing::warn!(status = %r.status(), "upstream reserve non-2xx; failing closed");
                return Err(fail_closed());
            }
            Err(e) => {
                tracing::warn!(error = %e, "upstream reserve unreachable; failing closed");
                return Err(fail_closed());
            }
        };
        match resp.json::<ReserveResp>().await {
            Ok(ReserveResp {
                reservation_id: Some(id),
                ..
            }) => Ok(Reservation {
                id: id as u64,
                principal: principal.clone(),
                reserved: max_tokens,
            }),
            Ok(ReserveResp {
                rejected:
                    Some(Rejection::InsufficientQuota {
                        requested,
                        available,
                    }),
                ..
            }) => Err(BudgetError::InsufficientQuota {
                requested,
                available,
            }),
            Ok(ReserveResp {
                rejected:
                    Some(Rejection::RateLimited {
                        requested,
                        available,
                        retry_after_secs,
                    }),
                ..
            }) => Err(BudgetError::RateLimited {
                requested,
                available,
                retry_after_secs,
            }),
            _ => Err(fail_closed()),
        }
    }

    async fn settle(&self, reservation: Reservation, actual_tokens: u64) {
        // Best-effort; a lost settle is reaped by the upstream sweeper (B2).
        let _ = self
            .client
            .post(self.url("/authz/v1/settle"))
            .bearer_auth(&self.bearer)
            .json(&serde_json::json!({
                "reservation_id": reservation.id as i64,
                "actual_tokens": actual_tokens,
            }))
            .send()
            .await
            .inspect_err(
                |e| tracing::warn!(error = %e, "upstream settle failed (sweeper will reap)"),
            );
    }

    async fn release(&self, reservation: Reservation) {
        let _ = self
            .client
            .post(self.url("/authz/v1/release"))
            .bearer_auth(&self.bearer)
            .json(&serde_json::json!({ "reservation_id": reservation.id as i64 }))
            .send()
            .await
            .inspect_err(
                |e| tracing::warn!(error = %e, "upstream release failed (sweeper will reap)"),
            );
    }

    async fn snapshot(&self, principal: &Principal) -> Option<BudgetSnapshot> {
        let resp = self
            .client
            .post(self.url("/authz/v1/snapshot"))
            .bearer_auth(&self.bearer)
            .json(&serde_json::json!({
                "account_id": principal.account_id,
                "key_id": principal.key_id,
            }))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let dto = resp.json::<SnapshotDto>().await.ok()?;
        Some(BudgetSnapshot {
            hard_cap: dto.hard_cap,
            spent: dto.spent,
            reserved: dto.reserved,
        })
    }
}
