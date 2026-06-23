//! `/authz/v1` — the machine surface cortex's `UpstreamEntitlementProvider`
//! (#57) consumes. It mirrors the `cortex_core::entitlements::EntitlementProvider`
//! trait 1:1 (resolve / reserve / settle / release / snapshot) over the B1
//! ledger.
//!
//! Contract notes for the cortex client:
//! - A **non-2xx** response means the authority could not give an
//!   authoritative answer (bad caller auth, malformed request, server
//!   error) → the client should **fail closed**.
//! - `reserve` returns **200** whether granted or budget-refused: the body
//!   carries either `reservation_id` or a `rejected` discriminant. A budget
//!   refusal is an authoritative answer, not a transport failure.
//! - Rejections that are genuinely auth failures use the #63 `OpenAiError`
//!   envelope so they can be surfaced verbatim.

use crate::crypto::sha256;
use crate::error::envelope_response;
use crate::ledger::{self, LedgerError};
use crate::state::AppState;
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use cortex_core::error_envelope::OpenAiError;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use uuid::Uuid;

/// The operator a validated client bearer identifies (served-usage
/// attribution, #58). Inserted into request extensions by [`client_auth`].
#[derive(Debug, Clone)]
pub struct OperatorId(pub String);

/// Build the `/authz/v1` router with the client-auth layer applied.
pub fn router(state: &AppState) -> Router<AppState> {
    Router::new()
        .route("/authz/v1/resolve", post(resolve))
        .route("/authz/v1/reserve", post(reserve))
        .route("/authz/v1/settle", post(settle))
        .route("/authz/v1/release", post(release))
        .route("/authz/v1/snapshot", post(snapshot))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            client_auth,
        ))
}

// ── client auth (shared bearer → operator_id) ───────────────────────

/// Validate the caller's `Authorization: Bearer` against the configured
/// client tokens (constant-time) and stamp the `operator_id`. When no tokens
/// are configured the surface is open (dev) and a synthetic operator is
/// used.
async fn client_auth(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    let tokens = &state.config.client_auth.tokens;
    if tokens.is_empty() {
        req.extensions_mut().insert(OperatorId("dev".into()));
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or("");
    let matched = tokens
        .iter()
        .find(|t| t.token.as_bytes().ct_eq(presented.as_bytes()).into());
    match matched {
        Some(t) => {
            req.extensions_mut()
                .insert(OperatorId(t.operator_id.clone()));
            next.run(req).await
        }
        None => envelope_response(OpenAiError::invalid_api_key(
            "missing or invalid client credentials",
        )),
    }
}

// ── DTOs ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ResolveReq {
    api_key: String,
}

#[derive(Serialize)]
struct PrincipalDto {
    account_id: String,
    key_id: String,
}

#[derive(Serialize)]
struct SnapshotDto {
    hard_cap: Option<i64>,
    spent: i64,
    reserved: i64,
}

#[derive(Serialize)]
struct ResolveResp {
    principal: PrincipalDto,
    snapshot: SnapshotDto,
}

#[derive(Deserialize)]
struct ReserveReq {
    account_id: String,
    key_id: String,
    max_tokens: i64,
}

#[derive(Serialize, Default)]
struct ReserveResp {
    #[serde(skip_serializing_if = "Option::is_none")]
    reservation_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rejected: Option<Rejection>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Rejection {
    InsufficientQuota {
        requested: i64,
        available: i64,
    },
    // Part of the frozen wire contract so the cortex client (#57) can map it
    // without a later breaking change. Not yet constructed: the B1 ledger
    // implements Balance caps only; rolling-window key sub-caps (which yield
    // this) land in a follow-up.
    #[allow(dead_code)]
    RateLimited {
        requested: i64,
        available: i64,
        retry_after_secs: u64,
    },
}

#[derive(Deserialize)]
struct SettleReq {
    reservation_id: i64,
    actual_tokens: i64,
}

#[derive(Deserialize)]
struct ReservationRef {
    reservation_id: i64,
}

#[derive(Deserialize)]
struct SnapshotReq {
    account_id: String,
    key_id: String,
}

// ── handlers ────────────────────────────────────────────────────────

/// `POST /authz/v1/resolve` — bearer key → principal + snapshot, or
/// `401 invalid_api_key` (also for a deactivated account: no clue).
async fn resolve(State(state): State<AppState>, Json(req): Json<ResolveReq>) -> Response {
    match ledger::resolve_key(&state.pool, &sha256(&req.api_key)).await {
        Ok(Some(p)) => Json(ResolveResp {
            principal: PrincipalDto {
                account_id: p.account_id.to_string(),
                key_id: p.key_id.to_string(),
            },
            snapshot: SnapshotDto {
                hard_cap: Some(p.hard_cap),
                spent: p.key_spent,
                reserved: p.key_reserved,
            },
        })
        .into_response(),
        Ok(None) => envelope_response(OpenAiError::invalid_api_key("invalid or unknown API key")),
        Err(e) => {
            tracing::error!(error = %e, "resolve query failed");
            envelope_response(OpenAiError::service_unavailable("authority error", Some(5)))
        }
    }
}

/// `POST /authz/v1/reserve` — 200 with `reservation_id` (granted) or
/// `rejected` (budget). Non-2xx only for bad input / server error.
async fn reserve(State(state): State<AppState>, Json(req): Json<ReserveReq>) -> Response {
    let (Ok(account_id), Ok(key_id)) = (
        Uuid::parse_str(&req.account_id),
        Uuid::parse_str(&req.key_id),
    ) else {
        return bad_request("account_id and key_id must be UUIDs");
    };
    match ledger::reserve(&state.pool, account_id, key_id, req.max_tokens).await {
        Ok(reservation_id) => Json(ReserveResp {
            reservation_id: Some(reservation_id),
            rejected: None,
        })
        .into_response(),
        Err(LedgerError::InsufficientQuota {
            requested,
            available,
        }) => Json(ReserveResp {
            reservation_id: None,
            rejected: Some(Rejection::InsufficientQuota {
                requested,
                available,
            }),
        })
        .into_response(),
        Err(LedgerError::AccountNotFound | LedgerError::KeyNotFound) => {
            // Resolve succeeded earlier; the principal vanished (archived /
            // deactivated). Treat as no budget — fail closed at the client.
            Json(ReserveResp {
                reservation_id: None,
                rejected: Some(Rejection::InsufficientQuota {
                    requested: req.max_tokens,
                    available: 0,
                }),
            })
            .into_response()
        }
        Err(LedgerError::Db(e)) => {
            tracing::error!(error = %e, "reserve failed");
            envelope_response(OpenAiError::service_unavailable("authority error", Some(5)))
        }
    }
}

/// `POST /authz/v1/settle` — idempotent; `204`.
async fn settle(State(state): State<AppState>, Json(req): Json<SettleReq>) -> Response {
    match ledger::settle(&state.pool, req.reservation_id, req.actual_tokens).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            tracing::error!(error = %e, "settle failed");
            envelope_response(OpenAiError::service_unavailable("authority error", Some(5)))
        }
    }
}

/// `POST /authz/v1/release` — idempotent; `204`.
async fn release(State(state): State<AppState>, Json(req): Json<ReservationRef>) -> Response {
    match ledger::release(&state.pool, req.reservation_id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            tracing::error!(error = %e, "release failed");
            envelope_response(OpenAiError::service_unavailable("authority error", Some(5)))
        }
    }
}

/// `POST /authz/v1/snapshot` — `{hard_cap, spent, reserved}` or `404`.
async fn snapshot(State(state): State<AppState>, Json(req): Json<SnapshotReq>) -> Response {
    let (Ok(account_id), Ok(key_id)) = (
        Uuid::parse_str(&req.account_id),
        Uuid::parse_str(&req.key_id),
    ) else {
        return bad_request("account_id and key_id must be UUIDs");
    };
    match ledger::snapshot(&state.pool, account_id, key_id).await {
        Ok(Some((hard_cap, spent, reserved))) => Json(SnapshotDto {
            hard_cap: Some(hard_cap),
            spent,
            reserved,
        })
        .into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::error!(error = %e, "snapshot failed");
            envelope_response(OpenAiError::service_unavailable("authority error", Some(5)))
        }
    }
}

fn bad_request(msg: &str) -> Response {
    envelope_response(OpenAiError::new(
        400,
        "invalid_request_error",
        "invalid_request",
        msg,
    ))
}
