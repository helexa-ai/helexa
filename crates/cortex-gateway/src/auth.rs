//! API-key authentication + principal resolution (#49).
//!
//! Identity rides standard bearer auth only — `Authorization: Bearer <key>`
//! — which is what keeps every tier OpenAI-compatible by construction (no
//! custom required headers or body fields, per #47). The middleware resolves
//! the key to a [`Principal`] via the [`EntitlementProvider`], carries it in
//! the request extensions for cortex-side metering/enforcement (#51/#52), and
//! stamps it as internal headers on the request so it reaches neuron, which
//! trusts cortex's assertion over WireGuard (#54).
//!
//! Anti-spoofing: any client-supplied principal header is **stripped** before
//! the authoritative value is stamped, so a client can never assert a
//! principal it didn't authenticate as.
//!
//! Rejection contract (#63): missing key under `require_auth`, or any present
//! but unresolvable key, yields `401 invalid_api_key` in the #60 envelope.

use crate::error::envelope_response;
use crate::state::CortexState;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;
use cortex_core::entitlements::{HEADER_ACCOUNT_ID, HEADER_KEY_ID};
use cortex_core::error_envelope::OpenAiError;
use std::sync::Arc;

/// Endpoints that never require auth: liveness/readiness probes. Everything
/// else flows through resolution.
fn is_public(path: &str) -> bool {
    path == "/health" || path == "/"
}

/// Extract the bearer token from an `Authorization` header value, if present
/// and well-formed. Scheme match is case-insensitive per RFC 7235.
fn parse_bearer(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = raw.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        let token = token.trim();
        (!token.is_empty()).then(|| token.to_string())
    } else {
        None
    }
}

/// Axum middleware: resolve the bearer key, attach the principal, stamp the
/// internal headers. Wired in `build_app` via `from_fn_with_state`.
pub async fn require_principal(
    State(fleet): State<Arc<CortexState>>,
    mut req: Request,
    next: Next,
) -> Response {
    if is_public(req.uri().path()) {
        return next.run(req).await;
    }

    // Anti-spoof: drop any client-supplied principal headers up front.
    {
        let headers = req.headers_mut();
        headers.remove(HEADER_ACCOUNT_ID);
        headers.remove(HEADER_KEY_ID);
    }

    match parse_bearer(req.headers()) {
        Some(key) => match fleet.entitlements.resolve(&key).await {
            Ok(principal) => {
                // Stamp the authoritative principal for neuron. Account/key
                // ids come from operator config, so they're valid header
                // values; guard anyway and skip a malformed one rather than
                // panic.
                if let (Ok(account), Ok(key_id)) = (
                    HeaderValue::from_str(&principal.account_id),
                    HeaderValue::from_str(&principal.key_id),
                ) {
                    let headers = req.headers_mut();
                    headers.insert(HEADER_ACCOUNT_ID, account);
                    headers.insert(HEADER_KEY_ID, key_id);
                }
                // Carry the typed principal for cortex-side metering (#51)
                // and budget enforcement (#52).
                req.extensions_mut().insert(principal);
                next.run(req).await
            }
            // A present-but-invalid credential is always an error, even when
            // anonymous access is otherwise allowed.
            Err(_) => unauthorized("invalid API key"),
        },
        None => {
            if fleet.require_auth {
                unauthorized("missing API key; supply 'Authorization: Bearer <key>'")
            } else {
                next.run(req).await
            }
        }
    }
}

/// `401 invalid_api_key` in the standard envelope (#63).
fn unauthorized(message: &str) -> Response {
    envelope_response(OpenAiError::invalid_api_key(message))
}

/// Copy the cortex-stamped principal headers from an inbound [`HeaderMap`]
/// onto an outbound reqwest builder. Used by the Anthropic proxy paths,
/// which construct their own upstream requests instead of going through
/// [`crate::proxy::forward_request`] (which forwards all headers verbatim).
pub fn forward_principal_headers(
    mut builder: reqwest::RequestBuilder,
    headers: &HeaderMap,
) -> reqwest::RequestBuilder {
    for name in [HEADER_ACCOUNT_ID, HEADER_KEY_ID] {
        if let Some(value) = headers.get(name) {
            builder = builder.header(name, value);
        }
    }
    builder
}
