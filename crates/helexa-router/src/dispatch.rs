//! Capacity-aware dispatch (#73) — the router's data path.
//!
//! Given an inbound request's `model`, pick a reachable cortex that can
//! serve it (preferring warm/loaded, region-affine, higher-headroom),
//! forward the client's bearer **unchanged** (auth stays at cortex), and
//! stream the response back verbatim via the shared [`helexa_stream`]
//! module. Cortex's #63-shaped rejections (`429 rate_limit_exceeded`,
//! `400 context_length_exceeded`, …) pass through untouched. Transport
//! failures fail over to the next feasible cortex; a genuine HTTP response —
//! any status — is returned as-is and never retried away.
//!
//! The router holds **no entitlement logic**: it routes on capacity, not
//! budget.

use crate::config::CortexEndpoint;
use crate::error::envelope_response;
use crate::state::RouterState;
use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::response::Response;
use cortex_core::error_envelope::OpenAiError;
use helexa_stream::{ChunkObserver, StreamError};
use std::cmp::Reverse;
use std::collections::HashMap;

/// Retry-After hint (seconds) on the router's own transient rejections.
const RETRY_AFTER_SECS: u64 = 5;

/// Outcome of choosing where to send a request.
#[derive(Debug, PartialEq, Eq)]
pub enum Selection {
    /// Feasible reachable cortexes, best-first (failover order).
    Candidates(Vec<CortexEndpoint>),
    /// Some cortex knows the model but none are reachable right now → 503.
    NoReachableCapacity,
    /// No configured cortex serves the model at all → 404.
    UnknownModel,
}

/// Rank the reachable cortexes that can serve `model`, best-first.
///
/// Ordering (each a tie-break for the next): loaded/warm before cold-loadable
/// · region match before not · more healthy nodes before fewer · name for
/// determinism.
pub async fn select_cortexes(state: &RouterState, model: &str) -> Selection {
    let topo = state.topology.read().await;
    let by_name: HashMap<&str, &CortexEndpoint> = state
        .cortexes
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    let mut ranked: Vec<Ranked> = Vec::new();
    let mut known_anywhere = false;

    for (name, t) in topo.iter() {
        let Some(entry) = t.models.get(model) else {
            continue;
        };
        if !crate::state::entry_feasible(entry) {
            continue;
        }
        // Known even via an unreachable cortex's last-good poll — lets us
        // tell "temporarily down" (503) from "nobody serves it" (404).
        known_anywhere = true;
        if !t.reachable {
            continue;
        }
        let Some(ep) = by_name.get(name.as_str()) else {
            continue;
        };
        let region_match = match (&state.region, &ep.region) {
            (Some(r), Some(cr)) => r == cr,
            _ => false,
        };
        ranked.push(Ranked {
            loaded: entry.loaded,
            region_match,
            healthy_nodes: t.healthy_nodes,
            endpoint: (*ep).clone(),
        });
    }

    if ranked.is_empty() {
        return if known_anywhere {
            Selection::NoReachableCapacity
        } else {
            Selection::UnknownModel
        };
    }

    ranked.sort_by(|a, b| {
        // false < true, so negate the "good" booleans to sort good first.
        (
            !a.loaded,
            !a.region_match,
            Reverse(a.healthy_nodes),
            &a.endpoint.name,
        )
            .cmp(&(
                !b.loaded,
                !b.region_match,
                Reverse(b.healthy_nodes),
                &b.endpoint.name,
            ))
    });

    Selection::Candidates(ranked.into_iter().map(|r| r.endpoint).collect())
}

struct Ranked {
    loaded: bool,
    region_match: bool,
    healthy_nodes: u32,
    endpoint: CortexEndpoint,
}

/// Proxy an inbound inference request to a capacity-bearing cortex.
///
/// `path` is the inference path to forward to (same on the cortex, e.g.
/// `/v1/chat/completions`). The body is parsed only to extract `model`.
pub async fn dispatch(
    state: &RouterState,
    path: &str,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(model) = extract_model(&body) else {
        return envelope_response(OpenAiError::new(
            400,
            "invalid_request_error",
            "missing_model_field",
            "missing 'model' field in request body",
        ));
    };

    // Product-tier aliases (#166): resolve federation-level names
    // (helexa/small, helexa/balanced, …) to the real operator model id
    // before selection, and rewrite the body so cortex-side validation,
    // routing and metering all see the true model. Non-alias requests
    // keep their original bytes verbatim.
    let (model, body) = match state.aliases.get(&model) {
        Some(real) => {
            tracing::debug!(alias = %model, model = %real, "resolving tier alias");
            (real.clone(), rewrite_model_in_body(&body, real))
        }
        None => (model, body),
    };

    let candidates = match select_cortexes(state, &model).await {
        Selection::Candidates(c) => c,
        Selection::UnknownModel => {
            return envelope_response(
                OpenAiError::new(
                    404,
                    "invalid_request_error",
                    "model_not_found",
                    format!("no operator serves model '{model}'"),
                )
                .with_param("model"),
            );
        }
        Selection::NoReachableCapacity => {
            return envelope_response(OpenAiError::service_unavailable(
                format!("model '{model}' is temporarily unavailable on all operators"),
                Some(RETRY_AFTER_SECS),
            ));
        }
    };

    // Try candidates in order, failing over only on transport errors. A
    // genuine HTTP response (any status — including cortex's #63 429/400)
    // is returned verbatim and never retried away.
    for ep in &candidates {
        // A candidate whose pinned TLS client failed to build (#74) is
        // disabled — skip it and fail over, same as an unreachable cortex.
        let Some(client) = state.client_for(&ep.name) else {
            tracing::warn!(cortex = %ep.name, "no TLS client (disabled); skipping candidate");
            continue;
        };
        let url = format!("{}{}", ep.endpoint, path);
        tracing::info!(cortex = %ep.name, url = %url, model = %model, "dispatching");
        match helexa_stream::forward_streaming(
            client,
            &url,
            headers.clone(),
            body.clone(),
            NoopObserver,
        )
        .await
        {
            Ok(resp) => return resp,
            Err(StreamError::Upstream(e)) => {
                tracing::warn!(
                    cortex = %ep.name,
                    url = %url,
                    error = %e,
                    "cortex unreachable; failing over"
                );
                continue;
            }
            Err(StreamError::ResponseBuild(msg)) => {
                tracing::error!(cortex = %ep.name, error = %msg, "failed to build proxied response");
                return envelope_response(OpenAiError::without_code(
                    500,
                    "api_error",
                    "failed to build proxied response",
                ));
            }
        }
    }

    // Every feasible cortex failed to connect.
    tracing::warn!(model = %model, tried = candidates.len(), "all feasible operators unreachable");
    envelope_response(OpenAiError::service_unavailable(
        format!("all operators able to serve '{model}' are unreachable"),
        Some(RETRY_AFTER_SECS),
    ))
}

/// Pull the `model` field out of a request body without re-serialising it.
fn extract_model(body: &Bytes) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(str::to_string)
}

/// Re-serialise the body with `model` replaced — only used on the alias
/// path, so ordinary requests still forward their original bytes. A body
/// that fails to parse is forwarded unchanged (cortex will reject it with
/// its own envelope).
fn rewrite_model_in_body(body: &Bytes, model: &str) -> Bytes {
    let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return body.clone();
    };
    if let Some(obj) = v.as_object_mut() {
        obj.insert("model".into(), serde_json::Value::String(model.into()));
    }
    serde_json::to_vec(&v)
        .map(Bytes::from)
        .unwrap_or_else(|_| body.clone())
}

/// The router proxies bytes verbatim and keeps no per-request policy, so it
/// needs no observation hooks. (Token metrics/metering stay at cortex.)
struct NoopObserver;

impl ChunkObserver for NoopObserver {
    fn observe(&mut self, _chunk: &[u8]) {}
    fn finish(&mut self) {}
}
