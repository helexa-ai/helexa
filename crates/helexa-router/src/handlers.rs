use crate::state::RouterState;
use crate::{catalogue, dispatch};
use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::{Json, Router, extract::State, routing::get, routing::post};
use serde_json::{Value, json};
use std::sync::Arc;

/// Routes served by the router. Inference paths are capacity-aware-dispatched
/// (#73) to a downstream cortex; `/health` and a stub `/v1/models` are local.
pub fn api_routes() -> Router<Arc<RouterState>> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/", get(health))
}

// ── Inference paths — forwarded verbatim to a chosen cortex ──────────
//
// Each handler dispatches to the same path on a capacity-bearing cortex.
// The body is parsed only to read `model`; the bearer and bytes are
// forwarded unchanged, and the SSE response streams back verbatim.

async fn chat_completions(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    dispatch::dispatch(&state, "/v1/chat/completions", headers, body).await
}

async fn completions(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    dispatch::dispatch(&state, "/v1/completions", headers, body).await
}

async fn responses(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    dispatch::dispatch(&state, "/v1/responses", headers, body).await
}

async fn messages(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    dispatch::dispatch(&state, "/v1/messages", headers, body).await
}

/// `GET /health` — router liveness plus a summary of downstream cortex
/// reachability from the topology poller (#72). `status` reflects the
/// router process itself (always `ok` if it answers); downstream health is
/// the informational `cortexes` block, so a fully-degraded fleet doesn't
/// make the router look dead to its own liveness probe.
async fn health(State(state): State<Arc<RouterState>>) -> Json<Value> {
    let topo = state.topology.read().await;
    let reachable = topo.values().filter(|t| t.reachable).count();
    Json(json!({
        "status": "ok",
        "cortexes": {
            "configured": state.cortexes.len(),
            "reachable": reachable,
        }
    }))
}

/// `GET /v1/models` — the federation catalogue (#75): the deduped union of
/// every reachable cortex's `/v1/models`, so a client doing discovery
/// against the router resolves the whole federation without knowing about
/// operators or cortexes.
async fn list_models(State(state): State<Arc<RouterState>>) -> Json<Value> {
    let topo = state.topology.read().await;
    let data: Vec<Value> = catalogue::aggregate_models(&topo)
        .iter()
        .map(|e| json!(e))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}
