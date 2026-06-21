use crate::state::RouterState;
use axum::{Json, Router, extract::State, routing::get};
use cortex_core::openai::ModelsResponse;
use serde_json::{Value, json};
use std::sync::Arc;

/// Routes served by the router skeleton. The inference paths
/// (`/v1/chat/completions`, `/v1/messages`, …) arrive with capacity-aware
/// dispatch (#73); for now the router only answers `/health` and a stub
/// `/v1/models`.
pub fn api_routes() -> Router<Arc<RouterState>> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/", get(health))
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

/// `GET /v1/models` — empty catalogue stub. The real cross-operator union
/// (catalogue × topology feasibility, aggregated from each cortex) is the
/// federation-catalogue issue (#75).
async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![],
    })
}
