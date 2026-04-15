//! HTTP API handlers for the neuron daemon.

use crate::health::HealthCache;
use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::get;
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use std::sync::Arc;

/// Shared state for the neuron HTTP server.
pub struct NeuronState {
    pub discovery: DiscoveryResponse,
    pub health_cache: Arc<HealthCache>,
}

/// Build the neuron API router.
pub fn neuron_routes() -> Router<Arc<NeuronState>> {
    Router::new()
        .route("/discovery", get(discovery_handler))
        .route("/health", get(health_handler))
}

async fn discovery_handler(State(state): State<Arc<NeuronState>>) -> Json<DiscoveryResponse> {
    Json(state.discovery.clone())
}

async fn health_handler(State(state): State<Arc<NeuronState>>) -> Json<HealthResponse> {
    Json(state.health_cache.snapshot().await)
}
