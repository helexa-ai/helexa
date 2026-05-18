//! HTTP API handlers for the neuron daemon.

use crate::harness::HarnessRegistry;
use crate::harness::candle::{CandleHarness, InferenceError};
use crate::health::HealthCache;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::harness::ModelSpec;
use cortex_core::openai::ChatCompletionRequest;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared state for the neuron HTTP server.
pub struct NeuronState {
    pub discovery: DiscoveryResponse,
    pub health_cache: Arc<HealthCache>,
    pub registry: RwLock<HarnessRegistry>,
    /// Typed handle to the candle harness for inference routes. Cached at
    /// startup so `/v1/chat/completions` doesn't have to hold the registry
    /// read lock or perform dyn-Trait dispatch per request.
    pub candle: Option<Arc<CandleHarness>>,
}

/// Build the neuron API router.
pub fn neuron_routes() -> Router<Arc<NeuronState>> {
    Router::new()
        .route("/discovery", get(discovery_handler))
        .route("/health", get(health_handler))
        .route("/models", get(list_models))
        .route("/models/load", post(load_model))
        .route("/models/unload", post(unload_model))
        .route("/models/{model_id}/endpoint", get(model_endpoint))
        .route("/v1/chat/completions", post(chat_completions))
}

async fn discovery_handler(State(state): State<Arc<NeuronState>>) -> Json<DiscoveryResponse> {
    Json(state.discovery.clone())
}

async fn health_handler(State(state): State<Arc<NeuronState>>) -> Json<HealthResponse> {
    Json(state.health_cache.snapshot().await)
}

async fn list_models(State(state): State<Arc<NeuronState>>) -> impl IntoResponse {
    let registry = state.registry.read().await;
    match registry.list_all_models().await {
        Ok(models) => Json(json!(models)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn load_model(
    State(state): State<Arc<NeuronState>>,
    Json(spec): Json<ModelSpec>,
) -> impl IntoResponse {
    let registry = state.registry.read().await;
    match registry.load_model(&spec).await {
        Ok(()) => Json(json!({"status": "loaded"})).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn unload_model(
    State(state): State<Arc<NeuronState>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let model_id = match body.get("model_id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "missing model_id"})),
            )
                .into_response();
        }
    };

    let registry = state.registry.read().await;
    match registry.unload_model(&model_id).await {
        Ok(()) => Json(json!({"status": "unloaded"})).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

async fn model_endpoint(
    State(state): State<Arc<NeuronState>>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    let registry = state.registry.read().await;
    match registry.inference_endpoint(&model_id).await {
        Some(url) => Json(json!({"url": url})).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("model '{}' not loaded", model_id)})),
        )
            .into_response(),
    }
}

/// OpenAI-compatible chat completions. Non-streaming for Stage 3; the
/// streaming path is added in Stage 4.
async fn chat_completions(
    State(state): State<Arc<NeuronState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let Some(candle) = state.candle.as_ref().map(Arc::clone) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "candle harness not enabled on this neuron"})),
        )
            .into_response();
    };

    if req.stream.unwrap_or(false) {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error": "streaming responses arrive in Stage 4"})),
        )
            .into_response();
    }

    match candle.chat_completion(req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(InferenceError::ModelNotLoaded(id)) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("model '{id}' not loaded on this neuron")})),
        )
            .into_response(),
        Err(InferenceError::Other(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
