//! HTTP API handlers for the neuron daemon.

use crate::activation::ActivationTracker;
use crate::harness::HarnessRegistry;
use crate::harness::candle::{CandleHarness, InferenceError};
use crate::health::HealthCache;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::harness::ModelSpec;
use cortex_core::openai::ChatCompletionRequest;
use futures::stream::{self, StreamExt};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;

/// Shared state for the neuron HTTP server.
pub struct NeuronState {
    pub discovery: DiscoveryResponse,
    pub health_cache: Arc<HealthCache>,
    pub registry: RwLock<HarnessRegistry>,
    /// Typed handle to the candle harness for inference routes. Cached at
    /// startup so `/v1/chat/completions` doesn't have to hold the registry
    /// read lock or perform dyn-Trait dispatch per request.
    pub candle: Option<Arc<CandleHarness>>,
    /// Activation-time pre-warm progress. Updated by the background
    /// `load_default_models` task, read by the `/health` handler.
    pub activation: Arc<ActivationTracker>,
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
    // HealthCache owns the uptime + per-device readings; the activation
    // tracker owns the pre-warm progress. We compose the response here
    // so the cache stays a thin runtime-state cache and doesn't need to
    // know about activation lifecycle.
    let mut snapshot = state.health_cache.snapshot().await;
    snapshot.activation = state.activation.snapshot().await;
    Json(snapshot)
}

async fn list_models(State(state): State<Arc<NeuronState>>) -> impl IntoResponse {
    let registry = state.registry.read().await;
    match registry.list_all_models().await {
        Ok(models) => Json(json!(models)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("{e:#}")})),
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
        Err(e) => {
            // Log the full anyhow chain server-side so journalctl shows
            // the underlying failure (hf-hub timeout, permission denied,
            // disk full, etc.) without needing to inspect the HTTP
            // response body separately.
            tracing::warn!(
                model = %spec.model_id,
                error = %format!("{e:#}"),
                "load_model failed"
            );
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("{e:#}")})),
            )
                .into_response()
        }
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
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("{e:#}")})),
        )
            .into_response(),
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

/// OpenAI-compatible chat completions. Dispatches to streaming SSE when
/// `stream: true` is set on the request; otherwise returns a single
/// `ChatCompletionResponse`.
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
        match candle.chat_completion_stream(req).await {
            Ok(rx) => {
                // Each chunk → one SSE `data: {json}` line. After the
                // channel closes, append the OpenAI [DONE] terminator.
                let body_stream = ReceiverStream::new(rx).map(|chunk| {
                    let body = serde_json::to_string(&chunk).unwrap_or_default();
                    Ok::<_, Infallible>(Event::default().data(body))
                });
                let done_stream =
                    stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
                Sse::new(body_stream.chain(done_stream))
                    .keep_alive(KeepAlive::default())
                    .into_response()
            }
            Err(InferenceError::ModelNotLoaded(id)) => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("model '{id}' not loaded on this neuron")})),
            )
                .into_response(),
            Err(InferenceError::PromptTooLong { prompt_len, max }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!("prompt has {prompt_len} tokens but max is {max}"),
                    "code": "prompt_too_long",
                    "prompt_len": prompt_len,
                    "max": max,
                })),
            )
                .into_response(),
            Err(InferenceError::InsufficientVram {
                free_mb,
                required_mb,
            }) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": format!(
                        "insufficient free VRAM: {free_mb} MiB free, need at least {required_mb} MiB"
                    ),
                    "code": "insufficient_vram",
                    "free_mb": free_mb,
                    "required_mb": required_mb,
                })),
            )
                .into_response(),
            Err(InferenceError::Other(e)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("{e:#}")})),
            )
                .into_response(),
        }
    } else {
        match candle.chat_completion(req).await {
            Ok(resp) => Json(resp).into_response(),
            Err(InferenceError::ModelNotLoaded(id)) => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("model '{id}' not loaded on this neuron")})),
            )
                .into_response(),
            Err(InferenceError::PromptTooLong { prompt_len, max }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!("prompt has {prompt_len} tokens but max is {max}"),
                    "code": "prompt_too_long",
                    "prompt_len": prompt_len,
                    "max": max,
                })),
            )
                .into_response(),
            Err(InferenceError::InsufficientVram {
                free_mb,
                required_mb,
            }) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": format!(
                        "insufficient free VRAM: {free_mb} MiB free, need at least {required_mb} MiB"
                    ),
                    "code": "insufficient_vram",
                    "free_mb": free_mb,
                    "required_mb": required_mb,
                })),
            )
                .into_response(),
            Err(InferenceError::Other(e)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("{e:#}")})),
            )
                .into_response(),
        }
    }
}
