//! HTTP API handlers for the neuron daemon.

use crate::activation::ActivationTracker;
use crate::harness::HarnessRegistry;
use crate::harness::candle::{CandleHarness, InferenceError};
use crate::harness::preflight::PreflightError;
use crate::health::HealthCache;
use crate::wire::{openai_chat, openai_responses};
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::harness::ModelSpec;
use cortex_core::openai::{ChatCompletionRequest, MessageContent};
use cortex_core::responses::{ResponsesRequest, ResponsesUsage};
use futures::stream::{self, StreamExt};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
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
        .route("/v1/responses", post(responses))
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
    // Driver/library mismatch preflight (#19): every CUDA load is
    // guaranteed to fail until the host reboots. Reject up front with
    // the operator-actionable reason instead of letting the load die
    // minutes later inside cuInit/NCCL with a cryptic error.
    if let Some(reason) = &state.discovery.cuda_unavailable_reason {
        tracing::warn!(model = %spec.model_id, reason = %reason, "load_model rejected: CUDA unavailable");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": reason,
                "code": "cuda_unavailable",
            })),
        )
            .into_response();
    }
    let registry = state.registry.read().await;
    match registry.load_model(&spec).await {
        Ok(()) => Json(json!({"status": "loaded"})).into_response(),
        Err(e) => {
            // If the underlying failure is a structured preflight
            // rejection, surface it as 422 Unprocessable Entity with
            // the typed JSON body. The kind/model_id/suggestion/etc.
            // fields let cortex (and operators reading the response
            // directly) act on the failure without parsing free text.
            if let Some(pf) = e.downcast_ref::<PreflightError>() {
                tracing::warn!(
                    model = %spec.model_id,
                    reason = preflight_kind(pf),
                    detail = %pf,
                    "load_model rejected by preflight"
                );
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({ "error": pf })),
                )
                    .into_response();
            }
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

/// Short kebab-case tag for a preflight failure, used as a structured
/// log field for journalctl-side filtering. Mirrors the same helper in
/// `startup.rs`; duplicated to keep the module surfaces independent.
fn preflight_kind(err: &PreflightError) -> &'static str {
    match err {
        PreflightError::RepoFetchFailed { .. } => "repo_fetch_failed",
        PreflightError::EmptyRepo { .. } => "empty_repo",
        PreflightError::TpRequiresSafetensors { .. } => "tp_requires_safetensors",
        PreflightError::QuantNotFound { .. } => "quant_not_found",
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
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let Some(candle) = state.candle.as_ref().map(Arc::clone) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "candle harness not enabled on this neuron"})),
        )
            .into_response();
    };

    // Reasoning-content opt-in. Off by default → naïve clients
    // (Zed's commit-message generator, vanilla OpenAI clients)
    // never see `<think>` blocks. On when the caller sends
    // `x-include-thinking: true` (helexa-acp does this so its
    // own ThinkParser keeps working unchanged).
    let include_thinking = headers
        .get("x-include-thinking")
        .and_then(|v| v.to_str().ok())
        .map(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let chat_config = openai_chat::ChatProjectionConfig {
        include_thinking,
        reasoning_markers: None, // filled in from the loaded model inside candle
    };

    if req.stream.unwrap_or(false) {
        match candle.chat_completion_stream_with(req, chat_config).await {
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
            Err(InferenceError::VisionUnsupported { model_id }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!(
                        "model '{model_id}' does not support image input"
                    ),
                    "code": "vision_unsupported",
                    "model_id": model_id,
                    "suggestion": "load a vision-capable model or remove image_url content parts",
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
            Err(InferenceError::VisionUnsupported { model_id }) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!(
                        "model '{model_id}' does not support image input"
                    ),
                    "code": "vision_unsupported",
                    "model_id": model_id,
                    "suggestion": "load a vision-capable model or remove image_url content parts",
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

/// OpenAI Responses API (`POST /v1/responses`). Translates the
/// Responses-shaped request into a chat-completions one the candle
/// harness already understands, then re-projects the harness's
/// event stream into the Responses event family.
async fn responses(
    State(state): State<Arc<NeuronState>>,
    Json(req): Json<ResponsesRequest>,
) -> impl IntoResponse {
    let Some(candle) = state.candle.as_ref().map(Arc::clone) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "candle harness not enabled on this neuron"})),
        )
            .into_response();
    };

    let stream_requested = req.stream;
    let model_id = req.model.clone();
    let response_id = mint_response_id();
    let message_item_id = mint_message_item_id();

    // Translate Responses → chat completions. The only failure
    // mode today is `previous_response_id` set, which we reject
    // with 400 — stateful conversations need a persistence layer
    // we haven't built.
    let mut chat_req = match openai_responses::request_to_chat(req) {
        Ok(r) => r,
        Err(openai_responses::TranslateError::ChainedConversationNotSupported) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "previous_response_id is not supported on this neuron",
                    "code": "chained_conversation_not_supported"
                })),
            )
                .into_response();
        }
    };
    chat_req.stream = Some(stream_requested);

    if stream_requested {
        match candle
            .responses_stream(chat_req, response_id, message_item_id)
            .await
        {
            Ok(rx) => {
                // Each ResponseStreamFrame → one SSE event carrying
                // both an event name and JSON data. The Responses
                // API doesn't use a `[DONE]` terminator — clients
                // see the `response.completed` event as the end of
                // the stream.
                let body_stream = ReceiverStream::new(rx).map(|frame| {
                    let body = serde_json::to_string(&frame.data).unwrap_or_else(|_| "{}".into());
                    Ok::<_, Infallible>(Event::default().event(frame.event_name).data(body))
                });
                Sse::new(body_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            }
            Err(e) => inference_error_response(e),
        }
    } else {
        // Non-streaming: drive the existing chat completion path
        // and translate the result. We don't currently re-tokenise
        // to compute usage; the harness returns it via the chat
        // response and we pass it through.
        match candle.chat_completion(chat_req).await {
            Ok(chat_resp) => {
                // Extract the assistant text (chat completions
                // always emits one choice on the candle path).
                let text = chat_resp
                    .choices
                    .first()
                    .map(|c| match &c.message.content {
                        MessageContent::Text(t) => t.clone(),
                        MessageContent::Parts(_) => {
                            // Candle output is always text today;
                            // a Parts response would be surprising.
                            // Empty-string fallback is safer than
                            // a panic.
                            String::new()
                        }
                    })
                    .unwrap_or_default();
                let finish = chat_resp
                    .choices
                    .first()
                    .and_then(|c| c.finish_reason.as_deref())
                    .map(finish_reason_from_str)
                    .unwrap_or(crate::wire::FinishReason::Stop);
                let usage = chat_resp.usage.as_ref().map(|u| ResponsesUsage {
                    input_tokens: u.prompt_tokens,
                    output_tokens: u.completion_tokens,
                    total_tokens: u.prompt_tokens + u.completion_tokens,
                });
                let meta = openai_responses::ResponseMeta {
                    response_id: mint_response_id(),
                    created_at: unix_now_secs(),
                    model_id,
                    message_item_id: mint_message_item_id(),
                };
                let _ = chat_resp; // make the borrow-checker happy if `text` consumed it
                let resp = openai_responses::build_response(&meta, text, finish, usage);
                Json(resp).into_response()
            }
            Err(e) => inference_error_response(e),
        }
    }
}

fn finish_reason_from_str(s: &str) -> crate::wire::FinishReason {
    use crate::wire::FinishReason;
    match s {
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

/// Centralised mapping from [`InferenceError`] to an HTTP response.
/// Lifted out so the chat-completions and responses handlers stay
/// readable and changes to error-code semantics happen in one spot.
fn inference_error_response(err: InferenceError) -> axum::response::Response {
    match err {
        InferenceError::ModelNotLoaded(id) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("model '{id}' not loaded on this neuron")})),
        )
            .into_response(),
        InferenceError::PromptTooLong { prompt_len, max } => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("prompt has {prompt_len} tokens but max is {max}"),
                "code": "prompt_too_long",
                "prompt_len": prompt_len,
                "max": max,
            })),
        )
            .into_response(),
        InferenceError::InsufficientVram {
            free_mb,
            required_mb,
        } => (
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
        InferenceError::VisionUnsupported { model_id } => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!(
                    "model '{model_id}' does not support image input"
                ),
                "code": "vision_unsupported",
                "model_id": model_id,
                "suggestion": "load a vision-capable model or remove image_url content parts",
            })),
        )
            .into_response(),
        InferenceError::Other(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("{e:#}")})),
        )
            .into_response(),
    }
}

fn mint_response_id() -> String {
    format!("resp_{:x}", unix_subsec_nanos())
}

fn mint_message_item_id() -> String {
    format!("msg_{:x}", unix_subsec_nanos())
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_subsec_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
