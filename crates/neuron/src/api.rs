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
        .route("/version", get(version_handler))
        .route("/discovery", get(discovery_handler))
        .route("/health", get(health_handler))
        .route("/models", get(list_models))
        .route("/models/load", post(load_model))
        .route("/models/unload", post(unload_model))
        .route("/models/{model_id}/endpoint", get(model_endpoint))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
}

/// `GET /version` — the daemon's own build identity (git SHA, enabled
/// features, rustc/candle versions). Static for the process lifetime, so
/// no state is touched. This is the canonical "which build is live"
/// probe for fleet validation and benchmark attribution.
async fn version_handler() -> Json<cortex_core::build_info::BuildInfo> {
    Json(crate::version::build_info())
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

/// Default `chat_template_kwargs.enable_thinking` to `include_thinking`
/// when the client didn't set it explicitly, leaving any explicit client
/// choice untouched. See the call site in [`chat_completions`] for the
/// rationale (reasoning eating the token budget for clients that drop it).
fn default_enable_thinking(req: &mut ChatCompletionRequest, include_thinking: bool) {
    if req
        .extra
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .is_some()
    {
        return; // client chose explicitly — respect it
    }
    if !req.extra.is_object() {
        req.extra = json!({});
    }
    let Some(obj) = req.extra.as_object_mut() else {
        return;
    };
    let kwargs = obj
        .entry("chat_template_kwargs")
        .or_insert_with(|| json!({}));
    if !kwargs.is_object() {
        *kwargs = json!({});
    }
    if let Some(kw) = kwargs.as_object_mut() {
        kw.insert("enable_thinking".into(), json!(include_thinking));
    }
}

/// OpenAI-compatible chat completions. Dispatches to streaming SSE when
/// `stream: true` is set on the request; otherwise returns a single
/// `ChatCompletionResponse`.
async fn chat_completions(
    State(state): State<Arc<NeuronState>>,
    headers: axum::http::HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
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

    // Couple reasoning *generation* to reasoning *surfacing*. Reasoning
    // models (Qwen3.6) think by default, and that `<think>` block can
    // consume the entire `max_tokens` budget — which, when we then drop
    // it (`include_thinking == false`, the default for OpenAI/Anthropic
    // clients like Claude Code), leaves the visible answer empty or
    // truncated. So when the caller isn't going to see the reasoning,
    // don't generate it: default `enable_thinking` to `include_thinking`.
    // A client that explicitly set `chat_template_kwargs.enable_thinking`
    // wins; thinking-aware clients (helexa-acp, `x-include-thinking:
    // true`) keep reasoning on.
    default_enable_thinking(&mut req, include_thinking);

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
            Err(e) => inference_error_response(e),
        }
    } else {
        match candle.chat_completion(req).await {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => inference_error_response(e),
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
                    // Non-streaming reasoning accounting deferred (#64).
                    output_tokens_details: None,
                    input_tokens_details: None,
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
///
/// Emits the OpenAI-standard *nested* error envelope:
///
/// ```json
/// { "error": { "message": "...", "type": "...", "code": "...", "param": null } }
/// ```
///
/// OpenAI-compatible clients (opencode, the openai SDK) reach into
/// `error.type` / `error.code` to drive behaviour — most importantly,
/// `code == "context_length_exceeded"` triggers auto-compaction and
/// retry rather than a hard failure. A flat `{"error": "..."}` string
/// is invisible to that logic, so every variant nests here. Diagnostic
/// extras (prompt_len, free_mb, …) ride *inside* the error object so
/// they don't break the envelope shape.
fn inference_error_response(err: InferenceError) -> axum::response::Response {
    use cortex_core::error_envelope::OpenAiError;
    let env = match err {
        InferenceError::ModelNotLoaded(id) => OpenAiError::new(
            404,
            "invalid_request_error",
            "model_not_found",
            format!("model '{id}' not loaded on this neuron"),
        )
        .with_extra("model_id", json!(id)),
        // OpenAI's canonical context-overflow error. opencode keys on
        // `code == "context_length_exceeded"` and the message phrasing
        // ("maximum context length is N tokens") to auto-compact+retry.
        InferenceError::PromptTooLong { prompt_len, max } => {
            OpenAiError::context_length_exceeded(format!(
                "This model's maximum context length is {max} tokens. \
                 However, your messages resulted in {prompt_len} tokens. \
                 Please reduce the length of the messages."
            ))
            .with_extra("prompt_len", json!(prompt_len))
            .with_extra("max", json!(max))
        }
        // VRAM frees as the in-flight request(s) complete, so this is a
        // transient 503 — advertise a short Retry-After (#63).
        InferenceError::InsufficientVram {
            free_mb,
            required_mb,
        } => OpenAiError::new(
            503,
            "api_error",
            "insufficient_vram",
            format!("insufficient free VRAM: {free_mb} MiB free, need at least {required_mb} MiB"),
        )
        .with_retry_after(5)
        .with_extra("free_mb", json!(free_mb))
        .with_extra("required_mb", json!(required_mb)),
        InferenceError::VisionUnsupported { model_id } => OpenAiError::new(
            400,
            "invalid_request_error",
            "vision_unsupported",
            format!("model '{model_id}' does not support image input"),
        )
        .with_extra("model_id", json!(model_id))
        .with_extra(
            "suggestion",
            json!("load a vision-capable model or remove image_url content parts"),
        ),
        InferenceError::TemplateRenderFailed { detail } => OpenAiError::new(
            422,
            "invalid_request_error",
            "template_render_failed",
            format!("chat template could not render this request: {detail}"),
        ),
        // Admission control refused (#53): a fast, retryable "busy" signal.
        // 503 (service busy) + Retry-After; opencode/AI SDK back off.
        InferenceError::Overloaded { retry_after_secs } => OpenAiError::new(
            503,
            "rate_limit_error",
            "rate_limit_exceeded",
            "model is busy (admission queue full); retry shortly",
        )
        .with_retry_after(retry_after_secs),
        InferenceError::Other(e) => OpenAiError::without_code(500, "api_error", format!("{e:#}")),
    };
    envelope_response(env)
}

/// Neuron adapter: turn the shared [`cortex_core::error_envelope::OpenAiError`]
/// into an axum response, setting `Retry-After` when the envelope carries one.
/// cortex-core owns the envelope shape (#60/#63); this is the only crossing
/// from that data into axum on the neuron side.
fn envelope_response(err: cortex_core::error_envelope::OpenAiError) -> axum::response::Response {
    let status = StatusCode::from_u16(err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let retry_after = err.retry_after_secs;
    let mut response = (status, Json(err.body())).into_response();
    if let Some(secs) = retry_after
        && let Ok(value) = axum::http::HeaderValue::from_str(&secs.to_string())
    {
        response
            .headers_mut()
            .insert(axum::http::header::RETRY_AFTER, value);
    }
    response
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

#[cfg(test)]
mod thinking_tests {
    use super::*;

    fn req(value: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("valid ChatCompletionRequest")
    }

    fn enable_thinking(r: &ChatCompletionRequest) -> Option<bool> {
        r.extra
            .get("chat_template_kwargs")
            .and_then(|k| k.get("enable_thinking"))
            .and_then(|v| v.as_bool())
    }

    #[test]
    fn defaults_enable_thinking_to_include_thinking_false() {
        let mut r = req(json!({"model": "m", "messages": []}));
        default_enable_thinking(&mut r, false);
        assert_eq!(enable_thinking(&r), Some(false));
    }

    #[test]
    fn defaults_enable_thinking_true_when_surfacing() {
        let mut r = req(json!({"model": "m", "messages": []}));
        default_enable_thinking(&mut r, true);
        assert_eq!(enable_thinking(&r), Some(true));
    }

    #[test]
    fn explicit_client_choice_is_respected() {
        let mut r = req(json!({
            "model": "m", "messages": [],
            "chat_template_kwargs": {"enable_thinking": true}
        }));
        // include_thinking=false would normally force false; explicit wins.
        default_enable_thinking(&mut r, false);
        assert_eq!(enable_thinking(&r), Some(true));
    }

    #[test]
    fn preserves_other_chat_template_kwargs() {
        let mut r = req(json!({
            "model": "m", "messages": [],
            "chat_template_kwargs": {"some_other": 42}
        }));
        default_enable_thinking(&mut r, false);
        assert_eq!(enable_thinking(&r), Some(false));
        assert_eq!(
            r.extra["chat_template_kwargs"]["some_other"],
            json!(42),
            "existing kwargs must survive"
        );
    }
}

#[cfg(test)]
mod error_envelope_tests {
    use super::*;
    use axum::http::StatusCode;

    /// Drive an `InferenceError` through the mapper and decode the
    /// `(status, json)` pair it produces.
    async fn map(err: InferenceError) -> (StatusCode, Value) {
        let resp = inference_error_response(err);
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("buffer error body");
        let body: Value = serde_json::from_slice(&bytes).expect("error body is JSON");
        (status, body)
    }

    #[tokio::test]
    async fn prompt_too_long_is_context_length_exceeded() {
        let (status, body) = map(InferenceError::PromptTooLong {
            prompt_len: 60_000,
            max: 49_152,
        })
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        // The envelope must be nested under `error`, not a flat string.
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error object");
        assert_eq!(error["type"], "invalid_request_error");
        assert_eq!(
            error["code"], "context_length_exceeded",
            "opencode keys on this code to auto-compact and retry"
        );
        assert_eq!(error["param"], Value::Null);
        // Phrasing opencode/openai clients pattern-match on.
        let msg = error["message"].as_str().unwrap();
        assert!(
            msg.contains("maximum context length is 49152 tokens"),
            "message was: {msg}"
        );
        // Diagnostics ride inside the error object.
        assert_eq!(error["prompt_len"], 60_000);
        assert_eq!(error["max"], 49_152);
    }

    #[tokio::test]
    async fn model_not_loaded_is_404_model_not_found() {
        let (status, body) = map(InferenceError::ModelNotLoaded("Qwen/X".into())).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let error = &body["error"];
        assert_eq!(error["type"], "invalid_request_error");
        assert_eq!(error["code"], "model_not_found");
        assert_eq!(error["model_id"], "Qwen/X");
    }

    #[tokio::test]
    async fn insufficient_vram_is_503_api_error() {
        let (status, body) = map(InferenceError::InsufficientVram {
            free_mb: 1_024,
            required_mb: 8_192,
        })
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let error = &body["error"];
        assert_eq!(error["type"], "api_error");
        assert_eq!(error["code"], "insufficient_vram");
        assert_eq!(error["free_mb"], 1_024);
        assert_eq!(error["required_mb"], 8_192);
    }

    #[tokio::test]
    async fn overloaded_is_503_rate_limited_with_retry_after() {
        // Admission rejection (#53) → fast, retryable backpressure.
        let resp = inference_error_response(InferenceError::Overloaded {
            retry_after_secs: 7,
        });
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let retry = resp
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .expect("admission rejection must advertise Retry-After");
        assert_eq!(retry.to_str().unwrap(), "7");

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn insufficient_vram_carries_retry_after() {
        // Transient 503 — VRAM frees as in-flight requests finish, so the
        // client should back off and retry (#63).
        let resp = inference_error_response(InferenceError::InsufficientVram {
            free_mb: 1_024,
            required_mb: 8_192,
        });
        let retry = resp
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .expect("transient 503 must advertise Retry-After");
        assert_eq!(retry.to_str().unwrap(), "5");
    }

    #[tokio::test]
    async fn permanent_rejections_have_no_retry_after() {
        // context_length_exceeded is permanent for this request — no hint.
        let resp = inference_error_response(InferenceError::PromptTooLong {
            prompt_len: 60_000,
            max: 49_152,
        });
        assert!(
            resp.headers()
                .get(axum::http::header::RETRY_AFTER)
                .is_none(),
            "permanent rejection must not advertise Retry-After"
        );
    }

    #[tokio::test]
    async fn other_is_500_with_null_code() {
        let (status, body) = map(InferenceError::Other(anyhow::anyhow!("kaboom"))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let error = &body["error"];
        assert_eq!(error["type"], "api_error");
        assert_eq!(error["code"], Value::Null);
        assert!(error["message"].as_str().unwrap().contains("kaboom"));
    }
}
