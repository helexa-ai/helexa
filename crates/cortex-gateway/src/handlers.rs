//! Axum HTTP handlers for the gateway API surface.

use crate::proxy;
use crate::router;
use crate::router::RouteDecision;
use crate::state::CortexState;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use chrono::Utc;
use cortex_core::node::{CortexModelEntry, ModelLocation};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;

pub fn api_routes() -> Router<Arc<CortexState>> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(list_models))
        .route("/v1/messages", post(anthropic_messages))
        .route("/health", get(health))
        .route("/", get(health))
}

/// `POST /v1/chat/completions` — proxy to the appropriate backend node.
async fn chat_completions(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    log_inbound("openai-chat", "/v1/chat/completions", &body);
    let model_id = match extract_model(&body) {
        Some(m) => m,
        None => {
            tracing::warn!(
                handler = "chat_completions",
                "rejected: missing 'model' field in request body"
            );
            return error_response(
                400,
                "invalid_request_error",
                "missing_model_field",
                "missing 'model' field in request body",
            );
        }
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                handler = "chat_completions",
                model = %model_id,
                error = %e,
                "route resolve failed"
            );
            return error_response(e.http_status(), e.broad_type(), e.code(), &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &route.resolved_model_id).await;

    let body = rewrite_model_in_body(body, &route.resolved_model_id);
    proxy_with_metrics(
        &fleet,
        &route,
        "/v1/chat/completions",
        headers,
        body,
        &route.resolved_model_id,
    )
    .await
}

/// `POST /v1/responses` — proxy to the appropriate backend node.
///
/// Same routing shape as [`chat_completions`]: extract `model` from
/// the body, resolve to a node, forward verbatim. No translation —
/// neuron speaks the Responses API natively (see
/// `crates/neuron/src/wire/openai_responses.rs`), so the gateway is
/// a pass-through. Streaming and non-streaming are handled
/// identically; the upstream `Content-Type` (text/event-stream vs.
/// application/json) propagates through the proxy.
async fn responses(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    log_inbound("openai-responses", "/v1/responses", &body);
    let model_id = match extract_model(&body) {
        Some(m) => m,
        None => {
            tracing::warn!(
                handler = "responses",
                "rejected: missing 'model' field in request body"
            );
            return error_response(
                400,
                "invalid_request_error",
                "missing_model_field",
                "missing 'model' field in request body",
            );
        }
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                handler = "responses",
                model = %model_id,
                error = %e,
                "route resolve failed"
            );
            return error_response(e.http_status(), e.broad_type(), e.code(), &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &route.resolved_model_id).await;

    let body = rewrite_model_in_body(body, &route.resolved_model_id);
    proxy_with_metrics(
        &fleet,
        &route,
        "/v1/responses",
        headers,
        body,
        &route.resolved_model_id,
    )
    .await
}

/// `POST /v1/completions` — proxy completions endpoint.
async fn completions(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    log_inbound("openai-completions", "/v1/completions", &body);
    let model_id = match extract_model(&body) {
        Some(m) => m,
        None => {
            tracing::warn!(
                handler = "completions",
                "rejected: missing 'model' field in request body"
            );
            return error_response(
                400,
                "invalid_request_error",
                "missing_model_field",
                "missing 'model' field in request body",
            );
        }
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                handler = "completions",
                model = %model_id,
                error = %e,
                "route resolve failed"
            );
            return error_response(e.http_status(), e.broad_type(), e.code(), &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &route.resolved_model_id).await;

    let body = rewrite_model_in_body(body, &route.resolved_model_id);
    proxy_with_metrics(
        &fleet,
        &route,
        "/v1/completions",
        headers,
        body,
        &route.resolved_model_id,
    )
    .await
}

/// `POST /v1/messages` — accept Anthropic format, translate, proxy, translate back.
async fn anthropic_messages(
    State(fleet): State<Arc<CortexState>>,
    _headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Parse as Anthropic request.
    let anth_req: cortex_core::anthropic::MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                handler = "anthropic_messages",
                error = %e,
                "rejected: invalid Anthropic request body"
            );
            return error_response(
                400,
                "invalid_request_error",
                "invalid_anthropic_body",
                "invalid Anthropic request body",
            );
        }
    };

    let model_id = anth_req.model.clone();
    let is_streaming = anth_req.stream.unwrap_or(false);

    // Wire-debug: make the exercised path and request shape concrete
    // rather than guesswork. `tool_history` flags whether the client is
    // continuing a tool conversation (tool_use/tool_result blocks in the
    // message history) vs. opening a fresh one. Full bodies ride at
    // trace! (cortex/neuron ship at info; operator infra runs at debug).
    if tracing::enabled!(tracing::Level::DEBUG) {
        let n_tools = anth_req
            .extra
            .get("tools")
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);
        let tool_history = anth_req
            .messages
            .iter()
            .any(|m| anthropic_message_has_tool_blocks(&m.content));
        tracing::debug!(
            wire = "anthropic",
            endpoint = "/v1/messages",
            model = %model_id,
            stream = is_streaming,
            messages = anth_req.messages.len(),
            tools = n_tools,
            tool_history,
            system = anth_req.system.is_some(),
            "inbound request"
        );
    }
    tracing::trace!(wire = "anthropic", body = %body_preview(&body), "inbound anthropic body");

    // Translate to OpenAI format.
    let openai_req = cortex_core::translate::anthropic_to_openai(anth_req);
    let openai_body = match serde_json::to_vec(&openai_req) {
        Ok(b) => Bytes::from(b),
        Err(e) => {
            tracing::error!(
                handler = "anthropic_messages",
                model = %model_id,
                error = %e,
                "internal: failed to serialise translated OpenAI request"
            );
            return error_response(
                500,
                "api_error",
                "internal_translation_error",
                "internal translation error",
            );
        }
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                handler = "anthropic_messages",
                model = %model_id,
                error = %e,
                "route resolve failed"
            );
            // RouteError's Display strings are short and informative
            // ("model 'X' not found...", "no healthy nodes available")
            // — fine to surface to the caller. The warn above carries
            // any extra context for operators.
            return error_response(e.http_status(), e.broad_type(), e.code(), &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &route.resolved_model_id).await;

    // Swap the alias for the concrete id in the translated body so
    // neuron's harness sees a model name that matches what it has
    // loaded.
    let openai_body = rewrite_model_in_body(openai_body, &route.resolved_model_id);
    // The translated body is what neuron actually sees — the reshaped
    // OpenAI-form tools live here. Tracing it makes "did the tool
    // definitions survive translation?" a log line, not a guess.
    tracing::trace!(
        wire = "anthropic",
        body = %body_preview(&openai_body),
        "translated openai body (sent upstream)"
    );

    let labels = [
        ("model", route.resolved_model_id.clone()),
        ("node", route.node_name.clone()),
    ];
    metrics::counter!("cortex_requests_total", &labels).increment(1);
    if route.cold_start {
        metrics::counter!("cortex_cold_starts_total", &labels).increment(1);
    }
    let start = Instant::now();

    if is_streaming {
        // Anthropic SSE translation (#24): upstream speaks OpenAI SSE;
        // re-frame it event-by-event into Anthropic's message_start /
        // content_block_* / message_delta / message_stop sequence.
        let resp = crate::anthropic_sse::stream_translated(
            &fleet.http_client,
            &route.endpoint,
            openai_body,
            &model_id,
            &route.node_name,
        )
        .await;
        metrics::histogram!("cortex_request_duration_seconds", &labels)
            .record(start.elapsed().as_secs_f64());
        if !resp.status().is_success() {
            metrics::counter!("cortex_request_errors_total", &labels).increment(1);
        }
        resp
    } else {
        // Non-streaming: proxy, buffer full response, translate back to Anthropic.
        let target_url = format!("{}/v1/chat/completions", route.endpoint);
        tracing::info!(
            handler = "anthropic_messages",
            model = %model_id,
            node = %route.node_name,
            url = %target_url,
            cold_start = route.cold_start,
            "proxying request"
        );
        let upstream_resp = fleet
            .http_client
            .post(&target_url)
            .body(openai_body)
            .header("content-type", "application/json")
            .send()
            .await;

        let upstream_resp = match upstream_resp {
            Ok(r) => r,
            Err(e) => {
                metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                tracing::warn!(
                    handler = "anthropic_messages",
                    model = %model_id,
                    node = %route.node_name,
                    url = %target_url,
                    error = %e,
                    "upstream request failed (network)"
                );
                return error_response(
                    502,
                    "api_error",
                    "upstream_connection_error",
                    "upstream request failed",
                );
            }
        };

        let upstream_status = upstream_resp.status();
        if !upstream_status.is_success() {
            metrics::counter!("cortex_request_errors_total", &labels).increment(1);
            let status = upstream_status.as_u16();
            let body = upstream_resp.text().await.unwrap_or_default();
            let body_snippet = body.chars().take(512).collect::<String>();
            tracing::warn!(
                handler = "anthropic_messages",
                model = %model_id,
                node = %route.node_name,
                url = %target_url,
                status,
                body = %body_snippet,
                "upstream returned non-2xx"
            );
            return error_response(
                status,
                "api_error",
                "upstream_error",
                &format!("upstream returned {status}"),
            );
        }

        let body_bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                tracing::warn!(
                    handler = "anthropic_messages",
                    model = %model_id,
                    node = %route.node_name,
                    url = %target_url,
                    error = %e,
                    "failed to read upstream response body"
                );
                return error_response(
                    502,
                    "api_error",
                    "upstream_connection_error",
                    "failed to read upstream response",
                );
            }
        };

        let openai_resp: cortex_core::openai::ChatCompletionResponse =
            match serde_json::from_slice(&body_bytes) {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                    let body_snippet = String::from_utf8_lossy(&body_bytes)
                        .chars()
                        .take(512)
                        .collect::<String>();
                    tracing::warn!(
                        handler = "anthropic_messages",
                        model = %model_id,
                        node = %route.node_name,
                        url = %target_url,
                        error = %e,
                        body = %body_snippet,
                        "failed to parse upstream response as OpenAI ChatCompletionResponse"
                    );
                    return error_response(
                        502,
                        "api_error",
                        "upstream_malformed_response",
                        "malformed upstream response",
                    );
                }
            };

        metrics::histogram!("cortex_request_duration_seconds", &labels)
            .record(start.elapsed().as_secs_f64());
        // Did the model actually produce a structured tool call, or just
        // text? This is the single most useful signal for "is tool
        // calling working end-to-end" — a `false` here alongside a
        // request that carried tools means the model improvised an
        // unparsed format (the original failure mode).
        let upstream_tool_calls = openai_resp.choices.iter().any(|c| {
            c.message
                .extra
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(|a| !a.is_empty())
                .unwrap_or(false)
        });
        let finish_reason = openai_resp
            .choices
            .first()
            .and_then(|c| c.finish_reason.clone());
        tracing::debug!(
            wire = "anthropic",
            model = %model_id,
            node = %route.node_name,
            upstream_tool_calls,
            finish_reason = ?finish_reason,
            "upstream non-streaming response"
        );
        let anthropic_resp = cortex_core::translate::openai_to_anthropic(openai_resp);
        Json(json!(anthropic_resp)).into_response()
    }
}

/// `GET /v1/models` — union of (catalogue × topology feasibility) and
/// (currently loaded somewhere). The result is what the fleet *could*
/// serve, not just what's already loaded — so OpenAI-compatible tools
/// see every model the operator has provisioned, and cortex
/// transparently cold-loads the first time one is requested.
async fn list_models(State(fleet): State<Arc<CortexState>>) -> Json<Value> {
    use std::collections::HashMap;
    let now = Utc::now().timestamp() as u64;
    let nodes = fleet.nodes.read().await;
    let catalogue = &fleet.catalogue;

    let mut entries: HashMap<String, CortexModelEntry> = HashMap::new();

    // Pass 1: catalogue × topology. For every catalogue profile, find
    // healthy neurons whose discovered devices satisfy the profile.
    // Catalogue-defined models surface here even if nothing has loaded
    // them yet — that's the point of the unified endpoint.
    for profile in &catalogue.models {
        let mut feasible_on = Vec::new();
        for node in nodes.values() {
            if !node.healthy {
                continue;
            }
            let Some(disc) = node.discovery.as_ref() else {
                continue;
            };
            if profile.is_feasible_on(&node.name, &disc.devices) {
                feasible_on.push(node.name.clone());
            }
        }
        if feasible_on.is_empty() {
            // The catalogue lists this model but no neuron's topology
            // matches — surface it as not-loaded with no feasible
            // location. Hides nothing; lets operators see why a
            // configured model isn't reachable.
            feasible_on.clear();
        }
        entries.insert(
            profile.id.clone(),
            CortexModelEntry {
                id: profile.id.clone(),
                object: "model".into(),
                created: now,
                owned_by: "helexa".into(),
                loaded: false,
                feasible_on,
                locations: Vec::new(),
                // Start with catalogue-declared capabilities; Pass 2 unions
                // runtime-detected ones from loaded neurons.
                capabilities: profile.capabilities.clone(),
                // Catalogue limit/cost flow through directly.
                limit: profile.limit.clone(),
                cost: profile.cost.clone(),
                // Runtime-detected — will be OR-ed in Pass 2 from neuron data.
                tool_call: false,
                reasoning: false,
            },
        );
    }

    // Pass 2: layer the actually-loaded state on top. For each
    // (node, model) entry, attach a ModelLocation. If the model isn't
    // in the catalogue, create a new CortexModelEntry from scratch —
    // cortex doesn't refuse to surface a manually-loaded model just
    // because the operator didn't enumerate it in models.toml.
    for node in nodes.values() {
        for (model_id, entry) in &node.models {
            let location = ModelLocation {
                node: node.name.clone(),
                status: entry.status,
                vram_estimate_mb: entry.vram_estimate_mb,
            };
            let was_loaded = matches!(entry.status, cortex_core::node::ModelStatus::Loaded);
            entries
                .entry(model_id.clone())
                .and_modify(|e| {
                    e.locations.push(location.clone());
                    if was_loaded {
                        e.loaded = true;
                    }
                    // Union the per-node capabilities so a model loaded
                    // on several neurons reports every modality any of
                    // them advertises.
                    for cap in &entry.capabilities {
                        if !e.capabilities.contains(cap) {
                            e.capabilities.push(cap.clone());
                        }
                    }
                    // OR-in runtime-detected capability flags from the neuron.
                    e.tool_call = e.tool_call || entry.tool_call;
                    e.reasoning = e.reasoning || entry.reasoning;
                })
                .or_insert_with(|| CortexModelEntry {
                    id: model_id.clone(),
                    object: "model".into(),
                    created: now,
                    owned_by: "helexa".into(),
                    loaded: was_loaded,
                    // Not in catalogue — cortex has no opinion on
                    // feasibility; leave empty.
                    feasible_on: Vec::new(),
                    locations: vec![location],
                    capabilities: entry.capabilities.clone(),
                    limit: None,
                    cost: None,
                    tool_call: entry.tool_call,
                    reasoning: entry.reasoning,
                });
        }
    }

    // Pass 3: surface pre-warming models. Each neuron's `/health`
    // activation snapshot (polled separately from /models) reports
    // `in_progress` (the model currently materialising) and `pending`
    // (queued behind it). Neither appears on the neuron's `/models`
    // yet — that endpoint only knows about fully-loaded handles — so
    // without this pass a client polling `/v1/models` during pre-warm
    // sees Qwen3.6-27B with no location and concludes "not there".
    // Synthesising a Loading location instead tells clients the model
    // is on its way. Idempotent against Pass 2: if a Loading location
    // for this node already exists (shouldn't, but be safe) we skip.
    for node in nodes.values() {
        let Some(activation) = node.activation.as_ref() else {
            continue;
        };
        let mut loading_ids: Vec<&str> = Vec::new();
        if let Some(id) = activation.in_progress.as_deref() {
            loading_ids.push(id);
        }
        for id in &activation.pending {
            loading_ids.push(id.as_str());
        }
        for model_id in loading_ids {
            let location = ModelLocation {
                node: node.name.clone(),
                status: cortex_core::node::ModelStatus::Loading,
                vram_estimate_mb: None,
            };
            entries
                .entry(model_id.to_string())
                .and_modify(|e| {
                    let already = e.locations.iter().any(|l| {
                        l.node == node.name && l.status == cortex_core::node::ModelStatus::Loading
                    });
                    if !already {
                        e.locations.push(location.clone());
                    }
                })
                .or_insert_with(|| CortexModelEntry {
                    id: model_id.to_string(),
                    object: "model".into(),
                    created: now,
                    owned_by: "helexa".into(),
                    loaded: false,
                    feasible_on: Vec::new(),
                    locations: vec![location],
                    // A model that's only mid-prewarm has no loaded
                    // location to read capabilities from yet.
                    capabilities: Vec::new(),
                    limit: None,
                    cost: None,
                    tool_call: false,
                    reasoning: false,
                });
        }
    }

    // Pass 4: surface aliases as their own entries pointing at the
    // same locations as the target id, so a client browsing /v1/models
    // sees "helexa/small" / "helexa/balanced" / "helexa/large" (or
    // whatever the operator defined) and can request inference
    // against them directly. Aliases that point at unknown targets
    // are skipped — surfacing a dead alias would be misleading.
    for (alias, target) in &catalogue.aliases {
        let Some(target_entry) = entries.get(target).cloned() else {
            tracing::warn!(
                alias = alias,
                target = target,
                "alias points at a model not present in catalogue or fleet; skipping"
            );
            continue;
        };
        entries.insert(
            alias.clone(),
            CortexModelEntry {
                id: alias.clone(),
                object: "model".into(),
                created: now,
                owned_by: "helexa".into(),
                loaded: target_entry.loaded,
                feasible_on: target_entry.feasible_on,
                locations: target_entry.locations,
                capabilities: target_entry.capabilities,
                limit: target_entry.limit.clone(),
                cost: target_entry.cost.clone(),
                tool_call: target_entry.tool_call,
                reasoning: target_entry.reasoning,
            },
        );
    }

    let data: Vec<Value> = entries.values().map(|e| json!(e)).collect();
    Json(json!({
        "object": "list",
        "data": data,
    }))
}

/// `GET /health`
async fn health(State(fleet): State<Arc<CortexState>>) -> Json<Value> {
    let nodes = fleet.nodes.read().await;
    let healthy_count = nodes.values().filter(|n| n.healthy).count();
    let total_count = nodes.len();

    Json(json!({
        "status": if healthy_count > 0 { "ok" } else { "degraded" },
        "nodes": {
            "healthy": healthy_count,
            "total": total_count,
        }
    }))
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Proxy a request with metrics instrumentation.
async fn proxy_with_metrics(
    fleet: &CortexState,
    route: &RouteDecision,
    path: &str,
    headers: HeaderMap,
    body: Bytes,
    model_id: &str,
) -> Response {
    let labels = [
        ("model", model_id.to_string()),
        ("node", route.node_name.clone()),
    ];

    metrics::counter!("cortex_requests_total", &labels).increment(1);
    if route.cold_start {
        metrics::counter!("cortex_cold_starts_total", &labels).increment(1);
    }

    let start = Instant::now();
    let result =
        proxy::forward_request(&fleet.http_client, route, path, headers, body, model_id).await;
    let duration = start.elapsed();

    match result {
        Ok(resp) => {
            metrics::histogram!("cortex_request_duration_seconds", &labels)
                .record(duration.as_secs_f64());
            resp
        }
        Err(e) => {
            metrics::counter!("cortex_request_errors_total", &labels).increment(1);
            // proxy::forward_request already warn'd with wire-level
            // detail (target URL, error, status). ProxyError::into_response
            // now returns a generic message — no body leak.
            e.into_response()
        }
    }
}

/// Update `last_accessed` timestamp for a model on a node (drives LRU eviction).
async fn touch_model(fleet: &CortexState, node_name: &str, model_id: &str) {
    let mut nodes = fleet.nodes.write().await;
    if let Some(node) = nodes.get_mut(node_name)
        && let Some(entry) = node.models.get_mut(model_id)
    {
        entry.last_accessed = Some(Utc::now());
    }
}

fn extract_model(body: &[u8]) -> Option<String> {
    let v: Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(|s| s.to_string())
}

/// Emit a uniform wire-debug summary for an OpenAI-family inbound
/// request (chat/completions, completions, responses). Makes which
/// surface a client exercised — and whether it sent tools / asked for
/// streaming — a concrete log line. The full body rides at trace!.
///
/// Parsing is gated on the debug level being enabled so info-level
/// deployments pay nothing.
fn log_inbound(wire: &str, endpoint: &str, body: &[u8]) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let v: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(_) => return,
        };
        let model = v.get("model").and_then(Value::as_str).unwrap_or("?");
        let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let tools = v
            .get("tools")
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);
        tracing::debug!(wire, endpoint, model, stream, tools, "inbound request");
    }
    tracing::trace!(wire, endpoint, body = %body_preview(body), "inbound body");
}

/// True if an Anthropic message's content carries any `tool_use` or
/// `tool_result` block — i.e. the client is mid tool-conversation.
fn anthropic_message_has_tool_blocks(content: &cortex_core::anthropic::AnthropicContent) -> bool {
    use cortex_core::anthropic::AnthropicContent;
    match content {
        AnthropicContent::Text(_) => false,
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b.block_type.as_str(), "tool_use" | "tool_result")),
    }
}

/// Render a UTF-8-safe, length-capped preview of a request/response
/// body for trace logging. Caps by characters (not bytes) so the slice
/// can never split a multi-byte codepoint.
fn body_preview(body: &[u8]) -> String {
    const MAX_CHARS: usize = 8192;
    let text = String::from_utf8_lossy(body);
    if text.chars().count() > MAX_CHARS {
        let head: String = text.chars().take(MAX_CHARS).collect();
        format!("{head}…<truncated, {} bytes total>", body.len())
    } else {
        text.into_owned()
    }
}

/// Rewrite the `model` field of an OpenAI-style JSON request body to
/// the resolved concrete id. Returns the original bytes if `new_model`
/// matches what's already there or the body fails to parse — the
/// caller has already extracted `model` via `extract_model`, so a
/// parse failure here would only happen on a body the client crafted
/// to defeat us, and we'd rather proxy it unchanged than 500.
///
/// Needed because neuron rejects requests whose `model` field doesn't
/// match a loaded model, so a client that sends `model: "helexa/small"`
/// would hit a 404 at the harness unless we swap it for the concrete
/// id the alias resolved to.
fn rewrite_model_in_body(body: Bytes, new_model: &str) -> Bytes {
    let Ok(mut v) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    let needs_rewrite = v
        .get("model")
        .and_then(|m| m.as_str())
        .map(|m| m != new_model)
        .unwrap_or(false);
    if !needs_rewrite {
        return body;
    }
    if let Value::Object(obj) = &mut v {
        obj.insert("model".into(), Value::String(new_model.to_string()));
    }
    match serde_json::to_vec(&v) {
        Ok(bytes) => Bytes::from(bytes),
        Err(_) => body,
    }
}

fn error_response(status: u16, typ: &str, code: &str, message: &str) -> Response {
    let status_code = axum::http::StatusCode::from_u16(status)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    let body = json!({
        "error": {
            "message": message,
            "type": typ,
            "code": code,
            "param": null,
        }
    });
    (status_code, Json(body)).into_response()
}
