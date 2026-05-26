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
    let model_id = match extract_model(&body) {
        Some(m) => m,
        None => {
            tracing::warn!(
                handler = "chat_completions",
                "rejected: missing 'model' field in request body"
            );
            return error_response(400, "missing 'model' field in request body");
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
            // RouteError's Display strings are short and informative
            // ("model 'X' not found...", "no healthy nodes available")
            // — fine to surface to the caller. The warn above carries
            // any extra context for operators.
            return error_response(404, &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &model_id).await;

    proxy_with_metrics(
        &fleet,
        &route,
        "/v1/chat/completions",
        headers,
        body,
        &model_id,
    )
    .await
}

/// `POST /v1/completions` — proxy completions endpoint.
async fn completions(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model_id = match extract_model(&body) {
        Some(m) => m,
        None => {
            tracing::warn!(
                handler = "completions",
                "rejected: missing 'model' field in request body"
            );
            return error_response(400, "missing 'model' field in request body");
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
            // RouteError's Display strings are short and informative
            // ("model 'X' not found...", "no healthy nodes available")
            // — fine to surface to the caller. The warn above carries
            // any extra context for operators.
            return error_response(404, &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &model_id).await;

    proxy_with_metrics(&fleet, &route, "/v1/completions", headers, body, &model_id).await
}

/// `POST /v1/messages` — accept Anthropic format, translate, proxy, translate back.
async fn anthropic_messages(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
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
            return error_response(400, "invalid Anthropic request body");
        }
    };

    let model_id = anth_req.model.clone();
    let is_streaming = anth_req.stream.unwrap_or(false);

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
            return error_response(500, "internal translation error");
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
            return error_response(404, &e.to_string());
        }
    };

    touch_model(&fleet, &route.node_name, &model_id).await;

    let labels = [
        ("model", model_id.clone()),
        ("node", route.node_name.clone()),
    ];
    metrics::counter!("cortex_requests_total", &labels).increment(1);
    if route.cold_start {
        metrics::counter!("cortex_cold_starts_total", &labels).increment(1);
    }
    let start = Instant::now();

    if is_streaming {
        // TODO: streaming Anthropic translation requires converting SSE format.
        // For now, proxy the OpenAI SSE stream directly (clients that can handle
        // OpenAI SSE will work; full Anthropic SSE translation is a follow-up).
        let result = proxy::forward_request(
            &fleet.http_client,
            &route,
            "/v1/chat/completions",
            headers,
            openai_body,
        )
        .await;
        metrics::histogram!("cortex_request_duration_seconds", &labels)
            .record(start.elapsed().as_secs_f64());
        match result {
            Ok(resp) => resp,
            Err(e) => {
                metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                // forward_request already warn'd with the wire-level
                // detail; no need to log again here.
                e.into_response()
            }
        }
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
                return error_response(502, "upstream request failed");
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
            return error_response(status, &format!("upstream returned {status}"));
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
                return error_response(502, "failed to read upstream response");
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
                    return error_response(502, "malformed upstream response");
                }
            };

        metrics::histogram!("cortex_request_duration_seconds", &labels)
            .record(start.elapsed().as_secs_f64());
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
                });
        }
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
    let result = proxy::forward_request(&fleet.http_client, route, path, headers, body).await;
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

fn error_response(status: u16, message: &str) -> Response {
    let code = axum::http::StatusCode::from_u16(status)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    let body = json!({
        "error": {
            "message": message,
            "type": "gateway_error",
        }
    });
    (code, Json(body)).into_response()
}
