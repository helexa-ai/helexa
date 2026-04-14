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
        None => return error_response(400, "missing 'model' field in request body"),
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => return error_response(404, &e.to_string()),
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
        None => return error_response(400, "missing 'model' field in request body"),
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => return error_response(404, &e.to_string()),
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
        Err(e) => return error_response(400, &format!("invalid Anthropic request: {e}")),
    };

    let model_id = anth_req.model.clone();
    let is_streaming = anth_req.stream.unwrap_or(false);

    // Translate to OpenAI format.
    let openai_req = cortex_core::translate::anthropic_to_openai(anth_req);
    let openai_body = match serde_json::to_vec(&openai_req) {
        Ok(b) => Bytes::from(b),
        Err(e) => return error_response(500, &format!("translation error: {e}")),
    };

    let route = match router::resolve(&fleet, &model_id).await {
        Ok(r) => r,
        Err(e) => return error_response(404, &e.to_string()),
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
                e.into_response()
            }
        }
    } else {
        // Non-streaming: proxy, buffer full response, translate back to Anthropic.
        let upstream_resp = fleet
            .http_client
            .post(format!("{}/v1/chat/completions", route.endpoint))
            .body(openai_body)
            .header("content-type", "application/json")
            .send()
            .await;

        let upstream_resp = match upstream_resp {
            Ok(r) => r,
            Err(e) => {
                metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                return error_response(502, &format!("upstream request failed: {e}"));
            }
        };

        if !upstream_resp.status().is_success() {
            metrics::counter!("cortex_request_errors_total", &labels).increment(1);
            let status = upstream_resp.status().as_u16();
            let body = upstream_resp.text().await.unwrap_or_default();
            return error_response(status, &format!("upstream error: {body}"));
        }

        let body_bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                return error_response(502, &format!("failed to read upstream response: {e}"));
            }
        };

        let openai_resp: cortex_core::openai::ChatCompletionResponse =
            match serde_json::from_slice(&body_bytes) {
                Ok(r) => r,
                Err(e) => {
                    metrics::counter!("cortex_request_errors_total", &labels).increment(1);
                    return error_response(502, &format!("failed to parse upstream response: {e}"));
                }
            };

        metrics::histogram!("cortex_request_duration_seconds", &labels)
            .record(start.elapsed().as_secs_f64());
        let anthropic_resp = cortex_core::translate::openai_to_anthropic(openai_resp);
        Json(json!(anthropic_resp)).into_response()
    }
}

/// `GET /v1/models` — aggregate models from all nodes.
async fn list_models(State(fleet): State<Arc<CortexState>>) -> Json<Value> {
    let nodes = fleet.nodes.read().await;
    let mut model_map: std::collections::HashMap<String, CortexModelEntry> =
        std::collections::HashMap::new();

    for node in nodes.values() {
        for (model_id, entry) in &node.models {
            let location = ModelLocation {
                node: node.name.clone(),
                status: entry.status,
                vram_estimate_mb: entry.vram_estimate_mb,
            };
            model_map
                .entry(model_id.clone())
                .and_modify(|e| e.locations.push(location.clone()))
                .or_insert_with(|| CortexModelEntry {
                    id: model_id.clone(),
                    object: "model".into(),
                    locations: vec![location],
                });
        }
    }

    let data: Vec<Value> = model_map.values().map(|e| json!(e)).collect();

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
