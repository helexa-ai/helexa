#![allow(dead_code)]

use axum::body::Body;
use axum::extract::Path;
use axum::http::header;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

/// Spawns a mock neuron that serves:
/// - GET /models (returns one loaded "test-model")
/// - GET /models/:id/endpoint (returns the inference URL)
/// - POST /models/unload (accepts unload requests)
/// - GET /v1/chat/completions + POST /v1/chat/completions (inference)
///
/// Returns the neuron base URL.
pub async fn spawn_mock_neuron() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();

    let app = Router::new()
        .route("/models", get(mock_neuron_list_models))
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_model_id): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .route(
            "/models/unload",
            post(|Json(_body): Json<Value>| async { Json(json!({"status": "unloaded"})) }),
        )
        .route("/v1/chat/completions", post(mock_chat_completions))
        .route("/v1/responses", post(mock_responses))
        .route("/v1/models", get(mock_v1_models));

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    base_url
}

async fn mock_neuron_list_models() -> Json<Value> {
    Json(json!([
        {"id": "test-model", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": 8000}
    ]))
}

async fn mock_v1_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": "test-model", "object": "model", "status": "loaded"}]
    }))
}

async fn mock_chat_completions(Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    Json(json!({
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "created": 1700000000_u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from mock backend"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }))
}

async fn mock_responses(Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    // Echo the model field back and synthesise a tiny ResponsesResponse.
    // Mirrors the shape neuron's /v1/responses handler emits so the
    // gateway test only needs to assert the proxy round-tripped it.
    Json(json!({
        "id": "resp-test-001",
        "object": "response",
        "created_at": 1700000000_u64,
        "status": "completed",
        "model": model,
        "output": [{
            "type": "message",
            "id": "msg-test-001",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "Hello from mock backend",
                "annotations": []
            }],
            "status": "completed"
        }],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 5,
            "total_tokens": 10
        }
    }))
}

/// Spawns a mock neuron that returns SSE streaming responses for chat completions.
pub async fn spawn_streaming_mock_neuron(chunk_count: usize, chunk_delay: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();

    let app = Router::new()
        .route("/models", get(mock_neuron_list_models))
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_model_id): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| async move {
                let model = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let chunks: Vec<String> = (0..chunk_count)
                    .map(|i| {
                        let content = format!("token{i}");
                        let chunk = json!({
                            "id": "chatcmpl-stream-001",
                            "object": "chat.completion.chunk",
                            "created": 1700000000_u64,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": content },
                                "finish_reason": null
                            }]
                        });
                        format!("data: {chunk}\n\n")
                    })
                    .collect();

                let delay = chunk_delay;
                let stream = stream::iter(
                    chunks
                        .into_iter()
                        .chain(std::iter::once("data: [DONE]\n\n".to_string())),
                )
                .then(move |chunk| async move {
                    tokio::time::sleep(delay).await;
                    Ok::<_, std::convert::Infallible>(chunk)
                });

                Response::builder()
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header(header::CACHE_CONTROL, "no-cache")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        );

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    base_url
}

/// Spawns a mock neuron with a custom models list.
pub async fn spawn_mock_neuron_with_models(models_response: Value) -> String {
    spawn_mock_neuron_with_models_and_health(models_response, default_health_response()).await
}

/// Default `/health` response used by mocks that don't care about the
/// activation field — empty devices, no in-flight pre-warm, state=ready.
pub fn default_health_response() -> Value {
    json!({
        "uptime_secs": 0,
        "devices": [],
        "activation": {
            "state": "ready",
            "pending": [],
            "in_progress": null,
            "completed": [],
            "failed": []
        }
    })
}

/// Variant of `spawn_mock_neuron_with_models` that also serves a
/// `/health` body. Used by tests that drive the gateway's activation
/// surface (poller reading /health, /v1/models synthesising Loading
/// locations from in_progress / pending).
pub async fn spawn_mock_neuron_with_models_and_health(
    models_response: Value,
    health_response: Value,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();

    let app = Router::new()
        .route(
            "/models",
            get(move || {
                let resp = models_response.clone();
                async move { Json(resp) }
            }),
        )
        .route(
            "/health",
            get(move || {
                let resp = health_response.clone();
                async move { Json(resp) }
            }),
        )
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_model_id): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .route(
            "/models/unload",
            post(|Json(_body): Json<Value>| async { Json(json!({"status": "unloaded"})) }),
        )
        .route("/v1/chat/completions", post(mock_chat_completions));

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    base_url
}

/// Spawns the cortex gateway with a single neuron pointing at `mock_url`.
/// The node is pre-seeded as healthy with one loaded model ("test-model").
/// Returns the gateway's base URL.
pub async fn spawn_gateway(mock_url: &str) -> String {
    let (_, url) = spawn_gateway_with_state(mock_url).await;
    url
}

/// Like `spawn_gateway` but also returns the shared `CortexState`.
pub async fn spawn_gateway_with_state(mock_url: &str) -> (Arc<CortexState>, String) {
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![NeuronEndpoint {
            name: "mock-node".into(),
            endpoint: mock_url.to_string(),
        }],
        models_config: "/dev/null".into(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Seed the node as healthy with a loaded model.
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").expect("node must exist");
        node.healthy = true;
        node.models.insert(
            "test-model".into(),
            ModelEntry {
                id: "test-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
            },
        );
    }

    let app = cortex_gateway::build_app(Arc::clone(&fleet));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (fleet, format!("http://{addr}"))
}
