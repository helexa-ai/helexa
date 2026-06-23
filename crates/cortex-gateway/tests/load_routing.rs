//! Load-aware routing across replicas (#55).
//!
//! When a model is loaded on more than one healthy neuron, the router picks
//! the least-busy replica using the per-model admission load each neuron
//! reports on `GET /health` (#53), rather than always taking the first.

mod common;

use axum::Json;
use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::discovery::ModelLoad;
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::net::TcpListener;

/// Seed a node as healthy with `test-model` loaded and a given admission load.
async fn seed_loaded(fleet: &CortexState, node: &str, in_flight: usize, queue_depth: usize) {
    let mut nodes = fleet.nodes.write().await;
    let n = nodes.get_mut(node).expect("node exists");
    n.healthy = true;
    n.models.insert(
        "test-model".into(),
        ModelEntry {
            id: "test-model".into(),
            status: ModelStatus::Loaded,
            last_accessed: None,
            vram_estimate_mb: Some(8000),
            capabilities: Vec::new(),
            tool_call: false,
            reasoning: false,
            limit: None,
        },
    );
    n.model_load.insert(
        "test-model".into(),
        ModelLoad {
            id: "test-model".into(),
            in_flight,
            queue_depth,
        },
    );
}

/// Build a gateway state over two mock neurons (no poller; we seed state).
async fn two_neuron_fleet(endpoint_a: &str, endpoint_b: &str) -> Arc<CortexState> {
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![
            NeuronEndpoint {
                name: "node-a".into(),
                endpoint: endpoint_a.to_string(),
            },
            NeuronEndpoint {
                name: "node-b".into(),
                endpoint: endpoint_b.to_string(),
            },
        ],
        models_config: "/dev/null".into(),
        entitlements: Default::default(),
        upstream: Default::default(),
    };
    Arc::new(CortexState::from_config(&config))
}

#[tokio::test]
async fn routes_to_least_busy_replica() {
    let neuron_a = common::spawn_mock_neuron().await;
    let neuron_b = common::spawn_mock_neuron().await;
    let fleet = two_neuron_fleet(&neuron_a, &neuron_b).await;

    // A is busy (1 running + 3 queued), B is idle.
    seed_loaded(&fleet, "node-a", 1, 3).await;
    seed_loaded(&fleet, "node-b", 0, 0).await;

    let route = cortex_gateway::router::resolve(&fleet, "test-model")
        .await
        .expect("model is loaded on both nodes");
    assert_eq!(route.node_name, "node-b", "should pick the idle replica");

    // Flip the load: now B is the busy one.
    seed_loaded(&fleet, "node-a", 0, 0).await;
    seed_loaded(&fleet, "node-b", 1, 5).await;
    let route = cortex_gateway::router::resolve(&fleet, "test-model")
        .await
        .expect("still loaded");
    assert_eq!(route.node_name, "node-a", "should follow the lighter load");
}

/// Mock neuron whose inference endpoint always returns a #63 backpressure
/// envelope (503 + Retry-After) — simulating a saturated neuron.
async fn spawn_busy_neuron() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();
    let app = axum::Router::new()
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({ "url": url })) }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|| async {
                let body = json!({"error": {
                    "message": "model is busy (admission queue full); retry shortly",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "param": null
                }});
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(header::RETRY_AFTER, "6")],
                    Json(body),
                )
                    .into_response()
            }),
        );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    base_url
}

#[tokio::test]
async fn neuron_backpressure_is_propagated_intact() {
    // A saturated neuron's 503 + Retry-After + envelope must reach the client
    // verbatim — not unwrapped, remapped, or stripped (#55 / #63).
    let neuron = spawn_busy_neuron().await;
    let fleet = two_neuron_fleet(&neuron, &neuron).await;
    seed_loaded(&fleet, "node-a", 1, 8).await;

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("6"),
        "Retry-After must survive the proxy"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn ties_break_deterministically_by_name() {
    let neuron_a = common::spawn_mock_neuron().await;
    let neuron_b = common::spawn_mock_neuron().await;
    let fleet = two_neuron_fleet(&neuron_a, &neuron_b).await;

    // Equal load on both → stable pick (lowest node name).
    seed_loaded(&fleet, "node-a", 0, 0).await;
    seed_loaded(&fleet, "node-b", 0, 0).await;

    let route = cortex_gateway::router::resolve(&fleet, "test-model")
        .await
        .expect("loaded");
    assert_eq!(route.node_name, "node-a", "ties break by name");
}
