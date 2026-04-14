use axum::routing::{get, post};
use axum::{Json, Router};
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NodeConfig,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::net::TcpListener;

/// Spawns a mock mistral.rs backend on a random port.
/// Returns the base URL (e.g. "http://127.0.0.1:12345").
pub async fn spawn_mock_backend() -> String {
    let app = Router::new()
        .route("/v1/chat/completions", post(mock_chat_completions))
        .route("/v1/models", get(mock_list_models));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    format!("http://{addr}")
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

async fn mock_list_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": "test-model",
            "object": "model",
            "status": "loaded"
        }]
    }))
}

/// Spawns the cortex gateway with a single node pointing at `mock_url`.
/// The node is pre-seeded as healthy with one loaded model ("test-model").
/// Returns the gateway's base URL.
pub async fn spawn_gateway(mock_url: &str) -> String {
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        nodes: vec![NodeConfig {
            name: "mock-node".into(),
            endpoint: mock_url.to_string(),
            vram_mb: 24000,
            pinned: vec![],
        }],
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Seed the node as healthy with a loaded model.
    // (Bypasses the poller, which is not running in tests.)
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

    let app = cortex_gateway::build_app(fleet);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    format!("http://{addr}")
}
