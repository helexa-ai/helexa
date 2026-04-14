mod common;

use chrono::Utc;
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NodeConfig,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::json;
use std::sync::Arc;

/// Spawn a mock backend that accepts `/v1/models/unload` and records the call.
async fn spawn_eviction_mock() -> (String, Arc<tokio::sync::Mutex<Vec<String>>>) {
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use serde_json::Value;

    let unloaded: Arc<tokio::sync::Mutex<Vec<String>>> = Arc::new(tokio::sync::Mutex::new(vec![]));
    let unloaded_clone = Arc::clone(&unloaded);

    let app = Router::new()
        .route(
            "/v1/models/unload",
            post(move |Json(body): Json<Value>| {
                let unloaded = Arc::clone(&unloaded_clone);
                async move {
                    let model_id = body
                        .get("model_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    unloaded.lock().await.push(model_id);
                    Json(json!({"status": "ok"}))
                }
            }),
        )
        .route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "object": "list",
                    "data": []
                }))
            }),
        );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), unloaded)
}

fn make_fleet(endpoint: &str, pinned: Vec<String>, defrag_after: u32) -> Arc<CortexState> {
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: defrag_after,
        },
        nodes: vec![NodeConfig {
            name: "gpu-node".into(),
            endpoint: endpoint.to_string(),
            vram_mb: 24000,
            pinned,
        }],
    };
    Arc::new(CortexState::from_config(&config))
}

#[tokio::test]
async fn test_evict_lru_model() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let fleet = make_fleet(&mock_url, vec![], 0);

    // Seed two loaded models. "old-model" was accessed earlier than "new-model".
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models.insert(
            "old-model".into(),
            ModelEntry {
                id: "old-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: Some(Utc::now() - chrono::Duration::hours(2)),
                vram_estimate_mb: Some(8000),
            },
        );
        node.models.insert(
            "new-model".into(),
            ModelEntry {
                id: "new-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: Some(Utc::now()),
                vram_estimate_mb: Some(8000),
            },
        );
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node")
        .await
        .expect("eviction should succeed");

    // The older model should be evicted.
    assert_eq!(evicted, Some("old-model".to_string()));

    // Mock received the unload call.
    let calls = unloaded.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], "old-model");

    // Local state updated.
    let nodes = fleet.nodes.read().await;
    let node = nodes.get("gpu-node").unwrap();
    assert_eq!(
        node.models.get("old-model").unwrap().status,
        ModelStatus::Unloaded
    );
    assert_eq!(
        node.models.get("new-model").unwrap().status,
        ModelStatus::Loaded
    );
}

#[tokio::test]
async fn test_eviction_skips_pinned_models() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    // Pin "old-model" so it can't be evicted.
    let fleet = make_fleet(&mock_url, vec!["old-model".into()], 0);

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        // old-model is pinned and older — normally it would be evicted.
        node.models.insert(
            "old-model".into(),
            ModelEntry {
                id: "old-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: Some(Utc::now() - chrono::Duration::hours(2)),
                vram_estimate_mb: Some(8000),
            },
        );
        node.models.insert(
            "new-model".into(),
            ModelEntry {
                id: "new-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: Some(Utc::now()),
                vram_estimate_mb: Some(8000),
            },
        );
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node")
        .await
        .expect("eviction should succeed");

    // new-model is evicted instead because old-model is pinned.
    assert_eq!(evicted, Some("new-model".to_string()));

    let calls = unloaded.lock().await;
    assert_eq!(calls[0], "new-model");
}

#[tokio::test]
async fn test_eviction_nothing_to_evict() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    // Pin the only model.
    let fleet = make_fleet(&mock_url, vec!["only-model".into()], 0);

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models.insert(
            "only-model".into(),
            ModelEntry {
                id: "only-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
            },
        );
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node")
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, None);

    // No unload call made.
    let calls = unloaded.lock().await;
    assert!(calls.is_empty());
}

#[tokio::test]
async fn test_eviction_increments_lifecycle_cycles() {
    let (mock_url, _) = spawn_eviction_mock().await;
    let fleet = make_fleet(&mock_url, vec![], 0);

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.lifecycle_cycles = 0;
        node.models.insert(
            "model-a".into(),
            ModelEntry {
                id: "model-a".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: None,
            },
        );
    }

    cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node")
        .await
        .expect("eviction should succeed");

    let nodes = fleet.nodes.read().await;
    assert_eq!(nodes.get("gpu-node").unwrap().lifecycle_cycles, 1);
}

#[tokio::test]
async fn test_last_accessed_updated_on_request() {
    let mock_url = common::spawn_mock_backend().await;
    let (fleet, gw_url) = common::spawn_gateway_with_state(&mock_url).await;

    // Verify last_accessed is None initially.
    {
        let nodes = fleet.nodes.read().await;
        let node = nodes.get("mock-node").unwrap();
        assert!(
            node.models
                .get("test-model")
                .unwrap()
                .last_accessed
                .is_none()
        );
    }

    // Make a request.
    let client = reqwest::Client::new();
    client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    // Verify last_accessed is now set.
    let nodes = fleet.nodes.read().await;
    let node = nodes.get("mock-node").unwrap();
    assert!(
        node.models
            .get("test-model")
            .unwrap()
            .last_accessed
            .is_some()
    );
}
