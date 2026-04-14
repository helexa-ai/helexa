mod common;

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NodeConfig,
};
use cortex_core::node::ModelStatus;
use cortex_gateway::state::CortexState;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn test_poller_discovers_models() {
    // Mock backend reports 2 models: one loaded, one unloaded.
    let mock_url = common::spawn_mock_backend_with_models(json!({
        "object": "list",
        "data": [
            { "id": "model-a", "object": "model", "status": "loaded" },
            { "id": "model-b", "object": "model", "status": "unloaded" }
        ]
    }))
    .await;

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
            name: "test-node".into(),
            endpoint: mock_url,
            vram_mb: 24000,
            pinned: vec![],
        }],
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Before polling: node is unhealthy, no models.
    {
        let nodes = fleet.nodes.read().await;
        let node = nodes.get("test-node").unwrap();
        assert!(!node.healthy);
        assert!(node.models.is_empty());
    }

    // Poll once.
    cortex_gateway::poller::poll_once(&fleet).await;

    // After polling: node is healthy, both models discovered with correct status.
    {
        let nodes = fleet.nodes.read().await;
        let node = nodes.get("test-node").unwrap();
        assert!(node.healthy);
        assert_eq!(node.models.len(), 2);

        let model_a = node.models.get("model-a").expect("model-a should exist");
        assert_eq!(model_a.status, ModelStatus::Loaded);

        let model_b = node.models.get("model-b").expect("model-b should exist");
        assert_eq!(model_b.status, ModelStatus::Unloaded);

        assert!(node.last_poll.is_some());
    }
}

#[tokio::test]
async fn test_poller_updates_gateway_models_endpoint() {
    // Mock backend with 2 models.
    let mock_url = common::spawn_mock_backend_with_models(json!({
        "object": "list",
        "data": [
            { "id": "model-x", "object": "model", "status": "loaded" },
            { "id": "model-y", "object": "model", "status": "loaded" }
        ]
    }))
    .await;

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
            name: "poll-node".into(),
            endpoint: mock_url,
            vram_mb: 24000,
            pinned: vec![],
        }],
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Poll to discover models and mark node healthy.
    cortex_gateway::poller::poll_once(&fleet).await;

    // Start gateway with the polled state.
    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Query /v1/models on the gateway.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    let data = body["data"].as_array().expect("data should be an array");
    assert_eq!(data.len(), 2);

    let ids: Vec<&str> = data.iter().filter_map(|m| m["id"].as_str()).collect();
    assert!(ids.contains(&"model-x"));
    assert!(ids.contains(&"model-y"));

    // Verify node attribution in locations.
    for model in data {
        let locations = model["locations"].as_array().expect("locations array");
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0]["node"], "poll-node");
    }
}

#[tokio::test]
async fn test_poller_marks_unreachable_node_unhealthy() {
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
            name: "dead-node".into(),
            endpoint: "http://127.0.0.1:1".into(), // unreachable
            vram_mb: 24000,
            pinned: vec![],
        }],
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Manually mark healthy to verify poller flips it.
    {
        let mut nodes = fleet.nodes.write().await;
        nodes.get_mut("dead-node").unwrap().healthy = true;
    }

    cortex_gateway::poller::poll_once(&fleet).await;

    let nodes = fleet.nodes.read().await;
    assert!(!nodes.get("dead-node").unwrap().healthy);
}

#[tokio::test]
async fn test_poller_removes_stale_models() {
    // Start with a mock that reports 2 models.
    let mock_url = common::spawn_mock_backend_with_models(json!({
        "object": "list",
        "data": [
            { "id": "keep-me", "object": "model", "status": "loaded" },
            { "id": "drop-me", "object": "model", "status": "loaded" }
        ]
    }))
    .await;

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
            name: "test-node".into(),
            endpoint: mock_url,
            vram_mb: 24000,
            pinned: vec![],
        }],
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    cortex_gateway::poller::poll_once(&fleet).await;

    // Verify both models exist.
    {
        let nodes = fleet.nodes.read().await;
        assert_eq!(nodes.get("test-node").unwrap().models.len(), 2);
    }

    // Now spin up a new mock that only reports one model, and re-point the node.
    let new_mock_url = common::spawn_mock_backend_with_models(json!({
        "object": "list",
        "data": [
            { "id": "keep-me", "object": "model", "status": "loaded" }
        ]
    }))
    .await;

    // Update the node endpoint to point at the new mock.
    // We can't change node_configs (they're immutable), so instead we'll
    // create a new fleet with the updated endpoint and poll that.
    let config2 = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        nodes: vec![NodeConfig {
            name: "test-node".into(),
            endpoint: new_mock_url,
            vram_mb: 24000,
            pinned: vec![],
        }],
    };

    let fleet2 = Arc::new(CortexState::from_config(&config2));

    // Seed the stale model so we can verify it gets removed.
    {
        let mut nodes = fleet2.nodes.write().await;
        let node = nodes.get_mut("test-node").unwrap();
        node.models.insert(
            "keep-me".into(),
            cortex_core::node::ModelEntry {
                id: "keep-me".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: None,
            },
        );
        node.models.insert(
            "drop-me".into(),
            cortex_core::node::ModelEntry {
                id: "drop-me".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: None,
            },
        );
    }

    cortex_gateway::poller::poll_once(&fleet2).await;

    let nodes = fleet2.nodes.read().await;
    let node = nodes.get("test-node").unwrap();
    assert_eq!(node.models.len(), 1);
    assert!(node.models.contains_key("keep-me"));
    assert!(!node.models.contains_key("drop-me"));
}
