mod common;

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::node::ModelStatus;
use cortex_gateway::state::CortexState;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn test_poller_discovers_models() {
    // Mock neuron reports 2 models via /models endpoint (neuron format).
    let mock_url = common::spawn_mock_neuron_with_models(json!([
        {"id": "model-a", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": 8000},
        {"id": "model-b", "harness": "candle", "status": "unloaded", "devices": [], "vram_used_mb": null}
    ]))
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
        neurons: vec![NeuronEndpoint {
            name: "test-node".into(),
            endpoint: mock_url,
        }],
        models_config: "/dev/null".into(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    {
        let nodes = fleet.nodes.read().await;
        let node = nodes.get("test-node").unwrap();
        assert!(!node.healthy);
        assert!(node.models.is_empty());
    }

    cortex_gateway::poller::poll_once(&fleet).await;

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
    let mock_url = common::spawn_mock_neuron_with_models(json!([
        {"id": "model-x", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": null},
        {"id": "model-y", "harness": "candle", "status": "loaded", "devices": [1], "vram_used_mb": null}
    ]))
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
        neurons: vec![NeuronEndpoint {
            name: "poll-node".into(),
            endpoint: mock_url,
        }],
        models_config: "/dev/null".into(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    cortex_gateway::poller::poll_once(&fleet).await;

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

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
        neurons: vec![NeuronEndpoint {
            name: "dead-node".into(),
            endpoint: "http://127.0.0.1:1".into(),
        }],
        models_config: "/dev/null".into(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

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
    let mock_url = common::spawn_mock_neuron_with_models(json!([
        {"id": "keep-me", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": null},
        {"id": "drop-me", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": null}
    ]))
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
        neurons: vec![NeuronEndpoint {
            name: "test-node".into(),
            endpoint: mock_url,
        }],
        models_config: "/dev/null".into(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    cortex_gateway::poller::poll_once(&fleet).await;

    {
        let nodes = fleet.nodes.read().await;
        assert_eq!(nodes.get("test-node").unwrap().models.len(), 2);
    }

    // New mock with only one model.
    let new_mock_url = common::spawn_mock_neuron_with_models(json!([
        {"id": "keep-me", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": null}
    ]))
    .await;

    let config2 = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![NeuronEndpoint {
            name: "test-node".into(),
            endpoint: new_mock_url,
        }],
        models_config: "/dev/null".into(),
    };

    let fleet2 = Arc::new(CortexState::from_config(&config2));

    // Seed stale model.
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
