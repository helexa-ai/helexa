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
        entitlements: Default::default(),
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
        entitlements: Default::default(),
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
async fn test_models_endpoint_unions_capabilities_across_nodes() {
    // C3: two neurons each have the same model loaded but advertise
    // different capability sets. The gateway's /v1/models must report
    // the union — a model loaded text-only on one node and
    // text+vision on another is vision-capable to the fleet.
    let node_a = common::spawn_mock_neuron_with_models(json!([
        {"id": "shared-model", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": null, "capabilities": ["text"]}
    ]))
    .await;
    let node_b = common::spawn_mock_neuron_with_models(json!([
        {"id": "shared-model", "harness": "candle", "status": "loaded", "devices": [1], "vram_used_mb": null, "capabilities": ["text", "vision"]}
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
        neurons: vec![
            NeuronEndpoint {
                name: "node-a".into(),
                endpoint: node_a,
            },
            NeuronEndpoint {
                name: "node-b".into(),
                endpoint: node_b,
            },
        ],
        models_config: "/dev/null".into(),
        entitlements: Default::default(),
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
    let body: serde_json::Value = client
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .expect("request should succeed")
        .json()
        .await
        .unwrap();

    let model = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .find(|m| m["id"] == "shared-model")
        .expect("shared-model should be present");

    let caps: Vec<&str> = model["capabilities"]
        .as_array()
        .expect("capabilities array")
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    assert!(caps.contains(&"text"), "union must include text: {caps:?}");
    assert!(
        caps.contains(&"vision"),
        "union must include vision: {caps:?}"
    );
    assert_eq!(caps.len(), 2, "union must not duplicate text: {caps:?}");

    // Both nodes hold the model, so two locations regardless of caps.
    assert_eq!(model["locations"].as_array().unwrap().len(), 2);
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
        entitlements: Default::default(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    {
        let mut nodes = fleet.nodes.write().await;
        nodes.get_mut("dead-node").unwrap().healthy = true;
    }

    // Debounce (#53 follow-up): a single missed poll must NOT evict a
    // previously-healthy node — a busy neuron briefly slow to answer
    // shouldn't yank its models out of routing.
    cortex_gateway::poller::poll_once(&fleet).await;
    assert!(
        fleet.nodes.read().await.get("dead-node").unwrap().healthy,
        "one failed poll should not mark a healthy node unhealthy"
    );

    // It flips unhealthy only after POLL_FAILURE_THRESHOLD (3) consecutive
    // failures.
    cortex_gateway::poller::poll_once(&fleet).await;
    cortex_gateway::poller::poll_once(&fleet).await;
    assert!(
        !fleet.nodes.read().await.get("dead-node").unwrap().healthy,
        "three consecutive failed polls should mark the node unhealthy"
    );

    // A subsequent successful poll would reset the counter and restore
    // health; covered implicitly by the discovery tests above.
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
        entitlements: Default::default(),
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
        entitlements: Default::default(),
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
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
            },
        );
        node.models.insert(
            "drop-me".into(),
            cortex_core::node::ModelEntry {
                id: "drop-me".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: None,
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
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

#[tokio::test]
async fn test_poller_captures_activation_from_health() {
    // Mock neuron is mid-prewarm: /models reports nothing (the loading
    // model hasn't been inserted into the harness map yet), but
    // /health's activation says model-x is in_progress and model-y is
    // queued behind it.
    let mock_url = common::spawn_mock_neuron_with_models_and_health(
        json!([]),
        json!({
            "uptime_secs": 30,
            "devices": [],
            "activation": {
                "state": "pre_warming",
                "pending": ["Qwen/model-y"],
                "in_progress": "Qwen/model-x",
                "completed": [],
                "failed": []
            }
        }),
    )
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
            name: "prewarm-node".into(),
            endpoint: mock_url,
        }],
        models_config: "/dev/null".into(),
        entitlements: Default::default(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    cortex_gateway::poller::poll_once(&fleet).await;

    let nodes = fleet.nodes.read().await;
    let node = nodes.get("prewarm-node").unwrap();
    assert!(node.healthy);
    // /models was empty — no entries in the per-node model map.
    assert!(node.models.is_empty());
    // But /health's activation should be captured.
    let activation = node
        .activation
        .as_ref()
        .expect("activation should be populated after /health poll");
    assert_eq!(activation.in_progress.as_deref(), Some("Qwen/model-x"));
    assert_eq!(activation.pending, vec!["Qwen/model-y".to_string()]);
}

#[tokio::test]
async fn test_poller_parses_recovering_status() {
    // #20: a model auto-recovering on a neuron (poisoned → unload →
    // reload, #17) is reported with status "recovering" and must land
    // in gateway state as the dedicated Recovering status — not fall
    // through the parser's catch-all to Loaded.
    let mock_url = common::spawn_mock_neuron_with_models(json!([
        {"id": "model-r", "harness": "candle", "status": "recovering", "devices": [0, 1], "vram_used_mb": null}
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
        entitlements: Default::default(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    cortex_gateway::poller::poll_once(&fleet).await;

    let nodes = fleet.nodes.read().await;
    let node = nodes.get("test-node").unwrap();
    let model_r = node.models.get("model-r").expect("model-r should exist");
    assert_eq!(model_r.status, ModelStatus::Recovering);
}
