mod common;

use chrono::Utc;
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::json;
use std::sync::Arc;

/// Spawn a mock neuron that accepts `/models/unload` and records unload calls.
async fn spawn_eviction_mock() -> (String, Arc<tokio::sync::Mutex<Vec<String>>>) {
    use axum::extract::Path;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use serde_json::Value;

    let unloaded: Arc<tokio::sync::Mutex<Vec<String>>> = Arc::new(tokio::sync::Mutex::new(vec![]));
    let unloaded_clone = Arc::clone(&unloaded);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();

    let app = Router::new()
        .route(
            "/models/unload",
            post(move |Json(body): Json<Value>| {
                let unloaded = Arc::clone(&unloaded_clone);
                async move {
                    let model_id = body
                        .get("model_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    unloaded.lock().await.push(model_id);
                    Json(json!({"status": "unloaded"}))
                }
            }),
        )
        .route("/models", get(|| async { Json(json!([])) }))
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_model_id): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({"url": url})) }
            }),
        );

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (base_url, unloaded)
}

/// A fleet whose catalogue ranks models, so the displacement rules are
/// exercised rather than defaulted. Writes the catalogue to a temp file
/// because `CortexState` loads it from a path.
fn make_fleet_with_catalogue(
    endpoint: &str,
    catalogue_toml: &str,
    tag: &str,
) -> (Arc<CortexState>, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!("cortex-evict-catalogue-{tag}.toml"));
    std::fs::write(&path, catalogue_toml).expect("write test catalogue");
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
            name: "gpu-node".into(),
            endpoint: endpoint.to_string(),
        }],
        models_config: path.to_string_lossy().into_owned(),
        entitlements: Default::default(),
        upstream: Default::default(),
    };
    (Arc::new(CortexState::from_config(&config)), path)
}

fn loaded(id: &str, age_secs: i64) -> ModelEntry {
    ModelEntry {
        id: id.into(),
        status: ModelStatus::Loaded,
        last_accessed: Some(Utc::now() - chrono::Duration::seconds(age_secs)),
        vram_estimate_mb: Some(8000),
        capabilities: Vec::new(),
        tool_call: false,
        reasoning: false,
        limit: None,
        servable: None,
        reasoning_budget: Vec::new(),
    }
}

/// The fleet policy under test. Two residency classes: image generation
/// and the mid tier share a node and take turns on it; the flagship and
/// the frontier model share a bigger one and take turns on that. Nothing
/// in the everyday class may touch the big-node class.
const TIERED: &str = r#"
[[models]]
id = "flagship"
harness = "candle"
residency_priority = 300

[[models]]
id = "frontier"
harness = "candle"
residency_priority = 300

[[models]]
id = "image"
harness = "candle"
residency_priority = 200

[[models]]
id = "mid"
harness = "candle"
residency_priority = 200
"#;

fn make_fleet(endpoint: &str, defrag_after: u32) -> Arc<CortexState> {
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: defrag_after,
        },
        neurons: vec![NeuronEndpoint {
            name: "gpu-node".into(),
            endpoint: endpoint.to_string(),
        }],
        models_config: "/dev/null".into(),
        entitlements: Default::default(),
        upstream: Default::default(),
    };
    Arc::new(CortexState::from_config(&config))
}

#[tokio::test]
async fn test_evict_lru_model() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let fleet = make_fleet(&mock_url, 0);

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
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
                servable: None,
                reasoning_budget: Vec::new(),
            },
        );
        node.models.insert(
            "new-model".into(),
            ModelEntry {
                id: "new-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: Some(Utc::now()),
                vram_estimate_mb: Some(8000),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
                servable: None,
                reasoning_budget: Vec::new(),
            },
        );
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", None)
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, Some("old-model".to_string()));

    let calls = unloaded.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], "old-model");

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
async fn test_eviction_nothing_to_evict() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let fleet = make_fleet(&mock_url, 0);

    // No models at all.
    {
        let mut nodes = fleet.nodes.write().await;
        nodes.get_mut("gpu-node").unwrap().healthy = true;
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", None)
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, None);
    let calls = unloaded.lock().await;
    assert!(calls.is_empty());
}

#[tokio::test]
async fn test_eviction_increments_lifecycle_cycles() {
    let (mock_url, _) = spawn_eviction_mock().await;
    let fleet = make_fleet(&mock_url, 0);

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
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
                servable: None,
                reasoning_budget: Vec::new(),
            },
        );
    }

    cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", None)
        .await
        .expect("eviction should succeed");

    let nodes = fleet.nodes.read().await;
    assert_eq!(nodes.get("gpu-node").unwrap().lifecycle_cycles, 1);
}

#[tokio::test]
async fn test_last_accessed_updated_on_request() {
    let mock_url = common::spawn_mock_neuron().await;
    let (fleet, gw_url) = common::spawn_gateway_with_state(&mock_url).await;

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

/// Image generation must take the mid tier's node when it needs it —
/// this is existing fleet behaviour and the change must preserve it.
#[tokio::test]
async fn image_generation_evicts_the_mid_tier() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let (fleet, path) = make_fleet_with_catalogue(&mock_url, TIERED, "image-takes-mid");

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models.insert("mid".into(), loaded("mid", 60));
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", Some("image"))
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, Some("mid".to_string()));
    assert_eq!(unloaded.lock().await.as_slice(), ["mid"]);
    std::fs::remove_file(path).ok();
}

/// Image generation must never take the flagship's node. Its device
/// constraints alone would let it land there, so nothing but priority
/// stops this.
#[tokio::test]
async fn image_generation_cannot_evict_the_flagship() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let (fleet, path) = make_fleet_with_catalogue(&mock_url, TIERED, "image-spares-flagship");

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models
            .insert("flagship".into(), loaded("flagship", 9999));
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", Some("image"))
        .await
        .expect("eviction should succeed");

    assert_eq!(
        evicted, None,
        "the flagship outranks image generation, however stale it is"
    );
    assert!(unloaded.lock().await.is_empty());
    std::fs::remove_file(path).ok();
}

/// The frontier tier may cold-swap the flagship off its node.
#[tokio::test]
async fn the_frontier_tier_evicts_the_flagship() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let (fleet, path) = make_fleet_with_catalogue(&mock_url, TIERED, "frontier-takes-flagship");

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models
            .insert("flagship".into(), loaded("flagship", 10));
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", Some("frontier"))
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, Some("flagship".to_string()));
    assert_eq!(unloaded.lock().await.as_slice(), ["flagship"]);
    std::fs::remove_file(path).ok();
}

/// LRU still decides *which* victim, but only among the models the
/// incoming one outranks. Here the flagship is by far the stalest, so a
/// purely age-ordered evictor would take it.
#[tokio::test]
async fn lru_picks_the_oldest_displaceable_model_not_the_oldest_model() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let (fleet, path) = make_fleet_with_catalogue(&mock_url, TIERED, "lru-within-rank");

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models
            .insert("flagship".into(), loaded("flagship", 9999));
        node.models.insert("mid".into(), loaded("mid", 60));
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", Some("image"))
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, Some("mid".to_string()));
    assert_eq!(unloaded.lock().await.as_slice(), ["mid"]);
    std::fs::remove_file(path).ok();
}

/// The other half of the cold-swap: after an image generation has taken
/// the node, the next text request must be able to take it back. A
/// strict priority order would let whichever model arrived first hold
/// the node forever, which looks to a user like the text tier vanishing
/// once somebody generated an image.
#[tokio::test]
async fn the_mid_tier_takes_its_node_back_from_image_generation() {
    let (mock_url, unloaded) = spawn_eviction_mock().await;
    let (fleet, path) = make_fleet_with_catalogue(&mock_url, TIERED, "mid-swaps-back");

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu-node").unwrap();
        node.healthy = true;
        node.models.insert("image".into(), loaded("image", 30));
    }

    let evicted = cortex_gateway::evictor::evict_lru_on_node(&fleet, "gpu-node", Some("mid"))
        .await
        .expect("eviction should succeed");

    assert_eq!(evicted, Some("image".to_string()));
    assert_eq!(unloaded.lock().await.as_slice(), ["image"]);
    std::fs::remove_file(path).ok();
}
