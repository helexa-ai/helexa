//! Alias resolution: a client request with `model: "helexa/small"`
//! routes to the concrete model id (e.g. `Qwen/Qwen3-1.7B`), with the
//! proxied request body rewritten so the upstream neuron sees a model
//! name that matches its loaded handle.

mod common;

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Write a `models.toml` with one alias to a unique temp path. Returns
/// the path; the file persists for the test process and gets reaped by
/// the OS at exit. Using $XDG_RUNTIME_DIR fallback for the temp dir
/// keeps the file off shared /tmp on CI without pulling in tempfile.
fn write_models_toml(alias: &str, target: &str) -> PathBuf {
    let contents = format!(
        r#"
[aliases]
"{alias}" = "{target}"
"#
    );
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("cortex-test-models-{pid}-{now}.toml"));
    std::fs::write(&path, contents).expect("write temp models.toml");
    path
}

#[tokio::test]
async fn test_alias_resolves_in_chat_completions() {
    let mock_url = common::spawn_mock_neuron().await;
    let models_path = write_models_toml("helexa/small", "test-model");

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
            endpoint: mock_url,
        }],
        models_config: models_path.to_string_lossy().to_string(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Seed the node as healthy with the concrete model loaded under
    // the target id. The poller doesn't run in this test; we just
    // populate state manually.
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
                vram_estimate_mb: None,
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
            },
        );
    }

    // Sanity: the catalogue actually picked up the alias.
    assert_eq!(
        fleet.catalogue.resolve_alias("helexa/small"),
        "test-model",
        "alias should resolve to target id"
    );

    // Spawn the gateway against this fleet.
    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let gateway_url = format!("http://{gateway_addr}");

    // Send a chat completion against the alias. The mock backend
    // echoes back the `model` field it received — so a body whose
    // model wasn't rewritten would come back as "helexa/small", and a
    // properly-rewritten one as "test-model".
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gateway_url}/v1/chat/completions"))
        .json(&json!({
            "model": "helexa/small",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("gateway should respond");

    assert!(resp.status().is_success(), "gateway returned non-2xx");
    let body: serde_json::Value = resp.json().await.expect("response is JSON");
    assert_eq!(
        body.get("model").and_then(|m| m.as_str()),
        Some("test-model"),
        "mock backend should have seen the resolved model id, not the alias"
    );
}

#[tokio::test]
async fn test_aliases_surface_in_v1_models() {
    let mock_url = common::spawn_mock_neuron().await;
    let models_path = write_models_toml("helexa/small", "test-model");

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
            endpoint: mock_url,
        }],
        models_config: models_path.to_string_lossy().to_string(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Seed the target as loaded so the alias's mirrored entry shows
    // loaded=true.
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
                vram_estimate_mb: Some(2000),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
            },
        );
    }

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let gateway_url = format!("http://{gateway_addr}");

    let resp = reqwest::get(format!("{gateway_url}/v1/models"))
        .await
        .expect("gateway should respond");
    let body: serde_json::Value = resp.json().await.unwrap();
    let entries = body
        .get("data")
        .and_then(|d| d.as_array())
        .expect("data array");

    // Both the alias and the target should be present.
    let ids: Vec<&str> = entries
        .iter()
        .filter_map(|e| e.get("id").and_then(|v| v.as_str()))
        .collect();
    assert!(ids.contains(&"test-model"), "target should be listed");
    assert!(ids.contains(&"helexa/small"), "alias should be listed");

    // The alias's `loaded` flag and locations should mirror the target.
    let alias_entry = entries
        .iter()
        .find(|e| e.get("id").and_then(|v| v.as_str()) == Some("helexa/small"))
        .expect("alias entry");
    assert_eq!(alias_entry.get("loaded"), Some(&json!(true)));
    let locations = alias_entry
        .get("locations")
        .and_then(|l| l.as_array())
        .expect("locations array");
    assert_eq!(locations.len(), 1);
    assert_eq!(
        locations[0].get("node").and_then(|n| n.as_str()),
        Some("mock-node")
    );
}

#[tokio::test]
async fn test_alias_falls_through_for_unmapped_model() {
    // Catalogue has an alias for some-other-thing but the request
    // model "test-model" isn't an alias; resolution should be a no-op.
    let mock_url = common::spawn_mock_neuron().await;
    let models_path = write_models_toml("helexa/large", "definitely-not-loaded");

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
            endpoint: mock_url,
        }],
        models_config: models_path.to_string_lossy().to_string(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));
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
                vram_estimate_mb: None,
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
            },
        );
    }

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let gateway_url = format!("http://{gateway_addr}");

    let resp = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/chat/completions"))
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body.get("model").and_then(|m| m.as_str()),
        Some("test-model")
    );
}
