//! Issue #62: `GET /v1/models` advertises a per-model serving budget so an
//! OpenAI-compatible client (opencode's helexa provider) can size and compact
//! its context without hand-configuration.
//!
//! Asserts the two composition sources land on the response:
//!   - `limit` + `cost` from the catalogue profile (operator-declared)
//!   - `tool_call` / `reasoning` from the neuron's runtime detection (OR-ed in)
//!
//! Also a regression guard for the removal of `max_model_len` — the misnamed,
//! unconsumed vLLM-ism that this contract replaces.

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::test]
async fn v1_models_surfaces_limit_cost_and_capability_flags() {
    // Catalogue declares the token budget + pricing for the model.
    let models_toml = r#"
[[models]]
id = "test-model"
harness = "candle"
limit.context = 49152
limit.input = 40960
limit.output = 8192
cost.input = 0.0
cost.output = 0.0
capabilities = ["text"]
"#;
    let cat_path = std::env::temp_dir().join("cortex_test_issue62_models.toml");
    std::fs::write(&cat_path, models_toml).unwrap();

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
            // Never contacted: build_app does not spawn the poller, so the
            // seeded state below is authoritative for /v1/models.
            endpoint: "http://127.0.0.1:1".into(),
        }],
        models_config: cat_path.to_string_lossy().into_owned(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Seed the model as loaded on the node with runtime-detected flags set —
    // these must OR into the catalogue entry, not be lost.
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").expect("node exists");
        node.healthy = true;
        node.models.insert(
            "test-model".into(),
            ModelEntry {
                id: "test-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
                capabilities: vec!["text".into()],
                tool_call: true,
                reasoning: true,
            },
        );
    }

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let entry = body["data"]
        .as_array()
        .expect("data is an array")
        .iter()
        .find(|m| m["id"] == "test-model")
        .expect("test-model present in /v1/models");

    // Catalogue-sourced budget + pricing flow through.
    assert_eq!(entry["limit"]["context"], 49152);
    assert_eq!(entry["limit"]["input"], 40960);
    assert_eq!(entry["limit"]["output"], 8192);
    assert_eq!(entry["cost"]["input"], 0.0);
    assert_eq!(entry["cost"]["output"], 0.0);

    // Runtime-detected capability flags OR-ed in from the neuron's ModelEntry.
    assert_eq!(entry["tool_call"], true);
    assert_eq!(entry["reasoning"], true);

    // Regression guard: the removed, unconsumed vLLM-ism must not reappear.
    assert!(
        entry.get("max_model_len").is_none(),
        "max_model_len was removed; /v1/models must not advertise it"
    );

    let _ = std::fs::remove_file(&cat_path);
}
