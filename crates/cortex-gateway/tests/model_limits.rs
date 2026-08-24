//! Issue #62 / #67: `GET /v1/models` advertises a per-model serving budget so
//! an OpenAI-compatible client (opencode's helexa provider) can size and
//! compact its context without hand-configuration.
//!
//! Asserts the composition sources land on the response:
//!   - `limit` from the neuron's self-derived value (#67) — NOT the catalogue;
//!     an operator-declared catalogue `limit` is deliberately ignored.
//!   - `cost` from the catalogue profile (operator-set pricing).
//!   - `tool_call` / `reasoning` from the neuron's runtime detection (OR-ed in)
//!
//! Also asserts the flat, vLLM-convention duplicates (`max_model_len`,
//! `max_input_tokens`, `max_output_tokens`) mirror `limit` (#78): the
//! earlier removal of `max_model_len` as "unconsumed" was wrong — Hermes
//! Agent (and the wider OpenAI client ecosystem) probes those flat keys
//! and cannot see `limit.context`.

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::harness::ModelLimit;
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::test]
async fn v1_models_surfaces_limit_cost_and_capability_flags() {
    // Catalogue declares pricing + an operator `limit` that must be IGNORED
    // (#67): the neuron's self-derived limit is authoritative.
    let models_toml = r#"
[[models]]
id = "test-model"
harness = "candle"
limit.context = 999999
limit.input = 999999
limit.output = 999999
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
        entitlements: Default::default(),
        upstream: Default::default(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));

    // Seed the model as loaded on the node with runtime-detected flags set —
    // these must OR into the catalogue entry, not be lost.
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").expect("node exists");
        node.healthy = true;
        // The ladder the neuron reported at poll time (#223) — host
        // configuration, held once per node.
        node.reasoning_budget = ["minimal", "low", "medium", "high"]
            .into_iter()
            .zip([1024usize, 4096, 12288, 32768])
            .map(
                |(effort, tokens)| cortex_core::harness::ReasoningBudgetRung {
                    effort: effort.into(),
                    tokens,
                },
            )
            .collect();
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
                // Neuron's self-derived limit (#67) — the authoritative
                // source. Distinct from the catalogue's (ignored) values.
                limit: Some(ModelLimit {
                    context: 49152,
                    input: Some(40960),
                    output: 8192,
                    // The ceiling a request may name, distinct from the
                    // 8192-token reserve above (#278).
                    output_ceiling: 32768,
                }),
                servable: None,
            },
        );
        // A model with no derivable limit: the flat #78 fields must be
        // OMITTED (absent-vs-zero is load-bearing), never 0 or a guess.
        node.models.insert(
            "no-limit-model".into(),
            ModelEntry {
                id: "no-limit-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: None,
                capabilities: vec!["text".into()],
                tool_call: false,
                reasoning: false,
                limit: None,
                servable: None,
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

    // `limit` is the neuron's self-derived value (#67), NOT the catalogue's
    // (which declared 999999 and must be ignored). `cost` still flows from
    // the catalogue.
    assert_eq!(entry["limit"]["context"], 49152);
    assert_eq!(entry["limit"]["input"], 40960);
    assert_eq!(entry["limit"]["output"], 8192);
    assert_eq!(entry["cost"]["input"], 0.0);
    assert_eq!(entry["cost"]["output"], 0.0);

    // Runtime-detected capability flags OR-ed in from the neuron's ModelEntry.
    assert_eq!(entry["tool_call"], true);
    assert_eq!(entry["reasoning"], true);

    // Flat ecosystem duplicates (#78) mirror the advertised `limit` so
    // vLLM-convention probes (Hermes Agent) auto-detect the window.
    assert_eq!(entry["max_model_len"], 49152);
    assert_eq!(entry["max_input_tokens"], 40960);
    // The ceiling, not the reserve (#278). pi-ai's discovery — and every
    // client like it — turns this field into the cap it sends on each
    // request; advertising the 8192 reserve told a reasoning model's
    // harness to stop below the cost of a single think block.
    assert_eq!(entry["max_output_tokens"], 32768);
    // What each effort level buys (#223). A client can only send the
    // rung names — it has no way to express a token count — so a ladder
    // it cannot see is a ladder it has to guess at.
    let rungs = entry["reasoning_budget"]
        .as_array()
        .expect("reasoning_budget advertised for a reasoning model");
    let names: Vec<&str> = rungs
        .iter()
        .map(|r| r["effort"].as_str().expect("effort"))
        .collect();
    assert_eq!(names, ["minimal", "low", "medium", "high"]);
    assert!(
        rungs
            .iter()
            .all(|r| r["tokens"].as_u64().is_some_and(|t| t > 0)),
        "every advertised rung must name a token count"
    );

    // The window under the two names generic clients actually read.
    // pi-ai looks for `context_window` then `context_length` and gives
    // up if neither is present — `max_model_len` is invisible to it.
    assert_eq!(entry["context_window"], 49152);
    assert_eq!(entry["context_length"], 49152);

    // No limit → flat fields omitted entirely, never 0 or a guess.
    let unknown = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "no-limit-model")
        .expect("no-limit-model present in /v1/models");
    assert!(unknown.get("limit").is_none());
    for key in [
        "max_model_len",
        "max_input_tokens",
        "max_output_tokens",
        "context_window",
        "context_length",
    ] {
        assert!(
            unknown.get(key).is_none(),
            "{key} must be omitted when the window is unknown"
        );
    }

    let _ = std::fs::remove_file(&cat_path);
}
