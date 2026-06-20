//! Issue #68: the `cost` wire contract on `GET /v1/models`.
//!
//! `cost` is operator-set pricing sourced from the `models.toml` catalogue
//! profile (the source of truth today; the marketplace clearing house #59
//! later — both must read the same value metering/#51 bills against). The
//! shape is the models.dev/opencode convention: **USD per 1,000,000 tokens,
//! as JSON numbers**, with optional `cache_read`/`cache_write` tiers. This
//! test pins:
//!   - the units/shape (per-million floats, not per-token, not strings);
//!   - that cache fields flow through when present and are omitted otherwise;
//!   - the load-bearing **absent vs `0.0`** distinction (#68): a model with
//!     no catalogue `cost` omits the key entirely (price unknown), distinct
//!     from an explicit `0.0` (intentionally free).
//!
//! Catalogue-only models surface via Pass 1 of `list_models` even with no
//! feasible neuron, so this is hermetic — no nodes or poller needed.

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_gateway::state::CortexState;
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::test]
async fn v1_models_cost_units_shape_and_absent_vs_zero() {
    // Three catalogue models exercise the whole contract: a priced model
    // with cache tiers, an intentionally-free model (explicit 0.0), and an
    // unpriced model (no `cost` block at all).
    let models_toml = r#"
[[models]]
id = "priced-model"
harness = "candle"
cost.input = 0.5
cost.output = 1.5
cost.cache_read = 0.05
cost.cache_write = 0.6

[[models]]
id = "free-model"
harness = "candle"
cost.input = 0.0
cost.output = 0.0

[[models]]
id = "unpriced-model"
harness = "candle"
"#;
    let cat_path = std::env::temp_dir().join("cortex_test_issue68_models.toml");
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
        // Never contacted: build_app does not spawn the poller, so the
        // catalogue alone drives /v1/models.
        neurons: vec![NeuronEndpoint {
            name: "mock-node".into(),
            endpoint: "http://127.0.0.1:1".into(),
        }],
        models_config: cat_path.to_string_lossy().into_owned(),
        entitlements: Default::default(),
    };

    let fleet = Arc::new(CortexState::from_config(&config));
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

    let data = body["data"].as_array().expect("data is an array");
    let entry = |id: &str| {
        data.iter()
            .find(|m| m["id"] == id)
            .unwrap_or_else(|| panic!("{id} present in /v1/models"))
            .clone()
    };

    // Priced model: exact values flow through as JSON numbers (USD per 1M
    // tokens). If anything rescaled by 10⁶ or stringified, these fail.
    let priced = entry("priced-model");
    assert_eq!(priced["cost"]["input"], 0.5);
    assert_eq!(priced["cost"]["output"], 1.5);
    assert_eq!(priced["cost"]["cache_read"], 0.05);
    assert_eq!(priced["cost"]["cache_write"], 0.6);
    assert!(
        priced["cost"]["input"].is_number(),
        "cost.input must be a JSON number, not a string"
    );

    // Intentionally free: cost present, rates explicitly 0.0. Unset cache
    // tiers are omitted (skip_serializing_if), not emitted as null/0.
    let free = entry("free-model");
    assert_eq!(free["cost"]["input"], 0.0);
    assert_eq!(free["cost"]["output"], 0.0);
    assert!(
        free["cost"].get("cache_read").is_none(),
        "absent cache tiers must be omitted, not null"
    );
    assert!(free["cost"].get("cache_write").is_none());

    // Unpriced: the whole `cost` object is omitted — "price unknown",
    // distinct from the free model's explicit 0.0. This is the #68
    // distinction opencode needs to avoid showing $0 for a model whose
    // price simply hasn't been declared.
    let unpriced = entry("unpriced-model");
    assert!(
        unpriced.get("cost").is_none(),
        "a model with no catalogue cost must omit `cost` entirely, got {:?}",
        unpriced.get("cost")
    );

    let _ = std::fs::remove_file(&cat_path);
}
