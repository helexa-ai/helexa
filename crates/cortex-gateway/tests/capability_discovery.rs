//! Cold models advertise what they can do (#241).
//!
//! Before this, `/v1/models` reported `capabilities: []` for anything not
//! currently loaded, because the only source of capabilities was a loaded
//! neuron reporting its own handle. That made image generation
//! undiscoverable in practice: the image model is evicted almost all of
//! the time, so a client listing our models saw an entry that looked like
//! a text model with nothing to say about itself.
//!
//! The fix is discovery, not configuration — neurons derive modalities
//! from their local model cache and cortex asks. These tests pin the two
//! properties that matter: a cold model's capabilities reach `/v1/models`,
//! and a node with no local evidence cannot erase them.

mod common;

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_gateway::state::CortexState;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A catalogue holding one image model that nothing has loaded.
fn write_catalogue() -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let toml = r#"
[[models]]
id = "Tongyi-MAI/Z-Image-Turbo"
harness = "candle"
min_devices = 1

[aliases]
"helexa/image" = "Tongyi-MAI/Z-Image-Turbo"
"#;
    let path = std::env::temp_dir().join(format!(
        "cortex_test_capability_models.{}.{}.toml",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, toml).unwrap();
    path
}

fn fleet_for(endpoint: String) -> Arc<CortexState> {
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
            name: "gpu".into(),
            endpoint,
        }],
        models_config: write_catalogue().to_string_lossy().into_owned(),
        ..Default::default()
    };
    Arc::new(CortexState::from_config(&config))
}

/// Serve `fleet` and return its base URL, so a test can assert the
/// actual HTTP surface rather than the state behind it.
async fn serve(fleet: Arc<CortexState>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = cortex_gateway::build_app(fleet);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// The headline case: nothing loaded anywhere, and `/v1/models` still
/// says the image model generates images. This is the acceptance
/// criterion from #241 — asserted on the response a client actually
/// receives, because that is where the bug was visible.
#[tokio::test]
async fn cold_image_model_advertises_image_capability() {
    let neuron = common::spawn_mock_neuron_with_capabilities(
        serde_json::json!([]),
        &[("Tongyi-MAI/Z-Image-Turbo", vec!["image"])],
    )
    .await;
    let fleet = fleet_for(neuron);

    cortex_gateway::poller::poll_once(&fleet).await;

    let gateway = serve(fleet).await;
    let body: serde_json::Value = reqwest::get(format!("{gateway}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let entry = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "Tongyi-MAI/Z-Image-Turbo")
        .expect("the catalogued image model should be listed");
    assert_eq!(
        entry["loaded"], false,
        "precondition: this asserts the *cold* path"
    );
    assert_eq!(
        entry["capabilities"],
        serde_json::json!(["image"]),
        "a cold image model must still advertise that it generates images"
    );

    // The alias is the id most clients will actually send, so it has to
    // carry the capability too — advertising it only on the concrete id
    // would leave the public-facing name looking like a text model.
    let alias = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "helexa/image")
        .expect("the tier alias should be listed");
    assert_eq!(
        alias["capabilities"],
        serde_json::json!(["image"]),
        "the alias must advertise the same capabilities as its target"
    );
}

/// A node that has never cached the weights answers 404. That must leave
/// the id unknown rather than recording an empty list, or the first node
/// polled would permanently pin a model to no capabilities — the exact
/// failure this work removes, reintroduced from a different direction.
#[tokio::test]
async fn node_without_local_evidence_records_nothing() {
    let neuron = common::spawn_mock_neuron_with_capabilities(serde_json::json!([]), &[]).await;
    let fleet = fleet_for(neuron);

    cortex_gateway::poller::poll_once(&fleet).await;

    assert!(
        fleet.discovered_capabilities.read().await.is_empty(),
        "a 404 means 'no local evidence', which must not be cached as an answer"
    );
}

/// An empty list is likewise not an answer worth keeping: a repo cached
/// without weights says nothing about what other nodes can serve.
#[tokio::test]
async fn empty_capability_list_is_not_recorded() {
    let neuron = common::spawn_mock_neuron_with_capabilities(
        serde_json::json!([]),
        &[("Tongyi-MAI/Z-Image-Turbo", vec![])],
    )
    .await;
    let fleet = fleet_for(neuron);

    cortex_gateway::poller::poll_once(&fleet).await;

    assert!(
        fleet.discovered_capabilities.read().await.is_empty(),
        "an empty list must not be cached as this model's capabilities"
    );
}
