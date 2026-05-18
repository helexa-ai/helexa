//! Activation-time behaviour: load_default_models continues past
//! individual failures so a single broken catalogue entry doesn't
//! prevent the rest of the fleet from starting.

use cortex_core::harness::{HarnessConfig, ModelSpec};
use neuron::config::HarnessSettings;
use neuron::harness::HarnessRegistry;
use neuron::startup;

#[tokio::test]
async fn test_load_default_models_skips_unknown_harness() {
    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );

    // Both entries fail synchronously inside the registry — no network
    // call escapes (the harness lookup mismatches before hf-hub is
    // touched). The function should still return cleanly.
    let specs = vec![
        ModelSpec {
            model_id: "model-a".into(),
            harness: "no-such-harness".into(),
            quant: None,
            tensor_parallel: None,
            devices: None,
        },
        ModelSpec {
            model_id: "model-b".into(),
            harness: "no-such-harness".into(),
            quant: None,
            tensor_parallel: None,
            devices: None,
        },
    ];

    startup::load_default_models(&registry, &specs).await;

    let listed = registry
        .list_all_models()
        .await
        .expect("list_all_models should succeed");
    assert!(
        listed.is_empty(),
        "no models should be loaded after failed entries"
    );
}

#[tokio::test]
async fn test_load_default_models_empty_is_noop() {
    let registry = HarnessRegistry::new();
    startup::load_default_models(&registry, &[]).await;
}
