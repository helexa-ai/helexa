//! Deactivation behaviour: unload_all_models tolerates an empty
//! registry and continues past per-model unload failures.

use cortex_core::harness::HarnessConfig;
use neuron::config::HarnessSettings;
use neuron::harness::HarnessRegistry;
use neuron::startup;

#[tokio::test]
async fn test_unload_all_models_empty_registry_is_noop() {
    let registry = HarnessRegistry::new();
    startup::unload_all_models(&registry).await;
}

#[tokio::test]
async fn test_unload_all_models_with_no_loaded_models() {
    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );

    startup::unload_all_models(&registry).await;

    let listed = registry
        .list_all_models()
        .await
        .expect("list_all_models should still succeed after shutdown cleanup");
    assert!(listed.is_empty());
}
