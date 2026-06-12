//! Activation-time behaviour: load_default_models continues past
//! individual failures so a single broken catalogue entry doesn't
//! prevent the rest of the fleet from starting.

use cortex_core::discovery::ActivationState;
use cortex_core::harness::{HarnessConfig, ModelSpec};
use neuron::activation::ActivationTracker;
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

    let activation = ActivationTracker::new(&specs);
    startup::load_default_models(&registry, &specs, &activation, None).await;

    let listed = registry
        .list_all_models()
        .await
        .expect("list_all_models should succeed");
    assert!(
        listed.is_empty(),
        "no models should be loaded after failed entries"
    );

    // Both specs should land in `failed`; tracker should flip to ready.
    let snapshot = activation.snapshot().await;
    assert_eq!(snapshot.state, ActivationState::Ready);
    assert!(snapshot.pending.is_empty());
    assert!(snapshot.in_progress.is_none());
    assert!(snapshot.completed.is_empty());
    assert_eq!(snapshot.failed.len(), 2);
    let failed_ids: Vec<&str> = snapshot
        .failed
        .iter()
        .map(|f| f.model_id.as_str())
        .collect();
    assert!(failed_ids.contains(&"model-a"));
    assert!(failed_ids.contains(&"model-b"));
}

#[tokio::test]
async fn test_load_default_models_empty_is_noop() {
    let registry = HarnessRegistry::new();
    let activation = ActivationTracker::new(&[]);
    startup::load_default_models(&registry, &[], &activation, None).await;
    let snapshot = activation.snapshot().await;
    assert_eq!(snapshot.state, ActivationState::Ready);
}

#[tokio::test]
async fn test_load_default_models_skipped_on_driver_mismatch() {
    // #19: when the host has a driver/library mismatch, no load is
    // attempted (it would die in cuInit/NCCL with a cryptic error);
    // every default model lands in `failed` carrying the actionable
    // reason, and the tracker still flips to ready so /health serves.
    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );
    let specs = vec![ModelSpec {
        model_id: "Qwen/Qwen3.6-27B".into(),
        harness: "candle".into(),
        quant: Some("q6k".into()),
        tensor_parallel: Some(2),
        devices: None,
    }];
    let activation = ActivationTracker::new(&specs);
    let reason = "host NVIDIA driver/library mismatch (userspace NVML 580.159 vs loaded \
                  kernel module 580.159.03) — reboot the host to reload the kernel module; \
                  all CUDA inference is unavailable until then";
    startup::load_default_models(&registry, &specs, &activation, Some(reason)).await;

    let listed = registry
        .list_all_models()
        .await
        .expect("list_all_models should succeed");
    assert!(
        listed.is_empty(),
        "no load may be attempted on a mismatch host"
    );

    let snapshot = activation.snapshot().await;
    assert_eq!(snapshot.state, ActivationState::Ready);
    assert_eq!(snapshot.failed.len(), 1);
    assert!(
        snapshot.failed[0].error.contains("driver/library mismatch"),
        "failure must carry the actionable reason, got: {}",
        snapshot.failed[0].error
    );
}
