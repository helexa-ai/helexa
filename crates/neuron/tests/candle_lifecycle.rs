//! Real model load/unload lifecycle through the candle harness.
//!
//! Gated behind the `cuda-integration` feature because it downloads a
//! real (small) GGUF from HuggingFace and materialises tensors on the
//! configured device. Run on a host with network access and either a
//! CUDA GPU (when built with `--features cuda`) or enough CPU RAM to
//! hold the model.
//!
//! Usage:
//!   cargo test -p neuron --features cuda-integration --test candle_lifecycle
//!
//! Optional environment variables:
//!   NEURON_TEST_MODEL_ID — HuggingFace repo to load (default: a small
//!     public Qwen3 GGUF repo).
//!   NEURON_TEST_QUANT    — quant substring matched against GGUF
//!     filenames (default: "Q4_K_M").
//!   HF_HOME              — HuggingFace cache directory.

#![cfg(feature = "cuda-integration")]

use cortex_core::harness::{HarnessConfig, ModelSpec};
use neuron::config::HarnessSettings;
use neuron::harness::HarnessRegistry;
use std::path::PathBuf;

#[tokio::test]
async fn test_candle_qwen3_load_unload_lifecycle() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("info,neuron=debug")
        .try_init();

    let model_id = std::env::var("NEURON_TEST_MODEL_ID")
        .unwrap_or_else(|_| "Qwen/Qwen3-0.6B-GGUF".to_string());
    let quant = std::env::var("NEURON_TEST_QUANT").unwrap_or_else(|_| "Q4_K_M".to_string());

    let mut settings = HarnessSettings::default();
    if let Ok(home) = std::env::var("HF_HOME") {
        settings.candle.hf_cache = Some(PathBuf::from(home));
    }

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:13131",
        &settings,
    );

    let spec = ModelSpec {
        model_id: model_id.clone(),
        harness: "candle".into(),
        quant: Some(quant),
        tensor_parallel: None,
        devices: Some(vec![0]),
    };

    registry
        .load_model(&spec)
        .await
        .expect("load_model should succeed");

    let models = registry
        .list_all_models()
        .await
        .expect("list_all_models");
    assert_eq!(models.len(), 1, "expected exactly one loaded model");
    assert_eq!(models[0].id, model_id);
    assert_eq!(models[0].harness, "candle");
    assert_eq!(models[0].status, "loaded");

    let url = registry.inference_endpoint(&model_id).await;
    assert_eq!(url, Some("http://localhost:13131".into()));

    // Re-loading the same model should be rejected.
    let again = registry.load_model(&spec).await;
    assert!(again.is_err(), "second load should error");

    registry
        .unload_model(&model_id)
        .await
        .expect("unload_model should succeed");

    let models = registry.list_all_models().await.expect("list_all_models");
    assert!(models.is_empty(), "registry should be empty after unload");

    // Unloading a model that isn't loaded should error.
    let err = registry.unload_model(&model_id).await;
    assert!(err.is_err(), "unload of missing model should error");
}
