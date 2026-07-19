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

/// Mock harness for the #189 retry tests: fails `load_model` with a
/// `PreflightError::RepoFetchFailed` for the first `fail_first` calls,
/// then succeeds. Counts attempts so the tests can assert the retry
/// schedule actually ran.
struct FlakyFetchHarness {
    fail_first: u32,
    calls: std::sync::atomic::AtomicU32,
    loaded: std::sync::Mutex<Vec<String>>,
}

impl FlakyFetchHarness {
    fn new(fail_first: u32) -> Self {
        Self {
            fail_first,
            calls: std::sync::atomic::AtomicU32::new(0),
            loaded: std::sync::Mutex::new(vec![]),
        }
    }
}

#[async_trait::async_trait]
impl cortex_core::harness::Harness for FlakyFetchHarness {
    fn name(&self) -> &str {
        "candle"
    }

    async fn health(&self) -> cortex_core::harness::HarnessHealth {
        cortex_core::harness::HarnessHealth {
            name: "candle".into(),
            running: true,
            uptime_secs: None,
        }
    }

    async fn list_models(&self) -> anyhow::Result<Vec<cortex_core::harness::ModelInfo>> {
        Ok(vec![])
    }

    async fn load_model(&self, spec: &ModelSpec) -> anyhow::Result<()> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n < self.fail_first {
            return Err(anyhow::Error::new(
                neuron::harness::preflight::PreflightError::RepoFetchFailed {
                    model_id: spec.model_id.clone(),
                    cause: "error sending request (mock)".into(),
                },
            ));
        }
        self.loaded.lock().unwrap().push(spec.model_id.clone());
        Ok(())
    }

    async fn unload_model(&self, _model_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    async fn inference_endpoint(&self, _model_id: &str) -> Option<String> {
        None
    }
}

fn flaky_registry(fail_first: u32) -> (HarnessRegistry, std::sync::Arc<FlakyFetchHarness>) {
    let harness = std::sync::Arc::new(FlakyFetchHarness::new(fail_first));
    let mut registry = HarnessRegistry::new();
    registry.register(harness.clone());
    (registry, harness)
}

fn qwen_spec() -> ModelSpec {
    ModelSpec {
        model_id: "Qwen/Qwen3-8B".into(),
        harness: "candle".into(),
        quant: None,
        tensor_parallel: None,
        devices: None,
    }
}

// start_paused: the retry backoff sleeps auto-advance, so the whole
// ten-minute schedule runs in milliseconds of wall clock.
#[tokio::test(start_paused = true)]
async fn test_load_default_models_retries_transient_repo_fetch() {
    let (registry, harness) = flaky_registry(2);
    let specs = vec![qwen_spec()];
    let activation = ActivationTracker::new(&specs);
    startup::load_default_models(&registry, &specs, &activation, None).await;

    let snapshot = activation.snapshot().await;
    assert_eq!(snapshot.state, ActivationState::Ready);
    assert!(snapshot.failed.is_empty(), "failed: {:?}", snapshot.failed);
    assert!(snapshot.pending.is_empty());
    assert_eq!(snapshot.completed, vec!["Qwen/Qwen3-8B".to_string()]);
    assert_eq!(
        harness.calls.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "two failures then one success"
    );
}

#[tokio::test(start_paused = true)]
async fn test_load_default_models_repo_fetch_exhausts_retries() {
    // Harness never recovers: after the initial attempt plus
    // MAX_LOAD_RETRIES (6) retry rounds the model must land in
    // `failed` and the tracker must still flip to ready.
    let (registry, harness) = flaky_registry(u32::MAX);
    let specs = vec![qwen_spec()];
    let activation = ActivationTracker::new(&specs);
    startup::load_default_models(&registry, &specs, &activation, None).await;

    let snapshot = activation.snapshot().await;
    assert_eq!(snapshot.state, ActivationState::Ready);
    assert!(snapshot.completed.is_empty());
    assert!(snapshot.pending.is_empty());
    assert_eq!(snapshot.failed.len(), 1);
    assert_eq!(snapshot.failed[0].model_id, "Qwen/Qwen3-8B");
    assert_eq!(
        harness.calls.load(std::sync::atomic::Ordering::SeqCst),
        7,
        "initial attempt + MAX_LOAD_RETRIES rounds"
    );
}

#[tokio::test(start_paused = true)]
async fn test_load_default_models_structural_failure_not_retried() {
    // An unknown harness is a structural failure, not a transient
    // fetch error — exactly one attempt, straight to `failed`.
    let (registry, harness) = flaky_registry(u32::MAX);
    let mut spec = qwen_spec();
    spec.harness = "no-such-harness".into();
    let specs = vec![spec];
    let activation = ActivationTracker::new(&specs);
    startup::load_default_models(&registry, &specs, &activation, None).await;

    let snapshot = activation.snapshot().await;
    assert_eq!(snapshot.state, ActivationState::Ready);
    assert_eq!(snapshot.failed.len(), 1);
    assert_eq!(
        harness.calls.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "registry rejects the spec before the harness is reached"
    );
}
