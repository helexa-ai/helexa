//! Hardware discovery and health types shared between cortex and neuron.

use serde::{Deserialize, Serialize};

/// Information about a single GPU device discovered on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
    pub vram_total_mb: u64,
    pub compute_capability: String,
}

/// Full discovery response from a neuron endpoint.
/// Returned by `GET /discovery`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResponse {
    pub hostname: String,
    pub os: String,
    pub kernel: String,
    pub cuda_version: Option<String>,
    pub driver_version: Option<String>,
    pub devices: Vec<DeviceInfo>,
    pub harnesses: Vec<String>,
    /// Set when the host has an NVIDIA stack that is currently
    /// unusable — specifically the userspace↔kernel-module version
    /// skew after an un-rebooted driver update ("Driver/library
    /// version mismatch"), where every CUDA call including nvidia-smi
    /// fails (#19). `None` on healthy hosts AND on hosts with no
    /// NVIDIA stack at all (CPU-only is not an error). Carries an
    /// operator-actionable description; cortex can read it to route
    /// around the node instead of cold-loading into a guaranteed
    /// failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_unavailable_reason: Option<String>,
    /// The neuron's effective maximum prompt size in tokens
    /// (`NEURON_MAX_PROMPT_TOKENS`) — the enforced prompt cap on this
    /// host. `#[serde(default)]` (→ 0) for forward-compat with neurons
    /// that predate this field; cortex treats 0 as "unknown".
    #[serde(default)]
    pub max_prompt_tokens: u64,
}

/// Runtime health metrics for a single GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHealth {
    pub index: u32,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    pub utilization_pct: u32,
    pub temp_c: u32,
}

/// Runtime health response from a neuron endpoint.
/// Returned by `GET /health`.
///
/// `activation` was added in 2026-05-26 to distinguish "process is up
/// and reachable" from "process is ready to serve traffic". A `Type=simple`
/// systemd unit reports `active` the moment the binary starts — but a
/// neuron whose `default_models` list takes minutes to materialise
/// won't bind its listener (or, in the new flow, won't have any models
/// loaded) until pre-warm completes. The new field is `#[serde(default)]`
/// so a pre-2026-05-26 gateway polling a new neuron — or vice versa —
/// keeps working.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub uptime_secs: u64,
    pub devices: Vec<DeviceHealth>,
    #[serde(default)]
    pub activation: ActivationStatus,
    /// Per-model admission load (#53): how many requests are running vs.
    /// queued on each loaded model right now. Cortex's load-aware router
    /// (#55) reads this to spread traffic across replicas and to propagate
    /// honest backpressure. `#[serde(default)]` keeps older gateways/neurons
    /// interoperable (absent → empty → treated as no load info).
    #[serde(default)]
    pub models: Vec<ModelLoad>,
}

/// Live admission load for one loaded model (#53).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoad {
    pub id: String,
    /// Requests currently running (batch-1 → 0 or 1).
    pub in_flight: usize,
    /// Requests waiting in the bounded admission queue.
    pub queue_depth: usize,
    /// Admission concurrency ceiling (#137) — the denominator for
    /// saturation = `in_flight / max_in_flight`. `#[serde(default)]` (→ 0)
    /// for pre-#137 neurons; cortex treats 0 as "unknown" and skips the
    /// ceiling gauge so a rolling deploy doesn't publish a bogus 0.
    #[serde(default)]
    pub max_in_flight: usize,
    /// Admission queue capacity (#137): how many requests may wait beyond
    /// the in-flight slots before the model sheds load. `#[serde(default)]`
    /// for back-compat with pre-#137 neurons.
    #[serde(default)]
    pub max_queue_depth: usize,
}

#[cfg(test)]
mod health_load_tests {
    use super::*;

    #[test]
    fn health_response_without_models_field_still_deserializes() {
        // A pre-#53 neuron's /health payload omits `models`; the gateway
        // must still parse it (serde default → empty).
        let json = r#"{"uptime_secs":42,"devices":[]}"#;
        let resp: HealthResponse = serde_json::from_str(json).expect("back-compat parse");
        assert_eq!(resp.uptime_secs, 42);
        assert!(resp.models.is_empty());
    }

    #[test]
    fn health_response_round_trips_model_load() {
        let resp = HealthResponse {
            uptime_secs: 1,
            devices: vec![],
            activation: ActivationStatus::default(),
            models: vec![ModelLoad {
                id: "Qwen/Qwen3.6-27B".into(),
                in_flight: 1,
                queue_depth: 3,
                max_in_flight: 8,
                max_queue_depth: 8,
            }],
        };
        let s = serde_json::to_string(&resp).unwrap();
        let back: HealthResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(back.models.len(), 1);
        assert_eq!(back.models[0].in_flight, 1);
        assert_eq!(back.models[0].queue_depth, 3);
        assert_eq!(back.models[0].max_in_flight, 8);
        assert_eq!(back.models[0].max_queue_depth, 8);
    }

    #[test]
    fn model_load_without_ceiling_fields_defaults_to_zero() {
        // A pre-#137 neuron omits max_in_flight/max_queue_depth; cortex must
        // still parse (serde default → 0, treated as "unknown").
        let json = r#"{"id":"m","in_flight":2,"queue_depth":0}"#;
        let m: ModelLoad = serde_json::from_str(json).expect("back-compat parse");
        assert_eq!(m.in_flight, 2);
        assert_eq!(m.max_in_flight, 0);
        assert_eq!(m.max_queue_depth, 0);
    }
}

/// High-level activation state of the neuron daemon. The HTTP listener
/// is bound during both states; what differs is whether the configured
/// `default_models` have finished loading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActivationState {
    /// At least one `default_models` entry is still loading. The
    /// neuron's other endpoints work, but inference against
    /// not-yet-loaded models will 404.
    PreWarming,
    /// Every `default_models` entry has either loaded or failed; the
    /// neuron is steady-state. Subsequent on-demand loads via
    /// `/models/load` don't flip back to PreWarming — that field
    /// reflects the activation-time set only.
    #[default]
    Ready,
}

/// Per-model failure record surfaced in [`ActivationStatus::failed`].
/// The error string is the rendered anyhow chain at the time of the
/// failure; operators read it from `/health` to decide whether to
/// retry, edit the spec, or unload+reload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreWarmFailure {
    pub model_id: String,
    pub error: String,
}

/// Activation-time progress snapshot. All four lists are populated by
/// the neuron's pre-warm task and read by the `/health` handler. The
/// snapshot is consistent: a model id appears in exactly one of
/// `pending`, `in_progress` (as `Option<String>`), `completed`, or
/// `failed` at any point in time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActivationStatus {
    pub state: ActivationState,
    /// Model ids queued but not yet started. Empty in `Ready` state.
    #[serde(default)]
    pub pending: Vec<String>,
    /// Model id currently materialising. None when between models or
    /// in `Ready` state.
    #[serde(default)]
    pub in_progress: Option<String>,
    /// Model ids that finished loading successfully during this
    /// activation. Cleared on process restart.
    #[serde(default)]
    pub completed: Vec<String>,
    /// Model ids that failed during this activation, with the rendered
    /// error chain. Cleared on process restart.
    #[serde(default)]
    pub failed: Vec<PreWarmFailure>,
}
