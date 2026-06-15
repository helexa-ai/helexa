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
    /// (`NEURON_MAX_PROMPT_TOKENS`). cortex surfaces this as
    /// `max_model_len` on `/v1/models` so clients can size and compact
    /// their context instead of blindly overflowing it into a 400.
    /// `#[serde(default)]` (→ 0) for forward-compat with neurons that
    /// predate this field; cortex treats 0 as "unknown" and skips it.
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
