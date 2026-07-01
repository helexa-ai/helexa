use crate::discovery::{ActivationStatus, DiscoveryResponse, ModelLoad};
use crate::harness::{ModelCost, ModelLimit};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Runtime state of a single neuron in the fleet.
#[derive(Debug, Clone)]
pub struct NodeState {
    pub name: String,
    /// Base URL of the neuron daemon (e.g. "http://beast.internal:13131").
    pub endpoint: String,
    pub healthy: bool,
    pub models: HashMap<String, ModelEntry>,
    /// Number of load/unload cycles since last process restart.
    pub lifecycle_cycles: u32,
    pub last_poll: Option<DateTime<Utc>>,
    /// Result of the most recent successful `GET /discovery` against
    /// this neuron. Cached forever once obtained — device topology is
    /// invariant for a given neuron process. `None` until the first
    /// successful poll. Used by the router and `/v1/models` to do
    /// catalogue × topology feasibility checks.
    pub discovery: Option<DiscoveryResponse>,
    /// Last-seen pre-warm progress from this neuron's `/health`
    /// endpoint. `None` until the first /health poll succeeds. The
    /// `/v1/models` handler reads `in_progress` + `pending` from here
    /// to synthesize `Loading` locations so clients see a catalogued
    /// model that's mid-prewarm as "loading", not "missing".
    pub activation: Option<ActivationStatus>,
    /// Last-seen per-model admission load from this neuron's `/health`
    /// (#53), keyed by model id. The router (#55) reads it to pick the
    /// least-busy replica when a model is loaded on more than one neuron.
    /// Empty until the first /health poll reports load.
    pub model_load: HashMap<String, ModelLoad>,
    /// Consecutive failed `/models` polls. The poller marks a node
    /// unhealthy only once this crosses a threshold, so a single transient
    /// miss (e.g. a neuron momentarily slow to answer while busy) doesn't
    /// yank the node — and all its models — out of routing. Reset to 0 on
    /// any successful poll.
    pub consecutive_poll_failures: u32,
}

/// A model registered on a node, with its runtime status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub status: ModelStatus,
    /// When this model was last used (for LRU eviction).
    pub last_accessed: Option<DateTime<Utc>>,
    /// Estimated VRAM usage in MB when loaded.
    pub vram_estimate_mb: Option<u64>,
    /// Modalities the loaded model advertises (e.g. `["text", "vision"]`),
    /// copied verbatim from the neuron's `ModelInfo.capabilities` at poll
    /// time. Empty when the neuron reports none. `#[serde(default)]` keeps
    /// older persisted/serialised entries deserialisable.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Runtime-detected capability flags from the neuron's `/models`
    /// response (`ModelInfo`). `false` when the neuron predates these
    /// fields or hasn't reported them yet.
    #[serde(default)]
    pub tool_call: bool,
    #[serde(default)]
    pub reasoning: bool,
    /// Self-derived token budget the neuron computed for this loaded
    /// model (#67), copied from `ModelInfo.limit` at poll time. `None`
    /// when the neuron doesn't compute one (arch without a context
    /// profile, or derivation disabled). This is the authoritative
    /// source the gateway advertises — operator-declared catalogue
    /// limits are no longer consulted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<ModelLimit>,
}

/// Model lifecycle status.
///
/// `Loading` is a gateway-side synthetic status: neurons never emit it
/// on `/models` (that endpoint only knows about already-loaded handles).
/// The gateway populates it from a neuron's `/health` activation
/// snapshot so the unified `/v1/models` can distinguish "model is
/// catalogued but no one has it" from "model is materialising on
/// neuron N right now". Other status values are reported verbatim by
/// neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
    Loading,
    /// Reported by neuron while a poisoned model auto-recovers via
    /// unload→reload (#17/#20). Temporarily unservable but NOT
    /// evicted: the gateway holds the route, answers with a transient
    /// retry error instead of 404, and must not race a second
    /// placement elsewhere.
    Recovering,
}

/// Unified model entry as exposed by the gateway's `/v1/models` endpoint.
///
/// The first four fields (`id`, `object`, `created`, `owned_by`) match
/// OpenAI's `/v1/models` shape verbatim, so existing OpenAI-aware
/// tooling deserialises this without custom code. The remaining fields
/// are helexa-specific extensions — OpenAI clients ignore unknown
/// fields and other consumers can read them for placement / debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexModelEntry {
    pub id: String,
    /// Always `"model"` per OpenAI's contract.
    pub object: String,
    /// Unix-second timestamp; cortex stamps this at response time.
    pub created: u64,
    /// OpenAI's "publisher" field — `"helexa"` for everything we serve.
    pub owned_by: String,
    /// True if any neuron currently has this model loaded. False for
    /// catalogue entries that are feasible but not yet loaded.
    pub loaded: bool,
    /// Neurons whose discovered topology can satisfy this model's
    /// catalogue placement constraints. Empty for models that are
    /// loaded somewhere but not present in the catalogue (cortex has
    /// no feasibility opinion on those).
    pub feasible_on: Vec<String>,
    /// Where this model is actually loaded right now. Subset of (or
    /// disjoint from) `feasible_on` depending on whether the catalogue
    /// covers this model.
    pub locations: Vec<ModelLocation>,
    /// Union of the modalities advertised by every neuron that has this
    /// model loaded (e.g. `["text", "vision"]`). Empty for catalogue-only
    /// entries with no loaded location — filled from catalogue profile
    /// capabilities when available, then unioned with runtime-detected
    /// values from loaded neurons.
    #[serde(default)]
    pub capabilities: Vec<String>,
    // ── Enrichment (issue #62) ────────────────────────────────
    /// Per-model token budget from the catalogue profile or discovered
    /// at load time. `None` when neither source provides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<ModelLimit>,
    /// Operator-set pricing from the catalogue profile — see
    /// [`cortex_core::harness::ModelCost`] for units (USD per 1M tokens) and
    /// the absent (not priced) vs `0.0` (intentionally free) distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// `true` when any neuron reports this model supports tool calls.
    #[serde(default)]
    pub tool_call: bool,
    /// `true` when any neuron reports this model supports reasoning tokens.
    #[serde(default)]
    pub reasoning: bool,
    // ── Flat ecosystem context-window fields (issue #78) ──────
    // Duplicates of `limit` under the flat, vLLM-convention key names
    // (`max_model_len` et al.) that OpenAI-ecosystem clients (Hermes
    // Agent, vLLM tooling) probe for — they cannot see `limit.context`.
    // Additive: `limit` stays the opencode-oriented source of truth.
    // Derived, never set directly — call [`sync_flat_limit`] after the
    // final `limit` value is known. Omitted (not `0`) when the window
    // is unknown; absent-vs-zero is load-bearing, as with `cost`.
    //
    // [`sync_flat_limit`]: CortexModelEntry::sync_flat_limit
    /// Served max-seq-len in tokens — mirrors `limit.context`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    /// Usable input budget — mirrors `limit.input` when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<usize>,
    /// Maximum generation tokens — mirrors `limit.output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
}

impl CortexModelEntry {
    /// Re-derive the flat ecosystem fields (#78) from `limit`.
    ///
    /// Must run after the final `limit` is known (post merge/tightening),
    /// immediately before serialization. Fully overwrites: a `None` limit
    /// clears the flat fields, so stale values can't survive a merge that
    /// dropped the limit.
    pub fn sync_flat_limit(&mut self) {
        self.max_model_len = self.limit.as_ref().map(|l| l.context);
        self.max_input_tokens = self.limit.as_ref().and_then(|l| l.input);
        self.max_output_tokens = self.limit.as_ref().map(|l| l.output);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLocation {
    pub node: String,
    pub status: ModelStatus,
    pub vram_estimate_mb: Option<u64>,
}
