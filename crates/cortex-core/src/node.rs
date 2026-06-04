use crate::discovery::{ActivationStatus, DiscoveryResponse};
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
    /// entries with no loaded location — the catalogue profile doesn't
    /// declare capabilities yet (tracked separately from C3).
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLocation {
    pub node: String,
    pub status: ModelStatus,
    pub vram_estimate_mb: Option<u64>,
}
