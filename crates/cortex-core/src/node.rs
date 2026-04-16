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
}

/// Model lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
}

/// Unified model entry as exposed by the gateway's `/v1/models` endpoint.
/// Includes which node(s) host this model and their status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexModelEntry {
    pub id: String,
    pub object: String,
    /// Which nodes have this model (and their status).
    pub locations: Vec<ModelLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLocation {
    pub node: String,
    pub status: ModelStatus,
    pub vram_estimate_mb: Option<u64>,
}
