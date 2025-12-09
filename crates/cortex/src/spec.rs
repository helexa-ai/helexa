/* helexa/crates/cortex/src/spec.rs */

// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use cache::JsonStore;
use protocol::ModelConfig;
use serde::{Deserialize, Serialize};

/// High-level specification used to bootstrap cortex with initial
/// model configurations and demand hints.
///
/// This is analogous to a "chainspec" in some blockchain systems:
/// it is intended for bootstrapping and long-lived policy/demand
/// hints, not as a rigid "run exactly these models forever" config.
///
/// At startup, cortex can:
/// - load this spec (if provided via `--spec`),
/// - seed an in-memory demand/model state,
/// - persist/overlay that state with runtime learnings via the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexSpec {
    /// Optional human-readable name for this spec.
    pub name: Option<String>,
    /// Optional version tag for the spec format / deployment.
    pub version: Option<String>,
    /// Initial model definitions and demand hints.
    pub models: Vec<ModelSpec>,
    /// Placeholder for future global policy fields.
    #[serde(default)]
    pub policy: Option<PolicySpec>,
}

/// Per-model bootstrapping configuration.
///
/// This wraps `ModelConfig` from the `protocol` crate with additional
/// hints that inform how the provisioner should treat this model at
/// startup and over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Protocol-level model configuration used when talking to neurons.
    pub config: ModelConfig,
    /// Optional demand weight; higher values indicate that this model
    /// is expected to be requested more frequently.
    #[serde(default)]
    pub weight: Option<f64>,
    /// Optional minimum number of replicas the provisioner should try
    /// to maintain across all neurons.
    #[serde(default)]
    pub min_replicas: Option<u32>,
    /// Optional maximum number of replicas the provisioner should
    /// allow across all neurons.
    #[serde(default)]
    pub max_replicas: Option<u32>,
}

/// Placeholder for future global policy fields.
///
/// Examples (not yet implemented):
/// - default maximum concurrent models per neuron
/// - default ramp-up/down rates for replicas
/// - default retry policies for failed spawns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicySpec {
    /// Optional free-form metadata for future use.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// In-memory representation of model demand and config state that
/// the orchestrator/provisioner can consume.
///
/// This is backed by a JSON cache via the `cache` crate so that
/// demand learnings survive cortex restarts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelDemandState {
    /// Per-model entries keyed by `ModelId` string.
    pub models: Vec<ModelDemandEntry>,
}

/// A single entry in the demand state.
///
/// Over time this can accumulate:
/// - rolling request rates,
/// - error/latency stats,
/// - learned capacity hints per backend/environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDemandEntry {
    /// Protocol-level model configuration.
    pub config: ModelConfig,
    /// Bootstrapped or learned demand weight.
    pub weight: f64,
    /// Current desired replica range.
    pub min_replicas: u32,
    pub max_replicas: u32,
}

/// Wrapper for the demand state cache store.
pub struct DemandStore {
    store: JsonStore,
}

impl DemandStore {
    /// Create a new demand store under the helexa cache root.
    ///
    /// The file on disk will be:
    ///   `${HOME}/.cache/helexa/cortex-model-demand.json`
    pub fn new() -> Result<Self> {
        let store = JsonStore::new("cortex-model-demand")?;
        Ok(Self { store })
    }

    /// Load the demand state from disk, or return an empty default.
    pub fn load_or_default(&self) -> Result<ModelDemandState> {
        self.store.load_or_default()
    }

    /// Persist the given demand state to disk.
    pub fn save(&self, state: &ModelDemandState) -> Result<()> {
        self.store.save(state)
    }
}

impl CortexSpec {
    /// Load a `CortexSpec` from a JSON file at the given path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let data = fs::read_to_string(path_ref).with_context(|| {
            format!("failed to read cortex spec file at {}", path_ref.display())
        })?;
        let spec: CortexSpec = serde_json::from_str(&data).with_context(|| {
            format!("failed to parse cortex spec JSON at {}", path_ref.display())
        })?;
        Ok(spec)
    }

    /// Seed a `ModelDemandState` from this spec.
    ///
    /// This is typically called at startup when a `--spec` is provided,
    /// and the resulting state can then be overlaid with runtime-derived
    /// demand metrics loaded from the `DemandStore`.
    pub fn to_initial_demand_state(&self) -> ModelDemandState {
        let mut entries = Vec::new();

        for model in &self.models {
            let cfg = model.config.clone();
            // Default weight and replica ranges if not provided.
            let weight = model.weight.unwrap_or(1.0);
            let min_replicas = model.min_replicas.unwrap_or(0);
            let max_replicas = model.max_replicas.unwrap_or(1);

            entries.push(ModelDemandEntry {
                config: cfg,
                weight,
                min_replicas,
                max_replicas,
            });
        }

        ModelDemandState { models: entries }
    }
}

/// Helper to load a spec (if present) and merge it with any cached demand
/// state from previous runs. The cache overlay semantics are:
///
/// - If a spec is provided:
///   - Start from `spec.to_initial_demand_state()`.
///   - Optionally merge in cached metrics (future work).
/// - If no spec is provided:
///   - Start from the cached demand state, or default if none exists.
pub fn load_combined_demand_state(
    spec_path: Option<PathBuf>,
    demand_store: &DemandStore,
) -> Result<ModelDemandState> {
    let cached = demand_store.load_or_default().unwrap_or_default();

    if let Some(path) = spec_path {
        let spec = CortexSpec::from_file(path)?;
        let mut initial = spec.to_initial_demand_state();

        // TODO: merge cached metrics into `initial` once we track them
        // per ModelDemandEntry (e.g. by matching on `config.id`).
        //
        // For now we simply prefer the spec definitions and ignore
        // the cached state if a spec is provided.
        if cached.models.is_empty() {
            Ok(initial)
        } else {
            // Placeholder: in the future, merge config from spec with learned
            // metrics from cache. For now, logically prefer spec but keep
            // the function signature ready for richer merging.
            initial.models.extend(cached.models);
            Ok(initial)
        }
    } else {
        Ok(cached)
    }
}
