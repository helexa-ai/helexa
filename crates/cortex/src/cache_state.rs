/* helexa/crates/cortex/src/cache_state.rs */

// SPDX-License-Identifier: PolyForm-Shield-1.0

//! Persistence helpers for cortex control-plane and observe state.
//!
//! This module provides a thin JSON-backed cache for:
//! - online neuron registry entries (descriptor + last heartbeat time),
//! - per-neuron model provisioning state.
//!
//! The goal is to give dashboards and higher-level components a way to
//! reconstruct "recently online" state across cortex restarts without
//! persisting offline or obviously stale entries. Offline neurons are
//! intentionally *not* written to cache and are forgotten between runs.
//!
//! Persistence is best-effort:
//! - On startup, callers should attempt to `load_cortex_state_from_cache`
//!   and hydrate in-memory registries from the result.
//! - On shutdown, callers should attempt to `save_cortex_state_to_cache`
//!   with a snapshot of the current registry and model store.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use cache::JsonStore;

use crate::control_plane::{ModelProvisioningStatus, NeuronDescriptor, NeuronRegistry, NeuronView};
use crate::ModelProvisioningStore;

/// Serializable snapshot of a single neuron suitable for on-disk caching.
///
/// This is intentionally narrower than the in-memory `ConnectedNeuron`
/// representation and focuses on data that is stable across restarts.
#[derive(Debug, Serialize, Deserialize)]
pub struct CachedNeuron {
    /// Descriptor as reported by the neuron during registration.
    pub descriptor: NeuronDescriptor,
    /// Best-effort wall-clock timestamp for the last observed heartbeat.
    ///
    /// This is derived from the in-memory `Instant` timer at the time we
    /// construct the cache snapshot. It is *not* used for correctness; it
    /// is only a hint for dashboards and future logic that might care about
    /// "last seen" information across restarts.
    pub last_heartbeat_at: Option<SystemTime>,
}

/// Serializable snapshot of cortex state for cache persistence.
///
/// This is a coarse, best-effort view used to seed in-memory state on
/// startup. It is *not* the source of truth; live control-plane traffic
/// always takes precedence and is expected to converge to the real state.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CachedCortexState {
    /// Online neurons as of the last successful cache save.
    ///
    /// Only neurons that were considered online (recent heartbeat and a
    /// stable `node_id`) at the time of saving are included here. Offline
    /// neurons are intentionally omitted and forgotten between runs.
    pub neurons: Vec<CachedNeuron>,
    /// Per-neuron model provisioning state as seen by cortex.
    ///
    /// The key is the neuron_id string (as reported by the neuron), and
    /// the value is the list of model provisioning statuses for that neuron.
    pub models_by_neuron: HashMap<String, Vec<ModelProvisioningStatus>>,
}

impl CachedCortexState {
    /// Name of the cache store on disk.
    ///
    /// The underlying path is:
    ///
    ///   `${HOME}/.cache/helexa/cortex-state.json`
    ///
    /// when using the default `JsonStore::new` root resolution.
    fn store_name() -> &'static str {
        "cortex-state"
    }
}

/// Persist a best-effort snapshot of cortex state (online neurons + model
/// provisioning state) to the JSON cache store.
///
/// This function is intended to be called during graceful shutdown of a
/// cortex node. Failures are non-fatal and should generally be logged at
/// WARN level rather than aborting shutdown.
///
/// Semantics:
///
/// - Only neurons that:
///   - have a stable `node_id`, and
///   - have a reasonably recent heartbeat
///   are persisted.
/// - Neurons that have never heartbeated, or whose last heartbeat is older
///   than `persist_threshold`, are treated as offline and omitted.
/// - Model state is persisted only for neurons that are included in the
///   cached neuron list.
pub async fn save_cortex_state_to_cache(
    registry: &NeuronRegistry,
    model_store: &ModelProvisioningStore,
) -> Result<()> {
    let store = JsonStore::new(CachedCortexState::store_name())?;

    // 1. Get health-enriched views of all neurons.
    let views: Vec<NeuronView> = registry.list_with_health().await;

    // 2. Build CachedNeuron list, but only for "online" neurons by a simple
    //    heartbeat recency heuristic.
    //
    //    This is intentionally conservative; if we are unsure, we err on
    //    the side of *not* persisting the entry so that stale/offline
    //    neurons do not get resurrected across restarts.
    let persist_threshold: Duration = Duration::from_secs(5 * 60);
    let now = SystemTime::now();

    let mut neurons: Vec<CachedNeuron> = Vec::new();
    let mut models_by_neuron: HashMap<String, Vec<ModelProvisioningStatus>> = HashMap::new();

    for view in views {
        let age = match view.last_heartbeat_age {
            Some(a) => a,
            None => {
                // No heartbeat yet; treat as offline/ephemeral and skip.
                continue;
            }
        };

        if age > persist_threshold {
            // Treat as offline; do not persist.
            continue;
        }

        let last_heartbeat_at = now.checked_sub(age);
        let descriptor = view.descriptor.clone();

        let neuron_id = match &descriptor.node_id {
            Some(id) => id.clone(),
            None => {
                // Without a stable id we can't safely key any restored state,
                // so skip this entry.
                continue;
            }
        };

        neurons.push(CachedNeuron {
            descriptor,
            last_heartbeat_at,
        });

        // Pull the current model provisioning state for this neuron. This
        // is optional; neurons with no recorded models simply won't have
        // an entry in `models_by_neuron`.
        let models = model_store.list_for_neuron(&neuron_id).await;
        if !models.is_empty() {
            models_by_neuron.insert(neuron_id, models);
        }
    }

    let state = CachedCortexState {
        neurons,
        models_by_neuron,
    };

    store.save(&state)?;
    Ok(())
}

/// Load a previously persisted snapshot of cortex state (if any) from the
/// JSON cache store and hydrate the in-memory registries from it.
///
/// This function is intended to be called during cortex startup *after*
/// constructing the shared `NeuronRegistry` and `ModelProvisioningStore`,
/// but before starting the control-plane server.
///
/// Semantics:
///
/// - If no cache file exists, this is a no-op.
/// - If the cache file cannot be parsed, the error is returned so that
///   callers can decide whether to proceed or log and continue.
/// - Only neurons and models that were persisted (i.e. considered online
///   at save time) are restored.
pub async fn load_cortex_state_from_cache(
    registry: &NeuronRegistry,
    model_store: &ModelProvisioningStore,
) -> Result<()> {
    let store = JsonStore::new(CachedCortexState::store_name())?;
    let state: CachedCortexState = store.load_or_default()?;

    // Rebuild registry from cached neurons.
    //
    // We intentionally treat restored neurons as if they had just
    // re-registered: their `last_heartbeat` in memory is set to "now"
    // via `upsert_neuron`. The persisted `last_heartbeat_at` is kept
    // only for potential UI/diagnostics use via `CachedNeuron`.
    for cached in state.neurons {
        let desc: NeuronDescriptor = cached.descriptor;
        registry.upsert_neuron(desc).await;
    }

    // Rebuild model provisioning state. The store is keyed by neuron_id,
    // which matches the `node_id` on NeuronDescriptor.
    for (neuron_id, models) in state.models_by_neuron {
        // Restore the full set of provisioning statuses for this neuron_id
        // into the in-memory ModelProvisioningStore. This replaces any
        // existing entries for the neuron and ensures that dashboards and
        // higher-level components see a consistent view immediately after
        // startup, even before new provisioning events occur.
        model_store
            .restore_statuses_for_neuron(&neuron_id, models)
            .await;
    }

    Ok(())
}
