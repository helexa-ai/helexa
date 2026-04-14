//! Model eviction logic.
//!
//! The evictor runs as a background task. When the router determines that a
//! model needs to be loaded on a node but VRAM is tight, it can request
//! eviction via a channel. The evictor then:
//!   1. Identifies the LRU model on that node (excluding pinned models)
//!   2. Calls `POST /v1/models/unload` on the node
//!   3. Increments the lifecycle cycle counter (for defrag tracking)

use crate::state::CortexState;
use cortex_core::node::{ModelLifecycleRequest, ModelStatus};
use std::sync::Arc;
use std::time::Duration;

/// Runs forever. Currently a placeholder that periodically checks for
/// eviction opportunities. In the future, this will be driven by a
/// channel from the router when VRAM pressure is detected.
pub async fn eviction_loop(fleet: Arc<CortexState>) {
    // TODO: Replace this polling approach with a channel-driven design
    // where the router sends eviction requests when it detects that a
    // model load would exceed available VRAM.
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        // Placeholder: the actual eviction logic is in `evict_lru_on_node`,
        // called on demand by the router.
        let _ = &fleet; // suppress unused warning
    }
}

/// Evict the least-recently-used model on a given node.
/// Returns the model ID that was evicted, or None if nothing could be evicted.
pub async fn evict_lru_on_node(
    fleet: &CortexState,
    node_name: &str,
) -> anyhow::Result<Option<String>> {
    let (endpoint, candidate) = {
        let nodes = fleet.nodes.read().await;
        let Some(node) = nodes.get(node_name) else {
            anyhow::bail!("node '{node_name}' not found");
        };

        // Find the loaded model with the oldest last_accessed, excluding pinned.
        let candidate = node
            .models
            .values()
            .filter(|m| m.status == ModelStatus::Loaded)
            .filter(|m| !node.pinned.contains(&m.id))
            .min_by_key(|m| m.last_accessed)
            .map(|m| m.id.clone());

        (node.endpoint.clone(), candidate)
    };

    let Some(model_id) = candidate else {
        tracing::info!(node = node_name, "no evictable models found");
        return Ok(None);
    };

    tracing::info!(node = node_name, model = %model_id, "evicting model");

    let url = format!("{endpoint}/v1/models/unload");
    let resp = fleet
        .http_client
        .post(&url)
        .json(&ModelLifecycleRequest {
            model_id: model_id.clone(),
        })
        .send()
        .await?;

    if resp.status().is_success() {
        // Update local state.
        let mut nodes = fleet.nodes.write().await;
        if let Some(node) = nodes.get_mut(node_name) {
            if let Some(entry) = node.models.get_mut(&model_id) {
                entry.status = ModelStatus::Unloaded;
            }
            node.lifecycle_cycles += 1;

            // Check if we should flag for defrag.
            if fleet.eviction.defrag_after_cycles > 0
                && node.lifecycle_cycles >= fleet.eviction.defrag_after_cycles
            {
                tracing::warn!(
                    node = node_name,
                    cycles = node.lifecycle_cycles,
                    "VRAM fragmentation threshold reached — consider restarting mistralrs"
                );
            }
        }

        tracing::info!(node = node_name, model = %model_id, "model evicted");
        Ok(Some(model_id))
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::error!(
            node = node_name,
            model = %model_id,
            status = %status,
            body = %body,
            "failed to evict model"
        );
        anyhow::bail!("eviction failed: {status} {body}");
    }
}
