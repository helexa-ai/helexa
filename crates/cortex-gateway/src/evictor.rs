//! Model eviction logic.
//!
//! The evictor identifies the LRU model on a node (excluding pinned models),
//! calls neuron's `POST /models/unload` to free the model, and updates
//! local state.

use crate::state::CortexState;
use cortex_core::node::ModelStatus;
use std::sync::Arc;
use std::time::Duration;

/// Runs forever. Placeholder for future channel-driven eviction.
pub async fn eviction_loop(fleet: Arc<CortexState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let _ = &fleet;
    }
}

/// Evict the least-recently-used model on a given node.
/// Returns the model ID that was evicted, or None if nothing could be evicted.
pub async fn evict_lru_on_node(
    fleet: &CortexState,
    node_name: &str,
) -> anyhow::Result<Option<String>> {
    let (neuron_endpoint, candidate) = {
        let nodes = fleet.nodes.read().await;
        let Some(node) = nodes.get(node_name) else {
            anyhow::bail!("node '{node_name}' not found");
        };

        // Find the loaded model with the oldest last_accessed,
        // excluding models pinned on this neuron (from catalogue).
        let candidate = node
            .models
            .values()
            .filter(|m| m.status == ModelStatus::Loaded)
            .filter(|m| !fleet.catalogue.is_pinned(&m.id, node_name))
            .min_by_key(|m| m.last_accessed)
            .map(|m| m.id.clone());

        (node.endpoint.clone(), candidate)
    };

    let Some(model_id) = candidate else {
        tracing::info!(node = node_name, "no evictable models found");
        return Ok(None);
    };

    tracing::info!(node = node_name, model = %model_id, "evicting model");

    // Call neuron's unload endpoint.
    let url = format!("{neuron_endpoint}/models/unload");
    let resp = fleet
        .http_client
        .post(&url)
        .json(&serde_json::json!({ "model_id": model_id }))
        .send()
        .await?;

    if resp.status().is_success() {
        let mut nodes = fleet.nodes.write().await;
        if let Some(node) = nodes.get_mut(node_name) {
            if let Some(entry) = node.models.get_mut(&model_id) {
                entry.status = ModelStatus::Unloaded;
            }
            node.lifecycle_cycles += 1;

            if fleet.eviction.defrag_after_cycles > 0
                && node.lifecycle_cycles >= fleet.eviction.defrag_after_cycles
            {
                tracing::warn!(
                    node = node_name,
                    cycles = node.lifecycle_cycles,
                    "VRAM fragmentation threshold reached — consider restarting harness"
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
