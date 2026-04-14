//! Background poller that periodically queries each node's `/v1/models`
//! endpoint to refresh the fleet state.

use crate::state::CortexState;
use chrono::Utc;
use cortex_core::node::{MistralModelsResponse, ModelEntry, ModelStatus};
use std::sync::Arc;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Runs forever, polling all nodes on a fixed interval.
pub async fn poll_loop(fleet: Arc<CortexState>) {
    loop {
        for nc in &fleet.node_configs {
            poll_node(&fleet, &nc.name, &nc.endpoint).await;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn poll_node(fleet: &CortexState, name: &str, endpoint: &str) {
    let url = format!("{endpoint}/v1/models");

    let result = fleet
        .http_client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    let mut nodes = fleet.nodes.write().await;
    let Some(node) = nodes.get_mut(name) else {
        return;
    };

    match result {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<MistralModelsResponse>().await {
                Ok(models_resp) => {
                    // Merge upstream model list into our state, preserving
                    // our local metadata (last_accessed, vram_estimate).
                    let mut seen = std::collections::HashSet::new();
                    for upstream in &models_resp.data {
                        seen.insert(upstream.id.clone());
                        let status = parse_status(upstream.status.as_deref());

                        node.models
                            .entry(upstream.id.clone())
                            .and_modify(|e| {
                                e.status = status;
                            })
                            .or_insert_with(|| ModelEntry {
                                id: upstream.id.clone(),
                                status,
                                last_accessed: None,
                                vram_estimate_mb: None,
                            });
                    }

                    // Remove models that are no longer reported by the node
                    // (e.g. after a config change / restart).
                    node.models.retain(|id, _| seen.contains(id));

                    node.healthy = true;
                    node.last_poll = Some(Utc::now());
                    tracing::debug!(node = name, models = models_resp.data.len(), "poll ok");
                }
                Err(e) => {
                    tracing::warn!(node = name, error = %e, "failed to parse /v1/models response");
                    node.healthy = false;
                }
            }
        }
        Ok(resp) => {
            tracing::warn!(
                node = name,
                status = %resp.status(),
                "node returned non-success status"
            );
            node.healthy = false;
        }
        Err(e) => {
            tracing::warn!(node = name, error = %e, "failed to reach node");
            node.healthy = false;
        }
    }
}

fn parse_status(s: Option<&str>) -> ModelStatus {
    match s {
        Some("loaded") => ModelStatus::Loaded,
        Some("unloaded") => ModelStatus::Unloaded,
        Some("reloading") => ModelStatus::Reloading,
        // If the status field is absent, assume loaded (older mistral.rs versions
        // may not include it).
        _ => ModelStatus::Loaded,
    }
}
