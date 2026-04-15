//! Background poller that periodically queries each neuron's API
//! to refresh the fleet state.

use crate::state::CortexState;
use chrono::Utc;
use cortex_core::harness::ModelInfo;
use cortex_core::node::{ModelEntry, ModelStatus};
use std::sync::Arc;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Runs forever, polling all neurons on a fixed interval.
pub async fn poll_loop(fleet: Arc<CortexState>) {
    loop {
        poll_once(&fleet).await;
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Poll all neurons once. Used by `poll_loop` and available for testing.
pub async fn poll_once(fleet: &CortexState) {
    for nc in &fleet.neuron_configs {
        poll_neuron(fleet, &nc.name, &nc.endpoint).await;
    }
}

async fn poll_neuron(fleet: &CortexState, name: &str, endpoint: &str) {
    let url = format!("{endpoint}/models");

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
            match resp.json::<Vec<ModelInfo>>().await {
                Ok(models) => {
                    let mut seen = std::collections::HashSet::new();
                    for upstream in &models {
                        seen.insert(upstream.id.clone());
                        let status = parse_status(&upstream.status);

                        node.models
                            .entry(upstream.id.clone())
                            .and_modify(|e| {
                                e.status = status;
                                e.vram_estimate_mb = upstream.vram_used_mb;
                            })
                            .or_insert_with(|| ModelEntry {
                                id: upstream.id.clone(),
                                status,
                                last_accessed: None,
                                vram_estimate_mb: upstream.vram_used_mb,
                            });
                    }

                    // Remove models no longer reported by the neuron.
                    node.models.retain(|id, _| seen.contains(id));

                    node.healthy = true;
                    node.last_poll = Some(Utc::now());
                    tracing::debug!(node = name, models = models.len(), "poll ok");
                }
                Err(e) => {
                    tracing::warn!(node = name, error = %e, "failed to parse /models response");
                    node.healthy = false;
                }
            }
        }
        Ok(resp) => {
            tracing::warn!(
                node = name,
                status = %resp.status(),
                "neuron returned non-success status"
            );
            node.healthy = false;
        }
        Err(e) => {
            tracing::warn!(node = name, error = %e, "failed to reach neuron");
            node.healthy = false;
        }
    }
}

fn parse_status(s: &str) -> ModelStatus {
    match s {
        "loaded" => ModelStatus::Loaded,
        "unloaded" => ModelStatus::Unloaded,
        "reloading" => ModelStatus::Reloading,
        _ => ModelStatus::Loaded,
    }
}
