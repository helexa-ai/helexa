//! Background poller that periodically queries each neuron's API
//! to refresh the fleet state.

use crate::state::CortexState;
use chrono::Utc;
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
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

/// One-shot fetch of `GET /discovery`. Cached on the NodeState forever
/// after the first success — topology is invariant for a given neuron
/// process. Skipped when the cache is already populated.
async fn maybe_poll_discovery(fleet: &CortexState, name: &str, endpoint: &str) {
    {
        let nodes = fleet.nodes.read().await;
        match nodes.get(name) {
            Some(n) if n.discovery.is_some() => return,
            _ => {}
        }
    }
    let url = format!("{endpoint}/discovery");
    let resp = match fleet
        .http_client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::debug!(node = name, status = %r.status(), "discovery probe non-success");
            return;
        }
        Err(e) => {
            tracing::debug!(node = name, error = %e, "discovery probe unreachable");
            return;
        }
    };
    match resp.json::<DiscoveryResponse>().await {
        Ok(d) => {
            let mut nodes = fleet.nodes.write().await;
            if let Some(node) = nodes.get_mut(name) {
                tracing::info!(
                    node = name,
                    hostname = %d.hostname,
                    devices = d.devices.len(),
                    "discovery cached"
                );
                node.discovery = Some(d);
            }
        }
        Err(e) => {
            tracing::warn!(node = name, error = %e, "failed to parse /discovery response");
        }
    }
}

async fn poll_neuron(fleet: &CortexState, name: &str, endpoint: &str) {
    // Topology first — cheap once cached, and the router needs it to
    // route requests against catalogue entries that aren't loaded yet.
    maybe_poll_discovery(fleet, name, endpoint).await;

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

    // Release the write lock before the next HTTP call.
    drop(nodes);

    // Poll /health for the activation snapshot. We don't want this to
    // flip the node to unhealthy on its own — a neuron that's serving
    // /models fine is still operational even if /health is briefly
    // unavailable — so failures are debug-level and leave the existing
    // activation reading in place.
    poll_health(fleet, name, endpoint).await;
}

/// Fetch `/health` and stash the activation snapshot on NodeState.
/// Decoupled from the /models poll so a /health glitch doesn't mark
/// the neuron unhealthy or evict the model list.
async fn poll_health(fleet: &CortexState, name: &str, endpoint: &str) {
    let url = format!("{endpoint}/health");
    let resp = match fleet
        .http_client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::debug!(node = name, status = %r.status(), "/health probe non-success");
            return;
        }
        Err(e) => {
            tracing::debug!(node = name, error = %e, "/health probe failed");
            return;
        }
    };
    match resp.json::<HealthResponse>().await {
        Ok(h) => {
            let mut nodes = fleet.nodes.write().await;
            if let Some(node) = nodes.get_mut(name) {
                node.activation = Some(h.activation);
            }
        }
        Err(e) => {
            tracing::debug!(node = name, error = %e, "failed to parse /health response");
        }
    }
}

fn parse_status(s: &str) -> ModelStatus {
    match s {
        "loaded" => ModelStatus::Loaded,
        "unloaded" => ModelStatus::Unloaded,
        "reloading" => ModelStatus::Reloading,
        "loading" => ModelStatus::Loading,
        _ => ModelStatus::Loaded,
    }
}
