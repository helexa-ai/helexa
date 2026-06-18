//! Background poller that periodically queries each neuron's API
//! to refresh the fleet state.

use crate::state::CortexState;
use chrono::Utc;
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::harness::ModelInfo;
use cortex_core::node::{ModelEntry, ModelStatus, NodeState};
use std::sync::Arc;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Consecutive failed `/models` polls before a node is marked unhealthy.
/// Debounces transient misses (a busy neuron briefly slow to answer) so a
/// single blip can't yank a node — and its models — out of routing. At the
/// 10s poll interval this tolerates ~20s of flapping before evicting.
const POLL_FAILURE_THRESHOLD: u32 = 3;

/// Record a failed poll for `node`, marking it unhealthy only once failures
/// reach [`POLL_FAILURE_THRESHOLD`]. Below the threshold the node keeps its
/// last-known health, riding over transient misses. A successful poll resets
/// the counter (see the success arm in `poll_once`).
fn record_poll_failure(node: &mut NodeState) {
    node.consecutive_poll_failures = node.consecutive_poll_failures.saturating_add(1);
    if node.consecutive_poll_failures >= POLL_FAILURE_THRESHOLD {
        node.healthy = false;
    }
}

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

/// Fetch `GET /discovery` and cache it on the NodeState — topology is
/// invariant for a given neuron process, so a successful fetch is kept.
/// Re-polled only while `max_prompt_tokens` is still unknown (0): on a
/// rolling deploy cortex can win the race and cache a neuron's discovery
/// before that neuron reports the field (it deserialises to 0). Re-polling
/// until a real cap arrives self-heals that without periodic polling.
async fn maybe_poll_discovery(fleet: &CortexState, name: &str, endpoint: &str) {
    {
        let nodes = fleet.nodes.read().await;
        match nodes.get(name) {
            Some(n)
                if n.discovery
                    .as_ref()
                    .is_some_and(|d| d.max_prompt_tokens > 0) =>
            {
                return;
            }
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
                                e.capabilities = upstream.capabilities.clone();
                                e.tool_call = upstream.tool_call;
                                e.reasoning = upstream.reasoning;
                                // Neuron's self-derived limit (#67) — the
                                // authoritative source the gateway advertises.
                                e.limit = upstream.limit.clone();
                            })
                            .or_insert_with(|| ModelEntry {
                                id: upstream.id.clone(),
                                status,
                                last_accessed: None,
                                vram_estimate_mb: upstream.vram_used_mb,
                                capabilities: upstream.capabilities.clone(),
                                tool_call: upstream.tool_call,
                                reasoning: upstream.reasoning,
                                limit: upstream.limit.clone(),
                            });
                    }

                    // Remove models no longer reported by the neuron.
                    node.models.retain(|id, _| seen.contains(id));

                    node.consecutive_poll_failures = 0;
                    node.healthy = true;
                    node.last_poll = Some(Utc::now());
                    tracing::debug!(node = name, models = models.len(), "poll ok");
                }
                Err(e) => {
                    tracing::warn!(node = name, error = %e, "failed to parse /models response");
                    record_poll_failure(node);
                }
            }
        }
        Ok(resp) => {
            tracing::warn!(
                node = name,
                status = %resp.status(),
                "neuron returned non-success status"
            );
            record_poll_failure(node);
        }
        Err(e) => {
            tracing::warn!(node = name, error = %e, "failed to reach neuron");
            record_poll_failure(node);
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
                // Per-model admission load (#53) → keyed by id for the
                // load-aware router (#55).
                node.model_load = h.models.into_iter().map(|m| (m.id.clone(), m)).collect();
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
        "recovering" => ModelStatus::Recovering,
        _ => ModelStatus::Loaded,
    }
}
