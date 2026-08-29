//! Background poller that periodically queries each neuron's API
//! to refresh the fleet state.

use crate::state::CortexState;
use chrono::Utc;
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::harness::ModelInfo;
use cortex_core::node::{ModelEntry, ModelStatus, NodeState};
use metrics::{counter, gauge};
use std::sync::Arc;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Consecutive failed `/models` polls before a node is marked unhealthy.
/// Debounces transient misses (a busy neuron briefly slow to answer) so a
/// single blip can't yank a node — and its models — out of routing. At the
/// 10s poll interval this tolerates ~20s of flapping before evicting.
const POLL_FAILURE_THRESHOLD: u32 = 3;

/// How long before re-asking the fleet about a model whose capabilities
/// nobody could derive (#241). Long, because the usual cause — no node
/// has the weights cached — is resolved by a deliberate operator action
/// (a load, a prefetch) rather than by time passing.
const CAPABILITY_RETRY: Duration = Duration::from_secs(600);

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
                    // Node-level ladder, kept only as a fallback for a
                    // model whose own entry carries none. It is NOT the
                    // source of truth any more: availability depends on
                    // the model's `max_in_flight`, so two models on one
                    // host can legitimately offer different rungs and a
                    // single node-level copy would attribute one's
                    // withheld rung to the other. Per-model is set below.
                    node.reasoning_budget = models
                        .iter()
                        .map(|m| m.reasoning_budget.clone())
                        .find(|rungs| !rungs.is_empty())
                        .unwrap_or_default();
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
                                e.servable = upstream.servable.clone();
                                e.reasoning_budget = upstream.reasoning_budget.clone();
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
                                servable: upstream.servable.clone(),
                                reasoning_budget: upstream.reasoning_budget.clone(),
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

    discover_capabilities(fleet, name, endpoint).await;

    // Poll /health for the activation snapshot. We don't want this to
    // flip the node to unhealthy on its own — a neuron that's serving
    // /models fine is still operational even if /health is briefly
    // unavailable — so failures are debug-level and leave the existing
    // activation reading in place.
    poll_health(fleet, name, endpoint).await;
}

/// Ask a neuron what the catalogue models it has never loaded can do
/// (#241).
///
/// A loaded model reports its modalities on every `/models` poll, but a
/// cold one reported nothing at all — which is why image generation was
/// invisible on `/v1/models`, the image model being evicted almost all
/// of the time. The neuron derives the answer from its local model cache
/// without loading anything.
///
/// Only ever asks about ids it has no answer for, so a steady fleet
/// costs nothing after the first cycle: what a given model id can do is
/// a property of the model, not of the moment. A 404 means this node has
/// no local evidence — another node may still have the weights cached,
/// so nothing is recorded and the id is retried against the next node.
///
/// Ids that stay unresolved back off to [`CAPABILITY_RETRY`] rather than
/// being re-asked every cycle. A catalogue entry whose weights nobody has
/// downloaded can never be answered, and at the poll interval that would
/// be a permanent trickle of requests and 404s. The retry still exists
/// because the answer *can* change: the first node to pull those weights
/// starts being able to reply.
async fn discover_capabilities(fleet: &CortexState, name: &str, endpoint: &str) {
    let unknown: Vec<String> = {
        let known = fleet.discovered_capabilities.read().await;
        let attempts = fleet.capability_probe_attempts.read().await;
        fleet
            .catalogue
            .models
            .iter()
            .map(|p| p.id.clone())
            .filter(|id| !known.contains_key(id))
            .filter(|id| {
                attempts
                    .get(&(name.to_string(), id.clone()))
                    .is_none_or(|at| at.elapsed() >= CAPABILITY_RETRY)
            })
            .collect()
    };
    if unknown.is_empty() {
        return;
    }

    {
        let now = std::time::Instant::now();
        let mut attempts = fleet.capability_probe_attempts.write().await;
        for id in &unknown {
            attempts.insert((name.to_string(), id.clone()), now);
        }
    }

    for model_id in unknown {
        let url = format!(
            "{endpoint}/models/{}/capabilities",
            urlencoding::encode(&model_id)
        );
        let resp = fleet
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        let Ok(resp) = resp else { continue };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            continue;
        };
        let Some(caps) = body.get("capabilities").and_then(|c| c.as_array()) else {
            continue;
        };
        let caps: Vec<String> = caps
            .iter()
            .filter_map(|c| c.as_str().map(str::to_string))
            .collect();
        // An empty list is a real answer ("cached, serves nothing"), but
        // recording it would pin a model to no capabilities on the word
        // of one node. Leave it unknown so a node with real weights can
        // still speak up.
        if caps.is_empty() {
            continue;
        }
        tracing::debug!(node = name, model = %model_id, ?caps, "discovered capabilities");
        fleet
            .discovered_capabilities
            .write()
            .await
            .insert(model_id, caps);
    }
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
            // Export the live load + device health to Prometheus (#137).
            // These values are already in hand from the routing scrape, so
            // publishing them adds no polling. Emitted as gauges (last-write
            // wins, refreshed every ~10s poll) outside the state lock.
            export_health_metrics(name, &h);

            let mut nodes = fleet.nodes.write().await;
            if let Some(node) = nodes.get_mut(name) {
                node.activation = Some(h.activation);
                // Per-model admission load (#53) → keyed by id for the
                // load-aware router (#55).
                node.model_load = h.models.into_iter().map(|m| (m.id.clone(), m)).collect();
                // Per-device VRAM readings (#203) for free-fit
                // cold-load placement.
                node.device_health = h.devices.clone();
            }
        }
        Err(e) => {
            tracing::debug!(node = name, error = %e, "failed to parse /health response");
        }
    }
}

/// Publish a neuron's `/health` snapshot to Prometheus (#137): live
/// per-model admission load + configured ceiling, and per-device GPU
/// headroom. Gauges are `{node,model}` / `{node,device}` labelled to match
/// the existing `cortex_*` set. Called on every successful poll so values
/// track the ~10s cadence; a model that unloads simply stops being
/// refreshed (its last gauge value goes stale — acceptable for the bounded
/// fleet cardinality here).
fn export_health_metrics(node: &str, h: &HealthResponse) {
    for m in &h.models {
        gauge!("cortex_model_in_flight", "node" => node.to_string(), "model" => m.id.clone())
            .set(m.in_flight as f64);
        gauge!("cortex_model_queue_depth", "node" => node.to_string(), "model" => m.id.clone())
            .set(m.queue_depth as f64);
        // Ceiling is the saturation denominator. 0 = pre-#137 neuron that
        // doesn't advertise it yet — skip rather than publish a bogus 0.
        if m.max_in_flight > 0 {
            gauge!("cortex_model_max_in_flight", "node" => node.to_string(), "model" => m.id.clone())
                .set(m.max_in_flight as f64);
            gauge!("cortex_model_max_queue_depth", "node" => node.to_string(), "model" => m.id.clone())
                .set(m.max_queue_depth as f64);
        }
        // Live throughput EMAs (#137) — decode tok/s is the headline
        // capacity number. Emitted unconditionally (0.0 = no sample yet).
        gauge!("cortex_model_tok_s_prefill", "node" => node.to_string(), "model" => m.id.clone())
            .set(m.tok_s_prefill);
        gauge!("cortex_model_tok_s_decode", "node" => node.to_string(), "model" => m.id.clone())
            .set(m.tok_s_decode);
        // Cumulative rejections by reason (#137) — the shedding signal.
        // Neuron reports counts-since-load; `.absolute` mirrors them onto a
        // counter (a model reload resets to 0, which Prometheus reads as a
        // normal counter reset).
        counter!("cortex_model_rejections_total",
            "node" => node.to_string(), "model" => m.id.clone(), "reason" => "queue_full")
        .absolute(m.rejected_queue_full);
        counter!("cortex_model_rejections_total",
            "node" => node.to_string(), "model" => m.id.clone(), "reason" => "wait_timeout")
        .absolute(m.rejected_timeout);
        counter!("cortex_model_rejections_total",
            "node" => node.to_string(), "model" => m.id.clone(), "reason" => "per_principal")
        .absolute(m.rejected_per_principal);
        counter!("cortex_model_rejections_total",
            "node" => node.to_string(), "model" => m.id.clone(), "reason" => "anon_yield")
        .absolute(m.rejected_anon_yield);
        // How much of this model's load is unattributable (#262), and the
        // ceiling that keeps it from starving identified callers. 0 for a
        // pre-#262 neuron, which had no ceiling — skip rather than publish
        // a 0 that reads as "anonymous is refused" when it means the
        // opposite.
        gauge!("cortex_model_anon_in_flight", "node" => node.to_string(), "model" => m.id.clone())
            .set(m.anon_in_flight as f64);
        if m.anon_max_in_flight > 0 {
            gauge!("cortex_model_anon_max_in_flight",
                "node" => node.to_string(), "model" => m.id.clone())
            .set(m.anon_max_in_flight as f64);
        }
    }
    for d in &h.devices {
        let device = d.index.to_string();
        gauge!("cortex_device_vram_used_mb", "node" => node.to_string(), "device" => device.clone())
            .set(d.vram_used_mb as f64);
        gauge!("cortex_device_vram_free_mb", "node" => node.to_string(), "device" => device.clone())
            .set(d.vram_free_mb as f64);
        gauge!("cortex_device_utilization_pct", "node" => node.to_string(), "device" => device.clone())
            .set(d.utilization_pct as f64);
        gauge!("cortex_device_temp_c", "node" => node.to_string(), "device" => device.clone())
            .set(d.temp_c as f64);
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
