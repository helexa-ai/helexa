//! Background poller that refreshes the multi-operator topology (#72).
//!
//! The same pattern as cortex↔neuron, one tier up: periodically poll each
//! configured cortex's `GET /v1/models` (catalogue × topology feasibility +
//! loaded state) and `GET /health` (coarse node-health/load), building the
//! live map the dispatcher (#73) routes on. An unreachable or erroring
//! cortex is debounced over [`POLL_FAILURE_THRESHOLD`] consecutive misses,
//! then flipped unhealthy and excluded from routing; it recovers on the
//! next successful poll.

use crate::state::RouterState;
use chrono::Utc;
use cortex_core::node::CortexModelEntry;
use serde::Deserialize;
use std::time::Duration;

/// Per-cortex HTTP timeout for each poll request.
const POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// Consecutive failed polls before a cortex is marked unreachable. Mirrors
/// cortex's neuron-poll debounce: a single blip (a busy cortex briefly slow
/// to answer) can't yank it — and all its models — out of routing.
pub const POLL_FAILURE_THRESHOLD: u32 = 3;

/// cortex's `/v1/models` envelope — `{ "object": "list", "data": [...] }`.
#[derive(Debug, Deserialize)]
struct ModelsEnvelope {
    #[serde(default)]
    data: Vec<CortexModelEntry>,
}

/// The subset of cortex's `/health` the router reads.
#[derive(Debug, Deserialize)]
struct CortexHealth {
    nodes: CortexHealthNodes,
}

#[derive(Debug, Deserialize)]
struct CortexHealthNodes {
    healthy: u32,
    total: u32,
}

/// Run forever, polling all cortexes on the configured interval.
pub async fn poll_loop(state: std::sync::Arc<RouterState>) {
    loop {
        poll_once(&state).await;
        tokio::time::sleep(state.poll_interval).await;
    }
}

/// Poll every configured cortex once. Public for testing.
pub async fn poll_once(state: &RouterState) {
    for cortex in &state.cortexes {
        poll_cortex(state, &cortex.name, &cortex.endpoint).await;
    }
}

/// Poll one cortex: refresh its model map from `/v1/models`, then its node
/// health from `/health`. A `/v1/models` failure debounces toward
/// unreachable; the `/health` poll is best-effort and never flips
/// reachability on its own (a cortex serving `/v1/models` is routable even
/// if `/health` momentarily isn't).
async fn poll_cortex(state: &RouterState, name: &str, endpoint: &str) {
    let models = fetch_models(state, endpoint).await;

    let mut topo = state.topology.write().await;
    let Some(entry) = topo.get_mut(name) else {
        return; // not a configured cortex (shouldn't happen)
    };

    match models {
        Ok(models) => {
            entry.models = models.into_iter().map(|m| (m.id.clone(), m)).collect();
            entry.reachable = true;
            entry.consecutive_failures = 0;
            entry.last_poll = Some(Utc::now());
            tracing::debug!(cortex = name, models = entry.models.len(), "poll ok");
        }
        Err(reason) => {
            entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
            if entry.consecutive_failures >= POLL_FAILURE_THRESHOLD {
                entry.reachable = false;
            }
            tracing::warn!(
                cortex = name,
                failures = entry.consecutive_failures,
                reachable = entry.reachable,
                reason,
                "cortex poll failed"
            );
        }
    }
    drop(topo);

    // Best-effort health (node counts). Never flips reachability.
    if let Some((healthy, total)) = fetch_health(state, endpoint).await {
        let mut topo = state.topology.write().await;
        if let Some(entry) = topo.get_mut(name) {
            entry.healthy_nodes = healthy;
            entry.total_nodes = total;
        }
    }
}

/// GET `/v1/models`, returning the parsed entries or a short failure reason.
async fn fetch_models(
    state: &RouterState,
    endpoint: &str,
) -> Result<Vec<CortexModelEntry>, &'static str> {
    let url = format!("{endpoint}/v1/models");
    let resp = state
        .http_client
        .get(&url)
        .timeout(POLL_TIMEOUT)
        .send()
        .await
        .map_err(|_| "unreachable")?;
    if !resp.status().is_success() {
        return Err("non-success status");
    }
    let envelope = resp
        .json::<ModelsEnvelope>()
        .await
        .map_err(|_| "bad json")?;
    Ok(envelope.data)
}

/// GET `/health`, returning `(healthy, total)` node counts. `None` on any
/// failure — the caller leaves the previous counts in place.
async fn fetch_health(state: &RouterState, endpoint: &str) -> Option<(u32, u32)> {
    let url = format!("{endpoint}/health");
    let resp = state
        .http_client
        .get(&url)
        .timeout(POLL_TIMEOUT)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let health = resp.json::<CortexHealth>().await.ok()?;
    Some((health.nodes.healthy, health.nodes.total))
}
