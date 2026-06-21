use crate::config::{CortexEndpoint, RouterConfig};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::RwLock;

/// Shared router state: the configured cortex list plus the live topology
/// map the poller (#72) maintains and the dispatcher (#73) will route on.
///
/// This is the router tier of the fractal neuron ← cortex ← router design:
/// just as cortex polls each neuron for capacity/catalogue, the router
/// polls each cortex's `/health` + `/v1/models`.
#[derive(Debug)]
pub struct RouterState {
    /// Downstream cortex endpoints, as configured.
    pub cortexes: Vec<CortexEndpoint>,
    /// Shared client for polling (and, later, proxying to) cortexes.
    pub http_client: reqwest::Client,
    /// How often the poller refreshes the topology.
    pub poll_interval: Duration,
    /// Live per-cortex topology, keyed by cortex name. Pre-populated from
    /// config (every configured cortex present, `reachable = false`) so the
    /// poller and handlers always find an entry; the poller flips
    /// reachability and fills the model map.
    pub topology: RwLock<HashMap<String, CortexTopology>>,
}

/// Live view of one downstream cortex, refreshed each poll.
#[derive(Debug, Clone, Default)]
pub struct CortexTopology {
    /// Whether the cortex is currently routable. Flipped `false` only after
    /// [`crate::poller::POLL_FAILURE_THRESHOLD`] consecutive failed polls
    /// (debounces transient blips); restored on the next successful poll.
    pub reachable: bool,
    /// Consecutive failed polls; reset to 0 on success.
    pub consecutive_failures: u32,
    /// Timestamp of the last successful poll.
    pub last_poll: Option<DateTime<Utc>>,
    /// Healthy / total neuron counts from the cortex's `/health` (coarse
    /// load signal; #73 refines headroom). 0/0 until first health poll.
    pub healthy_nodes: u32,
    pub total_nodes: u32,
    /// Per-model serveability, keyed by model id, from `/v1/models`.
    pub models: HashMap<String, RouterModelStatus>,
}

/// What a cortex can do with one model, distilled from its `/v1/models`
/// entry to the two facts the router routes on.
#[derive(Debug, Clone)]
pub struct RouterModelStatus {
    /// The model is loaded on at least one of the cortex's neurons.
    pub loaded: bool,
    /// The cortex can serve it — loaded now, or feasible to cold-load
    /// (catalogue × topology says some neuron can host it).
    pub feasible: bool,
}

impl RouterState {
    pub fn from_config(config: &RouterConfig) -> Self {
        let topology = config
            .cortexes
            .iter()
            .map(|c| (c.name.clone(), CortexTopology::default()))
            .collect();

        Self {
            cortexes: config.cortexes.clone(),
            http_client: reqwest::Client::new(),
            poll_interval: Duration::from_secs(config.router.poll_interval_secs),
            topology: RwLock::new(topology),
        }
    }

    /// Names of reachable cortexes that can serve `model_id` (loaded or
    /// feasible to cold-load). Groundwork for capacity-aware dispatch (#73);
    /// unreachable cortexes are excluded by construction.
    pub async fn cortexes_serving(&self, model_id: &str) -> Vec<String> {
        let topo = self.topology.read().await;
        topo.iter()
            .filter(|(_, t)| t.reachable)
            .filter(|(_, t)| t.models.get(model_id).is_some_and(|m| m.feasible))
            .map(|(name, _)| name.clone())
            .collect()
    }
}
