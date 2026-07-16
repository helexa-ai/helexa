use crate::config::{CortexEndpoint, RouterConfig};
use chrono::{DateTime, Utc};
use cortex_core::node::CortexModelEntry;
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
    /// Per-cortex HTTP client, keyed by cortex name (#74). A cortex enrolled
    /// with a `tls_ca` gets a client that trusts only that anchor; others
    /// get a default client. A cortex whose `tls_ca` failed to load is
    /// **absent** here — `client_for` returns `None` and it is never
    /// polled or routed to (fail closed: a misconfigured pin must not
    /// silently fall back to unpinned TLS).
    clients: HashMap<String, reqwest::Client>,
    /// This router instance's region, for dispatch geo affinity (#73).
    pub region: Option<String>,
    /// Product-tier aliases (#166): alias → real model id, applied before
    /// selection in dispatch.
    pub aliases: HashMap<String, String>,
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
    /// The cortex's full `/v1/models` entries, keyed by model id. Stored
    /// whole (not distilled to a loaded/feasible bool) so the federation
    /// catalogue (#75) can preserve per-model `limit`/`cost`/capabilities.
    pub models: HashMap<String, CortexModelEntry>,
}

/// Whether a cortex can serve this model — loaded now, or feasible to
/// cold-load (its catalogue × topology says some neuron can host it).
pub fn entry_feasible(entry: &CortexModelEntry) -> bool {
    entry.loaded || !entry.feasible_on.is_empty()
}

impl RouterState {
    pub fn from_config(config: &RouterConfig) -> Self {
        let topology = config
            .cortexes
            .iter()
            .map(|c| (c.name.clone(), CortexTopology::default()))
            .collect();

        // One client per cortex. A `tls_ca` that fails to load omits the
        // cortex from the map (fail closed) rather than degrading to an
        // unpinned client.
        let mut clients = HashMap::new();
        for c in &config.cortexes {
            match build_client(c.tls_ca.as_deref()) {
                Ok(client) => {
                    clients.insert(c.name.clone(), client);
                }
                Err(e) => {
                    tracing::error!(
                        cortex = %c.name,
                        tls_ca = c.tls_ca.as_deref().unwrap_or(""),
                        error = %e,
                        "failed to build pinned TLS client; cortex disabled (fail closed)"
                    );
                }
            }
        }

        Self {
            cortexes: config.cortexes.clone(),
            clients,
            region: config.router.region.clone(),
            aliases: config.aliases.clone(),
            poll_interval: Duration::from_secs(config.router.poll_interval_secs),
            topology: RwLock::new(topology),
        }
    }

    /// The HTTP client to use for `name`, or `None` if the cortex is
    /// disabled (its `tls_ca` failed to load). Callers must treat `None` as
    /// "not routable / not pollable".
    pub fn client_for(&self, name: &str) -> Option<&reqwest::Client> {
        self.clients.get(name)
    }

    /// Names of reachable cortexes that can serve `model_id` (loaded or
    /// feasible to cold-load). Groundwork for capacity-aware dispatch (#73);
    /// unreachable cortexes are excluded by construction.
    pub async fn cortexes_serving(&self, model_id: &str) -> Vec<String> {
        let topo = self.topology.read().await;
        topo.iter()
            .filter(|(_, t)| t.reachable)
            .filter(|(_, t)| t.models.get(model_id).is_some_and(entry_feasible))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Build a cortex HTTP client. With `tls_ca` set, the client trusts **only**
/// that PEM anchor (platform roots disabled) — pinning the router→cortex hop
/// to an enrolled cert (#74). Without it, standard platform-root validation.
pub fn build_client(tls_ca: Option<&str>) -> Result<reqwest::Client, BuildClientError> {
    let mut builder = reqwest::Client::builder();
    if let Some(path) = tls_ca {
        let pem = std::fs::read(path).map_err(|e| BuildClientError::Read(path.to_string(), e))?;
        let cert = reqwest::Certificate::from_pem(&pem).map_err(BuildClientError::Parse)?;
        builder = builder
            .tls_built_in_root_certs(false)
            .add_root_certificate(cert);
    }
    builder.build().map_err(BuildClientError::Build)
}

/// Why a cortex's pinned client could not be built (→ cortex disabled).
#[derive(Debug, thiserror::Error)]
pub enum BuildClientError {
    #[error("reading TLS anchor '{0}'")]
    Read(String, #[source] std::io::Error),
    #[error("parsing TLS anchor PEM")]
    Parse(#[source] reqwest::Error),
    #[error("building HTTP client")]
    Build(#[source] reqwest::Error),
}
