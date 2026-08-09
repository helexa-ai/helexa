use crate::entitlements_chain::ChainedEntitlementProvider;
use crate::entitlements_local::LocalEntitlementProvider;
use crate::entitlements_upstream::UpstreamEntitlementProvider;
use cortex_core::catalogue::ModelCatalogue;
use cortex_core::config::{EvictionSettings, GatewayConfig, NeuronEndpoint};
use cortex_core::entitlements::EntitlementProvider;
use cortex_core::node::NodeState;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared fleet state, protected by a RwLock for concurrent reader access.
pub struct CortexState {
    pub nodes: RwLock<HashMap<String, NodeState>>,
    pub neuron_configs: Vec<NeuronEndpoint>,
    pub eviction: EvictionSettings,
    pub catalogue: ModelCatalogue,
    pub http_client: reqwest::Client,
    /// Resolves bearer keys to principals and enforces token budgets (#47).
    /// A local/static provider today (#50); the upstream client later (#57).
    pub entitlements: Arc<dyn EntitlementProvider>,
    /// Whether to reject unauthenticated requests (#49). Read by the auth
    /// middleware once it lands.
    pub require_auth: bool,
    /// Per-principal served-token tally (#58), reported to upstream for
    /// operator reconciliation by the flush task when upstream is enabled.
    pub served_usage: Arc<crate::served_usage::ServedUsage>,
    /// Modalities discovered for models that are not currently loaded
    /// (#241), keyed by model id.
    ///
    /// A loaded model reports its own capabilities every poll, so this
    /// only carries the cold ones — which is the case that mattered,
    /// since the image model spends nearly all its time evicted and was
    /// therefore advertising nothing. Neurons derive these from their
    /// local model cache; cortex asks once and keeps the answer, because
    /// what a given model id can do does not change.
    pub discovered_capabilities: RwLock<HashMap<String, Vec<String>>>,
    /// When each unresolved model id was last probed for capabilities.
    ///
    /// A model no node has cached can never be answered, and the poll
    /// loop runs every few seconds — without this it would ask every
    /// node about every such model forever, turning a one-off lookup
    /// into steady background traffic and a stream of 404s in the logs.
    /// Resolved ids leave this map and are never probed again.
    pub capability_probe_attempts: RwLock<HashMap<String, std::time::Instant>>,
}

impl CortexState {
    pub fn from_config(config: &GatewayConfig) -> Self {
        let mut nodes = HashMap::new();
        for nc in &config.neurons {
            nodes.insert(
                nc.name.clone(),
                NodeState {
                    device_health: Vec::new(),
                    name: nc.name.clone(),
                    endpoint: nc.endpoint.clone(),
                    healthy: false,
                    models: HashMap::new(),
                    lifecycle_cycles: 0,
                    last_poll: None,
                    discovery: None,
                    activation: None,
                    model_load: HashMap::new(),
                    consecutive_poll_failures: 0,
                },
            );
        }

        let catalogue = ModelCatalogue::load(&config.models_config);

        // Local provider always handles operator + infra keys. When the
        // upstream client is enabled (#57), wrap it in the chain so locally
        // unknown keys fall through to the mesh authority; otherwise stay
        // purely local.
        let local = LocalEntitlementProvider::from_config(&config.entitlements);
        let entitlements: Arc<dyn EntitlementProvider> = if config.upstream.enabled {
            tracing::info!(url = %config.upstream.url, "upstream entitlement client enabled");
            Arc::new(ChainedEntitlementProvider::new(
                local,
                UpstreamEntitlementProvider::new(&config.upstream),
            ))
        } else {
            Arc::new(local)
        };

        Self {
            nodes: RwLock::new(nodes),
            neuron_configs: config.neurons.clone(),
            eviction: config.eviction.clone(),
            catalogue,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            entitlements,
            require_auth: config.entitlements.require_auth,
            served_usage: Arc::new(crate::served_usage::ServedUsage::new()),
            discovered_capabilities: RwLock::new(HashMap::new()),
            capability_probe_attempts: RwLock::new(HashMap::new()),
        }
    }
}
