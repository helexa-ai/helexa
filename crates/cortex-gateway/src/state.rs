use cortex_core::catalogue::ModelCatalogue;
use cortex_core::config::{EvictionSettings, GatewayConfig, NeuronEndpoint};
use cortex_core::node::NodeState;
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Shared fleet state, protected by a RwLock for concurrent reader access.
pub struct CortexState {
    pub nodes: RwLock<HashMap<String, NodeState>>,
    pub neuron_configs: Vec<NeuronEndpoint>,
    pub eviction: EvictionSettings,
    pub catalogue: ModelCatalogue,
    pub http_client: reqwest::Client,
}

impl CortexState {
    pub fn from_config(config: &GatewayConfig) -> Self {
        let mut nodes = HashMap::new();
        for nc in &config.neurons {
            nodes.insert(
                nc.name.clone(),
                NodeState {
                    name: nc.name.clone(),
                    endpoint: nc.endpoint.clone(),
                    healthy: false,
                    models: HashMap::new(),
                    lifecycle_cycles: 0,
                    last_poll: None,
                    discovery: None,
                    activation: None,
                },
            );
        }

        let catalogue = ModelCatalogue::load(&config.models_config);

        Self {
            nodes: RwLock::new(nodes),
            neuron_configs: config.neurons.clone(),
            eviction: config.eviction.clone(),
            catalogue,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
        }
    }
}
