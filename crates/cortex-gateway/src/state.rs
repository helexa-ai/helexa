use cortex_core::config::{EvictionSettings, GatewayConfig, NodeConfig};
use cortex_core::node::NodeState;
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Shared fleet state, protected by a RwLock for concurrent reader access.
pub struct CortexState {
    pub nodes: RwLock<HashMap<String, NodeState>>,
    pub node_configs: Vec<NodeConfig>,
    pub eviction: EvictionSettings,
    pub http_client: reqwest::Client,
}

impl CortexState {
    pub fn from_config(config: &GatewayConfig) -> Self {
        let mut nodes = HashMap::new();
        for nc in &config.nodes {
            nodes.insert(
                nc.name.clone(),
                NodeState {
                    name: nc.name.clone(),
                    endpoint: nc.endpoint.clone(),
                    vram_mb: nc.vram_mb,
                    pinned: nc.pinned.clone(),
                    healthy: false, // will be set by first poll
                    models: HashMap::new(),
                    lifecycle_cycles: 0,
                    last_poll: None,
                },
            );
        }

        Self {
            nodes: RwLock::new(nodes),
            node_configs: config.nodes.clone(),
            eviction: config.eviction.clone(),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
        }
    }
}
