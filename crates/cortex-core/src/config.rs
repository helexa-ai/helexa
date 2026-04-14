use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub gateway: GatewaySettings,
    pub eviction: EvictionSettings,
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewaySettings {
    /// Address to listen on for API requests (e.g. "0.0.0.0:8000")
    pub listen: String,
    /// Address to listen on for Prometheus metrics (e.g. "0.0.0.0:9100")
    pub metrics_listen: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionSettings {
    /// Eviction strategy: "lru" or "priority"
    pub strategy: EvictionStrategy,
    /// Restart the mistralrs process after this many load/unload cycles
    /// to reclaim fragmented VRAM. 0 = never.
    #[serde(default)]
    pub defrag_after_cycles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvictionStrategy {
    Lru,
    Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Human-readable node name (e.g. "gpu-large")
    pub name: String,
    /// Base URL of the mistralrs HTTP server (e.g. "http://gpu-large.internal:8080")
    pub endpoint: String,
    /// Total VRAM in MB across all GPUs on this node
    pub vram_mb: u64,
    /// Model IDs that should never be evicted from this node
    #[serde(default)]
    pub pinned: Vec<String>,
}

impl GatewayConfig {
    /// Load configuration from a TOML file, with environment variable overrides.
    /// Env vars are prefixed with `CORTEX_` and use `__` as a separator
    /// (e.g. `CORTEX_GATEWAY__LISTEN=0.0.0.0:9000`).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, figment::Error> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("CORTEX_").split("__"))
            .extract()
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            gateway: GatewaySettings {
                listen: "0.0.0.0:8000".into(),
                metrics_listen: "0.0.0.0:9100".into(),
            },
            eviction: EvictionSettings {
                strategy: EvictionStrategy::Lru,
                defrag_after_cycles: 50,
            },
            nodes: vec![],
        }
    }
}
