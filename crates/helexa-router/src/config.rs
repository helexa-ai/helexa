use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level `helexa-router` configuration.
///
/// Loaded from TOML with `HELEXA_ROUTER_`-prefixed env overrides (using
/// `__` as the nesting separator), matching the cortex/neuron convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub router: RouterSettings,
    /// Downstream cortex endpoints the router can dispatch to. The skeleton
    /// (#70) only loads these; capacity/catalogue polling (#72) and
    /// capacity-aware dispatch (#73) consume them later.
    #[serde(default)]
    pub cortexes: Vec<CortexEndpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterSettings {
    /// Address to listen on for the inbound API (e.g. "0.0.0.0:8088").
    ///
    /// Plaintext only — operator/edge nginx terminates client TLS in front
    /// of the router (see #69's TLS posture). The router never owns an
    /// inbound TLS listener.
    pub listen: String,
}

/// One downstream cortex the router may proxy to. The router verifies the
/// cortex's outbound TLS cert (#74) and routes on capacity (#73); it holds
/// no entitlement logic of its own and forwards the client bearer verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexEndpoint {
    /// Human-readable label (e.g. "lair-cafe").
    pub name: String,
    /// Base URL of the cortex gateway (e.g. "https://cortex.example.com").
    pub endpoint: String,
}

impl RouterConfig {
    /// Load configuration from a TOML file, with environment variable
    /// overrides prefixed with `HELEXA_ROUTER_` and `__` as the separator
    /// (e.g. `HELEXA_ROUTER_ROUTER__LISTEN=0.0.0.0:8088`).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("HELEXA_ROUTER_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            router: RouterSettings {
                listen: "0.0.0.0:8088".into(),
            },
            cortexes: vec![],
        }
    }
}
