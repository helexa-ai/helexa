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
    /// How often (seconds) the background poller refreshes each cortex's
    /// health + `/v1/models` topology (#72). Defaults to 10s, matching the
    /// cortex↔neuron poll cadence one tier down.
    #[serde(default = "default_poll_interval_secs")]
    pub poll_interval_secs: u64,
    /// This router instance's region (e.g. "eu-west"). When set, dispatch
    /// (#73) prefers cortexes whose `region` matches, before falling back to
    /// any feasible cortex. `None` → no geo affinity.
    #[serde(default)]
    pub region: Option<String>,
}

fn default_poll_interval_secs() -> u64 {
    10
}

/// One downstream cortex the router may proxy to. The router verifies the
/// cortex's outbound TLS cert (#74) and routes on capacity (#73); it holds
/// no entitlement logic of its own and forwards the client bearer verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CortexEndpoint {
    /// Human-readable label (e.g. "lair-cafe").
    pub name: String,
    /// Base URL of the cortex gateway (e.g. "https://cortex.example.com").
    pub endpoint: String,
    /// Optional region tag (e.g. "eu-west") for geo affinity in dispatch
    /// (#73). `None` → no region preference applies to this cortex.
    #[serde(default)]
    pub region: Option<String>,
    /// Path to a PEM trust anchor that **enrols** this cortex (#74): the
    /// expected CA (or self-signed cert) the cortex's TLS cert must chain
    /// to. When set on an `https://` endpoint, the router builds a client
    /// that trusts **only** this anchor (platform roots disabled), so the
    /// outbound router→cortex hop — which carries the client's bearer —
    /// reaches a cert the router was told to expect, and a rogue endpoint
    /// presenting any other (even publicly-valid) cert is rejected at the
    /// TLS handshake. A rejected handshake surfaces as a connection error,
    /// which the poller (#72) already treats as unreachable → excluded.
    ///
    /// `None` → standard platform-root validation (use for cortexes behind
    /// a publicly-trusted cert, or plaintext `http://` on a private network
    /// where the WireGuard mesh is the trust boundary).
    #[serde(default)]
    pub tls_ca: Option<String>,
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
                poll_interval_secs: default_poll_interval_secs(),
                region: None,
            },
            cortexes: vec![],
        }
    }
}
