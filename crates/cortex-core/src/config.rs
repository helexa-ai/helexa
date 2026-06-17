use crate::entitlements::CapWindow;
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
    /// Neuron endpoints (replaces old NodeConfig with static vram_mb/pinned).
    pub neurons: Vec<NeuronEndpoint>,
    /// Path to the model catalogue file. Defaults to the packaged
    /// location (`/etc/cortex/models.toml`); set explicitly for
    /// non-packaged / local runs.
    #[serde(default = "default_models_path")]
    pub models_config: String,
    /// Multi-tenant governance: auth + per-key token budgets (#47). Empty
    /// by default — anonymous, uncapped — so existing single-operator
    /// setups keep working until keys are configured.
    #[serde(default)]
    pub entitlements: EntitlementsConfig,
}

/// `[entitlements]` — the local/static [`crate::entitlements::EntitlementProvider`]
/// source of truth (#50). Accounts, keys, and hard caps live here; the
/// future upstream client (#57) ignores this section.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntitlementsConfig {
    /// Reject unauthenticated requests with `401 invalid_api_key` when
    /// true. Default `false` (allow-anonymous) for dev / single-operator
    /// continuity.
    #[serde(default)]
    pub require_auth: bool,
    /// Static API keys and their budgets, consumed by the local provider.
    #[serde(default)]
    pub keys: Vec<ApiKeyConfig>,
}

/// One configured API key: the bearer token, the account it bills to, and
/// its hard cap. `[[entitlements.keys]]` in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// The bearer token clients send in `Authorization: Bearer <key>`.
    pub key: String,
    /// Billable account. Multiple keys may share one account.
    pub account_id: String,
    /// Stable per-key identifier for ledger/metrics labels. Defaults to
    /// `account_id` when omitted, so the secret is never used as a label.
    #[serde(default)]
    pub key_id: Option<String>,
    /// Hard token cap. `None`/omitted = uncapped (e.g. operator infra key).
    #[serde(default)]
    pub hard_cap: Option<u64>,
    /// Cap-window semantics. Default: a non-resetting [`CapWindow::Balance`].
    #[serde(default)]
    pub window: CapWindow,
}

fn default_models_path() -> String {
    // Absolute, so the systemd-launched binary finds the catalogue
    // regardless of its working directory. The RPM installs the catalogue
    // here (`cortex.spec`); a relative "models.toml" silently resolved to
    // the service cwd and left the catalogue empty in production
    // (pinning / aliases / limits all no-ops). Override via `models_config`
    // in cortex.toml for local runs.
    "/etc/cortex/models.toml".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewaySettings {
    /// Address to listen on for API requests (e.g. "0.0.0.0:31313")
    pub listen: String,
    /// Address to listen on for Prometheus metrics (e.g. "0.0.0.0:31314")
    pub metrics_listen: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionSettings {
    /// Eviction strategy: "lru" or "priority"
    pub strategy: EvictionStrategy,
    /// Number of load/unload cycles before flagging for defrag. 0 = never.
    #[serde(default)]
    pub defrag_after_cycles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvictionStrategy {
    Lru,
    Priority,
}

/// A neuron endpoint in the fleet. Hardware details come from
/// neuron's /discovery endpoint, not from config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronEndpoint {
    /// Human-readable node name (e.g. "beast")
    pub name: String,
    /// Base URL of the neuron daemon (e.g. "http://beast.internal:13131")
    pub endpoint: String,
}

impl GatewayConfig {
    /// Load configuration from a TOML file, with environment variable overrides.
    /// Env vars are prefixed with `CORTEX_` and use `__` as a separator.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("CORTEX_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            gateway: GatewaySettings {
                listen: "0.0.0.0:31313".into(),
                metrics_listen: "0.0.0.0:31314".into(),
            },
            eviction: EvictionSettings {
                strategy: EvictionStrategy::Lru,
                defrag_after_cycles: 50,
            },
            neurons: vec![],
            models_config: default_models_path(),
            entitlements: EntitlementsConfig::default(),
        }
    }
}
