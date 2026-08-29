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
    /// helexa-upstream client (#57). When enabled, keys not found in the
    /// local `[entitlements]` config are validated against the mesh
    /// authority, and budget is reserved/settled there. Disabled by default
    /// — a single operator runs purely local.
    #[serde(default)]
    pub upstream: UpstreamClientConfig,
}

/// `[upstream]` — the helexa-upstream authority client (#57). Locally
/// unrecognised bearer keys are resolved against `url`'s `/authz/v1` surface
/// (mesh accounts); local keys (operator + infra) never leave the process.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpstreamClientConfig {
    /// Enable the upstream fallthrough. Off → purely local entitlements.
    #[serde(default)]
    pub enabled: bool,
    /// Base URL of helexa-upstream (e.g. "https://upstream.helexa.ai").
    #[serde(default)]
    pub url: String,
    /// Shared client bearer this cortex presents to `/authz/v1` (maps to an
    /// operator_id upstream). Sent as `Authorization: Bearer <bearer>`.
    #[serde(default)]
    pub bearer: String,
    /// Per-call timeout (seconds) to upstream.
    #[serde(default = "default_upstream_timeout")]
    pub timeout_secs: u64,
    /// How often (seconds) to flush served-usage counters to upstream for
    /// reconciliation (#58).
    #[serde(default = "default_served_usage_interval")]
    pub served_usage_report_interval_secs: u64,
}

fn default_upstream_timeout() -> u64 {
    5
}
fn default_served_usage_interval() -> u64 {
    60
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
    ///
    /// A sibling `secrets.toml`, if present, is merged **after** the main
    /// file and therefore wins. It exists so the two have different
    /// owners: CI writes the main config, and only the operator writes
    /// the secrets — see [`secrets_path`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        let path = path.as_ref();
        Figment::new()
            .merge(Toml::file(path))
            .merge(Toml::file(secrets_path(path)))
            .merge(Env::prefixed("CORTEX_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

/// `secrets.toml` beside the main config.
///
/// The split exists to make a class of outage impossible rather than
/// merely unlikely. Config that CI cannot write is config no check can
/// defend: `helexa-router.toml` was excluded from git because it sat
/// beside secret-bearing files, and its `helexa/balanced` alias then
/// pointed at a retired model for long enough to take a node down
/// (2026-08-27). The fix is not to hand CI the credentials — it is to
/// stop mixing the two in one file.
///
/// So `cortex.toml` holds structure and is deployed by CI, while
/// `secrets.toml` holds API keys and the upstream bearer, is written
/// only by the operator, and is never read, written or diffed by the
/// pipeline. Because it merges last it also wins, so a value present in
/// both is the operator's.
///
/// A missing file is not an error — figment treats an absent
/// `Toml::file` as an empty source — so a host that keeps everything in
/// one file, and every existing deployment, keeps working unchanged.
fn secrets_path(config: &Path) -> std::path::PathBuf {
    config
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("secrets.toml")
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
            upstream: UpstreamClientConfig::default(),
        }
    }
}

#[cfg(test)]
mod secrets_layer_tests {
    use super::*;
    use std::io::Write;

    /// Write `main` (and optionally `secrets.toml`) into a fresh directory
    /// under the target dir, and load through the real code path.
    fn load_with(dir: &str, main: &str, secrets: Option<&str>) -> GatewayConfig {
        // `CARGO_TARGET_TMPDIR` is only set for integration tests, not for
        // unit tests inside src/, so derive a unique directory instead.
        let base = std::env::temp_dir().join(format!("helexa-cortex-config-{dir}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create test dir");
        let cfg = base.join("cortex.toml");
        write!(
            std::fs::File::create(&cfg).expect("create config"),
            "{main}"
        )
        .expect("write");
        if let Some(s) = secrets {
            let p = base.join("secrets.toml");
            write!(std::fs::File::create(&p).expect("create secrets"), "{s}").expect("write");
        }
        GatewayConfig::load(&cfg).expect("config loads")
    }

    const MAIN: &str = r#"
[gateway]
listen = "0.0.0.0:31313"
metrics_listen = "0.0.0.0:31314"
[eviction]
strategy = "lru"
defrag_after_cycles = 50
[entitlements]
require_auth = true

[[neurons]]
name = "beast"
endpoint = "http://beast.invalid:13131"
"#;

    /// The whole point: a host with no secrets.toml behaves exactly as
    /// before. Every current deployment is in this state, so if this
    /// regressed the change would take the fleet down on rollout rather
    /// than on adoption.
    #[test]
    fn an_absent_secrets_file_is_not_an_error() {
        let cfg = load_with("secrets_absent", MAIN, None);
        assert!(cfg.entitlements.require_auth);
        assert!(
            cfg.entitlements.keys.is_empty(),
            "no keys configured anywhere means no keys"
        );
    }

    /// The array-of-tables case that ruled out env-var overrides:
    /// `[[entitlements.keys]]` cannot be expressed as `CORTEX_*` env vars,
    /// so the second TOML layer is what carries it.
    #[test]
    fn secrets_file_supplies_the_entitlement_keys() {
        let cfg = load_with(
            "secrets_keys",
            MAIN,
            Some(
                r#"
[[entitlements.keys]]
key = "sk-test-aaa"
account_id = "acct-a"
key_id = "a"
[[entitlements.keys]]
key = "sk-test-bbb"
account_id = "acct-b"
key_id = "b"
"#,
            ),
        );
        assert_eq!(cfg.entitlements.keys.len(), 2);
        assert_eq!(cfg.entitlements.keys[0].key, "sk-test-aaa");
        assert_eq!(cfg.entitlements.keys[1].account_id, "acct-b");
        assert!(
            cfg.entitlements.require_auth,
            "structure from the CI-owned file must survive the merge"
        );
    }

    /// Ordering is the safety property. If CI ever ships a placeholder in
    /// the main file, the operator's real value must still win — otherwise
    /// a deploy silently swaps a live credential for a dummy.
    #[test]
    fn the_operators_value_wins_over_the_deployed_one() {
        let cfg = load_with(
            "secrets_precedence",
            &format!(
                "{MAIN}\n[upstream]\nenabled = true\nurl = \"https://example.invalid\"\nbearer = \"PLACEHOLDER\"\n"
            ),
            Some("[upstream]\nbearer = \"real-token\"\n"),
        );
        assert_eq!(
            cfg.upstream.bearer, "real-token",
            "secrets.toml merges last, so it wins"
        );
        assert_eq!(
            cfg.upstream.url, "https://example.invalid",
            "fields the operator does not override keep the deployed value"
        );
    }
}
