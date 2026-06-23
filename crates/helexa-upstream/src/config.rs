//! helexa-upstream configuration: loaded from `helexa-upstream.toml` with
//! figment, `UPSTREAM_`-prefixed env overrides (mirrors the cortex/router
//! convention, e.g. `UPSTREAM_SERVER__LISTEN`, `UPSTREAM_DB__URL`).

use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    #[serde(default)]
    pub server: ServerSettings,
    pub db: DbSettings,
    #[serde(default)]
    pub grant: GrantSettings,
    #[serde(default)]
    pub abuse: AbuseSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    /// Address to listen on (e.g. "0.0.0.0:8090"). Plaintext — edge nginx
    /// terminates TLS, consistent with the rest of the stack.
    #[serde(default = "default_listen")]
    pub listen: String,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            listen: default_listen(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbSettings {
    /// PostgreSQL connection URL (e.g. "postgres://user:pass@host/helexa").
    pub url: String,
    /// Max pool connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

/// `[grant]` — the flat free token grant every email-verified account
/// receives (the floor of the hybrid allocation model; top-up codes extend
/// it).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrantSettings {
    #[serde(default = "default_free_grant")]
    pub free_token_grant: i64,
}

impl Default for GrantSettings {
    fn default() -> Self {
        Self {
            free_token_grant: default_free_grant(),
        }
    }
}

/// `[abuse]` — silent multi-account abuse detection. When at least
/// `fingerprint_account_threshold` accounts share one registration
/// fingerprint, all of them are silently deactivated (no notice to the
/// user; deactivation only surfaces as ordinary inference rejections).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbuseSettings {
    #[serde(default = "default_fingerprint_threshold")]
    pub fingerprint_account_threshold: i64,
}

impl Default for AbuseSettings {
    fn default() -> Self {
        Self {
            fingerprint_account_threshold: default_fingerprint_threshold(),
        }
    }
}

fn default_listen() -> String {
    "0.0.0.0:8090".into()
}
fn default_max_connections() -> u32 {
    16
}
fn default_free_grant() -> i64 {
    1_000_000
}
fn default_fingerprint_threshold() -> i64 {
    5
}

impl UpstreamConfig {
    /// Load from a TOML file with `UPSTREAM_`-prefixed env overrides
    /// (`__` nesting separator).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("UPSTREAM_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::result_large_err)]
    fn loads_toml_with_env_override_and_defaults() {
        figment::Jail::expect_with(|jail| {
            jail.create_file(
                "helexa-upstream.toml",
                r#"
[db]
url = "postgres://localhost/helexa"
"#,
            )?;
            jail.set_env("UPSTREAM_SERVER__LISTEN", "127.0.0.1:9099");

            let cfg = UpstreamConfig::load("helexa-upstream.toml").expect("load");
            assert_eq!(cfg.server.listen, "127.0.0.1:9099");
            assert_eq!(cfg.db.url, "postgres://localhost/helexa");
            // Defaults applied when sections omitted.
            assert_eq!(cfg.grant.free_token_grant, 1_000_000);
            assert_eq!(cfg.abuse.fingerprint_account_threshold, 5);
            assert_eq!(cfg.db.max_connections, 16);
            Ok(())
        });
    }
}
