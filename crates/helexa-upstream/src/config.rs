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
    #[serde(default)]
    pub client_auth: ClientAuthSettings,
    #[serde(default)]
    pub authz: AuthzSettings,
    #[serde(default)]
    pub auth: AuthSettings,
    #[serde(default)]
    pub email: EmailSettings,
}

/// `[auth]` — web-session signing + token lifetimes (B4). Web sessions are
/// JWTs, distinct from inference API keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSettings {
    /// HMAC secret for signing session JWTs. MUST be overridden in prod
    /// (env `UPSTREAM_AUTH__JWT_SECRET`); the default is dev-only.
    #[serde(default = "default_jwt_secret")]
    pub jwt_secret: String,
    /// Session token lifetime (seconds).
    #[serde(default = "default_session_ttl")]
    pub session_ttl_secs: u64,
    /// Email verification / password-reset token lifetime (seconds).
    #[serde(default = "default_email_token_ttl")]
    pub email_token_ttl_secs: u64,
    /// Public base URL of the frontend, used to build verify/reset links.
    #[serde(default = "default_app_base_url")]
    pub app_base_url: String,
}

impl Default for AuthSettings {
    fn default() -> Self {
        Self {
            jwt_secret: default_jwt_secret(),
            session_ttl_secs: default_session_ttl(),
            email_token_ttl_secs: default_email_token_ttl(),
            app_base_url: default_app_base_url(),
        }
    }
}

/// `[email]` — transactional email transport for verify/reset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSettings {
    /// `"log"` (dev — logs the link) or `"smtp"`.
    #[serde(default = "default_email_provider")]
    pub provider: String,
    /// SMTP relay URL (e.g. "smtp://user:pass@host:587") when provider=smtp.
    #[serde(default)]
    pub smtp_url: Option<String>,
    /// `From:` address.
    #[serde(default = "default_from_addr")]
    pub from_addr: String,
}

impl Default for EmailSettings {
    fn default() -> Self {
        Self {
            provider: default_email_provider(),
            smtp_url: None,
            from_addr: default_from_addr(),
        }
    }
}

/// `[client_auth]` — credentials operators' cortexes present to `/authz/v1`.
/// Each token maps to an `operator_id` (served-usage attribution, #58). This
/// transport credential is distinct from end-user API keys (which ride in
/// the `resolve` body). v2 adds mTLS.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientAuthSettings {
    /// When empty the authz surface is **open** (dev only; logged at warn).
    #[serde(default)]
    pub tokens: Vec<ClientToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToken {
    /// Shared bearer a cortex presents.
    pub token: String,
    /// Operator this token identifies.
    pub operator_id: String,
}

/// `[authz]` — reservation lifecycle knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthzSettings {
    /// Open reservations older than this are swept (released), self-healing
    /// a reservation whose settle/release from cortex was lost.
    #[serde(default = "default_reservation_ttl")]
    pub reservation_ttl_secs: u64,
    /// How often the sweeper runs.
    #[serde(default = "default_sweep_interval")]
    pub sweep_interval_secs: u64,
}

impl Default for AuthzSettings {
    fn default() -> Self {
        Self {
            reservation_ttl_secs: default_reservation_ttl(),
            sweep_interval_secs: default_sweep_interval(),
        }
    }
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
fn default_reservation_ttl() -> u64 {
    120
}
fn default_sweep_interval() -> u64 {
    60
}
fn default_jwt_secret() -> String {
    "dev-insecure-change-me".into()
}
fn default_session_ttl() -> u64 {
    7 * 24 * 3600
}
fn default_email_token_ttl() -> u64 {
    24 * 3600
}
fn default_app_base_url() -> String {
    "http://localhost:5173".into()
}
fn default_email_provider() -> String {
    "log".into()
}
fn default_from_addr() -> String {
    "helexa <no-reply@helexa.ai>".into()
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
