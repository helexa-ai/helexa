//! helexa-angels configuration: `helexa-angels.toml` via figment, with
//! `ANGELS_`-prefixed env overrides (the cortex/router/upstream
//! convention, e.g. `ANGELS_SERVER__LISTEN`, `ANGELS_DB__URL`).

use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngelsConfig {
    #[serde(default)]
    pub server: ServerSettings,
    pub db: DbSettings,
    #[serde(default)]
    pub session: SessionSettings,
    #[serde(default)]
    pub site: SiteSettings,
    #[serde(default)]
    pub content: ContentSettings,
    #[serde(default)]
    pub upstream: UpstreamSettings,
    #[serde(default)]
    pub email: EmailSettings,
}

/// `[email]` — where expressions of interest are announced.
///
/// `log` is a legitimate production choice while volumes are a handful of
/// people: the submission is stored in the database regardless, so this
/// only decides whether a mail also goes out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSettings {
    /// `"log"` or `"smtp"`.
    #[serde(default = "default_email_provider")]
    pub provider: String,
    pub smtp_url: Option<String>,
    #[serde(default = "default_from_addr")]
    pub from_addr: String,
    /// The operator inbox that receives interest notifications.
    #[serde(default = "default_notify_to")]
    pub notify_to: String,
}

impl Default for EmailSettings {
    fn default() -> Self {
        Self {
            provider: default_email_provider(),
            smtp_url: None,
            from_addr: default_from_addr(),
            notify_to: default_notify_to(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
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
    /// The **same** database helexa-upstream uses — credential auth is
    /// shared (D2). angels confines its own tables to the `angels` schema.
    pub url: String,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

/// `[session]` — the angels session realm, deliberately separate from
/// helexa.ai's. See `migrations/0001_init.sql` for why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    #[serde(default = "default_cookie_name")]
    pub cookie_name: String,
    /// Absolute lifetime.
    #[serde(default = "default_session_ttl")]
    pub ttl_secs: u64,
    /// Idle timeout — a session untouched for this long is dead even if
    /// its absolute lifetime has not expired.
    #[serde(default = "default_session_idle")]
    pub idle_secs: u64,
    /// `Secure` attribute on the cookie. Only ever false for local dev
    /// over plain HTTP; production is TLS-only.
    #[serde(default = "default_true")]
    pub secure: bool,
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            cookie_name: default_cookie_name(),
            ttl_secs: default_session_ttl(),
            idle_secs: default_session_idle(),
            secure: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteSettings {
    /// Public origin, used to build absolute links (invite URLs, mail).
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Where expressions of interest are routed.
    #[serde(default = "default_contact")]
    pub contact_email: String,
}

impl Default for SiteSettings {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            contact_email: default_contact(),
        }
    }
}

/// `[content]` — where round documents live on disk.
///
/// Deliberately outside any web root and outside the source repository:
/// `helexa/helexa` is open source, so a business plan committed there is
/// a business plan published.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSettings {
    #[serde(default = "default_content_dir")]
    pub dir: String,
}

impl Default for ContentSettings {
    fn default() -> Self {
        Self {
            dir: default_content_dir(),
        }
    }
}

/// `[upstream]` — helexa-upstream, reached over the mesh.
///
/// Registration is delegated there rather than reimplemented: upstream
/// already owns password policy, argon2 parameters, verification email,
/// unverified-signup reaping and registration fingerprinting. Two
/// divergent implementations against one `users` table is a defect
/// waiting to happen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamSettings {
    #[serde(default = "default_upstream_url")]
    pub base_url: String,
    #[serde(default = "default_upstream_timeout")]
    pub timeout_secs: u64,
}

impl Default for UpstreamSettings {
    fn default() -> Self {
        Self {
            base_url: default_upstream_url(),
            timeout_secs: default_upstream_timeout(),
        }
    }
}

fn default_listen() -> String {
    "127.0.0.1:8092".into()
}
fn default_max_connections() -> u32 {
    5
}
fn default_cookie_name() -> String {
    "angels_session".into()
}
fn default_session_ttl() -> u64 {
    7 * 24 * 3600
}
fn default_session_idle() -> u64 {
    12 * 3600
}
fn default_true() -> bool {
    true
}
fn default_base_url() -> String {
    "https://angels.helexa.ai".into()
}
fn default_contact() -> String {
    "angels@helexa.ai".into()
}
fn default_content_dir() -> String {
    "/var/lib/helexa-angels/content".into()
}
fn default_upstream_url() -> String {
    "http://localhost:8090".into()
}
fn default_upstream_timeout() -> u64 {
    30
}
fn default_email_provider() -> String {
    "log".into()
}
fn default_from_addr() -> String {
    "helexa <no-reply@helexa.ai>".into()
}
fn default_notify_to() -> String {
    "angels@helexa.ai".into()
}

impl AngelsConfig {
    /// A sibling `secrets.toml`, if present, merges after the main file
    /// and wins — see [`secrets_path`].
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let cfg: Self = Figment::new()
            .merge(Toml::file(path))
            .merge(Toml::file(secrets_path(path)))
            .merge(Env::prefixed("ANGELS_").split("__"))
            .extract()?;
        Ok(cfg)
    }
}

/// `secrets.toml` beside the main config.
///
/// The main file holds structure and is deployed by CI; this one holds
/// credentials, is written only by the operator, and is never read,
/// written or diffed by the pipeline. Merging it last means the
/// operator's value wins, so a deploy cannot swap a live credential for
/// a placeholder.
///
/// The split exists because config CI cannot write is config no check
/// can defend — `helexa-router.toml` was kept out of git for sitting
/// beside secret-bearing files, and its stale alias then took a node
/// down (2026-08-27). The answer is to separate secrets from structure,
/// not to hand the pipeline the credentials.
///
/// A missing file is not an error: figment treats an absent
/// `Toml::file` as an empty source, so existing single-file deployments
/// keep working untouched.
fn secrets_path(config: &Path) -> std::path::PathBuf {
    config
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("secrets.toml")
}
