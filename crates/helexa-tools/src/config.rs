//! Config for the helexa-tools service (figment: TOML + env override,
//! `HELEXA_TOOLS_` prefix), mirroring the other helexa daemons.

use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// Bind address for the HTTP API.
    pub listen: String,
    /// Per-request wall-clock budget for one upstream fetch (seconds).
    pub fetch_timeout_secs: u64,
    /// Maximum bytes read from an upstream response body.
    pub max_body_bytes: usize,
    /// Maximum characters of extracted text returned to the caller.
    /// The consumer is a model context — bound it hard.
    pub max_text_chars: usize,
    /// Maximum redirect hops followed (each hop is re-validated
    /// against the SSRF policy).
    pub max_redirects: u8,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8889".into(),
            fetch_timeout_secs: 20,
            max_body_bytes: 2 * 1024 * 1024,
            max_text_chars: 12_000,
            max_redirects: 5,
        }
    }
}

impl ToolsConfig {
    pub fn load(path: &str) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("HELEXA_TOOLS_"))
            .extract()
            .map_err(Box::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = ToolsConfig::default();
        assert_eq!(cfg.listen, "0.0.0.0:8889");
        assert!(cfg.max_text_chars > 0);
        assert!(cfg.max_redirects > 0);
    }

    #[test]
    fn loads_toml_with_env_override() {
        figment::Jail::expect_with(|jail| {
            jail.create_file(
                "helexa-tools.toml",
                r#"
                listen = "127.0.0.1:9999"
                max_text_chars = 5000
                "#,
            )?;
            jail.set_env("HELEXA_TOOLS_MAX_TEXT_CHARS", "7000");
            let cfg = ToolsConfig::load("helexa-tools.toml").expect("load");
            assert_eq!(cfg.listen, "127.0.0.1:9999");
            assert_eq!(cfg.max_text_chars, 7000);
            Ok(())
        });
    }
}
