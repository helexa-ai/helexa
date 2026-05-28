//! Configuration for the helexa-acp bridge.
//!
//! Loaded from `$XDG_CONFIG_HOME/helexa-acp/config.toml` (or
//! `~/.config/helexa-acp/config.toml` as a fallback). If no config file
//! exists, falls back to building a single anonymous endpoint from env
//! vars — that keeps "just point at one cortex" frictionless without
//! requiring a config file on disk.
//!
//! The design goal is "the missing ACP binary for users with multiple
//! API endpoints (possibly on a private LAN, possibly mixing wire
//! types)". Hence: every endpoint is named, has its own wire API, and
//! has its own default model. The agent's selected model id can be
//! prefixed `endpoint:model` to route across endpoints; a bare
//! `model` falls through to the configured `default_endpoint`.
//!
//! ### Example TOML
//!
//! ```toml
//! default_endpoint = "helexa"
//!
//! [[endpoints]]
//! name = "helexa"
//! base_url = "http://hanzalova.internal:31313/v1"
//! wire_api = "openai-chat"
//! default_model = "helexa/large"
//!
//! [[endpoints]]
//! name = "openrouter"
//! base_url = "https://openrouter.ai/api/v1"
//! wire_api = "openai-chat"
//! api_key_env = "OPENROUTER_API_KEY"
//! default_model = "anthropic/claude-opus-4"
//!
//! [[endpoints]]
//! name = "lmstudio"
//! base_url = "http://localhost:1234/v1"
//! wire_api = "openai-chat"
//! default_model = "auto"
//! ```

use anyhow::{Context, anyhow};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use url::Url;

const DEFAULT_BASE_URL: &str = "http://hanzalova.internal:31313/v1";
const DEFAULT_MODEL: &str = "helexa/large";
const DEFAULT_ENDPOINT_NAME: &str = "default";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Name of the endpoint used when a request doesn't pick one
    /// explicitly. Must reference an entry in `endpoints`. Defaults to
    /// the first endpoint declared if unset.
    #[serde(default)]
    pub default_endpoint: Option<String>,
    /// Per-endpoint configuration. At least one entry is required.
    #[serde(default)]
    pub endpoints: Vec<EndpointConfig>,
    /// Optional path to a system-prompt file. When unset, the built-in
    /// default prompt from `prompt.rs` is used.
    #[serde(default)]
    pub system_prompt_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Short identifier used in `endpoint:model` routing and in logs.
    pub name: String,
    /// Base URL of the OpenAI-compatible API. Must include the `/v1`
    /// (or equivalent) suffix — paths like `chat/completions` and
    /// `models` are joined onto this.
    pub base_url: Url,
    /// Wire protocol the endpoint speaks. Phase 1 supports
    /// [`WireApi::OpenAiChat`] only; `openai-responses` and
    /// `anthropic-messages` land later behind their own providers.
    #[serde(default)]
    pub wire_api: WireApi,
    /// Model to use when the client hasn't picked one via
    /// `session/set_model`.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Static API key to send as `Authorization: Bearer …`. Prefer
    /// `api_key_env` for anything sensitive — keys in plain TOML are a
    /// liability.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Env var name to read for the API key. Resolved at startup so a
    /// missing env var yields a clear error rather than silent
    /// unauthenticated calls.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WireApi {
    /// `POST {base}/chat/completions` returning OpenAI-format SSE.
    /// Compatible with cortex, LM Studio, Ollama (compat mode),
    /// OpenRouter, OpenAI itself.
    #[default]
    #[serde(rename = "openai-chat")]
    OpenAiChat,
    /// `POST {base}/responses` — OpenAI's newer Responses API. Not
    /// implemented yet; the variant is reserved so endpoint configs
    /// can be authored ahead of provider support.
    #[serde(rename = "openai-responses")]
    OpenAiResponses,
    /// `POST {base}/messages` — Anthropic format. Reserved.
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
}

impl EndpointConfig {
    /// Resolve the API key from `api_key` (literal) or `api_key_env`
    /// (env-var lookup). Returns `Ok(None)` when neither is set;
    /// `Err` when `api_key_env` references a missing variable.
    pub fn resolve_api_key(&self) -> anyhow::Result<Option<String>> {
        if let Some(literal) = &self.api_key {
            return Ok(Some(literal.clone()));
        }
        if let Some(var) = &self.api_key_env {
            return Ok(Some(std::env::var(var).with_context(|| {
                format!(
                    "endpoint '{}' references missing env var {}",
                    self.name, var
                )
            })?));
        }
        Ok(None)
    }

    /// `{base_url}/chat/completions`.
    pub fn chat_completions_url(&self) -> Url {
        join_segments(&self.base_url, &["chat", "completions"])
    }

    /// `{base_url}/models`. Called from `Provider::list_models`, which
    /// Stage 4 wires into the model-picker dropdown; until then it's
    /// reachable code with no in-tree callers.
    #[allow(dead_code)]
    pub fn models_url(&self) -> Url {
        join_segments(&self.base_url, &["models"])
    }
}

impl Config {
    /// Load from TOML at the standard config path, or build from env
    /// vars if no file exists. Env-fallback yields a single endpoint
    /// named `"default"`.
    pub fn load() -> anyhow::Result<Self> {
        let path = config_path();
        if let Some(path) = &path
            && path.exists()
        {
            return Self::from_file(path);
        }
        Self::from_env()
    }

    /// Single-endpoint config constructed from `HELEXA_ACP_BASE_URL`,
    /// `HELEXA_ACP_MODEL`, `HELEXA_ACP_API_KEY`,
    /// `HELEXA_ACP_SYSTEM_PROMPT_PATH`.
    pub fn from_env() -> anyhow::Result<Self> {
        let base_url = std::env::var("HELEXA_ACP_BASE_URL")
            .ok()
            .unwrap_or_else(|| DEFAULT_BASE_URL.into());
        let base_url = Url::parse(&base_url)
            .with_context(|| format!("HELEXA_ACP_BASE_URL is not a valid URL ({base_url})"))?;
        let default_model = std::env::var("HELEXA_ACP_MODEL")
            .ok()
            .unwrap_or_else(|| DEFAULT_MODEL.into());
        let api_key = std::env::var("HELEXA_ACP_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let system_prompt_path = std::env::var("HELEXA_ACP_SYSTEM_PROMPT_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        Ok(Self {
            default_endpoint: Some(DEFAULT_ENDPOINT_NAME.into()),
            endpoints: vec![EndpointConfig {
                name: DEFAULT_ENDPOINT_NAME.into(),
                base_url,
                wire_api: WireApi::OpenAiChat,
                default_model: Some(default_model),
                api_key,
                api_key_env: None,
            }],
            system_prompt_path,
        })
    }

    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        let mut cfg: Self =
            toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&mut self) -> anyhow::Result<()> {
        if self.endpoints.is_empty() {
            return Err(anyhow!("config has no [[endpoints]] entries"));
        }
        for (i, ep) in self.endpoints.iter().enumerate() {
            if ep.name.is_empty() {
                return Err(anyhow!("endpoints[{i}] has empty name"));
            }
            if ep.name.contains(':') {
                return Err(anyhow!(
                    "endpoints[{i}].name '{}' contains ':' which would clash \
                     with the endpoint:model selector syntax",
                    ep.name
                ));
            }
        }
        // Pick a default endpoint if none was named.
        if self.default_endpoint.is_none() {
            self.default_endpoint = Some(self.endpoints[0].name.clone());
        }
        let default_name = self.default_endpoint.as_deref().unwrap();
        if !self.endpoints.iter().any(|e| e.name == default_name) {
            return Err(anyhow!(
                "default_endpoint '{default_name}' is not declared in [[endpoints]]"
            ));
        }
        Ok(())
    }

    /// Look up an endpoint by name. Returns `None` if not configured.
    pub fn endpoint(&self, name: &str) -> Option<&EndpointConfig> {
        self.endpoints.iter().find(|e| e.name == name)
    }

    /// The default endpoint (guaranteed to exist after `validate`).
    pub fn default_endpoint(&self) -> &EndpointConfig {
        let name = self
            .default_endpoint
            .as_deref()
            .expect("default_endpoint set by validate");
        self.endpoint(name)
            .expect("default_endpoint resolves after validate")
    }
}

/// Parse an ACP-side `model` field into (endpoint name, raw model id).
///
/// `helexa:helexa/large` → (`Some("helexa")`, `"helexa/large"`).
/// `helexa/large` → (`None`, `"helexa/large"`).
///
/// The split happens at the FIRST colon. Model ids commonly contain
/// `/` (HuggingFace style) but rarely `:`; if a model id ever does, the
/// user can quote-prefix with the default endpoint name.
pub fn parse_model_selector(input: &str) -> (Option<&str>, &str) {
    match input.split_once(':') {
        Some((endpoint, model)) if !endpoint.is_empty() && !model.is_empty() => {
            (Some(endpoint), model)
        }
        _ => (None, input),
    }
}

fn config_path() -> Option<PathBuf> {
    if let Ok(override_path) = std::env::var("HELEXA_ACP_CONFIG_PATH") {
        return Some(PathBuf::from(override_path));
    }
    let xdg = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty());
    let base = xdg.map(PathBuf::from).or_else(|| {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".config"))
    })?;
    Some(base.join("helexa-acp").join("config.toml"))
}

fn join_segments(base: &Url, segments: &[&str]) -> Url {
    let mut out = base.clone();
    if let Ok(mut path) = out.path_segments_mut() {
        path.pop_if_empty().extend(segments.iter().copied());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_join_handles_trailing_slash() {
        let ep = EndpointConfig {
            name: "x".into(),
            base_url: Url::parse("http://h.internal:31313/v1").unwrap(),
            wire_api: WireApi::OpenAiChat,
            default_model: None,
            api_key: None,
            api_key_env: None,
        };
        assert_eq!(
            ep.chat_completions_url().as_str(),
            "http://h.internal:31313/v1/chat/completions"
        );
        assert_eq!(
            ep.models_url().as_str(),
            "http://h.internal:31313/v1/models"
        );
    }

    #[test]
    fn parses_model_selector() {
        assert_eq!(
            parse_model_selector("helexa:helexa/large"),
            (Some("helexa"), "helexa/large")
        );
        assert_eq!(parse_model_selector("helexa/large"), (None, "helexa/large"));
        assert_eq!(parse_model_selector("gpt-5"), (None, "gpt-5"));
        // Edge case: a leading colon → no endpoint.
        assert_eq!(parse_model_selector(":gpt-5"), (None, ":gpt-5"));
    }

    #[test]
    fn env_fallback_builds_single_endpoint() {
        // Don't actually set env vars (would race with other tests);
        // just confirm the default path constructs cleanly.
        unsafe {
            std::env::remove_var("HELEXA_ACP_BASE_URL");
            std::env::remove_var("HELEXA_ACP_MODEL");
            std::env::remove_var("HELEXA_ACP_API_KEY");
        }
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.endpoints.len(), 1);
        assert_eq!(cfg.endpoints[0].name, "default");
        assert_eq!(cfg.endpoints[0].base_url.as_str(), DEFAULT_BASE_URL);
        assert_eq!(
            cfg.endpoints[0].default_model.as_deref(),
            Some(DEFAULT_MODEL)
        );
    }

    #[test]
    fn toml_parses_multi_endpoint() {
        let toml_text = r#"
            default_endpoint = "helexa"

            [[endpoints]]
            name = "helexa"
            base_url = "http://hanzalova.internal:31313/v1"
            default_model = "helexa/large"

            [[endpoints]]
            name = "openrouter"
            base_url = "https://openrouter.ai/api/v1"
            wire_api = "openai-chat"
            api_key_env = "OPENROUTER_API_KEY"
            default_model = "anthropic/claude-opus-4"
        "#;
        let mut cfg: Config = toml::from_str(toml_text).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.endpoints.len(), 2);
        assert_eq!(cfg.default_endpoint().name, "helexa");
        assert_eq!(cfg.endpoints[0].wire_api, WireApi::OpenAiChat);
        assert_eq!(
            cfg.endpoints[1].api_key_env.as_deref(),
            Some("OPENROUTER_API_KEY")
        );
    }

    #[test]
    fn validate_rejects_colon_in_endpoint_name() {
        let toml_text = r#"
            [[endpoints]]
            name = "bad:name"
            base_url = "http://x/v1"
        "#;
        let mut cfg: Config = toml::from_str(toml_text).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("clash"));
    }
}
