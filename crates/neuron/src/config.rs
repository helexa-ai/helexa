//! Neuron configuration loaded from neuron.toml.

use cortex_core::harness::{HarnessConfig, ModelSpec};
use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Default scheme name applied to bare `org/name` model ids when no
/// `[harness.candle.default_source]` is set. Keeps existing operator
/// configs (which know nothing about schemes) working unchanged.
pub const DEFAULT_SOURCE_SCHEME: &str = "huggingface";

/// Endpoint URL for the default huggingface source, used when no
/// `[harness.candle.sources.huggingface]` is configured.
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub harnesses: Vec<HarnessConfig>,
    /// Per-harness configuration. Currently only `candle` is recognised.
    #[serde(default)]
    pub harness: HarnessSettings,
    /// Models to auto-load when the neuron service activates. Each entry
    /// is loaded sequentially before the HTTP listener binds. A failure
    /// on any single entry logs a warning and proceeds — broken entries
    /// don't prevent the rest of the fleet from starting.
    #[serde(default)]
    pub default_models: Vec<ModelSpec>,
}

/// Settings for individual harness implementations. Each harness owns
/// its own sub-table so users only configure the harnesses they enable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarnessSettings {
    #[serde(default)]
    pub candle: CandleHarnessConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandleHarnessConfig {
    /// HuggingFace cache directory for model weights.
    /// When unset, defers to hf-hub's default (~/.cache/huggingface).
    ///
    /// Retained for back-compat — operators with existing
    /// `hf_cache = "..."` configs continue to work. Treated as the
    /// `huggingface` source's cache_dir when a sources table isn't
    /// provided.
    #[serde(default)]
    pub hf_cache: Option<PathBuf>,

    /// Default source scheme applied to bare `org/name` model ids
    /// (those without an explicit `scheme:` prefix). When unset, falls
    /// back to `DEFAULT_SOURCE_SCHEME` ("huggingface").
    #[serde(default)]
    pub default_source: Option<String>,

    /// Per-scheme source endpoints. Each entry maps a scheme name
    /// (`huggingface`, `helexa`, an operator's mirror tag, …) to its
    /// endpoint URL, optional auth env var, and optional cache
    /// directory.
    ///
    /// When absent or missing the `huggingface` key, the loader
    /// synthesises a `huggingface` entry pointing at
    /// `https://huggingface.co` with `hf_cache` (above) as its
    /// cache_dir. This keeps single-source configs ergonomic.
    #[serde(default)]
    pub sources: HashMap<String, SourceConfig>,

    /// Prefix KV cache across requests (#11). Applies per loaded
    /// model, on architectures that support cache snapshots (qwen3_5).
    #[serde(default)]
    pub prefix_cache: PrefixCacheConfig,
}

/// `[harness.candle.prefix_cache]` settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    /// Master switch. On by default — set `false` to restore the
    /// clear-every-request behaviour.
    #[serde(default = "default_prefix_cache_enabled")]
    pub enabled: bool,
    /// Snapshot byte budget per loaded model, in MiB. Snapshots live
    /// on the model's device, so this comes out of the same VRAM that
    /// serves inference — size it against the device's headroom after
    /// the model weights.
    #[serde(default = "default_prefix_cache_budget_mb")]
    pub budget_mb: u64,
    /// Maximum live snapshots per loaded model, regardless of budget.
    #[serde(default = "default_prefix_cache_max_entries")]
    pub max_entries: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_prefix_cache_enabled(),
            budget_mb: default_prefix_cache_budget_mb(),
            max_entries: default_prefix_cache_max_entries(),
        }
    }
}

fn default_prefix_cache_enabled() -> bool {
    true
}

fn default_prefix_cache_budget_mb() -> u64 {
    1024
}

fn default_prefix_cache_max_entries() -> usize {
    8
}

/// Per-scheme source configuration. Mirrors the shape `hf_hub::ApiBuilder`
/// needs: endpoint URL, optional auth token (read from an env var so
/// secrets stay out of the config file), and optional cache directory
/// disambiguated per source to prevent mirror-vs-canonical collisions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Base URL of the registry. Must speak the HF-compatible wire
    /// format (siblings listing at
    /// `/api/models/{org}/{name}[/revision/{rev}]`, blob fetch at
    /// `/{org}/{name}/resolve/{rev}/{path}`).
    pub endpoint: String,

    /// Environment variable name to read for the bearer token used
    /// against this source. `None` = anonymous. Reading from env
    /// (vs. literal token in the config) keeps secrets out of TOML.
    #[serde(default)]
    pub auth_env: Option<String>,

    /// Cache directory for this source. The hf-hub
    /// `models--{org}--{name}/snapshots/...` tree lives directly
    /// under this path, so distinct sources serving the same
    /// `org/name` cannot collide on disk.
    ///
    /// `None` means "share the harness `hf_cache` directory" — only
    /// safe when the operator has exactly one source configured.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
}

impl CandleHarnessConfig {
    /// Resolve the effective sources map for this config, synthesising
    /// a `huggingface` entry from legacy fields (`hf_cache`) when the
    /// operator hasn't supplied a sources table. Idempotent.
    ///
    /// Returns a fresh map rather than mutating self so the original
    /// (operator-typed) config can still be serialized back to TOML
    /// for diagnostics.
    pub fn effective_sources(&self) -> HashMap<String, SourceConfig> {
        let mut out = self.sources.clone();
        out.entry(DEFAULT_SOURCE_SCHEME.to_string())
            .or_insert_with(|| SourceConfig {
                endpoint: DEFAULT_HF_ENDPOINT.to_string(),
                auth_env: Some("HF_TOKEN".to_string()),
                cache_dir: self.hf_cache.clone(),
            });
        out
    }

    /// Effective default scheme. Falls back to `DEFAULT_SOURCE_SCHEME`
    /// when the operator hasn't pinned one.
    pub fn effective_default_source(&self) -> &str {
        self.default_source
            .as_deref()
            .unwrap_or(DEFAULT_SOURCE_SCHEME)
    }
}

fn default_port() -> u16 {
    13131
}

impl NeuronConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("NEURON_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            port: 13131,
            harnesses: vec![],
            harness: HarnessSettings::default(),
            default_models: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_sources_synthesises_huggingface_when_absent() {
        let cfg = CandleHarnessConfig::default();
        let sources = cfg.effective_sources();
        assert!(sources.contains_key("huggingface"));
        let hf = &sources["huggingface"];
        assert_eq!(hf.endpoint, DEFAULT_HF_ENDPOINT);
        assert_eq!(hf.auth_env.as_deref(), Some("HF_TOKEN"));
        assert!(hf.cache_dir.is_none());
    }

    #[test]
    fn effective_sources_carries_legacy_hf_cache_into_synth_entry() {
        // Existing operator configs only set `hf_cache = "/archive3/..."`
        // — the synth must pick that up so the loader keeps using the
        // operator's storage.
        let cfg = CandleHarnessConfig {
            hf_cache: Some(PathBuf::from("/archive3/llm-cache")),
            ..Default::default()
        };
        let sources = cfg.effective_sources();
        assert_eq!(
            sources["huggingface"].cache_dir.as_deref(),
            Some(Path::new("/archive3/llm-cache"))
        );
    }

    #[test]
    fn effective_sources_preserves_explicit_huggingface_entry() {
        // When an operator types out `[harness.candle.sources.huggingface]`
        // explicitly, we must not clobber it with the synth defaults.
        let mut sources = HashMap::new();
        sources.insert(
            "huggingface".to_string(),
            SourceConfig {
                endpoint: "https://huggingface.example.org".into(),
                auth_env: Some("MY_TOKEN".into()),
                cache_dir: Some(PathBuf::from("/operator-cache")),
            },
        );
        let cfg = CandleHarnessConfig {
            hf_cache: Some(PathBuf::from("/legacy-cache")),
            sources,
            ..Default::default()
        };
        let effective = cfg.effective_sources();
        assert_eq!(
            effective["huggingface"].endpoint,
            "https://huggingface.example.org"
        );
        assert_eq!(
            effective["huggingface"].auth_env.as_deref(),
            Some("MY_TOKEN")
        );
        assert_eq!(
            effective["huggingface"].cache_dir.as_deref(),
            Some(Path::new("/operator-cache"))
        );
    }

    #[test]
    fn effective_sources_includes_helexa_alongside_synth_huggingface() {
        let mut sources = HashMap::new();
        sources.insert(
            "helexa".to_string(),
            SourceConfig {
                endpoint: "https://registry.helexa.ai".into(),
                auth_env: Some("HELEXA_TOKEN".into()),
                cache_dir: Some(PathBuf::from("/archive3/llm-cache/helexa")),
            },
        );
        let cfg = CandleHarnessConfig {
            hf_cache: Some(PathBuf::from("/archive3/llm-cache/huggingface")),
            sources,
            ..Default::default()
        };
        let effective = cfg.effective_sources();
        assert_eq!(effective.len(), 2);
        assert_eq!(effective["helexa"].endpoint, "https://registry.helexa.ai");
        // huggingface still gets synth-derived from legacy hf_cache.
        assert_eq!(
            effective["huggingface"].cache_dir.as_deref(),
            Some(Path::new("/archive3/llm-cache/huggingface"))
        );
    }

    #[test]
    fn effective_default_source_falls_back() {
        let cfg = CandleHarnessConfig::default();
        assert_eq!(cfg.effective_default_source(), DEFAULT_SOURCE_SCHEME);
    }

    #[test]
    fn effective_default_source_honours_explicit() {
        let cfg = CandleHarnessConfig {
            default_source: Some("helexa".into()),
            ..Default::default()
        };
        assert_eq!(cfg.effective_default_source(), "helexa");
    }
}
