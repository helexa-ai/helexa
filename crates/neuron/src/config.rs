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

    /// Self-derived context/token limits (#67). The neuron computes the
    /// most-efficient `limit{context,input,output}` that still allows
    /// coherent agentic performance from model architecture + live free
    /// VRAM + a self-measured throughput ceiling, advertises it on
    /// `/models`, and enforces it. These knobs tune that derivation.
    #[serde(default)]
    pub context_limit: ContextLimitConfig,

    /// Admission control (#53): bounds the per-model wait queue so a busy
    /// model returns a fast, retryable `429`/`503` instead of stalling new
    /// requests until their client times out.
    #[serde(default)]
    pub admission: AdmissionConfig,
}

/// `[harness.candle.admission]` settings (#53).
///
/// Inference is batch-1, so `max_in_flight` is 1 in practice; the queue
/// (`max_queue_depth`) absorbs short bursts, and `max_wait_secs` caps how
/// long a queued request waits before it's refused with backpressure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionConfig {
    /// Concurrent running requests per model. Batch-1 inference → 1.
    #[serde(default = "default_admission_max_in_flight")]
    pub max_in_flight: usize,
    /// Queued (waiting) requests allowed beyond the in-flight one. The
    /// `(max_in_flight + max_queue_depth + 1)`-th request is refused
    /// immediately with `429`/`503` + `Retry-After`.
    #[serde(default = "default_admission_max_queue_depth")]
    pub max_queue_depth: usize,
    /// Maximum seconds a queued request waits for the in-flight slot before
    /// it is refused (turns the old ~300s client-side hang into a fast,
    /// honest signal).
    #[serde(default = "default_admission_max_wait_secs")]
    pub max_wait_secs: u64,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            max_in_flight: default_admission_max_in_flight(),
            max_queue_depth: default_admission_max_queue_depth(),
            max_wait_secs: default_admission_max_wait_secs(),
        }
    }
}

fn default_admission_max_in_flight() -> usize {
    1
}

fn default_admission_max_queue_depth() -> usize {
    8
}

fn default_admission_max_wait_secs() -> u64 {
    30
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

/// `[harness.candle.context_limit]` settings (#67).
///
/// The derived limit is `context = min(max_position_embeddings,
/// vram_ceiling, throughput_ceiling)`, then `input = context −
/// output_reserve`. `vram_ceiling` and `throughput_ceiling` read live
/// state, so the advertised/enforced limit tracks the resident model and
/// rises automatically as efficiency work (e.g. prefix caching, #11)
/// frees headroom or speeds prefill — no operator action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLimitConfig {
    /// Master switch. On by default — set `false` to fall back to the
    /// static `NEURON_MAX_PROMPT_TOKENS` cap with no advertised limit.
    #[serde(default = "default_context_limit_enabled")]
    pub enabled: bool,

    /// Coherence target: the longest prefill-per-turn latency (seconds)
    /// considered acceptable agentic performance. The throughput ceiling
    /// is `target_prefill_latency_secs × measured_prefill_tok_per_sec`.
    /// Raise it once cross-request prefix caching (#11) makes long
    /// contexts cheap to re-prefill.
    #[serde(default = "default_target_prefill_latency_secs")]
    pub target_prefill_latency_secs: f64,

    /// Cold-start prefill speed (tokens/sec) used for the throughput
    /// ceiling until the model has served enough requests to measure its
    /// own rate. A conservative estimate; the live EMA supersedes it.
    #[serde(default = "default_bootstrap_prefill_tok_per_sec")]
    pub bootstrap_prefill_tok_per_sec: f64,

    /// VRAM (MiB) reserved per card for prefill activations on top of the
    /// resident weights and the KV cache, before computing the VRAM
    /// context ceiling.
    #[serde(default = "default_activation_headroom_mb")]
    pub activation_headroom_mb: u64,

    /// Free-VRAM floor (MiB) kept available per card — the VRAM ceiling
    /// leaves at least this much unused. Mirrors `NEURON_MIN_FREE_VRAM_MB`.
    #[serde(default = "default_context_min_free_floor_mb")]
    pub min_free_floor_mb: u64,

    /// Generation reserve (tokens) left below the context wall:
    /// `input = context − output_reserve_tokens`. Defaults to neuron's
    /// default `max_tokens`.
    #[serde(default = "default_output_reserve_tokens")]
    pub output_reserve_tokens: usize,
}

impl Default for ContextLimitConfig {
    fn default() -> Self {
        Self {
            enabled: default_context_limit_enabled(),
            target_prefill_latency_secs: default_target_prefill_latency_secs(),
            bootstrap_prefill_tok_per_sec: default_bootstrap_prefill_tok_per_sec(),
            activation_headroom_mb: default_activation_headroom_mb(),
            min_free_floor_mb: default_context_min_free_floor_mb(),
            output_reserve_tokens: default_output_reserve_tokens(),
        }
    }
}

fn default_context_limit_enabled() -> bool {
    true
}

fn default_target_prefill_latency_secs() -> f64 {
    // ~2 min/turn is the coherence wall observed pre-#11 on beast
    // (the issue's worked example). Raisable once prefix caching lands.
    120.0
}

fn default_bootstrap_prefill_tok_per_sec() -> f64 {
    // beast Qwen3.6-27B TP=2 measured ~850 tok/s prefill; a conservative
    // floor so the cold-start ceiling isn't wildly optimistic.
    800.0
}

fn default_activation_headroom_mb() -> u64 {
    2048
}

fn default_context_min_free_floor_mb() -> u64 {
    1500
}

fn default_output_reserve_tokens() -> usize {
    8192
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
