//! Harness trait and supporting types for inference engine management.
//!
//! Defined in cortex-core so both cortex (control plane) and neuron
//! (node plane) share the type definitions. neuron provides the
//! runtime implementations.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Configuration for a harness instance on a neuron.
///
/// All current harnesses are in-process (candle); per-harness tuning
/// (cache paths, device policies, etc.) lives in dedicated config
/// blocks rather than on this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub name: String,
}

/// Health status of a harness process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessHealth {
    pub name: String,
    pub running: bool,
    pub uptime_secs: Option<u64>,
}

/// Specification for loading a model through a harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub harness: String,
    pub quant: Option<String>,
    pub tensor_parallel: Option<u32>,
    pub devices: Option<Vec<u32>>,
}

/// Per-model token budget advertised by the catalogue or neuron.
///
/// `context` is the hard wall (the served max-seq-len).  `input` is the
/// compaction trigger — when set, opencode treats it as "usable context =
/// input − reserved".  When omitted, clients fall back to `context − output`.
/// `output` is the maximum number of generation tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLimit {
    /// Hard wall — served max-seq-len in tokens.
    pub context: usize,
    /// Compaction trigger / usable input budget.  When absent clients fall
    /// back to `context − output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<usize>,
    /// Maximum number of generation tokens.
    pub output: usize,
}

/// Operator-set pricing in USD per 1M tokens.
///
/// Self-hosted deployments typically leave both at `0.0`.  Cache fields are
/// optional — set when the backend supports a prefix-cache discount tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    /// USD per 1M input (prompt) tokens.
    #[serde(default)]
    pub input: f64,
    /// USD per 1M output (completion) tokens.
    #[serde(default)]
    pub output: f64,
    /// USD per 1M cache-hit tokens (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// USD per 1M cache-write tokens (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

/// A model as reported by a harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub harness: String,
    pub status: String,
    pub devices: Vec<u32>,
    pub vram_used_mb: Option<u64>,
    /// Modalities this loaded model supports. Today: `["text"]` for
    /// text-only checkpoints, `["text", "vision"]` for vision-capable
    /// ones (Stage B7). Clients like litellm / agent0 can gate
    /// `image_url` submission on the advertised set.
    ///
    /// Optional in the wire format so older clients that don't read
    /// it stay compatible. Default-empty for absent/older data, which
    /// callers can interpret as "text".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,

    // ── Enrichment (issue #62) ────────────────────────────────
    /// Token budget advertised by the catalogue or discovered at load time.
    /// `None` when neither the catalogue nor the loaded model can provide it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<ModelLimit>,
    /// Operator-set pricing in USD per 1M tokens (0.0 = free/self-hosted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// `true` when the model's tokenizer contains recognised tool-call
    /// marker tokens (`<tool_call>` / `<\/tool_call>` convention).
    #[serde(default)]
    pub tool_call: bool,
    /// `true` when the model's tokenizer contains recognised reasoning
    /// marker tokens (`<think>` / `<\/think>` or similar).
    #[serde(default)]
    pub reasoning: bool,
}

/// What an inference harness must do, from neuron's perspective.
///
/// All current harnesses are in-process — they share neuron's address
/// space and lifecycle. `start`/`stop` therefore default to no-ops; a
/// future process-supervising harness would override them.
#[async_trait]
pub trait Harness: Send + Sync {
    /// Human-readable name (e.g. "candle").
    fn name(&self) -> &str;

    /// Start the harness. Default no-op for in-process harnesses.
    async fn start(&self, _config: &HarnessConfig) -> Result<()> {
        Ok(())
    }

    /// Stop the harness. Default no-op for in-process harnesses.
    async fn stop(&self) -> Result<()> {
        Ok(())
    }

    /// Health check. Returns the harness process status.
    async fn health(&self) -> HarnessHealth;

    /// List models the harness knows about (loaded + unloaded).
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Load a model with the given spec (quant, TP, device assignment).
    async fn load_model(&self, spec: &ModelSpec) -> Result<()>;

    /// Unload a model, freeing device memory.
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// Return the URL where inference requests for this model should
    /// be sent. None if the model is not loaded.
    async fn inference_endpoint(&self, model_id: &str) -> Option<String>;
}
