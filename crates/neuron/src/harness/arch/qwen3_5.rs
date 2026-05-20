//! Qwen3-Next (`model_type = "qwen3_5"`) architecture — Qwen3.6's
//! upstream architecture revision.
//!
//! ## Naming
//!
//! The model release this targets is `Qwen/Qwen3.6-*` but the
//! architecture name in HuggingFace's `config.json` is `qwen3_5`.
//! mistralrs calls the same architecture `qwen3_next`; that label
//! ages poorly the next time Qwen ship a new arch, so we key on the
//! canonical `qwen3_5` from the model's own config.
//!
//! ## Status
//!
//! **Scaffold only.** `Config` deserialisation is real (so the dispatch
//! in `candle.rs::load_arch_dense` can route based on `model_type`
//! and the operator's diagnostic surfaces "qwen3_5" in the supported
//! set); the actual forward pass is `unimplemented!()`. Filling this
//! in is the substantive Stage 8c work.
//!
//! ## What the architecture needs (open work)
//!
//! Confirmed from `Qwen/Qwen3.6-27B/config.json`:
//! - Real hyperparams nested under `text_config: {...}`. The
//!   architecture is text-side; the multimodal vision tower is
//!   separate (`image_token_id`, `language_model_only=false`).
//! - `hidden_size: 5120`, `head_dim: 256`, `intermediate_size: 17408`,
//!   `num_attention_heads`, `num_key_value_heads`, etc. — bigger
//!   head_dim than plain Qwen3.
//! - `attn_output_gate: true` — a sigmoid gate multiplied into the
//!   attention output before the projection. ~10 LoC addition vs the
//!   plain Qwen3 attention.
//! - `layer_types: ["linear_attention", "linear_attention",
//!   "linear_attention", "full_attention", ...]` with
//!   `full_attention_interval: 4` — every 4th layer is full
//!   attention, the rest are linear-attention. The full-attention
//!   layers shape like a Qwen3 attention; the linear-attention
//!   layers are the hard part.
//!
//! ## Linear-attention layer
//!
//! Candle has nothing we can reuse — has to be written against the
//! reference Python in the Qwen3-Next HF repo. Likely Lightning
//! Attention-2 (state-space-ish recurrence) given the
//! `linear_attention` tag and Qwen3's prior `qwen3-omni` work. Needs:
//! - A persistent recurrent state per layer (replaces the explicit
//!   KV cache for full attention).
//! - Per-token update + readout primitives, fused if possible.
//! - Numerical-correctness validation against the Python reference
//!   on a fixed prompt before trusting any output downstream.
//!
//! ## TP-2 (the immediate motivator)
//!
//! Beast's 2x RTX 5090 needs tensor-parallel to fit Qwen3.6-27B.
//! TP-aware analogue lives at `harness/tp/tp_qwen3_5.rs` (not yet
//! created — added alongside the dense impl). Sharding strategy
//! diverges by layer type:
//! - Full-attention layers: column-parallel q/k/v + row-parallel o,
//!   same as `tp_qwen3.rs`. With `attn_output_gate`, the gate weight
//!   is also column-parallel (one gate scalar per head).
//! - Linear-attention layers: the recurrent state is per-token, not
//!   per-head, so head-dim sharding doesn't apply. Options are
//!   (a) replicate the linear-attention layers across ranks (cheap
//!   but wastes ~half the per-rank VRAM since 3 of every 4 layers
//!   replicate), or (b) shard along the recurrent-state dimension
//!   if the formulation allows. Decision deferred until the linear
//!   attention is actually implemented and profiled.

use anyhow::Result;
use candle_core::Tensor;
use serde::Deserialize;

/// `model_type` we deserialise from `config.json`. Const so the
/// dispatch in `candle.rs::load_arch_dense` can pattern-match without
/// magic strings.
pub const MODEL_TYPE: &str = "qwen3_5";

/// Top-level shape of Qwen3-Next's `config.json`. The real
/// hyperparameters live in `text_config`; the rest is multimodal /
/// tokeniser glue we don't need for the language-model forward.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Always `"qwen3_5"` for this architecture. Kept on the struct
    /// so the (eventual) dispatch / logging code can show it without
    /// re-parsing the JSON.
    pub model_type: String,
    /// The text-side hyperparameters. Everything we actually need.
    pub text_config: TextConfig,
}

/// Inner config (the `text_config` block). Mirrors the Qwen3 layout
/// but with the extras Qwen3-Next adds (`attn_output_gate`,
/// `layer_types`, `full_attention_interval`, larger `head_dim`).
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// New in Qwen3-Next: a sigmoid gate multiplied into the attention
    /// output before the o_proj. The Python reference applies it
    /// pointwise after softmax+matmul.
    #[serde(default)]
    pub attn_output_gate: bool,

    /// One entry per decoder layer; values are `"full_attention"` or
    /// `"linear_attention"`. Length must equal `num_hidden_layers`.
    /// `full_attention_interval` is a derived hint (every 4th layer
    /// by default) — `layer_types` is authoritative.
    #[serde(default)]
    pub layer_types: Vec<String>,

    /// Hint for the layer-type pattern (defaults to 4). Kept for
    /// logging / validation; the forward dispatches on `layer_types`.
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
}

/// Stub model. Fields are intentionally empty — filling in the
/// concrete architecture is the substantive Stage 8c work. The struct
/// exists so the `ModelArch::Qwen3_5Dense(_)` variant has a payload
/// and dispatch wiring compiles end-to-end.
///
/// To extend: add embed_tokens, decoder layers, final norm, and
/// lm_head fields here; implement `new`, `forward`, `clear_kv_cache`
/// in terms of them. Mirror the layout of `qwen3_dense::ModelForCausalLM`
/// (in candle-transformers) as a starting point.
pub struct Qwen3_5ForCausalLM {
    #[allow(dead_code)]
    config: Config,
}

impl Qwen3_5ForCausalLM {
    pub fn new(config: Config, _vb: candle_nn::VarBuilder) -> Result<Self> {
        // TODO(stage-8c): build embed_tokens, decoder layers (dispatching
        // on layer_types), final RmsNorm, lm_head from the VarBuilder.
        // For now we accept the construction so the load path can be
        // exercised end-to-end (config parse + safetensors mmap), and
        // bail at forward time with a clear marker.
        Ok(Self { config })
    }

    pub fn forward(&mut self, _input: &Tensor, _offset: usize) -> Result<Tensor> {
        anyhow::bail!(
            "Qwen3-Next ({}) forward not implemented yet (Stage 8c, TP-2 motivator)",
            self.config.model_type
        )
    }

    pub fn clear_kv_cache(&mut self) {
        // No-op for the stub. The real impl needs a `clear_kv_cache`
        // that resets the per-layer KV cache (full-attention layers)
        // and the per-layer recurrent state (linear-attention layers).
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms we can deserialise the real upstream config shape.
    /// Sample taken from `Qwen/Qwen3.6-27B/config.json`, trimmed to
    /// the fields the architecture cares about.
    #[test]
    fn config_deserialises_the_real_qwen3_6_shape() {
        let raw = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "language_model_only": false,
            "text_config": {
                "vocab_size": 248064,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "max_position_embeddings": 32768,
                "rope_theta": 5000000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "layer_types": [
                    "linear_attention", "linear_attention",
                    "linear_attention", "full_attention"
                ]
            }
        }"#;
        let cfg: Config = serde_json::from_str(raw).expect("parse Qwen3.6 config");
        assert_eq!(cfg.model_type, "qwen3_5");
        assert_eq!(cfg.text_config.hidden_size, 5120);
        assert_eq!(cfg.text_config.head_dim, 256);
        assert!(cfg.text_config.attn_output_gate);
        assert_eq!(cfg.text_config.full_attention_interval, Some(4));
        assert_eq!(cfg.text_config.layer_types.len(), 4);
    }
}
