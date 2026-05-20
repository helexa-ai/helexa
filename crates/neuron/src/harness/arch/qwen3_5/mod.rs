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
//! **Single-GPU dense path is real**. Both attention flavours
//! (`full_attention` with the output-gated GQA causal attention and
//! `linear_attention` with the Gated DeltaNet recurrent block) are
//! implemented. The model loads from upstream safetensors via the
//! existing `load_arch_dense` dispatch and runs forward end to end.
//!
//! Numerical correctness vs the reference Python is **not yet
//! validated** — the structural code path is right, weight tensor
//! names match the upstream layout, shapes flow through cleanly, but
//! the Tbilisi probe (and any other downstream test) is the next
//! step. Likely places a bug would surface:
//! - Per-rank vs per-token-position offsets in the recurrent delta
//!   rule (`linear_attn.rs`).
//! - Off-by-one in the conv state continuation across decode steps.
//! - RoPE phase mismatch from MRoPE simplification (we treat the
//!   three position grids as collapsed, which is correct only for
//!   text-only inference).
//!
//! ## Submodules
//!
//! - [`rmsnorm`] — `Qwen3_5RmsNorm` (`(1+w)*x` variant), the
//!   `Qwen3_5RmsNormGated` used after the delta rule, and the
//!   `l2norm` helper.
//! - [`rope`] — text-side rotary embedding (mrope simplified, GLM
//!   rotate-half).
//! - [`mlp`] — SwiGLU MLP (gate/up/down, no bias).
//! - [`full_attn`] — `Qwen3_5Attention` with the output-gate
//!   widening on `q_proj`.
//! - [`linear_attn`] — `GatedDeltaNet` recurrent delta-rule block
//!   (causal depthwise Conv1d → silu → split → L2norm → per-token
//!   delta rule → RMSNormGated → out_proj).
//! - [`decoder`] — `Qwen3_5DecoderLayer` dispatching to one of the
//!   two attention flavours per layer index.
//!
//! ## Open work
//!
//! - **TP variant.** `harness/tp/tp_qwen3_5.rs` is the next step.
//!   Sharding strategy diverges by layer type:
//!   - Full-attention layers: column-parallel q/k/v (including the
//!     gate half of `q_proj`) + row-parallel `o_proj`, mirroring
//!     `tp_qwen3.rs`.
//!   - Linear-attention layers: the recurrent state is per-V-head, so
//!     V-head-dimension sharding works cleanly — split `num_v_heads`
//!     across ranks (`num_v_heads / world_size` per rank), shard
//!     `in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` along
//!     the V-head dim, and row-parallel `out_proj`. The `A_log` /
//!     `dt_bias` per-head params shard with the heads.
//!
//! - **Chunked delta-rule prefill.** `linear_attn.rs` runs the
//!   per-token recurrent path for prefill too — correct but O(L).
//!   Porting `torch_chunk_gated_delta_rule` (chunk_size=64) speeds
//!   prefill substantially with no surface change.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::Embedding;
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;
use serde::Deserialize;
use std::sync::Arc;

pub mod decoder;
pub mod full_attn;
pub mod linear_attn;
pub mod mlp;
pub mod rmsnorm;
pub mod rope;

use decoder::Qwen3_5DecoderLayer;
use rmsnorm::Qwen3_5RmsNorm;
use rope::RotaryEmbedding;

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

    /// Hidden activation (`"silu"` for Qwen3-Next). Used by the MLP
    /// and the linear-attention conv1d.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    // --- Gated DeltaNet (linear-attention) hyperparams -----------------
    /// Per-layer linear-attention V-head count (Qwen3.6-27B: 48).
    /// More V-heads than K-heads is fine — query/key get
    /// `repeat_interleave`'d to match before the delta rule.
    #[serde(default)]
    pub linear_num_value_heads: usize,
    /// Per-layer linear-attention K-head count (Qwen3.6-27B: 16).
    #[serde(default)]
    pub linear_num_key_heads: usize,
    /// Per-head key dimension for the linear-attention path
    /// (Qwen3.6-27B: 128). Separate from `head_dim` which the
    /// full-attention layers use.
    #[serde(default)]
    pub linear_key_head_dim: usize,
    /// Per-head value dimension for the linear-attention path
    /// (Qwen3.6-27B: 128).
    #[serde(default)]
    pub linear_value_head_dim: usize,
    /// Causal Conv1d kernel size used before the delta rule
    /// (Qwen3.6-27B: 4).
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
}

fn default_hidden_act() -> String {
    "silu".into()
}

/// Qwen3-Next base transformer (embedding + decoder stack + final
/// norm). Public so a TP variant in `harness/tp/tp_qwen3_5.rs` can
/// also build on it later — for now only `Qwen3_5ForCausalLM` is the
/// loaded handle.
pub struct Qwen3_5Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    device: Device,
    dtype: DType,
}

impl Qwen3_5Model {
    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();

        let embed_vb = vb.pp("model.embed_tokens");
        let embed_weight = embed_vb
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .with_context(|| format!("load '{}/weight'", embed_vb.prefix()))?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);

        if cfg.layer_types.len() != cfg.num_hidden_layers {
            anyhow::bail!(
                "config.text_config.layer_types must have num_hidden_layers ({}) entries; \
                 got {}",
                cfg.num_hidden_layers,
                cfg.layer_types.len()
            );
        }

        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3_5DecoderLayer::load(
                cfg,
                rotary.clone(),
                i,
                &vb_l.pp(i),
            )?);
        }

        let norm = Qwen3_5RmsNorm::load(&vb.pp("model.norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device,
            dtype,
        })
    }

    pub fn embed_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> candle_core::Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Causal mask only needed for L > 1 prefill; full-attention
        // layers consume it via broadcast_add. Linear-attention layers
        // ignore the mask.
        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

pub struct Qwen3_5ForCausalLM {
    base: Qwen3_5Model,
    lm_head: Linear,
}

impl Qwen3_5ForCausalLM {
    pub fn new(config: Config, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = &config.text_config;
        let base = Qwen3_5Model::load(cfg, &vb)?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(base.embed_weight().clone(), None)
        } else {
            let weight = vb
                .pp("lm_head")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")
                .with_context(|| format!("load '{}/lm_head/weight'", vb.prefix()))?;
            Linear::new(weight, None)
        };
        Ok(Self { base, lm_head })
    }

    /// `input`: token-id tensor of shape `(B, L)`. Returns logits at
    /// the last position, shape `(B, 1, vocab_size)` — same contract
    /// as `qwen3::ModelForCausalLM::forward` so the harness's
    /// `squeeze_to_vocab` helper handles both uniformly.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden = self.base.forward(input, offset)?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
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
