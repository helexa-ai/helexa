//! Qwen3-Next decoder layer.
//!
//! Standard pre-norm transformer block (LN → attention → residual →
//! LN → MLP → residual) where the attention slot dispatches on the
//! per-layer `layer_types[i]` value in the config:
//!
//! - `"full_attention"` → [`Qwen3_5Attention`] (GQA causal + output
//!   gate + RoPE + KV cache).
//! - `"linear_attention"` → [`GatedDeltaNet`] (recurrent delta rule +
//!   causal conv + per-head state).
//!
//! In Qwen3.6-27B every 4th layer is full_attention; the rest are
//! linear_attention. `full_attention_interval` in the config is a
//! hint; `layer_types` is authoritative.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use std::sync::Arc;

use super::TextConfig;
use super::full_attn::Qwen3_5Attention;
use super::linear_attn::GatedDeltaNet;
use super::mlp::Qwen3_5MLP;
use super::rmsnorm::Qwen3_5RmsNorm;
use super::rope::RotaryEmbedding;

/// One of the two attention flavours sitting in a decoder layer's
/// attention slot. Full-attention layers need the rotary table and
/// take an attention mask; linear-attention layers carry their own
/// recurrent state and ignore the mask.
enum AttentionKind {
    Full(Qwen3_5Attention),
    Linear(GatedDeltaNet),
}

pub struct Qwen3_5DecoderLayer {
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
    mlp: Qwen3_5MLP,
    attention: AttentionKind,
}

impl Qwen3_5DecoderLayer {
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .map(String::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "layer_types[{layer_idx}] missing (have {} entries)",
                    cfg.layer_types.len()
                )
            })?;

        let attention = match layer_type {
            "full_attention" => {
                AttentionKind::Full(Qwen3_5Attention::load(cfg, rotary, &vb.pp("self_attn"))?)
            }
            "linear_attention" => {
                AttentionKind::Linear(GatedDeltaNet::load(cfg, &vb.pp("linear_attn"))?)
            }
            other => anyhow::bail!(
                "unknown layer_type '{other}' for layer {layer_idx} (expected \
                 'full_attention' or 'linear_attention')"
            ),
        };

        let mlp = Qwen3_5MLP::load(cfg, &vb.pp("mlp"))?;
        let input_layernorm =
            Qwen3_5RmsNorm::load(&vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RmsNorm::load(
            &vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            attention,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attention {
            AttentionKind::Full(attn) => attn.forward(&h, attn_mask, cos, sin)?,
            // Linear attention ignores attn_mask + rope; its causal
            // structure is baked into the recurrent state lifecycle.
            AttentionKind::Linear(net) => net.forward(&h)?,
        };
        let x = (x + attn_out)?;
        let h2 = self.post_attention_layernorm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.attention {
            AttentionKind::Full(attn) => attn.clear_kv_cache(),
            AttentionKind::Linear(net) => net.clear_kv_cache(),
        }
    }
}
