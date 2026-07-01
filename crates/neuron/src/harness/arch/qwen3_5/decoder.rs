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
use super::moe::Qwen3_5MoeBlock;
use super::rmsnorm::Qwen3_5RmsNorm;
use super::rope::RotaryEmbedding;
use super::snapshot::LayerKvSnapshot;

/// One of the two attention flavours sitting in a decoder layer's
/// attention slot. Full-attention layers need the rotary table and
/// take an attention mask; linear-attention layers carry their own
/// recurrent state and ignore the mask.
enum AttentionKind {
    Full(Qwen3_5Attention),
    Linear(GatedDeltaNet),
}

/// The FFN slot: dense SwiGLU (Qwen3.6) or the high-sparsity MoE block
/// (qwen3_next 80B-A3B family, #92), selected per layer by
/// [`TextConfig::layer_uses_moe`].
enum MlpKind {
    Dense(Qwen3_5MLP),
    Moe(Qwen3_5MoeBlock),
}

impl Module for MlpKind {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            MlpKind::Dense(mlp) => mlp.forward(x),
            MlpKind::Moe(moe) => moe.forward(x),
        }
    }
}

pub struct Qwen3_5DecoderLayer {
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
    mlp: MlpKind,
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

        let mlp = if cfg.layer_uses_moe(layer_idx) {
            MlpKind::Moe(Qwen3_5MoeBlock::load(cfg, &vb.pp("mlp"))?)
        } else {
            MlpKind::Dense(Qwen3_5MLP::load(cfg, &vb.pp("mlp"))?)
        };
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

    /// Capture this layer's cache state for a prefix snapshot.
    pub fn snapshot_kv(&self) -> candle_core::Result<LayerKvSnapshot> {
        Ok(match &self.attention {
            AttentionKind::Full(attn) => LayerKvSnapshot::Full(attn.snapshot_kv()),
            AttentionKind::Linear(net) => {
                let (conv_state, recurrent_state) = net.snapshot_state()?;
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                }
            }
        })
    }

    /// Replace this layer's cache state from a snapshot. The snapshot
    /// variant must match the layer's attention kind — a mismatch
    /// means the snapshot came from a different model.
    pub fn restore_kv(&mut self, snap: &LayerKvSnapshot) -> candle_core::Result<()> {
        match (&mut self.attention, snap) {
            (AttentionKind::Full(attn), LayerKvSnapshot::Full(kv)) => attn.restore_kv(kv.as_ref()),
            (
                AttentionKind::Linear(net),
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => net.restore_state(conv_state.as_ref(), recurrent_state.as_ref()),
            _ => candle_core::bail!(
                "restore_kv: snapshot layer kind does not match this layer's attention kind"
            ),
        }
    }
}
