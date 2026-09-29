//! The decision head that sits on top of the encoder.
//!
//! Mirrors the reference `DecisionModel` (the `laya` SDK's
//! `common.py`):
//!
//! 1. a per-question-type embedding added to every position;
//! 2. `head_layers` × PyTorch `nn.TransformerEncoderLayer` with
//!    `norm_first=True`: pre-norm self-attention through a fused
//!    `in_proj` with bias, then a pre-norm feed-forward whose
//!    activation is **ReLU** — PyTorch's default, which the reference
//!    never overrides;
//! 3. `scorer`, applied at each option's `[MASK]` marker: LayerNorm →
//!    Linear → exact GELU → Linear to one logit;
//! 4. `act_head`, over the first position's hidden state plus four
//!    summary features of the (uncalibrated) answer distribution.

use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};

use super::backbone::scaled_attention;

/// PyTorch's `LayerNorm` default.
const TORCH_LN_EPS: f64 = 1e-5;

struct HeadLayer {
    norm1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, d: usize) -> Result<Self> {
        // `nn.TransformerEncoder` is built with `nhead = max(1, d // 64)`
        // and `dim_feedforward = 4 * d`.
        let n_heads = (d / 64).max(1);
        Ok(Self {
            norm1: candle_nn::layer_norm(d, TORCH_LN_EPS, vb.pp("norm1"))?,
            in_proj: linear_torch_mha_in_proj(d, vb.pp("self_attn"))?,
            out_proj: candle_nn::linear(d, d, vb.pp("self_attn.out_proj"))?,
            norm2: candle_nn::layer_norm(d, TORCH_LN_EPS, vb.pp("norm2"))?,
            linear1: candle_nn::linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: candle_nn::linear(4 * d, d, vb.pp("linear2"))?,
            n_heads,
            head_dim: d / n_heads,
        })
    }

    fn forward(&self, xs: &Tensor, key_mask: &Tensor) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.norm1)?
            .apply(&self.in_proj)?
            .reshape((b, l, 3, self.n_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let ctx = scaled_attention(&q, &k, &v, key_mask, self.head_dim)?
            .transpose(1, 2)?
            .reshape((b, l, d))?;
        let xs = (xs + ctx.apply(&self.out_proj)?)?;
        let ff = xs
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        Ok((xs + ff)?)
    }
}

/// `nn.MultiheadAttention` stores q/k/v as one `in_proj_weight`
/// `(3d, d)` and `in_proj_bias` `(3d)`, rows ordered q then k then v —
/// exactly the layout one fused `Linear` wants.
fn linear_torch_mha_in_proj(d: usize, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get((3 * d, d), "in_proj_weight")?;
    let b = vb.get(3 * d, "in_proj_bias")?;
    Ok(Linear::new(w, Some(b)))
}

pub struct DecisionHead {
    type_emb: Embedding,
    layers: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer_fc: Linear,
    scorer_out: Linear,
    act_fc: Linear,
    act_out: Linear,
}

impl DecisionHead {
    /// `vb` is the checkpoint root; the head's tensors are top-level.
    pub fn load(vb: VarBuilder, d: usize, head_layers: usize, n_act: usize) -> Result<Self> {
        let layers = (0..head_layers)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), d))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            type_emb: candle_nn::embedding(3, d, vb.pp("type_emb"))?,
            layers,
            scorer_norm: candle_nn::layer_norm(d, TORCH_LN_EPS, vb.pp("scorer.0"))?,
            scorer_fc: candle_nn::linear(d, d, vb.pp("scorer.1"))?,
            scorer_out: candle_nn::linear(d, 1, vb.pp("scorer.3"))?,
            act_fc: candle_nn::linear(d + 4, 256, vb.pp("act_head.0"))?,
            act_out: candle_nn::linear(256, n_act, vb.pp("act_head.2"))?,
        })
    }

    /// Type embedding plus the transformer layers. `h` is the encoder
    /// output `(b, l, d)`, `qtype` is `(b,)` u32.
    pub fn contextualise(&self, h: &Tensor, qtype: &Tensor, key_mask: &Tensor) -> Result<Tensor> {
        let mut h = h.broadcast_add(&qtype.apply(&self.type_emb)?.unsqueeze(1)?)?;
        for layer in &self.layers {
            h = layer.forward(&h, key_mask)?;
        }
        Ok(h)
    }

    /// Score gathered marker states `(b, k, d)` → `(b, k)` f32 logits.
    pub fn score(&self, markers: &Tensor) -> Result<Tensor> {
        let x = markers
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_fc)?
            .gelu_erf()?
            .apply(&self.scorer_out)?;
        Ok(x.squeeze(2)?.to_dtype(DType::F32)?)
    }

    /// Act logits `(b, n_act)` f32 from the pooled first position
    /// `(b, d)` and the four answer-distribution features `(b, 4)`.
    pub fn act(&self, pooled: &Tensor, feats: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[pooled, &feats.to_dtype(pooled.dtype())?], 1)?;
        let x = self.act_fc.forward(&x)?.gelu_erf()?.apply(&self.act_out)?;
        Ok(x.to_dtype(DType::F32)?)
    }
}
