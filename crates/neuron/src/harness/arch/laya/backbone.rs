//! ModernBERT, the bidirectional encoder under every Laya checkpoint.
//!
//! Follows the structure of `candle_transformers::models::modernbert`
//! (itself a port of transformers' `ModernBertModel`), vendored rather
//! than used as-is because that module only runs correctly in f32:
//!
//! - its attention masks are built in F32 whatever the model dtype, so
//!   an f16/bf16 load fails at the first `broadcast_add`;
//! - it computes the rotary angles `position × inv_freq` in the model
//!   dtype, which in f16 already loses a quarter-radian of phase by
//!   position 500;
//! - it loads each layer's `attn_norm` with `.ok()`, so a weight-prefix
//!   mistake silently yields a model with no pre-attention norms.
//!
//! Here the angles are computed in f32 and only the resulting cos/sin
//! tables are cast; attention scores are softmaxed in f32 against f32
//! masks; and `attn_norm` is required on every layer except layer 0,
//! which ModernBERT builds as an identity.
//!
//! Masked positions use a large *finite* negative rather than `-inf`.
//! A padded query near the end of a short row can find every key in
//! its sliding window padded; with `-inf` that row's softmax is NaN,
//! and NaN in a padded row's value vector poisons the valid rows that
//! give it zero weight (`0 × NaN = NaN`). A finite value makes such a
//! row uniform instead, and valid rows never look at it.

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};

use super::config::{EncoderConfig, LayerKind};

/// Additive mask value for a position that must not be attended.
pub(crate) const MASKED: f32 = -1e9;

/// Precomputed rotary tables for one base, `(max_pos, head_dim / 2)`.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(cfg: &EncoderConfig, theta: f64, dtype: DType, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1.0 / theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), device)?;
        let max_pos = cfg.max_position_embeddings;
        let pos = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = pos.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
    kind: LayerKind,
}

impl Attention {
    fn forward(&self, xs: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.wqkv)?
            .reshape((b, l, 3, self.n_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?.contiguous()?;
        let (q, k) = rotary.apply(&q, &k, l)?;
        let ctx = scaled_attention(&q, &k, &v, mask, self.head_dim)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, d))?;
        Ok(ctx.apply(&self.wo)?)
    }
}

/// `softmax(q kᵀ / √d + mask) v`, scores and softmax in f32.
///
/// `q`, `k`, `v` are `(b, h, l, d)`; `mask` is an f32 additive mask
/// broadcastable to `(b, h, l, l)`.
pub(crate) fn scaled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    head_dim: usize,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let scale = (head_dim as f64).powf(-0.5);
    let scores = (q.matmul(&k.t()?)? * scale)?.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?;
    Ok(probs.to_dtype(dtype)?.matmul(v)?)
}

struct Mlp {
    wi: Linear,
    wo: Linear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let parts = xs.chunk(2, D::Minus1)?;
        (parts[0].gelu_erf()? * &parts[1])?.apply(&self.wo)
    }
}

struct Layer {
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
}

/// The two additive masks one forward needs, both f32.
pub struct Masks {
    /// `(b, 1, 1, l)`: padded keys masked. Global layers and the
    /// decision head use this one.
    pub global: Tensor,
    /// `(b, 1, l, l)`: padded keys, plus keys outside the window.
    pub sliding: Tensor,
}

impl Masks {
    /// Build both masks from per-row valid lengths. Rows are
    /// right-padded: positions `>= lens[i]` of row `i` are padding.
    pub fn new(
        lens: &[usize],
        seq_len: usize,
        half_window: usize,
        device: &Device,
    ) -> Result<Self> {
        let b = lens.len();
        let mut pad = Vec::with_capacity(b * seq_len);
        for &n in lens {
            pad.extend((0..seq_len).map(|j| if j < n { 0f32 } else { MASKED }));
        }
        let global = Tensor::from_vec(pad, (b, 1, 1, seq_len), device)?;
        // Built on the device from two aranges, not on the host: at
        // 1024 tokens a host-built window is a 4 MB upload per forward.
        let pos = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;
        let dist = pos
            .reshape((seq_len, 1))?
            .broadcast_sub(&pos.reshape((1, seq_len))?)?
            .abs()?;
        let outside = dist.gt(half_window as f64)?;
        let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
        let masked = Tensor::full(MASKED, (seq_len, seq_len), device)?;
        let window = outside
            .where_cond(&masked, &zeros)?
            .reshape((1, 1, seq_len, seq_len))?;
        let sliding = global.broadcast_add(&window)?;
        Ok(Self { global, sliding })
    }
}

pub struct ModernBert {
    embeddings: Embedding,
    embed_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    global_rotary: Rotary,
    sliding_rotary: Rotary,
    half_window: usize,
}

impl ModernBert {
    /// Load from a VarBuilder rooted at the encoder's prefix (`encoder`
    /// in a Laya checkpoint, `model` in a stock ModernBERT one).
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let embeddings =
            candle_nn::embedding(cfg.vocab_size, d, vb.pp("embeddings").pp("tok_embeddings"))
                .context("encoder.embeddings.tok_embeddings")?;
        let embed_norm = candle_nn::layer_norm_no_bias(d, cfg.norm_eps, vb.pp("embeddings.norm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, kind) in cfg.layer_kinds.iter().copied().enumerate() {
            let lvb = vb.pp(format!("layers.{i}"));
            // ModernBERT builds layer 0's attn_norm as an identity and
            // saves nothing for it; every other layer must have one.
            let attn_norm = if lvb.contains_tensor("attn_norm.weight") {
                Some(candle_nn::layer_norm_no_bias(
                    d,
                    cfg.norm_eps,
                    lvb.pp("attn_norm"),
                )?)
            } else {
                ensure!(
                    i == 0,
                    "encoder layer {i} has no attn_norm.weight — wrong weight prefix, \
                     or not a ModernBERT checkpoint"
                );
                None
            };
            let attn = Attention {
                wqkv: candle_nn::linear_no_bias(d, 3 * d, lvb.pp("attn.Wqkv"))?,
                wo: candle_nn::linear_no_bias(d, d, lvb.pp("attn.Wo"))?,
                n_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim(),
                kind,
            };
            let mlp_norm = candle_nn::layer_norm_no_bias(d, cfg.norm_eps, lvb.pp("mlp_norm"))?;
            let mlp = Mlp {
                wi: candle_nn::linear_no_bias(d, 2 * cfg.intermediate_size, lvb.pp("mlp.Wi"))?,
                wo: candle_nn::linear_no_bias(cfg.intermediate_size, d, lvb.pp("mlp.Wo"))?,
            };
            layers.push(Layer {
                attn_norm,
                attn,
                mlp_norm,
                mlp,
            });
        }
        let final_norm = candle_nn::layer_norm_no_bias(d, cfg.norm_eps, vb.pp("final_norm"))?;
        let device = vb.device();
        let dtype = vb.dtype();
        Ok(Self {
            embeddings,
            embed_norm,
            layers,
            final_norm,
            global_rotary: Rotary::new(cfg, cfg.global_rope_theta, dtype, device)?,
            sliding_rotary: Rotary::new(cfg, cfg.local_rope_theta, dtype, device)?,
            half_window: cfg.local_attention / 2,
        })
    }

    pub fn half_window(&self) -> usize {
        self.half_window
    }

    /// `ids` is `(b, l)` u32; returns the final hidden states `(b, l, d)`.
    pub fn forward(&self, ids: &Tensor, masks: &Masks) -> Result<Tensor> {
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.embed_norm)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => xs.apply(norm)?,
                None => xs.clone(),
            };
            let (rotary, mask) = match layer.attn.kind {
                LayerKind::Global => (&self.global_rotary, &masks.global),
                LayerKind::Sliding => (&self.sliding_rotary, &masks.sliding),
            };
            xs = (&xs + layer.attn.forward(&normed, rotary, mask)?)?;
            let mlp_out = xs.apply(&layer.mlp_norm)?.apply(&layer.mlp)?;
            xs = (xs + mlp_out)?;
        }
        Ok(xs.apply(&self.final_norm)?)
    }
}
