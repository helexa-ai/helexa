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
//! Precision follows PyTorch autocast, which is how the reference runs on
//! a GPU: the linear layers (and so the attention inputs) run in the
//! load dtype, while the residual stream, the embeddings and every
//! LayerNorm stay in f32. Casting the residual stream to bf16 as well
//! drifts the released English checkpoint's logits by ~0.25 over its 28
//! layers; keeping it in f32 is what autocast does and costs little.
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

/// Apply `lin` in its weight's dtype, casting `x` in if needed. The
/// result stays in that dtype; callers cast back to f32 where it joins
/// the residual stream.
pub(crate) fn project(lin: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    x.to_dtype(lin.weight().dtype())?.apply(lin)
}

/// A bias-free LayerNorm that takes candle's fused kernel.
///
/// `candle_nn::LayerNorm` dispatches to the fused `layer_norm` op only
/// when it has a bias; without one it falls back to six elementwise
/// kernels (mean, subtract, square, mean, divide, scale), which was a
/// quarter of the encoder's GPU time. ModernBERT's norms have no bias,
/// so they get an explicit zero one: same result, one kernel.
fn fused_layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = weight.zeros_like()?;
    Ok(LayerNorm::new(weight, bias, eps))
}

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
    fn forward(&self, xs: &Tensor, rotary: &Rotary, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let qkv = project(&self.wqkv, xs)?
            .reshape((b, l, 3, self.n_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        // Rotary in the projection dtype, as under autocast; the scores
        // are then taken in f32 (see `scaled_attention`).
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?.contiguous()?;
        let (q, k) = rotary.apply(&q, &k, l)?;
        let ctx = scaled_attention(&q, &k, &v, mask, self.head_dim)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, d))?;
        Ok(ctx.apply(&self.wo)?.to_dtype(DType::F32)?)
    }
}

/// A batch packed without padding for the flash-attention path: every
/// row's tokens back to back, `(total, …)`. Linear layers then do no work
/// for padding, and flash varlen keeps rows from attending to each other.
#[cfg(feature = "flash-attn")]
pub struct Packed {
    /// `(b + 1,)` u32 cumulative row lengths, as flash varlen takes them.
    pub cu_seqlens: Tensor,
    /// Longest row.
    pub max_len: usize,
    /// `(total,)` u32: each token's position within its own row.
    pub positions: Tensor,
    /// Where each row starts in the packed order.
    pub offsets: Vec<usize>,
}

#[cfg(feature = "flash-attn")]
impl Packed {
    pub fn new(lens: &[usize], device: &Device) -> Result<Self> {
        let mut cu = Vec::with_capacity(lens.len() + 1);
        let mut offsets = Vec::with_capacity(lens.len());
        let mut positions = Vec::with_capacity(lens.iter().sum());
        let mut total = 0u32;
        cu.push(0u32);
        for &n in lens {
            offsets.push(total as usize);
            positions.extend(0..n as u32);
            total += n as u32;
            cu.push(total);
        }
        let n = cu.len();
        Ok(Self {
            cu_seqlens: Tensor::from_vec(cu, n, device)?,
            max_len: lens.iter().copied().max().unwrap_or(0),
            positions: Tensor::from_vec(positions, total as usize, device)?,
            offsets,
        })
    }
}

/// Bidirectional flash attention over a packed batch. `q`, `k`, `v` are
/// `(total, h, dh)`; `half_window` limits each query to keys within that
/// distance on either side (ModernBERT's sliding layers), `None` attends
/// the whole row. Scores are accumulated in f32 inside the kernel.
#[cfg(feature = "flash-attn")]
pub(crate) fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    packed: &Packed,
    half_window: Option<usize>,
) -> Result<Tensor> {
    let head_dim = q.dim(2)?;
    let scale = (head_dim as f64).powf(-0.5) as f32;
    Ok(candle_flash_attn::flash_attn_varlen_windowed(
        q,
        k,
        v,
        &packed.cu_seqlens,
        &packed.cu_seqlens,
        packed.max_len,
        packed.max_len,
        scale,
        half_window,
        half_window,
    )?)
}

/// Split a packed `(total, 3·d)` qkv projection into contiguous
/// `(total, h, dh)` q, k and v.
#[cfg(feature = "flash-attn")]
pub(crate) fn split_qkv(
    qkv: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let t = qkv.dim(0)?;
    let qkv = qkv.reshape((t, 3, n_heads, head_dim))?;
    let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i, 1)?.squeeze(1)?.contiguous()?) };
    Ok((part(0)?, part(1)?, part(2)?))
}

#[cfg(feature = "flash-attn")]
impl Rotary {
    /// Rotate packed `(total, h, dh)` q and k by each token's position.
    fn apply_packed(&self, q: &Tensor, k: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        let rot = |x: &Tensor| -> Result<Tensor> {
            Ok(candle_nn::rotary_emb::rope_thd(&x.unsqueeze(0)?, &cos, &sin)?.squeeze(0)?)
        };
        Ok((rot(q)?, rot(k)?))
    }
}

#[cfg(feature = "flash-attn")]
impl Attention {
    /// `xs` is the packed f32 residual stream `(total, d)`.
    fn forward_packed(
        &self,
        xs: &Tensor,
        rotary: &Rotary,
        packed: &Packed,
        half_window: usize,
    ) -> Result<Tensor> {
        let (t, d) = xs.dims2()?;
        let (q, k, v) = split_qkv(&project(&self.wqkv, xs)?, self.n_heads, self.head_dim)?;
        let (q, k) = rotary.apply_packed(&q, &k, &packed.positions)?;
        let window = match self.kind {
            LayerKind::Sliding => Some(half_window),
            LayerKind::Global => None,
        };
        let ctx = flash_attention(&q, &k, &v, packed, window)?.reshape((t, d))?;
        Ok(ctx.apply(&self.wo)?.to_dtype(DType::F32)?)
    }
}

/// `softmax(q kᵀ / √d + mask) v`, scores and softmax in f32.
///
/// `q`, `k`, `v` are `(b, h, l, d)`; `mask` is an f32 additive mask
/// broadcastable to `(b, h, l, l)`, or `None` when nothing is masked —
/// which saves a bandwidth-bound pass over the `(b, h, l, l)` scores.
pub(crate) fn scaled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
) -> Result<Tensor> {
    let dtype = v.dtype();
    let scale = (head_dim as f64).powf(-0.5);
    // Scores in f32 end to end: a bf16 `q kᵀ` is rounded to 8 mantissa
    // bits before the softmax ever sees it, which on this model's
    // logit-sized scores is most of the drift from the reference's GPU
    // path (whose SDPA kernels keep scores in f32).
    // The scale goes on q, not on the `(b, h, l, l)` scores: the same
    // product, on a tensor `l / d` times smaller.
    let scores = (q.to_dtype(DType::F32)? * scale)?.matmul(&k.to_dtype(DType::F32)?.t()?)?;
    let scores = match mask {
        Some(mask) => scores.broadcast_add(mask)?,
        None => scores,
    };
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(probs.to_dtype(dtype)?.matmul(v)?)
}

struct Mlp {
    wi: Linear,
    wo: Linear,
}

impl Module for Mlp {
    /// f32 in, f32 out; the projections run in the weights' dtype.
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = project(&self.wi, xs)?;
        let parts = xs.chunk(2, D::Minus1)?;
        (parts[0].gelu_erf()? * &parts[1])?
            .apply(&self.wo)?
            .to_dtype(DType::F32)
    }
}

struct Layer {
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
}

/// The two additive masks one forward needs, both f32. Either is `None`
/// when it would mask nothing: `global` for a batch with no padding (every
/// single-question request), `sliding` additionally when no row is longer
/// than the window reaches.
pub struct Masks {
    /// `(b, 1, 1, l)`: padded keys masked. Global layers and the
    /// decision head use this one.
    pub global: Option<Tensor>,
    /// `(b, 1, l, l)`: padded keys, plus keys outside the window.
    pub sliding: Option<Tensor>,
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
        let padded = lens.iter().any(|&n| n < seq_len);
        let windowed = seq_len > half_window + 1;
        let global = if padded {
            let mut pad = Vec::with_capacity(b * seq_len);
            for &n in lens {
                pad.extend((0..seq_len).map(|j| if j < n { 0f32 } else { MASKED }));
            }
            Some(Tensor::from_vec(pad, (b, 1, 1, seq_len), device)?)
        } else {
            None
        };
        if !windowed {
            let sliding = global.clone();
            return Ok(Self { global, sliding });
        }
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
        let sliding = match &global {
            Some(g) => g.broadcast_add(&window)?,
            None => window,
        };
        Ok(Self {
            global,
            sliding: Some(sliding),
        })
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
    /// in a Laya checkpoint, `model` in a stock ModernBERT one). Linear
    /// weights load in the VarBuilder's dtype; embeddings and norms in
    /// f32.
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let f32_vb = vb.clone().set_dtype(DType::F32);
        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            d,
            f32_vb.pp("embeddings").pp("tok_embeddings"),
        )
        .context("encoder.embeddings.tok_embeddings")?;
        let embed_norm = fused_layer_norm_no_bias(d, cfg.norm_eps, f32_vb.pp("embeddings.norm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, kind) in cfg.layer_kinds.iter().copied().enumerate() {
            let lvb = vb.pp(format!("layers.{i}"));
            let norm_vb = f32_vb.pp(format!("layers.{i}"));
            // ModernBERT builds layer 0's attn_norm as an identity and
            // saves nothing for it; every other layer must have one.
            let attn_norm = if lvb.contains_tensor("attn_norm.weight") {
                Some(fused_layer_norm_no_bias(
                    d,
                    cfg.norm_eps,
                    norm_vb.pp("attn_norm"),
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
            let mlp_norm = fused_layer_norm_no_bias(d, cfg.norm_eps, norm_vb.pp("mlp_norm"))?;
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
        let final_norm = fused_layer_norm_no_bias(d, cfg.norm_eps, f32_vb.pp("final_norm"))?;
        let device = vb.device();
        Ok(Self {
            embeddings,
            embed_norm,
            layers,
            final_norm,
            global_rotary: Rotary::new(cfg, cfg.global_rope_theta, vb.dtype(), device)?,
            sliding_rotary: Rotary::new(cfg, cfg.local_rope_theta, vb.dtype(), device)?,
            half_window: cfg.local_attention / 2,
        })
    }

    pub fn half_window(&self) -> usize {
        self.half_window
    }

    /// `ids` is `(b, l)` u32; returns the final hidden states `(b, l, d)`
    /// in f32.
    pub fn forward(&self, ids: &Tensor, masks: &Masks) -> Result<Tensor> {
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.embed_norm)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => xs.apply(norm)?,
                None => xs.clone(),
            };
            let (rotary, mask) = match layer.attn.kind {
                LayerKind::Global => (&self.global_rotary, masks.global.as_ref()),
                LayerKind::Sliding => (&self.sliding_rotary, masks.sliding.as_ref()),
            };
            xs = (&xs + layer.attn.forward(&normed, rotary, mask)?)?;
            let mlp_out = xs.apply(&layer.mlp_norm)?.apply(&layer.mlp)?;
            xs = (xs + mlp_out)?;
        }
        Ok(xs.apply(&self.final_norm)?)
    }

    /// The flash path: `ids` is the packed `(total,)` u32 token stream;
    /// returns the packed final hidden states `(total, d)` in f32.
    #[cfg(feature = "flash-attn")]
    pub fn forward_packed(&self, ids: &Tensor, packed: &Packed) -> Result<Tensor> {
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.embed_norm)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => xs.apply(norm)?,
                None => xs.clone(),
            };
            let rotary = match layer.attn.kind {
                LayerKind::Global => &self.global_rotary,
                LayerKind::Sliding => &self.sliding_rotary,
            };
            xs = (&xs
                + layer
                    .attn
                    .forward_packed(&normed, rotary, packed, self.half_window)?)?;
            let mlp_out = xs.apply(&layer.mlp_norm)?.apply(&layer.mlp)?;
            xs = (xs + mlp_out)?;
        }
        Ok(xs.apply(&self.final_norm)?)
    }
}
