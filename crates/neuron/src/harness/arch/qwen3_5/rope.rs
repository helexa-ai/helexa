//! Rotary position embedding for Qwen3-Next's full-attention layers.
//!
//! Qwen3.6 ships with MRoPE (multimodal RoPE) machinery in the
//! reference Python — three position grids interleaved per
//! `mrope_section`. For text-only inference all three grids carry the
//! same position ids and the interleave is a no-op, so this module
//! implements the plain (non-mrope) flavour: the standard inv_freq
//! cosine/sine tables driven by `rope_theta` and `head_dim`.
//!
//! Rotation flavour: **GLM-style** rotate-half (the second half of the
//! head dim is negated and swapped into the first). The reference
//! Python uses `apply_rotary_pos_emb` with `rotate_half`; candle's
//! `rope_slow` is the matching helper.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::TextConfig;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &TextConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to q, k.
    ///
    /// `q`, `k` shape: `(B, H, L, head_dim)`. `offset` is the index
    /// into the cached cos/sin table — the position of the first token
    /// in the current step. `candle_nn::rotary_emb::rope_slow` does
    /// the GLM-style `x*cos + rotate_half(x)*sin` rotation and
    /// internally `cat`s cos/sin with themselves along the last dim,
    /// so we hand it the `(seq_len, head_dim/2)` slice it expects.
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}
