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
    /// Number of dims at the head's leading edge that the rotation
    /// covers. The remaining `head_dim - rotary_dim` dims pass through
    /// unchanged. Qwen3-Next uses `partial_rotary_factor = 0.25`, so
    /// for `head_dim = 256` only 64 dims rotate.
    rotary_dim: usize,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &TextConfig, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let rope = &cfg.rope_parameters;
        let rotary_dim = (head_dim as f32 * rope.partial_rotary_factor) as usize;
        if !rotary_dim.is_multiple_of(2) {
            anyhow::bail!(
                "rotary_dim = head_dim * partial_rotary_factor = {head_dim} * {} = {rotary_dim} \
                 must be even (cos/sin are paired)",
                rope.partial_rotary_factor
            );
        }
        if rotary_dim == 0 {
            anyhow::bail!(
                "rotary_dim = 0 (partial_rotary_factor = {} too small)",
                rope.partial_rotary_factor
            );
        }
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
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
            rotary_dim,
            head_dim,
        })
    }

    /// Apply RoPE to q, k.
    ///
    /// `q`, `k` shape: `(B, H, L, head_dim)`. `offset` is the index
    /// into the cached cos/sin table — the position of the first token
    /// in the current step.
    ///
    /// When `rotary_dim < head_dim` the rotation is applied only to the
    /// first `rotary_dim` dims of each head; the tail passes through
    /// unchanged (matches the reference Python's
    /// `apply_rotary_pos_emb` with non-trivial `partial_rotary_factor`).
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim_in) = q.dims4()?;
        debug_assert_eq!(head_dim_in, self.head_dim, "q head_dim mismatch");
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        if self.rotary_dim == self.head_dim {
            // Full rotation.
            let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            // Partial rotation: narrow → rotate → cat the untouched tail.
            let tail = self.head_dim - self.rotary_dim;
            let q_rot = q
                .narrow(candle_core::D::Minus1, 0, self.rotary_dim)?
                .contiguous()?;
            let q_pass = q.narrow(candle_core::D::Minus1, self.rotary_dim, tail)?;
            let k_rot = k
                .narrow(candle_core::D::Minus1, 0, self.rotary_dim)?
                .contiguous()?;
            let k_pass = k.narrow(candle_core::D::Minus1, self.rotary_dim, tail)?;
            let q_rotated = candle_nn::rotary_emb::rope_slow(&q_rot, &cos, &sin)?;
            let k_rotated = candle_nn::rotary_emb::rope_slow(&k_rot, &cos, &sin)?;
            let q_embed =
                Tensor::cat(&[&q_rotated, &q_pass.contiguous()?], candle_core::D::Minus1)?;
            let k_embed =
                Tensor::cat(&[&k_rotated, &k_pass.contiguous()?], candle_core::D::Minus1)?;
            Ok((q_embed, k_embed))
        }
    }
}
