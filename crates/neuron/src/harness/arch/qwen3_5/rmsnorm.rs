//! Norm primitives for Qwen3-Next.
//!
//! Two reasons we can't reuse `candle_nn::RmsNorm` directly:
//!
//! 1. **`(1.0 + weight)` scaling.** Qwen3-Next's `Qwen3_5RMSNorm`
//!    initialises `weight` to zeros and applies `(1.0 + weight)` to
//!    the normalised vector. `candle_nn::RmsNorm` applies `weight`
//!    directly. The two are equivalent only when the operator has
//!    pre-shifted the weights — the upstream checkpoints have not. See
//!    `huggingface/transformers#29402` for the upstream PR that
//!    introduced the `(1 + w)` form to recover from the zero-init.
//!
//! 2. **Gated variant.** The linear-attention layer post-normalises
//!    its output by an RMSNorm *gated* with a per-element SiLU on
//!    a sibling `z` projection — fused for numerical reasons (the
//!    norm's float32 promotion has to happen before the SiLU
//!    multiply). Not a single existing candle op.
//!
//! Both ops accept inputs in any compute dtype; promotion to f32 for
//! the variance calculation matches the Python reference.

use anyhow::{Context, Result};
use candle_core::{D, Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;

/// L2-normalise along the last dim with a small epsilon. Matches the
/// `l2norm` helper in `transformers/models/qwen3_5/modeling_qwen3_5.py`
/// — `x * rsqrt(sum(x*x) + eps)`. The linear-attention path uses this
/// on Q and K before the delta rule when
/// `use_qk_l2norm_in_kernel=True` (which Qwen3-Next always sets).
pub fn l2norm(x: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let sq = x_f32.sqr()?;
    let sum = sq.sum_keepdim(D::Minus1)?;
    let inv = (sum + eps as f64)?.sqrt()?.recip()?;
    x_f32.broadcast_mul(&inv)?.to_dtype(dtype)
}

/// Qwen3-Next's RMSNorm. Stores the raw weight tensor; forward applies
/// `(1.0 + weight) * x_normed`.
pub struct Qwen3_5RmsNorm {
    weight: Tensor,
    eps: f32,
    size: usize,
}

impl Qwen3_5RmsNorm {
    /// Load `weight` from the ShardedVarBuilder. `vb` should already be
    /// `.pp(...)`-ed to the norm's tensor prefix.
    pub fn load(vb: &ShardedVarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb
            .get(size, "weight")
            .with_context(|| format!("load '{}/weight'", vb.prefix()))?;
        Ok(Self {
            weight,
            eps: eps as f32,
            size,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Module for Qwen3_5RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_mul(&(var + self.eps as f64)?.sqrt()?.recip()?)?;
        // Promote weight to f32 and shift by 1.0 *before* multiplying.
        // Doing the (1 + w) operation in fp16 lands at -inf for the
        // bottom-of-range weights at load time.
        let w_f32 = self.weight.to_dtype(candle_core::DType::F32)?;
        let scale = (w_f32 + 1.0_f64)?;
        normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

/// Gated RMSNorm used at the tail of `Qwen3_5GatedDeltaNet`. Equivalent
/// to `x_normed * weight * silu(gate)` but with both the norm and the
/// gate evaluated in float32 to avoid mid-pipeline underflow.
///
/// Note: unlike `Qwen3_5RmsNorm`, this variant matches the Python
/// reference's `Qwen3_5RMSNormGated` which uses `weight` directly (not
/// `1.0 + weight`).
pub struct Qwen3_5RmsNormGated {
    weight: Tensor,
    eps: f32,
    size: usize,
}

impl Qwen3_5RmsNormGated {
    pub fn load(vb: &ShardedVarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb
            .get(size, "weight")
            .with_context(|| format!("load '{}/weight'", vb.prefix()))?;
        Ok(Self {
            weight,
            eps: eps as f32,
            size,
        })
    }

    /// Direct constructor — used by unit tests that build a layer
    /// without going through a VarBuilder.
    #[cfg(test)]
    pub(crate) fn from_weight(weight: Tensor, eps: f64) -> Self {
        let size = weight.dims()[0];
        Self {
            weight,
            eps: eps as f32,
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// `x` and `gate` share the same last-dim shape (`size`).
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_mul(&(var + self.eps as f64)?.sqrt()?.recip()?)?;
        let w = self.weight.to_dtype(candle_core::DType::F32)?;
        let out = normed.broadcast_mul(&w)?;
        // SiLU on the float32 gate, multiply back into the normed
        // tensor, then cast to the model dtype.
        let g = gate.to_dtype(candle_core::DType::F32)?;
        let silu_gate = candle_nn::ops::silu(&g)?;
        (out * silu_gate)?.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn l2norm_matches_hand_calc() {
        let x = Tensor::new(&[3.0_f32, 4.0_f32], &Device::Cpu).unwrap();
        let out = l2norm(&x, 1e-6).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        // |x| = 5, so x/|x| = [0.6, 0.8] (eps is tiny).
        assert!((v[0] - 0.6).abs() < 1e-4);
        assert!((v[1] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn l2norm_zero_vector_is_safe_via_epsilon() {
        let x = Tensor::new(&[0.0_f32, 0.0_f32], &Device::Cpu).unwrap();
        let out = l2norm(&x, 1e-6).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
