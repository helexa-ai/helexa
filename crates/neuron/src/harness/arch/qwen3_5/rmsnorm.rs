//! Norm primitives for Qwen3-Next, shared with `qwen4_exp`.
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
//!    its output by an RMSNorm *gated* with a per-element nonlinearity
//!    on a sibling `z` projection — fused for numerical reasons (the
//!    norm's float32 promotion has to happen before the gate
//!    multiply). Not a single existing candle op.
//!
//! Both ops accept inputs in any compute dtype; promotion to f32 for
//! the variance calculation matches the Python reference.
//!
//! `qwen4_exp` (`doc/qwen4_exp-port-spec.md`) reuses both with two
//! additions, because the conventions are otherwise identical:
//!
//! - **`group_size`** on the plain norm, for the 10240-wide norms whose
//!   four hidden-size slices normalise independently.
//! - **`OutputGate`** on the gated norm, because that arch sets
//!   `output_gate_type = "sigmoid"` where Qwen3.6 falls back to SiLU.
//!
//! Both default to the Qwen3.6 behaviour, so existing callers — the
//! decoder, the attention q/k norms, and the TP analogues in
//! `harness/tp/tp_qwen3_5.rs` — are unaffected.

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
///
/// With `group_size` set, the last axis is split into
/// `size / group_size` groups that are normalised **independently**
/// before the scale is applied across the whole axis — upstream's
/// `Qwen4ExpTextRMSNorm(dim, group_size=...)`. `qwen4_exp` uses this
/// for every 10240-wide norm (`hc_norm`, PLE's three norms,
/// `mtp.pre_fc_norm_hidden`), all with `group_size = hidden_size`, i.e.
/// four independent normalisations rather than one over 10240. A single
/// norm over the full axis compiles, runs, and is quietly wrong — see
/// `doc/qwen4_exp-port-spec.md` §8.
pub struct Qwen3_5RmsNorm {
    weight: Tensor,
    eps: f32,
    size: usize,
    /// `None` = one normalisation over the whole last axis (Qwen3.6).
    group_size: Option<usize>,
}

impl Qwen3_5RmsNorm {
    /// Load `weight` from the ShardedVarBuilder. `vb` should already be
    /// `.pp(...)`-ed to the norm's tensor prefix.
    pub fn load(vb: &ShardedVarBuilder, size: usize, eps: f64) -> Result<Self> {
        Self::load_inner(vb, size, eps, None)
    }

    /// Grouped variant: normalise each `group_size`-wide slice of the
    /// last axis independently.
    pub fn load_grouped(
        vb: &ShardedVarBuilder,
        size: usize,
        group_size: usize,
        eps: f64,
    ) -> Result<Self> {
        anyhow::ensure!(
            group_size > 0 && size.is_multiple_of(group_size),
            "grouped RMSNorm needs size ({size}) divisible by group_size ({group_size})"
        );
        Self::load_inner(vb, size, eps, Some(group_size))
    }

    fn load_inner(
        vb: &ShardedVarBuilder,
        size: usize,
        eps: f64,
        group_size: Option<usize>,
    ) -> Result<Self> {
        let weight = vb
            .get(size, "weight")
            .with_context(|| format!("load '{}/weight'", vb.prefix()))?;
        Ok(Self {
            weight,
            eps: eps as f32,
            size,
            group_size,
        })
    }

    /// Direct constructor — used by unit tests that build a norm
    /// without going through a VarBuilder.
    #[cfg(test)]
    pub(crate) fn from_weight(weight: Tensor, eps: f64, group_size: Option<usize>) -> Self {
        let size = weight.dims()[0];
        Self {
            weight,
            eps: eps as f32,
            size,
            group_size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Module for Qwen3_5RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        // Grouped: reshape the last axis into (groups, group_size),
        // normalise over group_size, then flatten back. The scale below
        // still applies across the full axis.
        let normed = match self.group_size {
            Some(gs) => {
                let rank = x_f32.rank();
                let mut shape = x_f32.dims().to_vec();
                let last = shape.pop().expect("tensor has at least one axis");
                shape.push(last / gs);
                shape.push(gs);
                let grouped = x_f32.reshape(shape)?;
                let var = grouped.sqr()?.mean_keepdim(D::Minus1)?;
                let normed = grouped.broadcast_mul(&(var + self.eps as f64)?.sqrt()?.recip()?)?;
                // rank-1 is the groups axis; flattening from there
                // restores the original rank.
                normed.flatten_from(rank - 1)?
            }
            None => {
                let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
                x_f32.broadcast_mul(&(var + self.eps as f64)?.sqrt()?.recip()?)?
            }
        };
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
    activation: OutputGate,
}

/// Which nonlinearity the output gate applies. Qwen3.6 leaves
/// `output_gate_type` unset and falls back to `hidden_act` (SiLU);
/// `qwen4_exp` sets it to `sigmoid` explicitly. Inheriting SiLU there
/// would be wrong on all 36 linear-attention layers, and wrong in a way
/// that still produces fluent output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputGate {
    Silu,
    Sigmoid,
}

impl OutputGate {
    /// Map a config's `output_gate_type` (or `hidden_act` fallback).
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "silu" => Ok(Self::Silu),
            "sigmoid" => Ok(Self::Sigmoid),
            other => anyhow::bail!("unsupported output_gate_type '{other}'"),
        }
    }

    fn apply(self, gate: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(gate),
            Self::Sigmoid => candle_nn::ops::sigmoid(gate),
        }
    }
}

impl Qwen3_5RmsNormGated {
    pub fn load(vb: &ShardedVarBuilder, size: usize, eps: f64) -> Result<Self> {
        Self::load_with_gate(vb, size, eps, OutputGate::Silu)
    }

    pub fn load_with_gate(
        vb: &ShardedVarBuilder,
        size: usize,
        eps: f64,
        activation: OutputGate,
    ) -> Result<Self> {
        let weight = vb
            .get(size, "weight")
            .with_context(|| format!("load '{}/weight'", vb.prefix()))?;
        Ok(Self {
            weight,
            eps: eps as f32,
            size,
            activation,
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
            activation: OutputGate::Silu,
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
        // Activate the float32 gate, multiply back into the normed
        // tensor, then cast to the model dtype.
        let g = gate.to_dtype(candle_core::DType::F32)?;
        let activated = self.activation.apply(&g)?;
        (out * activated)?.to_dtype(dtype)
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

    /// Grouped norm over `[1, 1, 4]` with `group_size = 2`: each pair is
    /// normalised on its own RMS, so the second pair — an order of
    /// magnitude larger — comes back at the same scale as the first.
    /// Ungrouped, the shared RMS would leave the pairs far apart, which
    /// is exactly the silent failure this guards.
    #[test]
    fn grouped_rmsnorm_normalises_each_group_independently() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[[3.0_f32, 4.0, 30.0, 40.0]]], &dev).unwrap();
        let w = Tensor::zeros(4, candle_core::DType::F32, &dev).unwrap();

        let grouped = Qwen3_5RmsNorm::from_weight(w.clone(), 1e-6, Some(2));
        let got: Vec<f32> = grouped
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // RMS([3,4]) = sqrt(12.5); RMS([30,40]) = sqrt(1250).
        let a = (12.5_f32).sqrt();
        let b = (1250.0_f32).sqrt();
        let want = [3.0 / a, 4.0 / a, 30.0 / b, 40.0 / b];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "grouped: got {got:?} want {want:?}");
        }
        // Both groups land at the same magnitude — the whole point.
        assert!((got[0] - got[2]).abs() < 1e-4);
    }

    #[test]
    fn ungrouped_rmsnorm_is_unchanged_by_the_group_option() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[[3.0_f32, 4.0, 30.0, 40.0]]], &dev).unwrap();
        let w = Tensor::zeros(4, candle_core::DType::F32, &dev).unwrap();

        let plain = Qwen3_5RmsNorm::from_weight(w, 1e-6, None);
        let got: Vec<f32> = plain
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // One RMS over all four: sqrt((9+16+900+1600)/4).
        let rms = ((9.0_f32 + 16.0 + 900.0 + 1600.0) / 4.0).sqrt();
        let want = [3.0 / rms, 4.0 / rms, 30.0 / rms, 40.0 / rms];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "plain: got {got:?} want {want:?}");
        }
    }

    /// The grouped reshape must not disturb the leading axes.
    #[test]
    fn grouped_rmsnorm_preserves_shape() {
        let dev = Device::Cpu;
        let x = Tensor::rand(0f32, 1f32, (2, 3, 8), &dev).unwrap();
        let w = Tensor::zeros(8, candle_core::DType::F32, &dev).unwrap();
        let n = Qwen3_5RmsNorm::from_weight(w, 1e-6, Some(4));
        assert_eq!(n.forward(&x).unwrap().dims(), &[2, 3, 8]);
    }

    #[test]
    fn output_gate_sigmoid_differs_from_silu() {
        let dev = Device::Cpu;
        let g = Tensor::new(&[1.0_f32, -2.0, 0.5], &dev).unwrap();

        let sig: Vec<f32> = OutputGate::Sigmoid.apply(&g).unwrap().to_vec1().unwrap();
        let silu: Vec<f32> = OutputGate::Silu.apply(&g).unwrap().to_vec1().unwrap();

        // sigmoid(1) = 0.7311; silu(1) = 1 * 0.7311 = 0.7311 — equal at
        // x = 1, so check a point where they must diverge.
        assert!(
            (sig[1] - 0.11920292).abs() < 1e-5,
            "sigmoid(-2) = {}",
            sig[1]
        );
        assert!(
            (silu[1] + 0.23840584).abs() < 1e-5,
            "silu(-2) = {}",
            silu[1]
        );
        assert!((sig[1] - silu[1]).abs() > 0.3);
    }

    #[test]
    fn output_gate_rejects_unknown_names() {
        assert!(OutputGate::from_name("silu").is_ok());
        assert!(OutputGate::from_name("sigmoid").is_ok());
        assert!(OutputGate::from_name("gelu").is_err());
    }
}
