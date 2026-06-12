//! Tensor-parallel linear layers built on candle's `ShardedVarBuilder`
//! and `Shard` sharding hints.
//!
//! candle reads only the rank's slice of each weight tensor from
//! safetensors via `view.slice(start..stop)` — no full-tensor host
//! materialisation. That's a memory-efficiency win over hand-rolled
//! "load full + narrow" sharding (which the earlier
//! `sharded_linear.rs` exploration demonstrated but didn't pay for).
//!
//! Two layer types:
//!
//! - [`ColumnParallelLinear`] — output-sharded; forward is a plain
//!   local matmul. The downstream consumer either accepts a sharded
//!   activation (next layer is also column-parallel) or all-gathers.
//! - [`RowParallelLinear`] — input-sharded; forward = local matmul
//!   then `AllReduce` `CustomOp1` to sum partials across ranks.
//!
//! Both assume **no bias** — every Qwen3-family weight layout we
//! actually target (Qwen3, Qwen3-Coder, Qwen3.6 base, etc.) sets
//! `attention_bias=false` and the MLP layers are no-bias. Adding bias
//! support is mechanical when a future model needs it; the design
//! choice would be: column-parallel shards the bias along dim 0;
//! row-parallel holds the bias only on rank 0 so the post-`AllReduce`
//! sum carries it exactly once.

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::{Shard, ShardedVarBuilder};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::all_reduce::AllReduce;

/// Linear primitive that holds either a plain `Linear` (bf16/f16/f32)
/// or a quantized `QMatMul` (Q4K/Q5K/Q6K/Q8_0/etc.).
///
/// Constructed via [`MaybeQuantLinear::from_weight`] — pass `None` to
/// keep the weight in its loaded dtype (no quantization), or
/// `Some(dtype)` to quantize at load time.
///
/// On the forward path the two arms dispatch identically: `Module::forward`
/// returns an output in the caller's input dtype (f32 fallback for the
/// quantized matmul). Subsequent ops don't need to know whether the
/// layer was quantized.
pub enum MaybeQuantLinear {
    Plain(Linear),
    Quant(QMatMul),
}

impl MaybeQuantLinear {
    /// Build a linear from a loaded weight tensor. If `quant` is set,
    /// the weight is quantized in-situ and stored as a `QMatMul`;
    /// otherwise it's wrapped in a plain `Linear`.
    pub fn from_weight(weight: Tensor, quant: Option<GgmlDType>) -> Result<Self> {
        match quant {
            Some(dtype) => {
                // Parallel ISQ (#1): same bytes as QTensor::quantize,
                // blocks fanned across the rayon pool.
                let qt = super::isq::quantize_parallel(&weight, dtype).with_context(|| {
                    format!(
                        "quantize_parallel to {dtype:?} for shape {:?}",
                        weight.shape()
                    )
                })?;
                let qmm = QMatMul::from_arc(Arc::new(qt))
                    .context("QMatMul::from_arc on freshly quantized weight")?;
                Ok(Self::Quant(qmm))
            }
            None => Ok(Self::Plain(Linear::new(weight, None))),
        }
    }
}

/// Above this M (the product of all input dims except the last)
/// dispatch the quantized matmul through `QMatMul::forward_via_f16`,
/// which dequantizes the weight to f16 once and runs cuBLAS GEMM.
/// At or below this M the GGUF GEMV kernel inside
/// `QMatMul::forward` wins (it operates on quantized blocks directly
/// and accumulates in registers).
///
/// Empirical: at M=30 on Qwen3.6-27B / RTX 5090, forward_via_f16 was
/// slightly *slower* than the GGUF GEMV kernel — the per-call dequant
/// cost (~30 MB f16 written to global memory per linear × ~480 calls
/// per prefill) eats the cuBLAS GEMM speedup at small M. The
/// crossover where the GEMM scaling actually beats the fixed dequant
/// tax sits well above M=8.
///
/// 64 is a conservative crossover that keeps short-prompt prefills
/// on the GGUF kernel (where the per-call cost is comparable to the
/// f16 path but the dequant tax is zero) and only activates the
/// dequant-then-GEMM path for long prefills where the GEMM size
/// makes amortising worth it. A proper fix is either a dequant
/// cache or a fused dequant+gemm cuda kernel — both larger projects.
const QUANT_PREFILL_M_THRESHOLD: usize = 64;

impl Module for MaybeQuantLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(x),
            Self::Quant(qm) => {
                // Decode vs prefill split. `M` is the "rows of x" the
                // matmul will iterate over — every dim except the last
                // (which is in_features). For decode (`seq_len == 1`
                // with batch 1) M is 1; for prefill with L>>1 it's L
                // (or B*L).
                let dims = x.dims();
                let m: usize = dims.iter().take(dims.len() - 1).product();

                if m > QUANT_PREFILL_M_THRESHOLD {
                    // Prefill: dequantize the weight once into f16,
                    // then run a real cuBLAS-backed GEMM. The cost of
                    // the dequant is amortised across all M tokens.
                    // `forward_via_f16` handles the dtype round-trip
                    // internally (output matches input dtype).
                    return qm.forward_via_f16(x);
                }

                // Decode (M <= threshold): use the on-the-fly GGUF
                // GEMV kernel via `QMatMul::forward`. That kernel
                // requires f32 inputs (it accumulates in f32 from the
                // dequantized quant blocks); cast in/out at the
                // boundary.
                let in_dtype = x.dtype();
                let x_f32 = if in_dtype == candle_core::DType::F32 {
                    x.clone()
                } else {
                    x.to_dtype(candle_core::DType::F32)?
                };
                let y = qm.forward(&x_f32)?;
                if y.dtype() == in_dtype {
                    Ok(y)
                } else {
                    y.to_dtype(in_dtype)
                }
            }
        }
    }
}

/// Helper to build a [`Shard`] hint for a given dimension.
pub(crate) fn shard(dim: usize, rank: u32, world_size: u32) -> Shard {
    Shard {
        dim,
        rank: rank as usize,
        world_size: world_size as usize,
    }
}

/// Output-dim sharded linear (column-parallel). Holds a
/// [`MaybeQuantLinear`] whose underlying weight is this rank's slice
/// of the full `[out_features, in_features]` tensor along dim 0.
pub struct ColumnParallelLinear {
    inner: MaybeQuantLinear,
}

impl ColumnParallelLinear {
    /// Load this rank's column-parallel slice from a
    /// `ShardedVarBuilder`. The provided `vb` must already be `pp`-ed
    /// to the layer's path (e.g. `vb.pp("model.layers.0.self_attn.q_proj")`).
    ///
    /// Backward-compatible variant — no in-situ quantization. For
    /// quantized loads, use [`Self::load_with_quant`].
    pub fn load(vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
        Self::load_with_quant(vb, rank, world_size, None)
    }

    /// Like [`Self::load`] but quantizes the per-rank weight in-situ
    /// when `quant` is `Some(dtype)`. Saves ~3-5x vs bf16/f16.
    pub fn load_with_quant(
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let weight = vb
            .get_with_hints((), "weight", shard(0, rank, world_size))
            .with_context(|| format!("load column-parallel '{}' weight", vb.prefix()))?;
        let inner = MaybeQuantLinear::from_weight(weight, quant)
            .with_context(|| format!("wrap column-parallel '{}'", vb.prefix()))?;
        Ok(Self { inner })
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

/// Input-dim sharded linear (row-parallel).
///
/// Holds a sharded [`MaybeQuantLinear`] plus an `AllReduce` op the
/// forward chains after the local matmul to recover the full activation.
pub struct RowParallelLinear {
    inner: MaybeQuantLinear,
    #[cfg(feature = "cuda")]
    all_reduce: AllReduce,
    /// Whether the AllReduce should run. Column-parallel ↔ row-parallel
    /// is a pair: the column output is sharded, the row input is
    /// sharded, and the AllReduce gives back the full output. For
    /// `world_size = 1` the AllReduce is a no-op so we skip it.
    needs_reduce: bool,
}

impl RowParallelLinear {
    /// Load this rank's row-parallel slice from a `ShardedVarBuilder`.
    ///
    /// Under `cuda`, `comm` is the NCCL communicator the row-parallel
    /// `AllReduce` runs against. On CPU builds the parameter is
    /// elided — forward returns the partial sum, which is the *wrong*
    /// answer for inference but lets us compile-check the model.
    ///
    /// Backward-compatible variant — no in-situ quantization. For
    /// quantized loads, use [`Self::load_with_quant`].
    #[cfg(feature = "cuda")]
    pub fn load(
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: std::sync::Arc<cudarc::nccl::Comm>,
    ) -> Result<Self> {
        Self::load_with_quant(vb, rank, world_size, comm, None)
    }

    /// Like [`Self::load`] but quantizes the per-rank weight in-situ.
    #[cfg(feature = "cuda")]
    pub fn load_with_quant(
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: std::sync::Arc<cudarc::nccl::Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let weight = vb
            .get_with_hints((), "weight", shard(1, rank, world_size))
            .with_context(|| format!("load row-parallel '{}' weight", vb.prefix()))?;
        let inner = MaybeQuantLinear::from_weight(weight, quant)
            .with_context(|| format!("wrap row-parallel '{}'", vb.prefix()))?;
        Ok(Self {
            inner,
            all_reduce: AllReduce::new(comm),
            needs_reduce: world_size > 1,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
        Self::load_with_quant(vb, rank, world_size, None)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load_with_quant(
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let weight = vb
            .get_with_hints((), "weight", shard(1, rank, world_size))
            .with_context(|| format!("load row-parallel '{}' weight", vb.prefix()))?;
        let inner = MaybeQuantLinear::from_weight(weight, quant)
            .with_context(|| format!("wrap row-parallel '{}'", vb.prefix()))?;
        Ok(Self {
            inner,
            needs_reduce: world_size > 1,
        })
    }
}

impl Module for RowParallelLinear {
    /// Local matmul followed by an `AllReduce` (when `cuda` and
    /// `world_size > 1`). On CPU or single-rank, returns the partial
    /// output directly — which is *only* correct for `world_size == 1`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let local = self.inner.forward(x)?;
        #[cfg(feature = "cuda")]
        if self.needs_reduce {
            return local.apply_op1_no_bwd(&self.all_reduce);
        }
        let _ = self.needs_reduce;
        Ok(local)
    }
}
