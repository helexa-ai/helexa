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
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::{Shard, ShardedVarBuilder};

#[cfg(feature = "cuda")]
use super::all_reduce::AllReduce;

/// Helper to build a [`Shard`] hint for a given dimension.
pub(crate) fn shard(dim: usize, rank: u32, world_size: u32) -> Shard {
    Shard {
        dim,
        rank: rank as usize,
        world_size: world_size as usize,
    }
}

/// Output-dim sharded linear (column-parallel). Holds a standard
/// `candle_nn::Linear` whose `weight` is the rank's slice of the full
/// `[out_features, in_features]` tensor along dim 0.
pub struct ColumnParallelLinear {
    inner: Linear,
}

impl ColumnParallelLinear {
    /// Load this rank's column-parallel slice from a
    /// `ShardedVarBuilder`. The provided `vb` must already be `pp`-ed
    /// to the layer's path (e.g. `vb.pp("model.layers.0.self_attn.q_proj")`).
    pub fn load(vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
        let weight = vb
            .get_with_hints((), "weight", shard(0, rank, world_size))
            .with_context(|| format!("load column-parallel '{}' weight", vb.prefix()))?;
        Ok(Self {
            inner: Linear::new(weight, None),
        })
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

/// Input-dim sharded linear (row-parallel).
///
/// Holds a sharded `Linear` plus an `AllReduce` op the forward chains
/// after the local matmul to recover the full activation.
pub struct RowParallelLinear {
    inner: Linear,
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
    #[cfg(feature = "cuda")]
    pub fn load(
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: std::sync::Arc<cudarc::nccl::Comm>,
    ) -> Result<Self> {
        let weight = vb
            .get_with_hints((), "weight", shard(1, rank, world_size))
            .with_context(|| format!("load row-parallel '{}' weight", vb.prefix()))?;
        Ok(Self {
            inner: Linear::new(weight, None),
            all_reduce: AllReduce::new(comm),
            needs_reduce: world_size > 1,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
        let weight = vb
            .get_with_hints((), "weight", shard(1, rank, world_size))
            .with_context(|| format!("load row-parallel '{}' weight", vb.prefix()))?;
        Ok(Self {
            inner: Linear::new(weight, None),
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
