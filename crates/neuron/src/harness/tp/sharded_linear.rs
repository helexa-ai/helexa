//! Tensor-parallel linear layers over `candle_nn::Linear`.
//!
//! Two sharding strategies, both following the Megatron-LM convention
//! that's also what mistral.rs uses for vanilla Qwen3:
//!
//! - [`ColumnParallelLinear`] — splits the **output** dimension. Each
//!   rank holds `out_features / world_size` rows of the weight matrix.
//!   The forward pass is a plain local matmul; the output is *sharded*
//!   (each rank produces a slice of the output vector). Used for
//!   `q_proj` / `k_proj` / `v_proj` (sharding by head) and the FFN's
//!   `gate_proj` / `up_proj`.
//!
//! - [`RowParallelLinear`] — splits the **input** dimension. Each
//!   rank holds `in_features / world_size` columns of the weight
//!   matrix and consumes a sharded input from upstream. Each rank's
//!   local matmul produces a *partial* output; an `all_reduce(Sum)`
//!   across ranks recovers the full activation. Used for `o_proj`
//!   (after attention) and `down_proj` (after the FFN).
//!
//! Stage 7b-ii (this commit): the layers, sharded loading, local
//! forward. The `all_reduce` collective lives in `forward_with_comm`
//! and is wired up in 7b-iii when the full TP-aware Qwen3 model is
//! assembled with an NCCL Comm in scope. Tests here exercise only
//! the local (no-NCCL) math against an unsharded reference.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Direction of the parallelism split — selects which axis of the
/// weight matrix the rank's local slice is taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardKind {
    /// Split the output dimension: rank `r` holds rows
    /// `[r * out/N .. (r+1) * out/N]` of the weight matrix. The
    /// downstream consumer either accepts a sharded activation
    /// (the next layer is also column-parallel) or merges via
    /// all-gather.
    Column,
    /// Split the input dimension: rank `r` holds columns
    /// `[r * in/N .. (r+1) * in/N]`. The forward pass produces a
    /// partial output; an `all_reduce(Sum)` across ranks yields the
    /// full activation.
    Row,
}

/// A linear layer whose weights have been sharded across NCCL ranks.
///
/// Holds a standard `candle_nn::Linear` constructed from the local
/// slice. The collective op (only meaningful for `Row`) is invoked
/// by [`forward_with_comm`] — the trait `Module::forward` does just
/// the local matmul, so callers that want correct semantics on a
/// Row-parallel layer must drive the collective themselves.
#[derive(Debug)]
pub struct ShardedLinear {
    inner: Linear,
    kind: ShardKind,
    rank: u32,
    world_size: u32,
    /// Captured for diagnostics ("rank 3 layer says X but should say Y").
    /// `out_features` reflects the **logical** size (pre-shard) so the
    /// caller can validate against the model config without doing the
    /// arithmetic itself.
    logical_out_features: usize,
    logical_in_features: usize,
}

impl ShardedLinear {
    /// Load a column-parallel slice from a `VarBuilder`. Reads the
    /// full weight (and bias, if any) from the safetensors and
    /// narrows on dim 0 to the rank's slice. The bias is sharded the
    /// same way (each rank holds its own bias slice).
    ///
    /// Bails if `out_features` is not divisible by `world_size` — the
    /// same divisibility precondition mistral.rs's PR #2054-era code
    /// added an explicit guard for after the first TP shard attempt.
    pub fn load_column(
        vb: &VarBuilder,
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        let path = vb.prefix();
        if !out_features.is_multiple_of(world_size as usize) {
            anyhow::bail!(
                "column-parallel '{path}': out_features={out_features} \
                 not divisible by world_size={world_size}"
            );
        }
        let shard = out_features / world_size as usize;
        let start = rank as usize * shard;

        let full_w = vb
            .get((out_features, in_features), "weight")
            .with_context(|| format!("load weight for column-parallel '{path}'"))?;
        let weight = full_w
            .narrow(0, start, shard)
            .with_context(|| format!("narrow weight rows for column-parallel '{path}'"))?
            .contiguous()
            .with_context(|| format!("contiguous weight for column-parallel '{path}'"))?;
        // Drop the full tensor as soon as we have the shard so peak
        // host RAM during load tracks shard-size, not full-size, once
        // all narrows complete (Rust's drop semantics handle this
        // because `full_w` goes out of scope here).
        drop(full_w);

        let bias = if has_bias {
            let full_b = vb
                .get(out_features, "bias")
                .with_context(|| format!("load bias for column-parallel '{path}'"))?;
            let b = full_b
                .narrow(0, start, shard)
                .with_context(|| format!("narrow bias for column-parallel '{path}'"))?
                .contiguous()
                .with_context(|| format!("contiguous bias for column-parallel '{path}'"))?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            inner: Linear::new(weight, bias),
            kind: ShardKind::Column,
            rank,
            world_size,
            logical_out_features: out_features,
            logical_in_features: in_features,
        })
    }

    /// Load a row-parallel slice from a `VarBuilder`. Reads the full
    /// weight and narrows on dim 1 to the rank's column slice. The
    /// bias, if any, lives **only on rank 0** — every other rank
    /// holds `None`. This keeps the post-`all_reduce` semantics
    /// correct: each rank contributes its partial sum without the
    /// bias, then rank 0's bias (added in `forward_with_comm`) lands
    /// on the result exactly once.
    pub fn load_row(
        vb: &VarBuilder,
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        let path = vb.prefix();
        if !in_features.is_multiple_of(world_size as usize) {
            anyhow::bail!(
                "row-parallel '{path}': in_features={in_features} \
                 not divisible by world_size={world_size}"
            );
        }
        let shard = in_features / world_size as usize;
        let start = rank as usize * shard;

        let full_w = vb
            .get((out_features, in_features), "weight")
            .with_context(|| format!("load weight for row-parallel '{path}'"))?;
        let weight = full_w
            .narrow(1, start, shard)
            .with_context(|| format!("narrow weight cols for row-parallel '{path}'"))?
            .contiguous()
            .with_context(|| format!("contiguous weight for row-parallel '{path}'"))?;
        drop(full_w);

        let bias = if has_bias && rank == 0 {
            let b = vb
                .get(out_features, "bias")
                .with_context(|| format!("load bias for row-parallel '{path}'"))?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            inner: Linear::new(weight, bias),
            kind: ShardKind::Row,
            rank,
            world_size,
            logical_out_features: out_features,
            logical_in_features: in_features,
        })
    }

    pub fn kind(&self) -> ShardKind {
        self.kind
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    pub fn logical_in_features(&self) -> usize {
        self.logical_in_features
    }

    pub fn logical_out_features(&self) -> usize {
        self.logical_out_features
    }
}

impl Module for ShardedLinear {
    /// Local matmul only. For `Row`-parallel layers, the output is a
    /// *partial sum* — call [`Self::forward_with_comm`] to get the
    /// reduced result. Implementing `Module` lets a `ShardedLinear`
    /// be drop-in for any `Module`-shaped consumer that doesn't need
    /// the reduce step (column-parallel layers; tests).
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

#[cfg(feature = "cuda")]
impl ShardedLinear {
    /// Forward pass that issues an `all_reduce(Sum)` for row-parallel
    /// layers. Column-parallel layers just delegate to the local
    /// matmul (their output is naturally sharded; the next consumer
    /// will either gather or accept the shard).
    pub fn forward_with_comm(&self, x: &Tensor, comm: &cudarc::nccl::Comm) -> Result<Tensor> {
        let local = self
            .inner
            .forward(x)
            .map_err(|e| anyhow::anyhow!("local matmul: {e}"))?;
        match self.kind {
            ShardKind::Column => Ok(local),
            ShardKind::Row => {
                // TODO Stage 7b-iii: wrap `local`'s CudaSlice with a
                // matching output buffer, call comm.all_reduce(Sum),
                // return the result. The cudarc::nccl all_reduce
                // signature takes `&S: DevicePtr<T>` + `&mut R: DevicePtrMut<T>`,
                // both backed by `CudaSlice<T>`. candle stores its
                // Tensor data behind its own slab — extracting the
                // underlying CudaSlice safely is a separate piece of
                // plumbing best landed alongside the model assembly,
                // so this body is a placeholder.
                let _ = comm;
                anyhow::bail!(
                    "ShardedLinear::forward_with_comm row-parallel reduce \
                     lands in Stage 7b-iii alongside the model assembly; \
                     7b-ii ships only the local matmul"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::var_builder::VarBuilderArgs;
    use std::collections::HashMap;

    /// Build a VarBuilder over an in-memory map of tensors. Used by
    /// the tests to fake a safetensors source without touching disk.
    fn vb_from_map(tensors: HashMap<String, Tensor>, device: &Device) -> VarBuilder<'static> {
        VarBuilderArgs::from_tensors(tensors, DType::F32, device)
    }

    /// World_size=2 column-parallel split of a 4x3 weight. Each rank's
    /// local matmul on the same input should be 2 rows of the
    /// reference (full) matmul.
    #[test]
    fn column_parallel_shards_output_correctly() {
        let device = Device::Cpu;
        // weight (out=4, in=3): rows are easy to identify by value.
        let w = Tensor::from_slice(
            &[
                1f32, 2., 3., // row 0
                4., 5., 6., // row 1
                7., 8., 9., // row 2
                10., 11., 12., // row 3
            ],
            (4, 3),
            &device,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("foo.weight".into(), w.clone());
        let vb_root = vb_from_map(tensors, &device);
        let vb_foo = vb_root.pp("foo");

        // rank 0 of world_size 2 gets rows 0..2.
        let r0 = ShardedLinear::load_column(&vb_foo, 3, 4, false, 0, 2).unwrap();
        // rank 1 gets rows 2..4.
        let r1 = ShardedLinear::load_column(&vb_foo, 3, 4, false, 1, 2).unwrap();

        let x = Tensor::from_slice(&[1f32, 0., 0.], (1, 3), &device).unwrap();
        let y0 = r0.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        let y1 = r1.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        // Full reference: x @ w.T → [1, 4, 7, 10]. Rank 0 owns [1, 4],
        // rank 1 owns [7, 10].
        assert_eq!(y0, vec![vec![1.0, 4.0]]);
        assert_eq!(y1, vec![vec![7.0, 10.0]]);
    }

    /// World_size=2 row-parallel split of a 4x4 weight. Each rank's
    /// local matmul on its half of the input should be a partial sum;
    /// summing the two partials should equal the unsharded reference.
    #[test]
    fn row_parallel_partials_sum_to_full() {
        let device = Device::Cpu;
        // weight (out=4, in=4): use distinct values per column so the
        // partial sums are obviously different.
        let w = Tensor::from_slice(
            &[
                1f32, 2., 3., 4., // row 0
                5., 6., 7., 8., // row 1
                9., 10., 11., 12., // row 2
                13., 14., 15., 16., // row 3
            ],
            (4, 4),
            &device,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("bar.weight".into(), w.clone());
        let vb_root = vb_from_map(tensors, &device);
        let vb_bar = vb_root.pp("bar");

        let r0 = ShardedLinear::load_row(&vb_bar, 4, 4, false, 0, 2).unwrap();
        let r1 = ShardedLinear::load_row(&vb_bar, 4, 4, false, 1, 2).unwrap();

        // x split: rank 0 takes x[..2], rank 1 takes x[2..].
        let x_full = Tensor::from_slice(&[1f32, 1., 1., 1.], (1, 4), &device).unwrap();
        let x0 = x_full.narrow(1, 0, 2).unwrap();
        let x1 = x_full.narrow(1, 2, 2).unwrap();

        let y0 = r0.forward(&x0).unwrap();
        let y1 = r1.forward(&x1).unwrap();
        let summed = (y0 + y1).unwrap().to_vec2::<f32>().unwrap();

        // Reference: x_full @ w.T = [1+2+3+4, 5+6+7+8, 9+10+11+12, 13+14+15+16]
        //                         = [10, 26, 42, 58].
        assert_eq!(summed, vec![vec![10.0, 26.0, 42.0, 58.0]]);
    }

    /// Row-parallel bias lives only on rank 0; other ranks have None.
    /// (Verifies the rank-0-only bias contract.)
    #[test]
    fn row_parallel_bias_only_on_rank_zero() {
        let device = Device::Cpu;
        let w = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let b = Tensor::from_slice(&[1f32, 1., 1., 1.], 4, &device).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("baz.weight".into(), w);
        tensors.insert("baz.bias".into(), b);
        let vb_root = vb_from_map(tensors, &device);
        let vb_baz = vb_root.pp("baz");

        let r0 = ShardedLinear::load_row(&vb_baz, 4, 4, true, 0, 2).unwrap();
        let r1 = ShardedLinear::load_row(&vb_baz, 4, 4, true, 1, 2).unwrap();

        // We can't introspect the Linear's bias from the public API,
        // but we can run forward of zero-weight rank 1 and confirm
        // the output is zero (no bias added on non-zero ranks).
        let x = Tensor::ones((1, 2), DType::F32, &device).unwrap();
        let y1 = r1.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(y1, vec![vec![0.0, 0.0, 0.0, 0.0]]);

        let y0 = r0.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        // Rank 0 weight is zero but bias is [1,1,1,1] → output should be [1,1,1,1].
        assert_eq!(y0, vec![vec![1.0, 1.0, 1.0, 1.0]]);
    }

    #[test]
    fn column_parallel_rejects_non_divisible_out_features() {
        let device = Device::Cpu;
        let w = Tensor::zeros((5, 3), DType::F32, &device).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("nope.weight".into(), w);
        let vb_root = vb_from_map(tensors, &device);
        let vb_nope = vb_root.pp("nope");

        let err = ShardedLinear::load_column(&vb_nope, 3, 5, false, 0, 2).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not divisible by world_size"),
            "expected divisibility error, got: {msg}"
        );
    }

    #[test]
    fn row_parallel_rejects_non_divisible_in_features() {
        let device = Device::Cpu;
        let w = Tensor::zeros((4, 5), DType::F32, &device).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("nope.weight".into(), w);
        let vb_root = vb_from_map(tensors, &device);
        let vb_nope = vb_root.pp("nope");

        let err = ShardedLinear::load_row(&vb_nope, 5, 4, false, 0, 2).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not divisible by world_size"),
            "expected divisibility error, got: {msg}"
        );
    }
}
