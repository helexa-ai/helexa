//! Tensor-parallel Qwen3-Next (`qwen3_5`) model.
//!
//! Two distinct sharding strategies coexist in the same model because
//! `layer_types[i]` dispatches per layer:
//!
//! - **Full-attention layers** (`Qwen3_5Attention`): column-parallel
//!   `q_proj` (the doubled `2 * num_heads * head_dim` output sharded
//!   on the head axis, including the gate half), `k_proj`, `v_proj`;
//!   row-parallel `o_proj` with the trailing `AllReduce`. Same shape
//!   of work as `tp_qwen3.rs` apart from the gate.
//!
//! - **Linear-attention layers** (`Qwen3_5GatedDeltaNet`): V-head-dim
//!   sharding. Per rank: `num_v_heads / world_size` value heads and
//!   `num_k_heads / world_size` key heads. The recurrent state shards
//!   1:1 with the V-heads; no cross-rank sync inside the delta-rule
//!   loop. `out_proj` is row-parallel + AllReduce — the only
//!   collective inside the block.
//!
//!   The `in_proj_qkv` and `conv1d` weights are *fused* tensors with
//!   three regions sequentially along dim 0:
//!   `[first key_dim, second key_dim, value_dim]`. Uniform
//!   slicing-along-dim-0 (the standard `ShardedSafeTensors` behaviour)
//!   does **not** align with these head boundaries — rank 0 would end
//!   up with `[first half of key_dim_0, full key_dim_1, first half of
//!   value_dim]`, garbage. So we load the full tensor and re-slice it
//!   per-region per-rank, dropping the unused portion. Net memory is
//!   the same as proper per-rank loading; transient peak is one
//!   full-tensor allocation per layer during construction.
//!
//! Replicated: embedding, all RmsNorms, the gated RMSNorm tail of the
//! linear-attention block, lm_head, the rotary table.

use anyhow::{Context, Result, bail};
use candle_core::quantized::GgmlDType;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use candle_nn::{Embedding, kv_cache::ConcatKvCache};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::nccl::Comm;

use super::tp_linear::{ColumnParallelLinear, RowParallelLinear};
use crate::harness::arch::qwen3_5::linear_attn::repeat_interleave;
use crate::harness::arch::qwen3_5::rmsnorm::{Qwen3_5RmsNorm, Qwen3_5RmsNormGated, l2norm};
use crate::harness::arch::qwen3_5::rope::RotaryEmbedding;
use crate::harness::arch::qwen3_5::snapshot::{KvCacheSnapshot, LayerKvSnapshot};
use crate::harness::arch::qwen3_5::splice_runs;
use crate::harness::arch::qwen3_5::vision::VisionTower;
pub use crate::harness::arch::qwen3_5::{Config, TextConfig};

// ─── linear-attention (Gated DeltaNet) ──────────────────────────────

/// Per-rank, per-layer state for the TP linear-attention block.
/// Identical shape to the single-GPU `GatedDeltaNetState` but with
/// `num_v_heads` replaced by `per_rank_num_v_heads`.
#[derive(Default)]
pub struct TpGatedDeltaNetState {
    pub conv_state: Option<Tensor>,
    pub recurrent_state: Option<Tensor>,
}

pub(crate) struct TpQwen3_5GatedDeltaNet {
    in_proj_qkv: super::tp_linear::MaybeQuantLinear,
    in_proj_z: ColumnParallelLinear,
    in_proj_b: ColumnParallelLinear,
    in_proj_a: ColumnParallelLinear,
    out_proj: RowParallelLinear,

    /// Depthwise causal Conv1d weight, sharded per-region by V-head.
    /// Shape: `(per_rank_conv_dim, 1, conv_kernel_size)`.
    conv1d_weight: Tensor,

    /// Per-V-head discretisation params, sharded along `num_v_heads`.
    a_log: Tensor,
    dt_bias: Tensor,

    /// Output gated RMSNorm (replicated; the norm dim is `head_v_dim`
    /// which doesn't change with sharding).
    norm: Qwen3_5RmsNormGated,

    // Per-rank shape hyperparams.
    per_rank_num_v_heads: usize,
    per_rank_num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    per_rank_key_dim: usize,
    per_rank_value_dim: usize,
    per_rank_conv_dim: usize,
    conv_kernel_size: usize,

    state: TpGatedDeltaNetState,
}

impl TpQwen3_5GatedDeltaNet {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        Self::load_inner(cfg, vb, mmap, rank, world_size, comm, quant)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        Self::load_inner(cfg, vb, mmap, rank, world_size, quant)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_inner(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        #[cfg(feature = "cuda")] comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let ws = world_size as usize;
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        if num_v_heads == 0 || num_k_heads == 0 {
            bail!(
                "Qwen3-Next linear_num_*_heads must be set; got v={num_v_heads}, k={num_k_heads}"
            );
        }
        if !num_v_heads.is_multiple_of(num_k_heads) {
            bail!(
                "linear_num_value_heads ({num_v_heads}) must be a multiple of \
                 linear_num_key_heads ({num_k_heads}) for GQA-style head expansion"
            );
        }
        if !num_v_heads.is_multiple_of(ws) {
            bail!("linear_num_value_heads ({num_v_heads}) not divisible by world_size {ws}");
        }
        if !num_k_heads.is_multiple_of(ws) {
            bail!("linear_num_key_heads ({num_k_heads}) not divisible by world_size {ws}");
        }

        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;
        let per_rank_num_v_heads = num_v_heads / ws;
        let per_rank_num_k_heads = num_k_heads / ws;
        let per_rank_key_dim = head_k_dim * per_rank_num_k_heads;
        let per_rank_value_dim = head_v_dim * per_rank_num_v_heads;
        let per_rank_conv_dim = per_rank_key_dim * 2 + per_rank_value_dim;

        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let _conv_dim = key_dim * 2 + value_dim;
        let hidden_size = cfg.hidden_size;

        // ----- Fused `in_proj_qkv` and `conv1d` (direct safetensors slicing).
        // Reads only this rank's per-region byte slices from the mmap
        // and uploads as one device allocation per fused tensor — no
        // full-fused-tensor device materialisation, which on the prior
        // narrow+cat approach was the main allocator-fragmentation
        // source on consumer GPUs near their VRAM ceiling.
        let dtype = vb.dtype();
        let device = vb.device().clone();

        // Two checkpoint layouts (#92, mirroring the single-GPU loader):
        // - Qwen3.6: separate `in_proj_qkv` ([Q|K|V] contiguous) +
        //   `in_proj_z`/`in_proj_b`/`in_proj_a` — per-region mmap
        //   slicing for qkv, uniform column-parallel for the rest.
        // - Qwen3-Next 80B-A3B: fused `in_proj_qkvz` + `in_proj_ba`,
        //   interleaved per key-head group. K- and V-heads shard
        //   uniformly together, so each rank owns a CONTIGUOUS span of
        //   whole groups — a uniform dim-0 shard of the fused tensor is
        //   exactly the rank's groups, and the per-rank de-interleave
        //   (`split_fused_qkvz`/`split_fused_ba` with per-rank head
        //   counts) restores the contiguous [Q|K|V] + Z / B + A layout
        //   the forward path expects.
        let (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) =
            if vb.contains_tensor("in_proj_qkvz.weight") {
                let qkvz_slice = vb
                    .pp("in_proj_qkvz")
                    .get_with_hints((), "weight", super::tp_linear::shard(0, rank, world_size))
                    .with_context(|| format!("load '{}/in_proj_qkvz' rank slice", vb.prefix()))?;
                let ba_slice = vb
                    .pp("in_proj_ba")
                    .get_with_hints((), "weight", super::tp_linear::shard(0, rank, world_size))
                    .with_context(|| format!("load '{}/in_proj_ba' rank slice", vb.prefix()))?;
                let (qkv_w, z_w) = crate::harness::arch::qwen3_5::linear_attn::split_fused_qkvz(
                    &qkvz_slice,
                    per_rank_num_k_heads,
                    per_rank_num_v_heads,
                    head_k_dim,
                    head_v_dim,
                )
                .with_context(|| format!("de-interleave '{}/in_proj_qkvz'", vb.prefix()))?;
                let (b_w, a_w) = crate::harness::arch::qwen3_5::linear_attn::split_fused_ba(
                    &ba_slice,
                    per_rank_num_k_heads,
                    per_rank_num_v_heads,
                )
                .with_context(|| format!("de-interleave '{}/in_proj_ba'", vb.prefix()))?;
                (
                    super::tp_linear::MaybeQuantLinear::from_weight(qkv_w, quant)
                        .with_context(|| format!("wrap fused in_proj_qkv '{}'", vb.prefix()))?,
                    ColumnParallelLinear::from_weight(z_w, quant)?,
                    ColumnParallelLinear::from_weight(b_w, quant)?,
                    ColumnParallelLinear::from_weight(a_w, quant)?,
                )
            } else {
                let in_proj_qkv_name = format!("{}.in_proj_qkv.weight", vb.prefix());
                let in_proj_qkv_weight = super::fused_load::load_fused_qkv_2d(
                    mmap,
                    &in_proj_qkv_name,
                    hidden_size,
                    key_dim,
                    value_dim,
                    rank,
                    world_size,
                    dtype,
                    &device,
                )?;
                (
                    super::tp_linear::MaybeQuantLinear::from_weight(in_proj_qkv_weight, quant)
                        .with_context(|| format!("wrap fused in_proj_qkv for '{}'", vb.prefix()))?,
                    // in_proj_z: hidden → value_dim, sharded along value_dim
                    // (V-head); in_proj_b / in_proj_a: hidden → num_v_heads.
                    ColumnParallelLinear::load_with_quant(
                        &vb.pp("in_proj_z"),
                        rank,
                        world_size,
                        quant,
                    )?,
                    ColumnParallelLinear::load_with_quant(
                        &vb.pp("in_proj_b"),
                        rank,
                        world_size,
                        quant,
                    )?,
                    ColumnParallelLinear::load_with_quant(
                        &vb.pp("in_proj_a"),
                        rank,
                        world_size,
                        quant,
                    )?,
                )
            };

        let conv1d_name = format!("{}.conv1d.weight", vb.prefix());
        let conv1d_weight = super::fused_load::load_fused_qkv_3d(
            mmap,
            &conv1d_name,
            1,
            conv_kernel_size,
            key_dim,
            value_dim,
            rank,
            world_size,
            dtype,
            &device,
        )?;

        // ----- Per-V-head 1D params (sharded uniformly). -----
        let a_log = vb
            .get_with_hints((), "A_log", super::tp_linear::shard(0, rank, world_size))
            .with_context(|| format!("load '{}/A_log'", vb.prefix()))?;
        let dt_bias = vb
            .get_with_hints((), "dt_bias", super::tp_linear::shard(0, rank, world_size))
            .with_context(|| format!("load '{}/dt_bias'", vb.prefix()))?;

        // ----- Output gated RMSNorm (replicated, norm dim is head_v_dim). -----
        let norm = Qwen3_5RmsNormGated::load(&vb.pp("norm"), head_v_dim, cfg.rms_norm_eps)?;

        // ----- Output projection: row-parallel + AllReduce. -----
        #[cfg(feature = "cuda")]
        let out_proj =
            RowParallelLinear::load_with_quant(&vb.pp("out_proj"), rank, world_size, comm, quant)?;
        #[cfg(not(feature = "cuda"))]
        let out_proj =
            RowParallelLinear::load_with_quant(&vb.pp("out_proj"), rank, world_size, quant)?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            a_log,
            dt_bias,
            norm,
            per_rank_num_v_heads,
            per_rank_num_k_heads,
            head_k_dim,
            head_v_dim,
            per_rank_key_dim,
            per_rank_value_dim,
            per_rank_conv_dim,
            conv_kernel_size,
            state: TpGatedDeltaNetState::default(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.state = TpGatedDeltaNetState::default();
    }

    /// Deep-copy this rank's recurrent state for a prefix snapshot.
    /// Same in-place-kernel rationale as the single-GPU
    /// `GatedDeltaNet::snapshot_state`.
    pub fn snapshot_state(&self) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let conv = self
            .state
            .conv_state
            .as_ref()
            .map(Tensor::copy)
            .transpose()?;
        let rec = self
            .state
            .recurrent_state
            .as_ref()
            .map(Tensor::copy)
            .transpose()?;
        Ok((conv, rec))
    }

    /// Replace this rank's live recurrent state with a deep copy of a
    /// snapshot. See the single-GPU `GatedDeltaNet::restore_state`.
    pub fn restore_state(
        &mut self,
        conv_state: Option<&Tensor>,
        recurrent_state: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        self.state = TpGatedDeltaNetState {
            conv_state: conv_state.map(Tensor::copy).transpose()?,
            recurrent_state: recurrent_state.map(Tensor::copy).transpose()?,
        };
        Ok(())
    }

    /// `x` shape: `(B, L, hidden_size)`. Returns `(B, L, hidden_size)`
    /// after the row-parallel AllReduce.
    pub fn forward(&mut self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device().clone();

        // ----- Projections (per-rank). -----
        let mixed_qkv = self.in_proj_qkv.forward(x)?; // (B, L, per_rank_conv_dim)
        let mixed_qkv_chw = mixed_qkv.transpose(1, 2)?.contiguous()?;

        let z = self.in_proj_z.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.per_rank_num_v_heads,
            self.head_v_dim,
        ))?;

        let b = self.in_proj_b.forward(x)?; // (B, L, per_rank_num_v_heads)
        let a = self.in_proj_a.forward(x)?;

        // ----- State-aware causal Conv1d + SiLU. -----
        // Same shared helper as single-GPU — cuda kernel when available.
        let (conv_out, new_state) = crate::harness::arch::qwen3_5::linear_attn::run_causal_conv1d(
            &mixed_qkv_chw,
            &self.conv1d_weight,
            self.state.conv_state.take(),
            batch_size,
            self.per_rank_conv_dim,
            seq_len,
            self.conv_kernel_size,
        )?;
        self.state.conv_state = Some(new_state);
        let mixed_qkv = conv_out.transpose(1, 2)?.contiguous()?;

        // ----- Split into q, k, v (per-rank head counts). -----
        let q = mixed_qkv.narrow(2, 0, self.per_rank_key_dim)?;
        let k = mixed_qkv.narrow(2, self.per_rank_key_dim, self.per_rank_key_dim)?;
        let v = mixed_qkv.narrow(2, 2 * self.per_rank_key_dim, self.per_rank_value_dim)?;

        let q = q.reshape((
            batch_size,
            seq_len,
            self.per_rank_num_k_heads,
            self.head_k_dim,
        ))?;
        let k = k.reshape((
            batch_size,
            seq_len,
            self.per_rank_num_k_heads,
            self.head_k_dim,
        ))?;
        let v = v.reshape((
            batch_size,
            seq_len,
            self.per_rank_num_v_heads,
            self.head_v_dim,
        ))?;

        // ----- beta + g (per-V-head, per-token). -----
        // Same fused gating helper as single-GPU — cuda kernel when
        // available, per-op Rust fallback otherwise.
        let (beta, g) = crate::harness::arch::qwen3_5::linear_attn::run_fused_gating(
            &b,
            &a,
            &self.a_log,
            &self.dt_bias,
        )?;

        // ----- GQA expansion if per-rank ratio > 1. -----
        let (q, k) = if self.per_rank_num_v_heads > self.per_rank_num_k_heads {
            let rep = self.per_rank_num_v_heads / self.per_rank_num_k_heads;
            (
                repeat_interleave(&q, rep, 2)?,
                repeat_interleave(&k, rep, 2)?,
            )
        } else {
            (q, k)
        };

        // ----- L2norm on q, k. -----
        let q = l2norm(&q, 1e-6)?;
        let k = l2norm(&k, 1e-6)?;

        // ----- Transpose to (B, H, L, D) for delta-rule loop. -----
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let g = g.transpose(1, 2)?.contiguous()?;
        let beta = beta.transpose(1, 2)?.contiguous()?;

        let scale = 1.0_f64 / (self.head_k_dim as f64).sqrt();
        let q = (q.to_dtype(DType::F32)? * scale)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;

        let state_init = match self.state.recurrent_state.take() {
            Some(s) => s.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (
                    batch_size,
                    self.per_rank_num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                ),
                DType::F32,
                &device,
            )?,
        };

        // Hand off to the shared delta-rule runner — same cuda-kernel
        // dispatch as the single-GPU `arch::qwen3_5::linear_attn`, just
        // with per-rank head counts. CPU path falls back to a per-token
        // Rust loop; cuda path is the V-tiled register-resident kernel
        // imported from mistralrs.
        let (core_attn_out, new_state) =
            crate::harness::arch::qwen3_5::linear_attn::run_delta_rule(
                &q,
                &k,
                &v,
                &g,
                &beta,
                state_init,
                batch_size,
                self.per_rank_num_v_heads,
                seq_len,
                self.head_k_dim,
                self.head_v_dim,
            )?;
        self.state.recurrent_state = Some(new_state.to_dtype(dtype)?);

        let core_attn_out = core_attn_out.transpose(1, 2)?.contiguous()?;
        let core_attn_out = core_attn_out.to_dtype(dtype)?;
        let core_attn_flat = core_attn_out.reshape((
            batch_size * seq_len * self.per_rank_num_v_heads,
            self.head_v_dim,
        ))?;
        let z_flat = z.reshape((
            batch_size * seq_len * self.per_rank_num_v_heads,
            self.head_v_dim,
        ))?;
        let normed = self.norm.forward(&core_attn_flat, &z_flat)?;
        let normed = normed.reshape((
            batch_size,
            seq_len,
            self.per_rank_num_v_heads * self.head_v_dim,
        ))?;

        // Row-parallel out_proj + AllReduce.
        self.out_proj.forward(&normed)
    }
}

// ─── full-attention layer ───────────────────────────────────────────

pub(crate) struct TpQwen3_5Attention {
    q_proj: ColumnParallelLinear, // output = 2 * num_heads * head_dim
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    q_norm: Qwen3_5RmsNorm,
    k_norm: Qwen3_5RmsNorm,
    per_rank_num_heads: usize,
    per_rank_num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    per_rank_hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl TpQwen3_5Attention {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        Self::load_inner(cfg, rotary, vb, rank, world_size, comm, quant)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        Self::load_inner(cfg, rotary, vb, rank, world_size, quant)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_inner(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        #[cfg(feature = "cuda")] comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let ws = world_size as usize;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        if !num_heads.is_multiple_of(ws) {
            bail!("num_attention_heads ({num_heads}) not divisible by world_size {ws}");
        }
        if !num_kv_heads.is_multiple_of(ws) {
            bail!("num_key_value_heads ({num_kv_heads}) not divisible by world_size {ws}");
        }
        let per_rank_num_heads = num_heads / ws;
        let per_rank_num_kv_heads = num_kv_heads / ws;
        let num_kv_groups = per_rank_num_heads / per_rank_num_kv_heads;
        let head_dim = cfg.head_dim;
        let per_rank_hidden_size = head_dim * per_rank_num_heads;

        // q_proj has 2x output width (query + gate halves). Column-parallel
        // sharding along the output (head) axis splits both halves
        // consistently — rank R holds heads `[R*per_rank, (R+1)*per_rank)`
        // for both query AND gate, so the post-attention `gate.sigmoid()`
        // multiply against the per-rank attention output matches up.
        let q_proj =
            ColumnParallelLinear::load_with_quant(&vb.pp("q_proj"), rank, world_size, quant)?;
        let k_proj =
            ColumnParallelLinear::load_with_quant(&vb.pp("k_proj"), rank, world_size, quant)?;
        let v_proj =
            ColumnParallelLinear::load_with_quant(&vb.pp("v_proj"), rank, world_size, quant)?;
        #[cfg(feature = "cuda")]
        let o_proj =
            RowParallelLinear::load_with_quant(&vb.pp("o_proj"), rank, world_size, comm, quant)?;
        #[cfg(not(feature = "cuda"))]
        let o_proj = RowParallelLinear::load_with_quant(&vb.pp("o_proj"), rank, world_size, quant)?;

        let q_norm = Qwen3_5RmsNorm::load(&vb.pp("q_norm"), head_dim, cfg.rms_norm_eps)?;
        let k_norm = Qwen3_5RmsNorm::load(&vb.pp("k_norm"), head_dim, cfg.rms_norm_eps)?;

        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            per_rank_num_heads,
            per_rank_num_kv_heads,
            num_kv_groups,
            head_dim,
            per_rank_hidden_size,
            rotary,
            kv_cache,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. q_proj outputs (B, L, per_rank_num_heads * head_dim * 2)
        //    — split into (query, gate) per rank.
        let q_raw =
            self.q_proj
                .forward(x)?
                .reshape((b, l, self.per_rank_num_heads, self.head_dim * 2))?;
        let q = q_raw.narrow(3, 0, self.head_dim)?;
        let gate = q_raw.narrow(3, self.head_dim, self.head_dim)?;
        let gate = gate
            .contiguous()?
            .reshape((b, l, self.per_rank_num_heads * self.head_dim))?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let q = q.transpose(1, 2)?.contiguous()?; // (B, H, L, D)

        let k =
            self.k_proj
                .forward(x)?
                .reshape((b, l, self.per_rank_num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let k = k.transpose(1, 2)?.contiguous()?;

        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.per_rank_num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary.apply_cos_sin(&q, &k, cos, sin)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // Attention core — FlashAttention when available, eager
        // GQA-repeat + masked softmax otherwise (#95). Per-rank heads,
        // same kernel semantics as the single-GPU path.
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let ctx = crate::harness::arch::qwen3_5::full_attn::attention_context(
            &q,
            &k,
            &v,
            attn_mask,
            self.num_kv_groups,
            scale,
        )?;

        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.per_rank_hidden_size))?;
        let gate_sig = candle_nn::ops::sigmoid(&gate)?;
        let gated = (ctx * gate_sig)?;
        self.o_proj.forward(&gated)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Capture this rank's KV cache for a prefix snapshot. Shallow
    /// clones are safe — see the single-GPU
    /// `Qwen3_5Attention::snapshot_kv`.
    pub fn snapshot_kv(&self) -> Option<(Tensor, Tensor)> {
        match (self.kv_cache.k(), self.kv_cache.v()) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }

    /// Replace this rank's live KV cache with a snapshot.
    pub fn restore_kv(&mut self, snap: Option<&(Tensor, Tensor)>) -> candle_core::Result<()> {
        self.kv_cache.reset();
        if let Some((k, v)) = snap {
            self.kv_cache.append(k, v)?;
        }
        Ok(())
    }
}

// ─── MLP ────────────────────────────────────────────────────────────

pub(crate) struct TpQwen3_5MLP {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
}

impl TpQwen3_5MLP {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        if !cfg.intermediate_size.is_multiple_of(world_size as usize) {
            bail!(
                "intermediate_size {} not divisible by world_size {}",
                cfg.intermediate_size,
                world_size
            );
        }
        Ok(Self {
            gate_proj: ColumnParallelLinear::load_with_quant(
                &vb.pp("gate_proj"),
                rank,
                world_size,
                quant,
            )?,
            up_proj: ColumnParallelLinear::load_with_quant(
                &vb.pp("up_proj"),
                rank,
                world_size,
                quant,
            )?,
            down_proj: RowParallelLinear::load_with_quant(
                &vb.pp("down_proj"),
                rank,
                world_size,
                comm,
                quant,
            )?,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        if !cfg.intermediate_size.is_multiple_of(world_size as usize) {
            bail!(
                "intermediate_size {} not divisible by world_size {}",
                cfg.intermediate_size,
                world_size
            );
        }
        Ok(Self {
            gate_proj: ColumnParallelLinear::load_with_quant(
                &vb.pp("gate_proj"),
                rank,
                world_size,
                quant,
            )?,
            up_proj: ColumnParallelLinear::load_with_quant(
                &vb.pp("up_proj"),
                rank,
                world_size,
                quant,
            )?,
            down_proj: RowParallelLinear::load_with_quant(
                &vb.pp("down_proj"),
                rank,
                world_size,
                quant,
            )?,
        })
    }
}

impl Module for TpQwen3_5MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let rhs = self.up_proj.forward(x)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ─── MoE FFN (qwen3_next 80B-A3B, #92) ──────────────────────────────

/// One routed expert's per-rank slice: `gate_proj`/`up_proj`
/// column-sharded along `moe_intermediate_size`, `down_proj`
/// input-sharded — `forward_partial` yields a PARTIAL hidden output.
///
/// Deliberately NOT built from [`RowParallelLinear`]: that embeds an
/// AllReduce per call, which with top-10 routing would mean ten
/// collectives per layer. Partial sums from every selected expert and
/// the shared expert add linearly, so the whole MoE block needs
/// exactly ONE AllReduce at the end — preserving the existing
/// one-reduce-per-FFN pattern.
struct TpExpert {
    gate_proj: super::tp_linear::MaybeQuantLinear,
    up_proj: super::tp_linear::MaybeQuantLinear,
    down_proj: super::tp_linear::MaybeQuantLinear,
}

impl TpExpert {
    fn load(
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let col = |name: &str| -> Result<super::tp_linear::MaybeQuantLinear> {
            let w = vb
                .pp(name)
                .get_with_hints((), "weight", super::tp_linear::shard(0, rank, world_size))
                .with_context(|| format!("load expert '{}/{name}'", vb.prefix()))?;
            super::tp_linear::MaybeQuantLinear::from_weight(w, quant)
        };
        let gate_proj = col("gate_proj")?;
        let up_proj = col("up_proj")?;
        let down_w = vb
            .pp("down_proj")
            .get_with_hints((), "weight", super::tp_linear::shard(1, rank, world_size))
            .with_context(|| format!("load expert '{}/down_proj'", vb.prefix()))?;
        let down_proj = super::tp_linear::MaybeQuantLinear::from_weight(down_w, quant)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// SwiGLU over this rank's intermediate slice; the output is a
    /// partial sum awaiting the block-end AllReduce.
    fn forward_partial(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

/// How the routed experts are stored and dispatched (#92 slice 4).
///
/// `Scatter` is the correctness-first path: per-expert
/// `MaybeQuantLinear`s driven by a host-side token scatter. Correct
/// everywhere, but at batch-1 decode it costs a GPU→CPU routing sync
/// plus ~top-k tiny GEMV launches per layer per token — measured at
/// **4.3 tok/s** on the 80B (vs ~27 for the dense 27B).
///
/// `Fused` holds each projection as ONE stacked per-rank `QTensor`
/// (`[num_experts, out/ws, in]`) driven by candle-nn's grouped-GEMM
/// GGUF kernels (`moe_gemm_gguf`): routing, index sort, and all expert
/// GEMMs stay on-device — three kernel launches per layer regardless
/// of k. Chosen at load on CUDA devices when ISQ is active and the
/// block dims satisfy the GGUF constraints; `NEURON_MOE_FUSED=0`
/// forces Scatter as an escape hatch / A-B lever.
enum TpExpertStore {
    Scatter(Vec<TpExpert>),
    #[cfg(feature = "cuda")]
    Fused {
        gate_experts: Arc<candle_core::quantized::QTensor>,
        up_experts: Arc<candle_core::quantized::QTensor>,
        down_experts: Arc<candle_core::quantized::QTensor>,
        num_experts: usize,
    },
}

/// TP counterpart of `arch::qwen3_5::moe::Qwen3_5MoeBlock`. The router
/// and the shared-expert sigmoid gate are replicated (tiny; every rank
/// computes identical routing with zero communication); expert FFNs
/// shard per-expert along the intermediate dim; one AllReduce at block
/// end recovers the full activation.
pub(crate) struct TpQwen3_5MoeBlock {
    gate: candle_nn::Linear,
    experts: TpExpertStore,
    shared_expert: Option<TpExpert>,
    shared_expert_gate: Option<candle_nn::Linear>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    #[cfg(feature = "cuda")]
    all_reduce: super::all_reduce::AllReduce,
    needs_reduce: bool,
}

impl TpQwen3_5MoeBlock {
    fn check_and_gate(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        world_size: u32,
    ) -> Result<candle_nn::Linear> {
        let ws = world_size as usize;
        if !cfg.moe_intermediate_size.is_multiple_of(ws) {
            bail!(
                "moe_intermediate_size {} not divisible by world_size {ws}",
                cfg.moe_intermediate_size
            );
        }
        if cfg.shared_expert_intermediate_size > 0
            && !cfg.shared_expert_intermediate_size.is_multiple_of(ws)
        {
            bail!(
                "shared_expert_intermediate_size {} not divisible by world_size {ws}",
                cfg.shared_expert_intermediate_size
            );
        }
        let gate_w = load_replicated(&vb.pp("gate"), (cfg.num_experts, cfg.hidden_size), "weight")?;
        Ok(candle_nn::Linear::new(gate_w, None))
    }

    fn load_experts_and_shared(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<(TpExpertStore, Option<TpExpert>, Option<candle_nn::Linear>)> {
        let experts = Self::load_expert_store(cfg, vb, rank, world_size, quant)?;
        let (shared_expert, shared_expert_gate) = if cfg.shared_expert_intermediate_size > 0 {
            let shared = TpExpert::load(&vb.pp("shared_expert"), rank, world_size, quant)
                .context("load TP shared_expert")?;
            let gate_w =
                load_replicated(&vb.pp("shared_expert_gate"), (1, cfg.hidden_size), "weight")?;
            (Some(shared), Some(candle_nn::Linear::new(gate_w, None)))
        } else {
            (None, None)
        };
        Ok((experts, shared_expert, shared_expert_gate))
    }

    fn load_expert_store(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<TpExpertStore> {
        #[cfg(feature = "cuda")]
        if Self::fused_eligible(cfg, vb, world_size, quant) {
            return Self::load_fused_experts(cfg, vb, rank, world_size, quant);
        }
        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for i in 0..cfg.num_experts {
            experts.push(
                TpExpert::load(&experts_vb.pp(i), rank, world_size, quant)
                    .with_context(|| format!("load TP expert {i}"))?,
            );
        }
        Ok(TpExpertStore::Scatter(experts))
    }

    /// Whether the fused grouped-GEMM path can serve this block: CUDA
    /// device, ISQ active with a kernel-supported GGML dtype, GGUF
    /// block alignment on both GEMM K dims (hidden for gate/up, the
    /// per-rank intermediate slice for down), and not vetoed by the
    /// `NEURON_MOE_FUSED=0` escape hatch.
    #[cfg(feature = "cuda")]
    fn fused_eligible(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> bool {
        if std::env::var("NEURON_MOE_FUSED").is_ok_and(|v| v == "0" || v == "false") {
            tracing::info!("NEURON_MOE_FUSED=0 — MoE block using scatter dispatch");
            return false;
        }
        if !vb.device().is_cuda() {
            return false;
        }
        let Some(q) = quant else { return false };
        let kernel_supported = matches!(
            q,
            GgmlDType::Q8_0
                | GgmlDType::Q4K
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        );
        let per_rank_inter = cfg.moe_intermediate_size / world_size as usize;
        let aligned = cfg.hidden_size.is_multiple_of(q.block_size())
            && per_rank_inter.is_multiple_of(q.block_size());
        if !kernel_supported || !aligned {
            tracing::warn!(
                quant = ?q,
                hidden = cfg.hidden_size,
                per_rank_inter,
                "MoE fused path ineligible — falling back to scatter dispatch"
            );
            return false;
        }
        true
    }

    /// Build the stacked per-rank expert QTensors for the fused path:
    /// read each expert's rank slice, stack into `[E, out/ws, in]`
    /// (gate/up) and `[E, hidden, inter/ws]` (down), ISQ the stack in
    /// one parallel pass per projection. Transient cost: one bf16
    /// stack per projection (~0.5 GB at 80B dims) alive on-device
    /// until its QTensor replaces it.
    #[cfg(feature = "cuda")]
    fn load_fused_experts(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<TpExpertStore> {
        let q = quant.expect("fused_eligible ensured quant");
        let experts_vb = vb.pp("experts");

        let stack_proj = |name: &str, shard_dim: usize| -> Result<Tensor> {
            let mut slices = Vec::with_capacity(cfg.num_experts);
            for i in 0..cfg.num_experts {
                let w = experts_vb
                    .pp(i)
                    .pp(name)
                    .get_with_hints(
                        (),
                        "weight",
                        super::tp_linear::shard(shard_dim, rank, world_size),
                    )
                    .with_context(|| format!("load expert {i} '{name}' rank slice"))?;
                slices.push(w);
            }
            Tensor::stack(&slices, 0).with_context(|| format!("stack {name} experts"))
        };

        let quantize =
            |stack: Tensor, name: &str| -> Result<Arc<candle_core::quantized::QTensor>> {
                let qt = super::isq::quantize_parallel(&stack, q)
                    .with_context(|| format!("ISQ {name} expert stack to {q:?}"))?;
                Ok(Arc::new(qt))
            };

        let gate_experts = quantize(stack_proj("gate_proj", 0)?, "gate_proj")?;
        let up_experts = quantize(stack_proj("up_proj", 0)?, "up_proj")?;
        let down_experts = quantize(stack_proj("down_proj", 1)?, "down_proj")?;

        Ok(TpExpertStore::Fused {
            gate_experts,
            up_experts,
            down_experts,
            num_experts: cfg.num_experts,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let gate = Self::check_and_gate(cfg, vb, world_size)?;
        let (experts, shared_expert, shared_expert_gate) =
            Self::load_experts_and_shared(cfg, vb, rank, world_size, quant)?;
        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            all_reduce: super::all_reduce::AllReduce::new(comm),
            needs_reduce: world_size > 1,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let gate = Self::check_and_gate(cfg, vb, world_size)?;
        let (experts, shared_expert, shared_expert_gate) =
            Self::load_experts_and_shared(cfg, vb, rank, world_size, quant)?;
        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            needs_reduce: world_size > 1,
        })
    }
}

impl TpQwen3_5MoeBlock {
    /// Correctness-first dispatch: shared `route_scatter` host routing
    /// with per-expert GEMMs. Returns the rank's PARTIAL routed
    /// output, `(tokens, hidden)`, in the input dtype.
    fn forward_scatter(
        &self,
        experts: &[TpExpert],
        xs_flat: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (tokens_for, weights_for) = crate::harness::arch::qwen3_5::moe::route_scatter(
            &self.gate,
            xs_flat,
            experts.len(),
            self.num_experts_per_tok,
            self.norm_topk_prob,
        )?;

        let mut ys = xs_flat.zeros_like()?;
        for (e, expert) in experts.iter().enumerate() {
            if tokens_for[e].is_empty() {
                continue;
            }
            let rows = Tensor::new(tokens_for[e].as_slice(), xs_flat.device())?;
            let picked = xs_flat.index_select(&rows, 0)?;
            let out = expert.forward_partial(&picked)?;
            let w = Tensor::new(weights_for[e].as_slice(), xs_flat.device())?
                .to_dtype(out.dtype())?
                .reshape(((), 1))?;
            ys = ys.index_add(&rows, &out.broadcast_mul(&w)?, 0)?;
        }
        Ok(ys)
    }

    /// Fused grouped-GEMM dispatch (#92 slice 4): routing, index sort,
    /// and all expert GEMMs stay on-device — the port of
    /// candle-transformers' `FusedMoeGGUF::forward` onto per-rank
    /// expert stacks. Returns the rank's PARTIAL routed output,
    /// `(tokens, hidden)`, in the input dtype.
    ///
    /// Kernel contract (candle-nn `moe_gemm_gguf`): decode wants an
    /// F32 input and always emits F32; the `dtype` argument only
    /// selects the f16/bf16 conversion used by the prefill kernel.
    /// gate/up run with `topk_weights: None` → `tokens×topk` output
    /// rows; the down GEMM folds the routing weights in-kernel and the
    /// final `(tokens, topk, hidden)` view sums over `topk`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_fused(
        &self,
        gate_experts: &candle_core::quantized::QTensor,
        up_experts: &candle_core::quantized::QTensor,
        down_experts: &candle_core::quantized::QTensor,
        num_experts: usize,
        xs_flat: &Tensor,
        is_prefill: bool,
    ) -> candle_core::Result<Tensor> {
        use candle_core::D;

        let original_dtype = xs_flat.dtype();
        let n_tokens = xs_flat.dim(0)?;
        let xs_f32 = if original_dtype == DType::F32 {
            xs_flat.clone()
        } else {
            xs_flat.to_dtype(DType::F32)?
        };

        // Replicated routing, all on-device (contrast route_scatter's
        // host round-trip): softmax over all experts → top-k → renorm.
        // The router runs in the ORIGINAL activation dtype — its
        // replicated weight is bf16, and a bf16×f32 matmul is a dtype
        // error (the first live fused request poisoned on exactly
        // this). Only the softmax and everything downstream are f32,
        // matching route_scatter.
        let router_logits = self.gate.forward(xs_flat)?;
        let probs = candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;
        let topk_ids = probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let mut topk_weights = probs.gather(&topk_ids, D::Minus1)?;
        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;
        let _ = num_experts;

        let gate = candle_nn::moe::moe_gemm_gguf(
            &xs_f32,
            gate_experts,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            DType::BF16,
        )?;
        let up = candle_nn::moe::moe_gemm_gguf(
            &xs_f32,
            up_experts,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            DType::BF16,
        )?;
        let down_inputs = (up * candle_nn::ops::silu(&gate)?)?;
        let ys = candle_nn::moe::moe_gemm_gguf(
            &down_inputs,
            down_experts,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            DType::BF16,
        )?;

        let hidden = xs_flat.dim(1)?;
        let ys = ys.reshape((n_tokens, (), hidden))?.sum(D::Minus2)?;
        if ys.dtype() == original_dtype {
            Ok(ys)
        } else {
            ys.to_dtype(original_dtype)
        }
    }
}

impl Module for TpQwen3_5MoeBlock {
    /// Route + expert dispatch (scatter or fused per the store), then
    /// the shared expert's sigmoid-gated partial, then ONE AllReduce
    /// recovering the full activation. The shared expert's per-token
    /// mix is a replicated scalar, so scaling the partial slice
    /// commutes with the reduce.
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;

        let mut ys = match &self.experts {
            TpExpertStore::Scatter(experts) => self.forward_scatter(experts, &xs_flat)?,
            #[cfg(feature = "cuda")]
            TpExpertStore::Fused {
                gate_experts,
                up_experts,
                down_experts,
                num_experts,
            } => self.forward_fused(
                gate_experts,
                up_experts,
                down_experts,
                *num_experts,
                &xs_flat,
                l > 1,
            )?,
        };

        if let (Some(shared), Some(gate)) = (&self.shared_expert, &self.shared_expert_gate) {
            let mix = candle_nn::ops::sigmoid(&gate.forward(&xs_flat)?)?;
            let shared_out = shared.forward_partial(&xs_flat)?.broadcast_mul(&mix)?;
            ys = (ys + shared_out)?;
        }

        #[cfg(feature = "cuda")]
        if self.needs_reduce {
            ys = ys.apply_op1_no_bwd(&self.all_reduce)?;
        }
        let _ = self.needs_reduce;
        ys.reshape((b, l, hidden))
    }
}

/// The FFN slot: dense SwiGLU (Qwen3.6) or the sharded MoE block
/// (qwen3_next 80B-A3B, #92) — mirrors the single-GPU `MlpKind`.
enum TpMlpKind {
    Dense(TpQwen3_5MLP),
    Moe(TpQwen3_5MoeBlock),
}

impl Module for TpMlpKind {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            TpMlpKind::Dense(mlp) => mlp.forward(x),
            TpMlpKind::Moe(moe) => moe.forward(x),
        }
    }
}

// ─── decoder layer ──────────────────────────────────────────────────

enum TpAttentionKind {
    Full(TpQwen3_5Attention),
    Linear(TpQwen3_5GatedDeltaNet),
}

pub struct TpQwen3_5DecoderLayer {
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
    mlp: TpMlpKind,
    attention: TpAttentionKind,
}

impl TpQwen3_5DecoderLayer {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .map(String::as_str)
            .ok_or_else(|| anyhow::anyhow!("layer_types[{layer_idx}] missing"))?;
        let attention = match layer_type {
            "full_attention" => TpAttentionKind::Full(TpQwen3_5Attention::load(
                cfg,
                rotary,
                &vb.pp("self_attn"),
                rank,
                world_size,
                comm.clone(),
                quant,
            )?),
            "linear_attention" => TpAttentionKind::Linear(TpQwen3_5GatedDeltaNet::load(
                cfg,
                &vb.pp("linear_attn"),
                mmap,
                rank,
                world_size,
                comm.clone(),
                quant,
            )?),
            other => bail!("unknown layer_type '{other}' for layer {layer_idx}"),
        };
        let mlp = if cfg.layer_uses_moe(layer_idx) {
            TpMlpKind::Moe(TpQwen3_5MoeBlock::load(
                cfg,
                &vb.pp("mlp"),
                rank,
                world_size,
                comm,
                quant,
            )?)
        } else {
            TpMlpKind::Dense(TpQwen3_5MLP::load(
                cfg,
                &vb.pp("mlp"),
                rank,
                world_size,
                comm,
                quant,
            )?)
        };
        let input_layernorm =
            Qwen3_5RmsNorm::load(&vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RmsNorm::load(
            &vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            attention,
        })
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .map(String::as_str)
            .ok_or_else(|| anyhow::anyhow!("layer_types[{layer_idx}] missing"))?;
        let attention = match layer_type {
            "full_attention" => TpAttentionKind::Full(TpQwen3_5Attention::load(
                cfg,
                rotary,
                &vb.pp("self_attn"),
                rank,
                world_size,
                quant,
            )?),
            "linear_attention" => TpAttentionKind::Linear(TpQwen3_5GatedDeltaNet::load(
                cfg,
                &vb.pp("linear_attn"),
                mmap,
                rank,
                world_size,
                quant,
            )?),
            other => bail!("unknown layer_type '{other}' for layer {layer_idx}"),
        };
        let mlp = if cfg.layer_uses_moe(layer_idx) {
            TpMlpKind::Moe(TpQwen3_5MoeBlock::load(
                cfg,
                &vb.pp("mlp"),
                rank,
                world_size,
                quant,
            )?)
        } else {
            TpMlpKind::Dense(TpQwen3_5MLP::load(
                cfg,
                &vb.pp("mlp"),
                rank,
                world_size,
                quant,
            )?)
        };
        let input_layernorm =
            Qwen3_5RmsNorm::load(&vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RmsNorm::load(
            &vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            attention,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attention {
            TpAttentionKind::Full(attn) => attn.forward(&h, attn_mask, cos, sin)?,
            TpAttentionKind::Linear(net) => net.forward(&h)?,
        };
        let x = (x + attn_out)?;
        let h2 = self.post_attention_layernorm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.attention {
            TpAttentionKind::Full(a) => a.clear_kv_cache(),
            TpAttentionKind::Linear(n) => n.clear_kv_cache(),
        }
    }

    /// Capture this layer's per-rank cache state for a prefix
    /// snapshot. Reuses the single-GPU snapshot types — the shard
    /// state has the same shape, just sharded head dims.
    pub fn snapshot_kv(&self) -> candle_core::Result<LayerKvSnapshot> {
        Ok(match &self.attention {
            TpAttentionKind::Full(a) => LayerKvSnapshot::Full(a.snapshot_kv()),
            TpAttentionKind::Linear(n) => {
                let (conv_state, recurrent_state) = n.snapshot_state()?;
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                }
            }
        })
    }

    /// Replace this layer's per-rank cache state from a snapshot.
    pub fn restore_kv(&mut self, snap: &LayerKvSnapshot) -> candle_core::Result<()> {
        match (&mut self.attention, snap) {
            (TpAttentionKind::Full(a), LayerKvSnapshot::Full(kv)) => a.restore_kv(kv.as_ref()),
            (
                TpAttentionKind::Linear(n),
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => n.restore_state(conv_state.as_ref(), recurrent_state.as_ref()),
            _ => candle_core::bail!(
                "restore_kv: snapshot layer kind does not match this layer's attention kind"
            ),
        }
    }
}

// ─── base Model ─────────────────────────────────────────────────────

pub struct TpQwen3_5Model {
    embed_tokens: Embedding,
    layers: Vec<TpQwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    /// Replicated rotary, shared with every full-attention layer. The
    /// model builds the per-forward cos/sin (interleaved M-RoPE for image
    /// tokens, plain for text) once and the layers apply it. Identical on
    /// every rank, so per-rank position ids stay consistent.
    rotary: Arc<RotaryEmbedding>,
    /// `offset + rope_delta` is the text-axis decode position; set from
    /// `get_rope_index` during a vision prefill, reset in `clear_kv_cache`.
    /// See `Qwen3_5Model::rope_delta`.
    rope_delta: i64,
    device: Device,
    dtype: DType,
}

impl TpQwen3_5Model {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
        text_prefix: &str,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let text_vb = vb.pp(text_prefix);

        let embed_weight = load_replicated(
            &text_vb.pp("embed_tokens"),
            (cfg.vocab_size, cfg.hidden_size),
            "weight",
        )?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);

        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "layer_types must have num_hidden_layers ({}) entries; got {}",
                cfg.num_hidden_layers,
                cfg.layer_types.len()
            );
        }

        let vb_l = text_vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        log_vram(&device, rank, "before layer 0");
        // Per-phase timing (#1): the layer loop is where ISQ cost
        // concentrates; the per-layer line is debug, the loop total
        // info, so journalctl always shows where a cold load went.
        let layers_start = std::time::Instant::now();
        for i in 0..cfg.num_hidden_layers {
            let layer_start = std::time::Instant::now();
            let layer = TpQwen3_5DecoderLayer::load(
                cfg,
                rotary.clone(),
                i,
                &vb_l.pp(i),
                mmap,
                rank,
                world_size,
                comm.clone(),
                quant,
            )
            .with_context(|| {
                let (free_mb, total_mb) = cuda_mem_mb(&device);
                format!("load layer {i} (rank {rank}): free={free_mb}MB / total={total_mb}MB")
            })?;
            layers.push(layer);
            tracing::debug!(
                rank,
                layer = i,
                elapsed_ms = layer_start.elapsed().as_millis() as u64,
                "TP layer loaded"
            );
            log_vram(&device, rank, &format!("after layer {i}"));
        }
        tracing::info!(
            rank,
            layers = cfg.num_hidden_layers,
            elapsed_ms = layers_start.elapsed().as_millis() as u64,
            "TP layer loop complete"
        );

        let norm = Qwen3_5RmsNorm::load(&text_vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            rope_delta: 0,
            device,
            dtype,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
        text_prefix: &str,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let text_vb = vb.pp(text_prefix);

        let embed_weight = load_replicated(
            &text_vb.pp("embed_tokens"),
            (cfg.vocab_size, cfg.hidden_size),
            "weight",
        )?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);

        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "layer_types must have num_hidden_layers ({}) entries; got {}",
                cfg.num_hidden_layers,
                cfg.layer_types.len()
            );
        }

        let vb_l = text_vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_start = std::time::Instant::now();
        for i in 0..cfg.num_hidden_layers {
            layers.push(TpQwen3_5DecoderLayer::load(
                cfg,
                rotary.clone(),
                i,
                &vb_l.pp(i),
                mmap,
                rank,
                world_size,
                quant,
            )?);
        }
        tracing::info!(
            rank,
            layers = cfg.num_hidden_layers,
            elapsed_ms = layers_start.elapsed().as_millis() as u64,
            "TP layer loop complete"
        );

        let norm = Qwen3_5RmsNorm::load(&text_vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            rope_delta: 0,
            device,
            dtype,
        })
    }

    pub fn embed_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
        self.rope_delta = 0;
    }

    /// Capture this rank's per-layer cache state plus the rope
    /// position counter as one consistent prefix snapshot (#11).
    /// Mirrors `Qwen3_5Model::snapshot_kv_cache`.
    pub fn snapshot_kv_cache(&self) -> candle_core::Result<KvCacheSnapshot> {
        let layers = self
            .layers
            .iter()
            .map(|l| l.snapshot_kv())
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(KvCacheSnapshot {
            layers,
            rope_delta: self.rope_delta,
        })
    }

    /// Replace this rank's live cache state with a snapshot. The
    /// snapshot stays valid for further restores.
    pub fn restore_kv_cache(&mut self, snap: &KvCacheSnapshot) -> candle_core::Result<()> {
        if snap.layers.len() != self.layers.len() {
            candle_core::bail!(
                "restore_kv_cache: snapshot has {} layers, model has {}",
                snap.layers.len(),
                self.layers.len()
            );
        }
        for (layer, layer_snap) in self.layers.iter_mut().zip(snap.layers.iter()) {
            layer.restore_kv(layer_snap)?;
        }
        self.rope_delta = snap.rope_delta;
        Ok(())
    }

    /// Set the decode `rope_delta` computed by `get_rope_index` during a
    /// vision prefill, so decode after the image resumes text positions
    /// from the image-compressed counter.
    pub fn set_rope_delta(&mut self, delta: i64) {
        self.rope_delta = delta;
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> candle_core::Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        self.forward_inner(input, offset, None, None, None)
    }

    /// Forward for a vision-prefill chunk: optional image-embedding
    /// splice plus explicit interleaved-M-RoPE `position_ids` (the
    /// chunk's slice of the full prompt's 3D positions). Used by
    /// `TpQwen3_5ForCausalLM::prefill_with_images_chunked`, which
    /// computes the positions once over the whole prompt and slices them
    /// per chunk so every rank steps in lockstep.
    pub fn forward_with_positions(
        &mut self,
        input: &Tensor,
        offset: usize,
        position_ids: &Tensor,
        image_embeds: Option<&Tensor>,
        image_token_id: Option<u32>,
    ) -> candle_core::Result<Tensor> {
        self.forward_inner(
            input,
            offset,
            image_embeds,
            image_token_id,
            Some(position_ids),
        )
    }

    /// Shared forward. Splices image embeddings at `image_token_id`
    /// positions when present, then builds the rotary cos/sin — from the
    /// explicit `position_ids` (interleaved M-RoPE, vision) when given,
    /// else plain positions at `offset + rope_delta` (text / decode) —
    /// and runs the sharded decoder stack. The TP replicated-hidden-state
    /// invariant holds because every rank encodes the same pixels and
    /// computes the same positions.
    fn forward_inner(
        &mut self,
        input: &Tensor,
        offset: usize,
        image_embeds: Option<&Tensor>,
        image_token_id: Option<u32>,
        position_ids: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        if let (Some(img), Some(tok_id)) = (image_embeds, image_token_id) {
            let ids: Vec<u32> = input.flatten_all()?.to_vec1()?;
            let mut positions: Vec<u32> = Vec::with_capacity(img.dim(0)?);
            for (idx, id) in ids.iter().enumerate() {
                if *id == tok_id {
                    positions.push(idx as u32);
                }
            }
            let n_img_tokens = img.dim(0)?;
            if positions.len() != n_img_tokens {
                candle_core::bail!(
                    "TP forward: chunk has {} image-token positions but image_embeds carries \
                     {} tokens — patch-count expansion / chunk slicing mismatch",
                    positions.len(),
                    n_img_tokens,
                );
            }
            if !positions.is_empty() {
                let img = img.to_dtype(self.dtype)?;
                h = splice_runs(&h, &img, &positions)?;
            }
        }

        let (cos, sin) = match position_ids {
            Some(pos) => self.rotary.mrope_cos_sin(pos)?,
            None => {
                let base = (offset as i64 + self.rope_delta).max(0) as usize;
                self.rotary.plain_cos_sin(base, l)?
            }
        };

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), &cos, &sin)?;
        }
        self.norm.forward(&h)
    }
}

pub struct TpQwen3_5ForCausalLM {
    base: TpQwen3_5Model,
    lm_head: super::tp_linear::MaybeQuantLinear,
    /// Replicated vision tower (TP-vision). Loaded on every rank from
    /// the full, unsharded `model.visual.*` weights; `None` for
    /// text-only checkpoints. Each rank encodes the same image
    /// independently — no sharding, no broadcast — which keeps the
    /// spliced input embeddings identical across ranks (the
    /// replicated-hidden-state invariant the sharded layers rely on).
    vision: Option<VisionTower>,
    /// `<|image_pad|>` sentinel id (mirrors `Config::image_token_id`);
    /// the splice target for `forward_with_vision`.
    image_token_id: Option<u32>,
}

/// Load the replicated vision tower from the unsharded `model.visual.*`
/// weights when the config carries a `vision_config` block. Shared by
/// the cuda and non-cuda `load` variants. `vb.pp("model.visual")`
/// resolves against the same full safetensors every rank mmaps; plain
/// `.get()` on a `ShardedVarBuilder` returns the full (replicated)
/// tensor, so this loads identically regardless of `world_size`.
fn load_replicated_vision_tower(
    config: &Config,
    vb: &ShardedVarBuilder,
) -> Result<Option<VisionTower>> {
    match config.vision_config.clone() {
        Some(vcfg) => {
            tracing::info!(
                depth = vcfg.depth,
                hidden_size = vcfg.hidden_size,
                "loading qwen3_5 vision tower (TP replicated)"
            );
            let tower = VisionTower::load(vcfg, vb.pp("model.visual"))
                .context("load qwen3_5 vision tower (model.visual.*) [TP replicated]")?;
            Ok(Some(tower))
        }
        None => Ok(None),
    }
}

impl TpQwen3_5ForCausalLM {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        config: Config,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let cfg = &config.text_config;
        let text_prefix = crate::harness::arch::qwen3_5::text_weight_prefix(&config.model_type);
        let base = TpQwen3_5Model::load(cfg, vb, mmap, rank, world_size, comm, quant, text_prefix)?;
        let lm_head = build_lm_head(cfg, vb, &base, quant)?;
        let vision = load_replicated_vision_tower(&config, vb)?;
        let image_token_id = config.image_token_id;
        let model = Self {
            base,
            lm_head,
            vision,
            image_token_id,
        };
        log_construction_complete(cfg, rank, world_size, quant, model.device());
        Ok(model)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        config: Config,
        vb: &ShardedVarBuilder,
        mmap: &MmapedSafetensors,
        rank: u32,
        world_size: u32,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let cfg = &config.text_config;
        let text_prefix = crate::harness::arch::qwen3_5::text_weight_prefix(&config.model_type);
        let base = TpQwen3_5Model::load(cfg, vb, mmap, rank, world_size, quant, text_prefix)?;
        let lm_head = build_lm_head(cfg, vb, &base, quant)?;
        let vision = load_replicated_vision_tower(&config, vb)?;
        let image_token_id = config.image_token_id;
        let model = Self {
            base,
            lm_head,
            vision,
            image_token_id,
        };
        log_construction_complete(cfg, rank, world_size, quant, model.device());
        Ok(model)
    }

    /// True when this TP load materialised a replicated vision tower.
    /// Drives capability advertising and the Stage 3 vision dispatch.
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// `<|image_pad|>` sentinel id, when known.
    pub fn image_token_id(&self) -> Option<u32> {
        self.image_token_id
    }

    /// Encode one preprocessed `(C, H, W)` image into LM-side patch
    /// embeddings `(N_lm, hidden)` via this rank's replicated tower.
    /// Errors when loaded without a vision tower.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        self.vision
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "encode_image: this TP Qwen3.6 load has no vision tower \
                     (config.json::vision_config absent or weights missing)"
                )
            })?
            .forward(image)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden = self.base.forward(input, offset)?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    /// Forward for a vision-prefill chunk (optional image splice +
    /// explicit interleaved-M-RoPE `position_ids`). Mirrors `forward`
    /// but routes through `TpQwen3_5Model::forward_with_positions`.
    pub fn forward_with_positions(
        &mut self,
        input: &Tensor,
        offset: usize,
        position_ids: &Tensor,
        image_embeds: Option<&Tensor>,
        image_token_id: Option<u32>,
    ) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden = self.base.forward_with_positions(
            input,
            offset,
            position_ids,
            image_embeds,
            image_token_id,
        )?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    /// End-to-end image prefill on one rank: encode each preprocessed
    /// `(C, H, W)` pixel tensor through this rank's replicated tower,
    /// concatenate the per-image embeddings along the patch axis, and
    /// forward with the splice. Shared by the leader (`TpLeaderModel`)
    /// and the subprocess worker (`WorkerModel`) so every rank runs the
    /// identical encode → splice → forward and keeps the replicated
    /// hidden state in lockstep. Returns last-position logits
    /// `(B, 1, vocab)`, same contract as `forward`.
    /// Encode every preprocessed `(C,H,W)` image once through this
    /// rank's replicated tower and concatenate along the patch axis →
    /// `(sum_patches, hidden)`. Done once per prefill, not per chunk.
    fn encode_images_concat(&self, image_pixels: &[Tensor]) -> candle_core::Result<Tensor> {
        let mut per_image = Vec::with_capacity(image_pixels.len());
        for (idx, img) in image_pixels.iter().enumerate() {
            let embed = self
                .encode_image(img)
                .map_err(|e| candle_core::Error::Msg(format!("encode image[{idx}]: {e:#}")))?;
            per_image.push(embed);
        }
        Tensor::cat(&per_image.iter().collect::<Vec<_>>(), 0)
    }

    /// Chunked image prefill on one rank. Encodes the image(s) once,
    /// then walks the (pre-expanded) prompt in `chunk_size`-token
    /// windows — exactly like the text `chunked_prefill_tp` — splicing
    /// the patch embeddings into whichever chunk(s) carry `<|image_pad|>`
    /// positions. Activation memory is bounded by the chunk, not the
    /// full prompt, so a long vision context no longer single-shot-OOMs.
    ///
    /// Every rank runs the identical chunk sequence (same `tokens.len()`
    /// and `chunk_size`), so the row-parallel `AllReduce`s pair up
    /// chunk-by-chunk across ranks with no extra synchronisation. The KV
    /// cache accumulates across chunks via the growing offset; only the
    /// final chunk's last-position logits are returned (intermediate
    /// chunks just populate the cache, same as the text path).
    pub fn prefill_with_images_chunked(
        &mut self,
        tokens: &[u32],
        base_offset: usize,
        image_pixels: &[Tensor],
        image_token_id: u32,
        chunk_size: usize,
    ) -> candle_core::Result<Tensor> {
        if image_pixels.is_empty() {
            candle_core::bail!("prefill_with_images_chunked: called with zero images");
        }
        if tokens.is_empty() {
            candle_core::bail!("prefill_with_images_chunked: empty prompt");
        }
        let chunk_size = chunk_size.max(1);
        let device = self.device().clone();
        let image_embeds = self.encode_images_concat(image_pixels)?;

        // Each image's LM grid (lm_gh, lm_gw) = (h/factor, w/factor),
        // factor = patch×merge. Recomputed per rank from this rank's own
        // pixel tensors — deterministic, so every rank's grids (and hence
        // M-RoPE positions) match without crossing the RPC (#14).
        let factor = self
            .vision
            .as_ref()
            .map(|v| {
                let c = v.config();
                c.patch_size * c.spatial_merge_size
            })
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "prefill_with_images_chunked: loaded without a vision tower".into(),
                )
            })?;
        let grids: Vec<(usize, usize)> = image_pixels
            .iter()
            .map(|t| {
                let (_, h, w) = t.dims3()?;
                Ok::<(usize, usize), candle_core::Error>((h / factor, w / factor))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        // Interleaved-M-RoPE 3D position ids for the whole prompt,
        // computed once and sliced per chunk so every rank assigns image
        // tokens their grid coordinates (and text after an image resumes
        // from the compressed counter). `rope_delta` is stored on the base
        // model for the decode that follows this prefill. Every chunk —
        // text or image — uses the M-RoPE slice, because each image shifts
        // the positions of the text around it.
        let (text, height, width, delta) =
            crate::harness::arch::qwen3_5::rope::get_rope_index(tokens, image_token_id, &grids)
                .map_err(|e| candle_core::Error::Msg(format!("get_rope_index: {e}")))?;
        self.base.set_rope_delta(delta);
        let full_pos = crate::harness::arch::qwen3_5::rope::mrope_position_tensor(
            &text, &height, &width, &device,
        )?;

        let mut last_logits: Option<Tensor> = None;
        // Rows of `image_embeds` already spliced by earlier chunks. The
        // `<|image_pad|>` run is contiguous, so chunks consume embedding
        // rows in order.
        let mut img_off = 0usize;
        let mut start = 0usize;
        while start < tokens.len() {
            let end = (start + chunk_size).min(tokens.len());
            let chunk = &tokens[start..end];
            let input = Tensor::new(chunk, &device)?.unsqueeze(0)?;
            let pos_slice = full_pos.narrow(1, start, end - start)?;
            let n_here = chunk.iter().filter(|&&t| t == image_token_id).count();
            let logits = if n_here == 0 {
                self.forward_with_positions(&input, base_offset + start, &pos_slice, None, None)?
            } else {
                // Splice the next `n_here` patch rows at this chunk's
                // local image-pad positions.
                let rows = image_embeds.narrow(0, img_off, n_here)?;
                img_off += n_here;
                self.forward_with_positions(
                    &input,
                    base_offset + start,
                    &pos_slice,
                    Some(&rows),
                    Some(image_token_id),
                )?
            };
            last_logits = Some(logits);
            start = end;
        }
        last_logits
            .ok_or_else(|| candle_core::Error::Msg("prefill_with_images_chunked: no chunks".into()))
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }

    /// See [`TpQwen3_5Model::snapshot_kv_cache`].
    pub fn snapshot_kv_cache(&self) -> candle_core::Result<KvCacheSnapshot> {
        self.base.snapshot_kv_cache()
    }

    /// See [`TpQwen3_5Model::restore_kv_cache`].
    pub fn restore_kv_cache(&mut self, snap: &KvCacheSnapshot) -> candle_core::Result<()> {
        self.base.restore_kv_cache(snap)
    }

    pub fn device(&self) -> &Device {
        &self.base.device
    }
}

fn build_lm_head(
    cfg: &TextConfig,
    vb: &ShardedVarBuilder,
    base: &TpQwen3_5Model,
    quant: Option<GgmlDType>,
) -> Result<super::tp_linear::MaybeQuantLinear> {
    if cfg.tie_word_embeddings {
        // Tied: lm_head shares the embedding weight. Quantizing the
        // shared tensor would corrupt the embedding lookup, so keep
        // the lm_head plain even when `quant` is set. The memory win
        // is already taken: only one copy of the (vocab, hidden) weight
        // lives in VRAM in the tied case.
        super::tp_linear::MaybeQuantLinear::from_weight(base.embed_weight().clone(), None)
            .context("wrap tied lm_head")
    } else {
        // lm_head sits at the top level (sibling of `model.*`), NOT
        // under `model.language_model`.
        let lm_head_start = std::time::Instant::now();
        let weight = load_replicated(
            &vb.pp("lm_head"),
            (cfg.vocab_size, cfg.hidden_size),
            "weight",
        )?;
        let head = super::tp_linear::MaybeQuantLinear::from_weight(weight, quant)
            .context("wrap lm_head")?;
        tracing::info!(
            elapsed_ms = lm_head_start.elapsed().as_millis() as u64,
            quantized = quant.is_some(),
            "lm_head loaded"
        );
        Ok(head)
    }
}

// ─── load helpers ───────────────────────────────────────────────────

/// Load a tensor that's the SAME on every rank by asking the
/// ShardedVarBuilder with the default `Shard { world_size: 1 }` hint
/// (which falls through to the unsharded backend).
fn load_replicated<S: Into<candle_core::Shape>>(
    vb: &ShardedVarBuilder,
    shape: S,
    name: &str,
) -> Result<Tensor> {
    vb.get(shape, name)
        .with_context(|| format!("load replicated '{}/{name}'", vb.prefix()))
}

/// Query the cuda driver for free/total VRAM on the current device.
/// Returns `(free_mb, total_mb)`. Returns `(0, 0)` if the query fails
/// (so logging never crashes the load path).
#[cfg(feature = "cuda")]
fn cuda_mem_mb(device: &Device) -> (usize, usize) {
    use candle_core::cuda::cudarc::driver::result;
    use candle_core::cuda_backend::WrapErr;
    let Device::Cuda(dev) = device else {
        return (0, 0);
    };
    let Ok(()) = dev.cuda_stream().context().bind_to_thread().w() else {
        return (0, 0);
    };
    match result::mem_get_info() {
        Ok((free, total)) => (free / (1024 * 1024), total / (1024 * 1024)),
        Err(_) => (0, 0),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn cuda_mem_mb(_device: &Device) -> (usize, usize) {
    (0, 0)
}

/// Info-log the current device's free VRAM with a tag. No-op when the
/// query fails or on cpu.
#[cfg(feature = "cuda")]
fn log_vram(device: &Device, rank: u32, tag: &str) {
    let (free_mb, total_mb) = cuda_mem_mb(device);
    if total_mb > 0 {
        tracing::info!(
            target: "neuron::tp::load",
            rank,
            free_mb,
            total_mb,
            "{tag}"
        );
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn log_vram(_device: &Device, _rank: u32, _tag: &str) {}

/// Summary line emitted at end of `TpQwen3_5ForCausalLM::load`, after
/// the per-layer load loop AND after the lm_head + any post-construct
/// allocations. Logs the resolved config knobs (the ones an operator
/// would want to know when chasing a numerical or OOM issue) plus a
/// final free/total VRAM snapshot per rank.
///
/// The free_mb here is the most diagnostic number we have at this
/// stage: the gap between the last "after layer N" log and this line
/// is everything else the model construction allocated — lm_head,
/// embedding (if not tied), per-layer buffers held by candle's
/// allocator, the RotaryEmbedding tables, and any working space.
///
/// `kv_cache_per_layer_per_token_bytes` is a back-of-envelope estimate
/// — the actual cache grows as inference proceeds, but knowing the
/// per-token cost at this point lets an operator estimate "for a
/// 14k-token prompt I need ~X GB extra VRAM" without having to dig
/// into the architecture's attention modules.
fn log_construction_complete(
    cfg: &TextConfig,
    rank: u32,
    world_size: u32,
    quant: Option<GgmlDType>,
    device: &Device,
) {
    let (free_mb, total_mb) = cuda_mem_mb(device);
    // Distribution of attention kinds across layers. Qwen3-Next is
    // hybrid: most layers are linear (Gated DeltaNet), a few are full
    // softmax attention. Knowing the split at a glance helps when
    // reasoning about KV cache size — only full-attention layers
    // contribute to the standard kv cache.
    let mut full_attn_layers = 0;
    let mut linear_attn_layers = 0;
    for kind in &cfg.layer_types {
        match kind.as_str() {
            "full_attention" => full_attn_layers += 1,
            "linear_attention" => linear_attn_layers += 1,
            _ => {}
        }
    }
    // KV cache per-layer-per-token byte estimate for the per-rank
    // full-attention layers. bf16 = 2 bytes, K + V doubles it, and
    // sharded across world_size. Linear-attention layers carry a
    // fixed-size state instead of a growing cache.
    let per_rank_num_kv_heads = (cfg.num_key_value_heads / world_size as usize).max(1);
    // Only full-attention layers grow a KV cache (linear layers carry a
    // fixed-size recurrent state). Shared helper (#67) — the same
    // per-card math drives the derived context limit.
    let kv_bytes_per_token = crate::harness::context_limit::kv_bytes_per_token(
        full_attn_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        crate::harness::context_limit::KV_CACHE_DTYPE_BYTES,
        world_size,
    );
    tracing::info!(
        target: "neuron::tp::load",
        rank,
        world_size,
        quant = ?quant,
        free_mb,
        total_mb,
        vocab_size = cfg.vocab_size,
        hidden_size = cfg.hidden_size,
        num_hidden_layers = cfg.num_hidden_layers,
        num_attention_heads = cfg.num_attention_heads,
        num_key_value_heads = cfg.num_key_value_heads,
        head_dim = cfg.head_dim,
        max_position_embeddings = cfg.max_position_embeddings,
        full_attn_layers,
        linear_attn_layers,
        linear_num_value_heads = cfg.linear_num_value_heads,
        linear_num_key_heads = cfg.linear_num_key_heads,
        linear_key_head_dim = cfg.linear_key_head_dim,
        linear_value_head_dim = cfg.linear_value_head_dim,
        per_rank_num_kv_heads,
        kv_bytes_per_token,
        "Qwen3-Next model construction complete"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "cuda"))]
    use crate::harness::arch::qwen3_5::moe::Qwen3_5MoeBlock;
    #[cfg(not(feature = "cuda"))]
    use std::collections::HashMap;

    /// Write a tiny MoE-block checkpoint (router + experts + shared
    /// expert) and return a ShardedVarBuilder over it plus the config.
    /// Non-cuda only: the cuda `TpQwen3_5MoeBlock::load` takes an NCCL
    /// `Comm`, which tests cannot construct — the CPU Test job covers
    /// this; the CUDA job type-checks the cuda variants.
    #[cfg(not(feature = "cuda"))]
    fn tiny_moe_fixture(dir: &std::path::Path) -> (TextConfig, std::path::PathBuf) {
        let dev = Device::Cpu;
        let randn = |shape: &[usize]| Tensor::randn(0f32, 0.3f32, shape, &dev).unwrap();
        let (h, inter, n_exp) = (8usize, 4usize, 6usize);

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("mlp.gate.weight".into(), randn(&[n_exp, h]));
        for e in 0..n_exp {
            t.insert(
                format!("mlp.experts.{e}.gate_proj.weight"),
                randn(&[inter, h]),
            );
            t.insert(
                format!("mlp.experts.{e}.up_proj.weight"),
                randn(&[inter, h]),
            );
            t.insert(
                format!("mlp.experts.{e}.down_proj.weight"),
                randn(&[h, inter]),
            );
        }
        t.insert(
            "mlp.shared_expert.gate_proj.weight".into(),
            randn(&[inter, h]),
        );
        t.insert(
            "mlp.shared_expert.up_proj.weight".into(),
            randn(&[inter, h]),
        );
        t.insert(
            "mlp.shared_expert.down_proj.weight".into(),
            randn(&[h, inter]),
        );
        t.insert("mlp.shared_expert_gate.weight".into(), randn(&[1, h]));

        let path = dir.join("moe.safetensors");
        candle_core::safetensors::save(&t, &path).expect("save moe safetensors");

        // Minimal TextConfig via the flat-config parser: only the MoE
        // fields matter for the block loaders.
        let cfg = Config::from_config_json(
            r#"{
                "model_type": "qwen3_next",
                "vocab_size": 16, "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4,
                "max_position_embeddings": 32, "rms_norm_eps": 1e-6,
                "num_experts": 6, "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
                "norm_topk_prob": true
            }"#,
        )
        .expect("parse tiny moe config")
        .text_config;
        (cfg, path)
    }

    #[cfg(not(feature = "cuda"))]
    fn vb_over(path: &std::path::Path) -> ShardedVarBuilder {
        // SAFETY: mmap of a file the test just wrote; nothing mutates it.
        unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path.to_path_buf()),
                DType::F32,
                &Device::Cpu,
            )
            .expect("build ShardedVarBuilder")
        }
    }

    /// world_size = 2 on CPU: the block-end AllReduce is elided, so
    /// each rank's forward returns its PARTIAL output. Summing the two
    /// ranks' partials must reproduce the single-GPU output — this
    /// pins the expert slicing (column gate/up, row down), the
    /// replicated routing, and the shared-expert partial scaling,
    /// i.e. everything the real AllReduce would combine.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn tp_moe_ws2_partials_sum_to_single_gpu_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (cfg, path) = tiny_moe_fixture(dir.path());

        let single =
            Qwen3_5MoeBlock::load(&cfg, &vb_over(&path).pp("mlp")).expect("single-GPU load");
        let rank0 = TpQwen3_5MoeBlock::load(&cfg, &vb_over(&path).pp("mlp"), 0, 2, None)
            .expect("TP rank 0 load");
        let rank1 = TpQwen3_5MoeBlock::load(&cfg, &vb_over(&path).pp("mlp"), 1, 2, None)
            .expect("TP rank 1 load");

        let xs = Tensor::randn(0f32, 1.0f32, (1, 4, 8), &Device::Cpu).unwrap();
        let full: Vec<f32> = single
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let p0 = rank0.forward(&xs).unwrap();
        let p1 = rank1.forward(&xs).unwrap();
        let summed: Vec<f32> = (p0 + p1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for (i, (x, y)) in full.iter().zip(&summed).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "dim {i}: single {x} vs summed partials {y}"
            );
        }
    }

    /// Per-rank fused-qkvz de-interleave: sharding the fused tensor by
    /// whole k-head groups and then splitting with per-rank head counts
    /// must equal the corresponding row-ranges of the full split.
    #[test]
    fn per_rank_qkvz_split_matches_full_split_slices() {
        use crate::harness::arch::qwen3_5::linear_attn::{split_fused_ba, split_fused_qkvz};
        let dev = Device::Cpu;
        let (num_k, num_v, head_k, head_v, hidden) = (4usize, 8usize, 3usize, 2usize, 5usize);
        let r = num_v / num_k;
        let stride = 2 * head_k + 2 * head_v * r;
        let fused = Tensor::randn(0f32, 1.0f32, (num_k * stride, hidden), &dev).unwrap();
        let ba = Tensor::randn(0f32, 1.0f32, (2 * num_v, hidden), &dev).unwrap();

        let (full_qkv, full_z) = split_fused_qkvz(&fused, num_k, num_v, head_k, head_v).unwrap();
        let (full_b, full_a) = split_fused_ba(&ba, num_k, num_v).unwrap();
        let key_dim = num_k * head_k;

        let ws = 2usize;
        for rank in 0..ws {
            let (pk, pv) = (num_k / ws, num_v / ws);
            let group_rows = pk * stride;
            let rank_fused = fused.narrow(0, rank * group_rows, group_rows).unwrap();
            let (qkv, z) = split_fused_qkvz(&rank_fused, pk, pv, head_k, head_v).unwrap();

            // Expected: the rank's row-ranges of the full [Q|K|V] and Z.
            let (prk, prv) = (pk * head_k, pv * head_v);
            let expect_qkv = Tensor::cat(
                &[
                    full_qkv.narrow(0, rank * prk, prk).unwrap(),
                    full_qkv.narrow(0, key_dim + rank * prk, prk).unwrap(),
                    full_qkv.narrow(0, 2 * key_dim + rank * prv, prv).unwrap(),
                ],
                0,
            )
            .unwrap();
            let expect_z = full_z.narrow(0, rank * prv, prv).unwrap();
            let d1: f32 = (qkv - expect_qkv)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            let d2: f32 = (z - expect_z)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            assert_eq!(d1, 0.0, "rank {rank} qkv slice mismatch");
            assert_eq!(d2, 0.0, "rank {rank} z slice mismatch");

            // ba: rank's groups are 2r rows each.
            let rank_ba = ba.narrow(0, rank * pk * 2 * r, pk * 2 * r).unwrap();
            let (b, a) = split_fused_ba(&rank_ba, pk, pv).unwrap();
            let expect_b = full_b.narrow(0, rank * pv, pv).unwrap();
            let expect_a = full_a.narrow(0, rank * pv, pv).unwrap();
            let d3: f32 = (b - expect_b)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            let d4: f32 = (a - expect_a)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            assert_eq!(d3, 0.0, "rank {rank} b slice mismatch");
            assert_eq!(d4, 0.0, "rank {rank} a slice mismatch");
        }
    }
}
