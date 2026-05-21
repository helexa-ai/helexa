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
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use candle_nn::{Embedding, Linear, kv_cache::ConcatKvCache};
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::nccl::Comm;

use super::tp_linear::{ColumnParallelLinear, RowParallelLinear};
use crate::harness::arch::qwen3_5::linear_attn::repeat_interleave;
use crate::harness::arch::qwen3_5::rmsnorm::{Qwen3_5RmsNorm, Qwen3_5RmsNormGated, l2norm};
use crate::harness::arch::qwen3_5::rope::RotaryEmbedding;
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
    in_proj_qkv: Linear,
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
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        Self::load_inner(cfg, vb, rank, world_size, comm)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        Self::load_inner(cfg, vb, rank, world_size)
    }

    fn load_inner(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        #[cfg(feature = "cuda")] comm: Arc<Comm>,
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
        let conv_dim = key_dim * 2 + value_dim;
        let hidden_size = cfg.hidden_size;

        // ----- Fused `in_proj_qkv` and `conv1d` (per-region slicing). -----
        let in_proj_qkv_weight = load_fused_qkv_slice_2d(
            vb,
            "in_proj_qkv",
            conv_dim,
            hidden_size,
            key_dim,
            value_dim,
            rank,
            world_size,
        )?;
        let in_proj_qkv = Linear::new(in_proj_qkv_weight, None);

        let conv1d_weight = load_fused_qkv_slice_3d(
            &vb.pp("conv1d"),
            (conv_dim, 1, conv_kernel_size),
            key_dim,
            value_dim,
            rank,
            world_size,
        )?;

        // ----- Uniformly-sharded projections (along output dim 0). -----
        // in_proj_z: hidden → value_dim, sharded along value_dim (V-head).
        let in_proj_z = ColumnParallelLinear::load(&vb.pp("in_proj_z"), rank, world_size)?;
        // in_proj_b, in_proj_a: hidden → num_v_heads, sharded along output.
        let in_proj_b = ColumnParallelLinear::load(&vb.pp("in_proj_b"), rank, world_size)?;
        let in_proj_a = ColumnParallelLinear::load(&vb.pp("in_proj_a"), rank, world_size)?;

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
        let out_proj = RowParallelLinear::load(&vb.pp("out_proj"), rank, world_size, comm)?;
        #[cfg(not(feature = "cuda"))]
        let out_proj = RowParallelLinear::load(&vb.pp("out_proj"), rank, world_size)?;

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
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        Self::load_inner(cfg, rotary, vb, rank, world_size, comm)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        Self::load_inner(cfg, rotary, vb, rank, world_size)
    }

    fn load_inner(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        #[cfg(feature = "cuda")] comm: Arc<Comm>,
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
        let q_proj = ColumnParallelLinear::load(&vb.pp("q_proj"), rank, world_size)?;
        let k_proj = ColumnParallelLinear::load(&vb.pp("k_proj"), rank, world_size)?;
        let v_proj = ColumnParallelLinear::load(&vb.pp("v_proj"), rank, world_size)?;
        #[cfg(feature = "cuda")]
        let o_proj = RowParallelLinear::load(&vb.pp("o_proj"), rank, world_size, comm)?;
        #[cfg(not(feature = "cuda"))]
        let o_proj = RowParallelLinear::load(&vb.pp("o_proj"), rank, world_size)?;

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
        offset: usize,
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

        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

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
}

// ─── MLP ────────────────────────────────────────────────────────────

pub(crate) struct TpQwen3_5MLP {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
}

impl TpQwen3_5MLP {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        if !cfg.intermediate_size.is_multiple_of(world_size as usize) {
            bail!(
                "intermediate_size {} not divisible by world_size {}",
                cfg.intermediate_size,
                world_size
            );
        }
        Ok(Self {
            gate_proj: ColumnParallelLinear::load(&vb.pp("gate_proj"), rank, world_size)?,
            up_proj: ColumnParallelLinear::load(&vb.pp("up_proj"), rank, world_size)?,
            down_proj: RowParallelLinear::load(&vb.pp("down_proj"), rank, world_size, comm)?,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        if !cfg.intermediate_size.is_multiple_of(world_size as usize) {
            bail!(
                "intermediate_size {} not divisible by world_size {}",
                cfg.intermediate_size,
                world_size
            );
        }
        Ok(Self {
            gate_proj: ColumnParallelLinear::load(&vb.pp("gate_proj"), rank, world_size)?,
            up_proj: ColumnParallelLinear::load(&vb.pp("up_proj"), rank, world_size)?,
            down_proj: RowParallelLinear::load(&vb.pp("down_proj"), rank, world_size)?,
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

// ─── decoder layer ──────────────────────────────────────────────────

enum TpAttentionKind {
    Full(TpQwen3_5Attention),
    Linear(TpQwen3_5GatedDeltaNet),
}

pub struct TpQwen3_5DecoderLayer {
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
    mlp: TpQwen3_5MLP,
    attention: TpAttentionKind,
}

impl TpQwen3_5DecoderLayer {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
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
            )?),
            "linear_attention" => TpAttentionKind::Linear(TpQwen3_5GatedDeltaNet::load(
                cfg,
                &vb.pp("linear_attn"),
                rank,
                world_size,
                comm.clone(),
            )?),
            other => bail!("unknown layer_type '{other}' for layer {layer_idx}"),
        };
        let mlp = TpQwen3_5MLP::load(cfg, &vb.pp("mlp"), rank, world_size, comm)?;
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
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
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
            )?),
            "linear_attention" => TpAttentionKind::Linear(TpQwen3_5GatedDeltaNet::load(
                cfg,
                &vb.pp("linear_attn"),
                rank,
                world_size,
            )?),
            other => bail!("unknown layer_type '{other}' for layer {layer_idx}"),
        };
        let mlp = TpQwen3_5MLP::load(cfg, &vb.pp("mlp"), rank, world_size)?;
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
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attention {
            TpAttentionKind::Full(attn) => attn.forward(&h, attn_mask, offset)?,
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
}

// ─── base Model ─────────────────────────────────────────────────────

pub struct TpQwen3_5Model {
    embed_tokens: Embedding,
    layers: Vec<TpQwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    device: Device,
    dtype: DType,
}

impl TpQwen3_5Model {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let text_vb = vb.pp("model.language_model");

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
        for i in 0..cfg.num_hidden_layers {
            let layer = TpQwen3_5DecoderLayer::load(
                cfg,
                rotary.clone(),
                i,
                &vb_l.pp(i),
                rank,
                world_size,
                comm.clone(),
            )
            .with_context(|| {
                let (free_mb, total_mb) = cuda_mem_mb(&device);
                format!("load layer {i} (rank {rank}): free={free_mb}MB / total={total_mb}MB")
            })?;
            layers.push(layer);
            log_vram(&device, rank, &format!("after layer {i}"));
        }

        let norm = Qwen3_5RmsNorm::load(&text_vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device,
            dtype,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &TextConfig,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let text_vb = vb.pp("model.language_model");

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
        for i in 0..cfg.num_hidden_layers {
            layers.push(TpQwen3_5DecoderLayer::load(
                cfg,
                rotary.clone(),
                i,
                &vb_l.pp(i),
                rank,
                world_size,
            )?);
        }

        let norm = Qwen3_5RmsNorm::load(&text_vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
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
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> candle_core::Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

pub struct TpQwen3_5ForCausalLM {
    base: TpQwen3_5Model,
    lm_head: Linear,
}

impl TpQwen3_5ForCausalLM {
    #[cfg(feature = "cuda")]
    pub fn load(
        config: Config,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        let cfg = &config.text_config;
        let base = TpQwen3_5Model::load(cfg, vb, rank, world_size, comm)?;
        let lm_head = build_lm_head(cfg, vb, &base)?;
        Ok(Self { base, lm_head })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        config: Config,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        let cfg = &config.text_config;
        let base = TpQwen3_5Model::load(cfg, vb, rank, world_size)?;
        let lm_head = build_lm_head(cfg, vb, &base)?;
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden = self.base.forward(input, offset)?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }

    pub fn device(&self) -> &Device {
        &self.base.device
    }
}

fn build_lm_head(
    cfg: &TextConfig,
    vb: &ShardedVarBuilder,
    base: &TpQwen3_5Model,
) -> Result<Linear> {
    if cfg.tie_word_embeddings {
        Ok(Linear::new(base.embed_weight().clone(), None))
    } else {
        // lm_head sits at the top level (sibling of `model.*`), NOT
        // under `model.language_model`.
        let weight = load_replicated(
            &vb.pp("lm_head"),
            (cfg.vocab_size, cfg.hidden_size),
            "weight",
        )?;
        Ok(Linear::new(weight, None))
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

/// Load a fused QKV-style 2D weight tensor that stores three regions
/// sequentially along dim 0: `[first key_dim, second key_dim, value_dim]`.
/// Returns the per-rank slice formed by extracting the rank's share
/// from each region and concatenating along dim 0.
///
/// The full tensor materialises briefly on the device before the
/// slices are extracted (`narrow` views + `contiguous` copy). Memory
/// peak is one full-tensor load per layer during construction; only
/// the per-rank concatenation stays after `full` drops.
#[allow(clippy::too_many_arguments)]
fn load_fused_qkv_slice_2d(
    vb: &ShardedVarBuilder,
    name: &str,
    conv_dim: usize,
    hidden_size: usize,
    key_dim: usize,
    value_dim: usize,
    rank: u32,
    world_size: u32,
) -> Result<Tensor> {
    let ws = world_size as usize;
    let r = rank as usize;
    if !key_dim.is_multiple_of(ws) || !value_dim.is_multiple_of(ws) {
        bail!(
            "fused qkv shard: key_dim ({key_dim}) and value_dim ({value_dim}) \
             must each be divisible by world_size ({ws})"
        );
    }
    let per_rank_key = key_dim / ws;
    let per_rank_value = value_dim / ws;

    // Force full-tensor load via `vb.get`, which defaults to
    // `Shard { world_size: 1 }` and falls through to SimpleBackend.
    let full = vb
        .pp(name)
        .get((conv_dim, hidden_size), "weight")
        .with_context(|| format!("load fused qkv '{}/{}/weight'", vb.prefix(), name))?;

    let q = full.narrow(0, r * per_rank_key, per_rank_key)?;
    let k = full.narrow(0, key_dim + r * per_rank_key, per_rank_key)?;
    let v = full.narrow(0, 2 * key_dim + r * per_rank_value, per_rank_value)?;

    Tensor::cat(&[&q, &k, &v], 0)?
        .contiguous()
        .with_context(|| format!("materialise fused qkv slice for rank {r}"))
}

/// Same per-region slicing pattern for a 3D fused tensor (the depthwise
/// `conv1d.weight` of the linear-attention block: shape
/// `(conv_dim, 1, kernel_size)`).
fn load_fused_qkv_slice_3d(
    vb: &ShardedVarBuilder,
    shape: (usize, usize, usize),
    key_dim: usize,
    value_dim: usize,
    rank: u32,
    world_size: u32,
) -> Result<Tensor> {
    let (conv_dim, mid, kernel_size) = shape;
    let ws = world_size as usize;
    let r = rank as usize;
    if !key_dim.is_multiple_of(ws) || !value_dim.is_multiple_of(ws) {
        bail!(
            "fused conv shard: key_dim ({key_dim}) and value_dim ({value_dim}) \
             must each be divisible by world_size ({ws})"
        );
    }
    let per_rank_key = key_dim / ws;
    let per_rank_value = value_dim / ws;

    let full = vb
        .get((conv_dim, mid, kernel_size), "weight")
        .with_context(|| format!("load fused conv '{}/weight'", vb.prefix()))?;

    let q = full.narrow(0, r * per_rank_key, per_rank_key)?;
    let k = full.narrow(0, key_dim + r * per_rank_key, per_rank_key)?;
    let v = full.narrow(0, 2 * key_dim + r * per_rank_value, per_rank_value)?;

    Tensor::cat(&[&q, &k, &v], 0)?
        .contiguous()
        .with_context(|| format!("materialise fused conv slice for rank {r}"))
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
