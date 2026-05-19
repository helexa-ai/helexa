//! Tensor-parallel Qwen3 dense model.
//!
//! Mirrors `candle_transformers::models::qwen3` structurally, but with:
//!
//! - Attention's `q_proj` / `k_proj` / `v_proj` as
//!   [`ColumnParallelLinear`] (output sharded along the head dimension —
//!   per-rank `num_heads = total/world_size`, ditto for kv heads).
//! - Attention's `o_proj` as [`RowParallelLinear`] (input sharded; the
//!   trailing `AllReduce` recovers the full activation).
//! - MLP's `gate_proj` / `up_proj` as [`ColumnParallelLinear`] (sharded
//!   along `intermediate_size`).
//! - MLP's `down_proj` as [`RowParallelLinear`].
//! - `embed_tokens`, all `RmsNorm`s, and `lm_head` **replicated** on
//!   every rank. The per-rank duplicate weight is bounded and lets us
//!   skip the embedding all-gather and the lm-head column-shard +
//!   all-gather; both are pure latency optimisations that don't change
//!   correctness.
//! - `kv_cache` holds the per-rank slice of K/V already (because they
//!   came out of a column-parallel projection). No cache resharding
//!   needed across ranks.
//!
//! Divisibility requirement, checked at load time:
//!
//! - `num_attention_heads % world_size == 0`
//! - `num_key_value_heads % world_size == 0`
//! - `intermediate_size  % world_size == 0`
//!
//! Anything else bails — the safetensors slice would lose data otherwise.
//! This is the same divisibility-bail pattern that landed in
//! `EricLBuehler/mistral.rs` PR #2054.
//!
//! Replicated tensors (norms, embedding, lm_head) are loaded by asking
//! the `ShardedVarBuilder` for the full tensor via `vb.get(shape, name)`
//! — which defaults to `Shard { world_size: 1 }` and falls through to
//! the unsharded backend path.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use candle_nn::{Activation, Embedding, Linear, RmsNorm, kv_cache::ConcatKvCache};
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::nccl::Comm;

use super::tp_linear::{ColumnParallelLinear, RowParallelLinear};

pub use candle_transformers::models::qwen3::Config;

/// Replicated rotary-embedding lookup. Re-implementation of the
/// `pub(crate)` candle equivalent — we can't reach into the upstream
/// type, so the inv-freq / sin / cos construction lives here.
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Helper: load a replicated tensor by asking the ShardedVarBuilder for
/// the full tensor (world_size=1 hint falls through to SimpleBackend).
fn load_replicated<S: Into<candle_core::Shape>>(
    vb: &ShardedVarBuilder,
    shape: S,
    name: &str,
) -> Result<Tensor> {
    vb.get(shape, name)
        .with_context(|| format!("load replicated '{}/{name}'", vb.prefix()))
}

fn load_rms_norm(vb: &ShardedVarBuilder, size: usize, eps: f64) -> Result<RmsNorm> {
    let weight = load_replicated(vb, size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}

/// TP MLP. SwiGLU = `down(silu(gate(x)) * up(x))`.
pub(crate) struct TpQwen3MLP {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
    act_fn: Activation,
}

impl TpQwen3MLP {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &Config,
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
            act_fn: cfg.hidden_act,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(cfg: &Config, vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
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
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for TpQwen3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// TP attention. Carries per-rank head counts and the q/k per-head
/// RmsNorms (which are replicated and operate on a flattened B*H*L
/// axis, so the same code path works irrespective of how H was split).
pub(crate) struct TpQwen3Attention {
    q_proj: ColumnParallelLinear,
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    local_num_heads: usize,
    local_num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    local_hidden_size: usize,
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl TpQwen3Attention {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        Self::load_inner(
            cfg,
            rotary_emb,
            vb,
            rank,
            world_size,
            #[cfg(feature = "cuda")]
            comm,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        Self::load_inner(cfg, rotary_emb, vb, rank, world_size)
    }

    fn load_inner(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        #[cfg(feature = "cuda")] comm: Arc<Comm>,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            bail!("sliding window is not yet supported in the TP path");
        }
        if cfg.attention_bias {
            bail!("attention_bias=true is not supported by ColumnParallel/RowParallelLinear yet");
        }
        let ws = world_size as usize;
        if !cfg.num_attention_heads.is_multiple_of(ws) {
            bail!(
                "num_attention_heads {} not divisible by world_size {}",
                cfg.num_attention_heads,
                world_size
            );
        }
        if !cfg.num_key_value_heads.is_multiple_of(ws) {
            bail!(
                "num_key_value_heads {} not divisible by world_size {}",
                cfg.num_key_value_heads,
                world_size
            );
        }
        let head_dim = cfg.head_dim;
        let local_num_heads = cfg.num_attention_heads / ws;
        let local_num_kv_heads = cfg.num_key_value_heads / ws;
        let num_kv_groups = local_num_heads / local_num_kv_heads;
        let local_hidden_size = head_dim * local_num_heads;

        let q_proj = ColumnParallelLinear::load(&vb.pp("q_proj"), rank, world_size)?;
        let k_proj = ColumnParallelLinear::load(&vb.pp("k_proj"), rank, world_size)?;
        let v_proj = ColumnParallelLinear::load(&vb.pp("v_proj"), rank, world_size)?;
        #[cfg(feature = "cuda")]
        let o_proj = RowParallelLinear::load(&vb.pp("o_proj"), rank, world_size, comm)?;
        #[cfg(not(feature = "cuda"))]
        let o_proj = RowParallelLinear::load(&vb.pp("o_proj"), rank, world_size)?;

        let q_norm = load_rms_norm(&vb.pp("q_norm"), head_dim, cfg.rms_norm_eps)?;
        let k_norm = load_rms_norm(&vb.pp("k_norm"), head_dim, cfg.rms_norm_eps)?;

        // dim=2 because we cat along the seq axis of (B, H, L, D) tensors.
        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            local_num_heads,
            local_num_kv_heads,
            num_kv_groups,
            head_dim,
            local_hidden_size,
            rotary_emb,
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

        // 1. Projections (column-parallel → output is sharded).
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) → (B, H, L, D).
        let q = q
            .reshape((b, l, self.local_num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.local_num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.local_num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head RmsNorm (replicated weight, flat input).
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.local_num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.local_num_kv_heads, l, self.head_dim))?;

        // 4. Rotary.
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV.
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // 6. GQA repeat_kv on the rank-local K/V.
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // 7. Attention scores.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // 8. Output projection (row-parallel → AllReduce inside).
        ctx.transpose(1, 2)?
            .reshape((b, l, self.local_hidden_size))?
            .apply(&self.o_proj)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

struct TpDecoderLayer {
    self_attn: TpQwen3Attention,
    mlp: TpQwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl TpDecoderLayer {
    #[cfg(feature = "cuda")]
    fn load(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        let self_attn = TpQwen3Attention::load(
            cfg,
            rotary_emb,
            &vb.pp("self_attn"),
            rank,
            world_size,
            comm.clone(),
        )?;
        let mlp = TpQwen3MLP::load(cfg, &vb.pp("mlp"), rank, world_size, comm)?;
        let ln1 = load_rms_norm(&vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ln2 = load_rms_norm(
            &vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn load(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
    ) -> Result<Self> {
        let self_attn =
            TpQwen3Attention::load(cfg, rotary_emb, &vb.pp("self_attn"), rank, world_size)?;
        let mlp = TpQwen3MLP::load(cfg, &vb.pp("mlp"), rank, world_size)?;
        let ln1 = load_rms_norm(&vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ln2 = load_rms_norm(
            &vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Base TP Qwen3 transformer — embedding, decoder stack, final norm.
/// The lm_head sits on top in [`TpQwen3ForCausalLM`].
pub struct TpQwen3Model {
    embed_tokens: Embedding,
    layers: Vec<TpDecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl TpQwen3Model {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &Config,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(dtype, cfg, &device)?);

        let embed_vb = vb.pp("model.embed_tokens");
        let embed_weight = load_replicated(&embed_vb, (cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TpDecoderLayer::load(
                cfg,
                rotary.clone(),
                &vb_l.pp(i),
                rank,
                world_size,
                comm.clone(),
            )?);
        }
        let norm = load_rms_norm(&vb.pp("model.norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device,
            dtype,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(cfg: &Config, vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(dtype, cfg, &device)?);

        let embed_vb = vb.pp("model.embed_tokens");
        let embed_weight = load_replicated(&embed_vb, (cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TpDecoderLayer::load(
                cfg,
                rotary.clone(),
                &vb_l.pp(i),
                rank,
                world_size,
            )?);
        }
        let norm = load_rms_norm(&vb.pp("model.norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

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

/// TP Qwen3 with a (replicated) language-model head on top.
pub struct TpQwen3ForCausalLM {
    base: TpQwen3Model,
    lm_head: Linear,
}

impl TpQwen3ForCausalLM {
    #[cfg(feature = "cuda")]
    pub fn load(
        cfg: &Config,
        vb: &ShardedVarBuilder,
        rank: u32,
        world_size: u32,
        comm: Arc<Comm>,
    ) -> Result<Self> {
        let base = TpQwen3Model::load(cfg, vb, rank, world_size, comm)?;
        let lm_head = build_lm_head(cfg, vb, &base)?;
        Ok(Self { base, lm_head })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(cfg: &Config, vb: &ShardedVarBuilder, rank: u32, world_size: u32) -> Result<Self> {
        let base = TpQwen3Model::load(cfg, vb, rank, world_size)?;
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

    pub fn dtype(&self) -> DType {
        self.base.dtype
    }
}

fn build_lm_head(cfg: &Config, vb: &ShardedVarBuilder, base: &TpQwen3Model) -> Result<Linear> {
    if cfg.tie_word_embeddings {
        Ok(Linear::new(base.embed_weight().clone(), None))
    } else {
        let weight = load_replicated(
            &vb.pp("lm_head"),
            (cfg.vocab_size, cfg.hidden_size),
            "weight",
        )?;
        Ok(Linear::new(weight, None))
    }
}
