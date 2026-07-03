//! Qwen3-Next's `full_attention` layer.
//!
//! Standard GQA causal attention with two Qwen3-Next-specific quirks:
//!
//! 1. **Output gate (`attn_output_gate=True`).** `q_proj` is widened
//!    to `num_heads * head_dim * 2`. The second half is reshaped to
//!    `(B, L, num_heads * head_dim)` and fed through a sigmoid; the
//!    attention output is pointwise-multiplied by this gate before
//!    `o_proj`. Effectively a per-head per-position attenuation on
//!    the attention output.
//!
//! 2. **`(1 + w) * x` RmsNorm** on q and k (see `rmsnorm::Qwen3_5RmsNorm`).
//!    candle_nn's RmsNorm applies `w * x`; the upstream Qwen3-Next
//!    checkpoints expect the `(1 + w)` form.
//!
//! Otherwise: GQA with `num_attention_heads / num_key_value_heads`
//! repeat, q_norm + k_norm on the head dim, GLM-style rotary (see
//! `rope::RotaryEmbedding`), and the usual causal mask.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::var_builder::ShardedVarBuilder;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

use super::TextConfig;
use super::rmsnorm::Qwen3_5RmsNorm;
use super::rope::RotaryEmbedding;

/// Runtime kill-switch for the FlashAttention path (#95):
/// `NEURON_FLASH_ATTN=0` (or `false`) forces the eager fallback
/// without a rebuild — the A/B lever and the rollback if the kernels
/// misbehave on some device. Read once.
#[cfg(feature = "flash-attn")]
fn flash_attn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = !std::env::var("NEURON_FLASH_ATTN").is_ok_and(|v| v == "0" || v == "false");
        tracing::info!(enabled = on, "FlashAttention path (#95)");
        on
    })
}

/// Attention core shared by the single-GPU and TP full-attention
/// layers (#95): `(B, H, L, D)` query and `(B, H_kv, S, D)` key/value
/// (post-KV-cache, NOT GQA-repeated) → `(B, H, L, D)` context.
///
/// With the `flash-attn` feature on a CUDA device in f16/bf16, this
/// dispatches to the FlashAttention kernel: GQA is native (no
/// repeated-K/V materialisation) and causality is a kernel flag, so
/// the O(L²) mask/score tensors never exist. The kernels align the
/// causal mask to the BOTTOM-RIGHT when `seqlen_q != seqlen_k`
/// (flash-attention v2.1+ semantics), which is exactly what chunked
/// prefill continuation needs: a chunk of L new queries against
/// `offset + L` cached keys masks correctly.
///
/// INVARIANT: `attn_mask` is either `None` (decode / single position)
/// or the standard causal mask — the only mask the qwen3_5 forward
/// constructs. The flash path encodes it as `causal = attn_mask
/// .is_some()`; a future non-causal mask must extend this signature,
/// not silently pass through.
///
/// Falls back to the eager matmul→softmax→matmul everywhere else
/// (CPU, f32, feature off, or `NEURON_FLASH_ATTN=0`).
pub(crate) fn attention_context(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    num_kv_groups: usize,
    scale: f64,
) -> candle_core::Result<Tensor> {
    #[cfg(feature = "flash-attn")]
    {
        use candle_core::DType;
        let dtype = q.dtype();
        // Prefill only (q_len > 1): measured on beast (27B, 30k-token
        // prompt, 2x RTX 5090), flash cuts prefill 24.8s → 22.1s but
        // REGRESSES decode ~20% (50 → 60 ms/token at 30k KV) — FA2
        // without flash-decoding is weak at query-length 1 and the
        // per-step layout transposes add overhead. Greedy outputs are
        // byte-identical either way, so this split is purely a
        // performance routing decision.
        if flash_attn_enabled()
            && q.dim(2)? > 1
            && q.device().is_cuda()
            && (dtype == DType::F16 || dtype == DType::BF16)
        {
            // flash_attn wants (B, L, H, D); the callers carry (B, H, L, D).
            let qf = q.transpose(1, 2)?.contiguous()?;
            let kf = k.transpose(1, 2)?.contiguous()?;
            let vf = v.transpose(1, 2)?.contiguous()?;
            let causal = attn_mask.is_some();
            let ctx = candle_flash_attn::flash_attn(&qf, &kf, &vf, scale as f32, causal)?;
            return ctx.transpose(1, 2)?.contiguous();
        }
    }

    // Eager fallback: materialise GQA-repeated K/V and the score matrix.
    let k = repeat_kv(k.clone(), num_kv_groups)?.contiguous()?;
    let v = repeat_kv(v.clone(), num_kv_groups)?.contiguous()?;
    let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    if let Some(m) = attn_mask {
        scores = scores.broadcast_add(m)?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(&v)
}

pub struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3_5RmsNorm,
    k_norm: Qwen3_5RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Qwen3_5Attention {
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            anyhow::bail!(
                "num_attention_heads ({num_heads}) must be a positive multiple of \
                 num_key_value_heads ({num_kv_heads})"
            );
        }
        let num_kv_groups = num_heads / num_kv_heads;

        // q_proj is 2x wide: the extra `num_heads * head_dim` slice is
        // the gate (see attn_output_gate notes above).
        let q_proj = load_linear_no_bias(vb, "q_proj", cfg.hidden_size, num_heads * head_dim * 2)?;
        let k_proj = load_linear_no_bias(vb, "k_proj", cfg.hidden_size, num_kv_heads * head_dim)?;
        let v_proj = load_linear_no_bias(vb, "v_proj", cfg.hidden_size, num_kv_heads * head_dim)?;
        let o_proj = load_linear_no_bias(vb, "o_proj", num_heads * head_dim, cfg.hidden_size)?;

        let q_norm = Qwen3_5RmsNorm::load(&vb.pp("q_norm"), head_dim, cfg.rms_norm_eps)?;
        let k_norm = Qwen3_5RmsNorm::load(&vb.pp("k_norm"), head_dim, cfg.rms_norm_eps)?;

        let hidden_size = head_dim * num_heads;
        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
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

        // 1. q_proj — widened output, split into (query, gate).
        let q_raw = self
            .q_proj
            .forward(x)?
            .reshape((b, l, self.num_heads, self.head_dim * 2))?;
        let q = q_raw.narrow(3, 0, self.head_dim)?;
        let gate = q_raw.narrow(3, self.head_dim, self.head_dim)?;
        // Flatten the gate's head dim back into hidden_size for the
        // post-attention pointwise multiply.
        let gate = gate
            .contiguous()?
            .reshape((b, l, self.num_heads * self.head_dim))?;

        // 2. q_norm + k_norm + reshape to (B, H, L, D).
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let q = q.transpose(1, 2)?.contiguous()?; // (B, H, L, D)

        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let k = k.transpose(1, 2)?.contiguous()?;

        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // 3. RoPE on q, k (cos/sin built once per forward by the model —
        //    interleaved M-RoPE for image tokens, plain for text).
        let (q, k) = self.rotary.apply_cos_sin(&q, &k, cos, sin)?;

        // 4. KV cache.
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // 5+6. Attention core — FlashAttention when available, eager
        // GQA-repeat + masked softmax otherwise (#95).
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let ctx = attention_context(&q, &k, &v, attn_mask, self.num_kv_groups, scale)?; // (B, H, L, D)

        // 7. Reshape back, apply the output gate, project.
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.hidden_size))?;
        let gate_sig = candle_nn::ops::sigmoid(&gate)?;
        let gated = (ctx * gate_sig)?;
        self.o_proj.forward(&gated)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Capture the KV cache contents for a prefix snapshot. Shallow
    /// clones: `ConcatKvCache::append` cats into fresh allocations and
    /// never mutates stored tensors in place, so the captured tensors
    /// stay valid after the live cache moves on.
    pub fn snapshot_kv(&self) -> Option<(Tensor, Tensor)> {
        match (self.kv_cache.k(), self.kv_cache.v()) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }

    /// Replace the live KV cache with a previously captured snapshot.
    pub fn restore_kv(&mut self, snap: Option<&(Tensor, Tensor)>) -> candle_core::Result<()> {
        self.kv_cache.reset();
        if let Some((k, v)) = snap {
            self.kv_cache.append(k, v)?;
        }
        Ok(())
    }
}

fn load_linear_no_bias(
    vb: &ShardedVarBuilder,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
}
