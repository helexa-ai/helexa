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

        // 5. GQA repeat (cheap shape op).
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // 6. Scaled dot-product + causal mask.
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

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
