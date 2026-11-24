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

/// Which mask an attention call carries — and whether the flash kernel
/// can express it.
///
/// The flash path encodes causality as a boolean flag and never reads a
/// mask tensor. While the only mask this crate built was the causal
/// one, `Option<&Tensor>` was enough: `Some` meant causal. It is no
/// longer, because `qwen4_exp`'s QSA selection is an arbitrary additive
/// mask — and handed to the flash path it would be silently replaced by
/// a dense causal one. The model would run, the sparsity would do
/// nothing, and the only symptom would be that long context cost what
/// it always did.
///
/// So the distinction is a type rather than a comment: an
/// [`AttnMask::Additive`] mask always takes the eager path.
#[derive(Clone, Copy)]
pub(crate) enum AttnMask<'a> {
    /// Decode, or a single position — nothing to mask.
    None,
    /// The standard causal mask, which the flash kernel applies as a
    /// flag rather than reading.
    Causal(&'a Tensor),
    /// A mask the kernel cannot express. Forces the eager path.
    Additive(&'a Tensor),
}

impl<'a> AttnMask<'a> {
    /// The historical `Option<&Tensor>` contract: present means causal.
    pub(crate) fn causal(mask: Option<&'a Tensor>) -> Self {
        match mask {
            Some(m) => Self::Causal(m),
            None => Self::None,
        }
    }

    /// The tensor the eager path adds to the scores, if any.
    fn tensor(&self) -> Option<&'a Tensor> {
        match self {
            Self::None => None,
            Self::Causal(m) | Self::Additive(m) => Some(m),
        }
    }

    /// Whether the flash kernel may be used at all for this mask.
    /// Only the flash path asks, so only that build compiles it.
    #[cfg(feature = "flash-attn")]
    fn reaches_flash(&self) -> bool {
        !matches!(self, Self::Additive(_))
    }

    /// The kernel's `causal` flag.
    #[cfg(feature = "flash-attn")]
    fn is_causal(&self) -> bool {
        matches!(self, Self::Causal(_))
    }
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
/// The mask arrives as an [`AttnMask`] rather than an `Option`, because
/// the flash path cannot express an arbitrary one and would discard it
/// in silence — see that type.
///
/// Falls back to the eager matmul→softmax→matmul everywhere else
/// (CPU, f32, feature off, `NEURON_FLASH_ATTN=0`, or a mask the kernel
/// cannot express).
pub(crate) fn attention_context(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: AttnMask<'_>,
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
            && attn_mask.reaches_flash()
            && q.dim(2)? > 1
            && q.device().is_cuda()
            && (dtype == DType::F16 || dtype == DType::BF16)
        {
            // flash_attn wants (B, L, H, D); the callers carry (B, H, L, D).
            let qf = q.transpose(1, 2)?.contiguous()?;
            let kf = k.transpose(1, 2)?.contiguous()?;
            let vf = v.transpose(1, 2)?.contiguous()?;
            let causal = attn_mask.is_causal();
            let ctx = candle_flash_attn::flash_attn(&qf, &kf, &vf, scale as f32, causal)?;
            return ctx.transpose(1, 2)?.contiguous();
        }
    }

    if let AttnMask::Additive(m) = attn_mask {
        return masked_attention(q, k, v, m, num_kv_groups, scale);
    }

    // Eager fallback: materialise GQA-repeated K/V and the score matrix.
    let k = repeat_kv(k.clone(), num_kv_groups)?.contiguous()?;
    let v = repeat_kv(v.clone(), num_kv_groups)?.contiguous()?;
    let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    if let Some(m) = attn_mask.tensor() {
        scores = scores.broadcast_add(m)?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(&v)
}

/// Score elements one query tile may hold: 2^26, about 0.75 GB across
/// the bf16 scores and the f32 softmax. A tile is as many queries as fit.
const SCORE_TILE_ELEMENTS: usize = 1 << 26;

/// Attention under an arbitrary additive mask, `(B, 1, L, S)` — the
/// path every [`AttnMask::Additive`] takes.
///
/// Equal to plain eager attention, with two changes
/// that bound its memory so long prompts can be read in large chunks:
///
/// - GQA without copying K/V: each KV head's query heads are folded
///   into the row axis, so the matmul reads the shared K/V once rather
///   than a 12x repeat of the whole cache.
/// - Queries are processed in tiles sized to [`SCORE_TILE_ELEMENTS`],
///   so a chunk of `L` queries over `S` positions never holds the full
///   `heads * L * S` scores.
///
/// The mask add and softmax run in f32, as the reference does.
fn masked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    num_kv_groups: usize,
    scale: f64,
) -> candle_core::Result<Tensor> {
    masked_attention_tiled(q, k, v, mask, num_kv_groups, scale, SCORE_TILE_ELEMENTS)
}

fn masked_attention_tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    num_kv_groups: usize,
    scale: f64,
    tile_elements: usize,
) -> candle_core::Result<Tensor> {
    let (b, heads, l, d) = q.dims4()?;
    let (_, kv_heads, s, _) = k.dims4()?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;
    let tile = (tile_elements / (heads * s).max(1)).clamp(1, l);
    let mut out = Vec::with_capacity(l.div_ceil(tile));
    let mut start = 0;
    while start < l {
        let t = tile.min(l - start);
        // (B, H, t, D) -> (B, Hkv, G * t, D): head h = kv * G + g.
        let qt =
            q.narrow(2, start, t)?
                .contiguous()?
                .reshape((b, kv_heads, num_kv_groups * t, d))?;
        let scores = (qt.matmul(&kt)? * scale)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((b, kv_heads, num_kv_groups, t, s))?;
        let m = mask
            .narrow(2, start, t)?
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(1)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&m)?)?
            .to_dtype(q.dtype())?
            .reshape((b, kv_heads, num_kv_groups * t, s))?;
        out.push(probs.matmul(&v)?.reshape((b, heads, t, d))?);
        start += t;
    }
    if out.len() == 1 {
        return Ok(out.remove(0));
    }
    Tensor::cat(&out, 2)
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
        let ctx = attention_context(
            &q,
            &k,
            &v,
            AttnMask::causal(attn_mask),
            self.num_kv_groups,
            scale,
        )?; // (B, H, L, D)

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Plain eager attention: K/V repeated per query head, one score
    /// matrix, f32 throughout. Independent of the code under test.
    fn reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        groups: usize,
        scale: f64,
    ) -> Vec<f32> {
        let k = repeat_kv(k.clone(), groups).unwrap();
        let v = repeat_kv(v.clone(), groups).unwrap();
        let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * scale)
            .unwrap()
            .broadcast_add(mask)
            .unwrap();
        candle_nn::ops::softmax_last_dim(&scores)
            .unwrap()
            .matmul(&v.contiguous().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// The tiled, copy-free GQA attention equals plain eager attention
    /// under an arbitrary mask, for every tile size: one query per tile,
    /// a ragged tail, and the whole chunk at once. Each KV head and each
    /// row's mask differ, so a wrong head fold or a misaligned mask tile
    /// changes the answer.
    #[test]
    fn masked_attention_matches_eager_attention_at_any_tile() {
        let dev = Device::Cpu;
        let (heads, kv_heads, l, s, d) = (4, 2, 5, 7, 8);
        let values = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                    ((x >> 8) % 1000) as f32 / 500.0 - 1.0
                })
                .collect()
        };
        let q = Tensor::from_vec(values(heads * l * d, 1), (1, heads, l, d), &dev).unwrap();
        let k = Tensor::from_vec(values(kv_heads * s * d, 2), (1, kv_heads, s, d), &dev).unwrap();
        let v = Tensor::from_vec(values(kv_heads * s * d, 3), (1, kv_heads, s, d), &dev).unwrap();
        // Row i sees an irregular subset that always includes position i.
        let mut mask = vec![f32::NEG_INFINITY; l * s];
        for i in 0..l {
            for p in 0..s {
                if p == i || (p * 3 + i) % 4 == 0 {
                    mask[i * s + p] = 0.0;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, l, s), &dev).unwrap();
        let scale = 0.35;
        let want = reference(&q, &k, &v, &mask, 2, scale);
        for tile_queries in [1, 2, l] {
            let got: Vec<f32> =
                masked_attention_tiled(&q, &k, &v, &mask, 2, scale, tile_queries * heads * s)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() < 1e-5,
                    "tile of {tile_queries}: got {got:?} want {want:?}"
                );
            }
        }
        // And the dispatcher routes an additive mask here.
        let routed: Vec<f32> = attention_context(&q, &k, &v, AttnMask::Additive(&mask), 2, scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (g, w) in routed.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5);
        }
    }
}
