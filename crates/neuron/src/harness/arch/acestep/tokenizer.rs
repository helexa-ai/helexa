//! Audio tokenizer and detokenizer for ACE-Step.
//!
//! Implements Finite Scalar Quantization (FSQ) based audio tokenization.
//! These components are used in the cover/audio-code pipeline (LM + DiT hybrid
//! mode) but are **not exercised** by the Text2Music path.
//!
//! **Validation status** (against Python `vector_quantize_pytorch.ResidualFSQ`):
//! - `AudioTokenDetokenizer`: bit-exact (diff_max < 0.00001)
//! - `ResidualFSQ` (quantizer): FSQ bound/round matches; small rounding
//!   boundary differences (diff_max ~0.018) are expected for discrete
//!   quantization due to float precision at level boundaries.
//! - `AttentionPooler` / `AudioTokenizer`: structurally correct, minor
//!   float precision differences propagate through FSQ rounding.

use super::linear::{Linear, linear, linear_no_bias};
use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::with_tracing::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

use super::AceStepConfig;
use super::dit::RotaryEmbedding;

// ---------------------------------------------------------------------------
// Encoder MLP (same SiLU-gated pattern as Qwen3MLP)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Encoder Attention (bidirectional, GQA, with RoPE)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct TokenizerAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl TokenizerAttention {
    fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        let num_heads = cfg.encoder_num_attention_heads();
        let num_kv_heads = cfg.encoder_num_key_value_heads();
        let head_dim = cfg.head_dim;
        let hidden_size = num_heads * head_dim;

        Ok(Self {
            q_proj: linear_no_bias(enc_hs, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(enc_hs, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(enc_hs, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(num_heads * head_dim, enc_hs, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm
        let q = self.q_norm.forward(&q.flatten(0, 1)?)?.reshape(q.shape())?;
        let k = self.k_norm.forward(&k.flatten(0, 1)?)?.reshape(k.shape())?;

        // RoPE
        let (q, k) = rotary.apply(&q, &k, 0)?;

        // GQA
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

// ---------------------------------------------------------------------------
// Encoder Layer
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: TokenizerAttention,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: MLP,
}

impl EncoderLayer {
    fn new(cfg: &AceStepConfig, _layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        Ok(Self {
            self_attn: TokenizerAttention::new(cfg, vb.pp("self_attn"))?,
            input_layernorm: RmsNorm::new(enc_hs, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                enc_hs,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: MLP::new(
                enc_hs,
                cfg.encoder_intermediate_size(),
                cfg.hidden_act,
                vb.pp("mlp"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, attn_mask, rotary)?;
        let x = (residual + h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// AttentionPooler - CLS-token based attention pooling
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AttentionPooler {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    special_token: Tensor,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl AttentionPooler {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_attention_pooler_hidden_layers {
            layers.push(EncoderLayer::new(cfg, i, vb_l.pp(i))?);
        }
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        Ok(Self {
            embed_tokens: linear(enc_hs, enc_hs, vb.pp("embed_tokens"))?,
            layers,
            norm: RmsNorm::new(enc_hs, cfg.rms_norm_eps, vb.pp("norm"))?,
            special_token: vb.get((1, 1, enc_hs), "special_token")?,
            rotary_emb,
        })
    }

    /// Pool patches into sequence-level representations.
    /// Input: (B, T, pool_window_size, D) → Output: (B, T, D)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, p, _d) = x.dims4()?;
        let x = x.apply(&self.embed_tokens)?;

        // Prepend CLS token to each patch group
        let cls = self
            .special_token
            .broadcast_as((b * t, 1, x.dim(D::Minus1)?))?;
        let x = x.reshape((b * t, p, ()))?;
        let x = Tensor::cat(&[&cls, &x], 1)?;

        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None, &self.rotary_emb)?;
        }
        let hidden = self.norm.forward(&hidden)?;

        // Extract CLS output (position 0)
        hidden.narrow(1, 0, 1)?.squeeze(1)?.reshape((b, t, ()))
    }
}

// ---------------------------------------------------------------------------
// ResidualFSQ - Finite Scalar Quantization
// ---------------------------------------------------------------------------

/// Residual Finite Scalar Quantization.
///
/// Matches `vector_quantize_pytorch.ResidualFSQ` 1.27 as ACE-Step builds it
/// (one quantizer, the library defaults): a top-level `project_in`
/// (dim → codebook_dim) and `project_out` (codebook_dim → dim), a soft clamp
/// of the projected input to `±(1 + 1/(L-1))`, and an FSQ layer constructed
/// with `preserve_symmetry=True` and a hard-clamped bound. Symmetric FSQ puts
/// level `k` of `L` at `2k/(L-1) - 1`, spanning `[-1, 1]`; the asymmetric
/// formula (`(k - L/2) / (L/2)`) is not what the checkpoint was trained with.
#[derive(Debug, Clone)]
pub struct ResidualFSQ {
    project_in: Linear,
    project_out: Linear,
    /// Basis for the flat index: `[1, L0, L0*L1, ...]`.
    basis: Vec<i64>,
    /// Quantization levels per codebook dimension (e.g. [8, 8, 8, 5, 5, 5]).
    levels: Vec<usize>,
    /// Model working dtype (inferred from VarBuilder at construction time).
    model_dtype: DType,
}

impl ResidualFSQ {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let model_dtype = vb.dtype();
        let dim = cfg.fsq_dim;
        let levels = &cfg.fsq_input_levels;
        let codebook_dim = levels.len();

        // Single project_in/project_out at the ResidualFSQ level
        let project_in = linear(dim, codebook_dim, vb.pp("project_in"))?;
        let project_out = linear(codebook_dim, dim, vb.pp("project_out"))?;

        let mut basis = vec![1i64];
        for i in 0..levels.len() - 1 {
            basis.push(basis[i] * levels[i] as i64);
        }

        Ok(Self {
            project_in,
            project_out,
            basis,
            levels: levels.to_vec(),
            model_dtype,
        })
    }

    /// A per-codebook-dimension constant as a `(1, 1, d)` f32 tensor.
    fn per_dim(&self, dev: &candle_core::Device, f: impl Fn(f32) -> f32) -> Result<Tensor> {
        let v: Vec<f32> = self.levels.iter().map(|&l| f(l as f32)).collect();
        Tensor::new(&v[..], dev)?.unsqueeze(0)?.unsqueeze(0)
    }

    /// Soft clamp, then the symmetric hard-clamped bound:
    /// `2/(L-1) * floor((L-1) * (clamp(z, -1, 1) + 1) / 2 + 0.5) - 1`.
    /// Returns `(codes, level_indices)`.
    fn fsq_quantize(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let dev = z.device();
        let z = z.to_dtype(DType::F32)?;
        let clamp = self.per_dim(dev, |l| 1.0 + 1.0 / (l - 1.0))?;
        let z = z.broadcast_div(&clamp)?.tanh()?.broadcast_mul(&clamp)?;
        let lm1 = self.per_dim(dev, |l| l - 1.0)?;
        let levels = z
            .clamp(-1f32, 1f32)?
            .affine(0.5, 0.5)?
            .broadcast_mul(&lm1)?
            .affine(1.0, 0.5)?
            .floor()?;
        let codes = levels.broadcast_div(&lm1)?.affine(2.0, -1.0)?;
        Ok((codes, levels))
    }

    /// Quantize continuous representations into discrete tokens.
    /// Input: `(B, N, dim)`. Returns `(quantized_out, indices)` where
    /// `quantized_out` has the same shape as input and `indices` is `(B, N, num_quantizers)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let projected = x.apply(&self.project_in)?;
        let (codes, levels) = self.fsq_quantize(&projected)?;
        let basis: Vec<f32> = self.basis.iter().map(|&b| b as f32).collect();
        let basis = Tensor::new(&basis[..], x.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let indices = levels
            .broadcast_mul(&basis)?
            .sum(2)?
            .round()?
            .to_dtype(DType::I64)?
            .unsqueeze(2)?; // (B, N, 1)
        let quantized_out = codes.to_dtype(self.model_dtype)?.apply(&self.project_out)?;
        Ok((quantized_out, indices))
    }

    /// Reconstruct from quantized codes (not indices).
    /// Input: `(B, N, codebook_dim)` → Output: `(B, N, dim)`.
    pub fn get_output_from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        codes.apply(&self.project_out)
    }

    /// The codes for packed indices: level `k = (index / basis) % L` of each
    /// dimension, at `2k/(L-1) - 1`. Input `(B, N, 1)` → `(B, N, codebook_dim)`.
    pub fn codes_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let dev = indices.device();
        let indices = indices.squeeze(D::Minus1)?.to_dtype(DType::F32)?;
        // F32 holds every index exactly: the codebook has 64000 entries.
        let mut levels = Vec::with_capacity(self.levels.len());
        for (i, &l) in self.levels.iter().enumerate() {
            let l = l as f64;
            let v = indices.affine(1.0 / self.basis[i] as f64, 0.0)?.floor()?;
            let v = (&v - v.affine(1.0 / l, 0.0)?.floor()?.affine(l, 0.0)?)?;
            levels.push(v);
        }
        let levels = Tensor::stack(&levels, D::Minus1)?;
        let lm1 = self.per_dim(dev, |l| l - 1.0)?;
        levels.broadcast_div(&lm1)?.affine(2.0, -1.0)
    }

    /// Reconstruct from packed indices.
    /// Input: `(B, N, 1)` → Output: `(B, N, dim)`.
    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        self.codes_from_indices(indices)?
            .to_dtype(self.model_dtype)?
            .apply(&self.project_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A ResidualFSQ with identity projections, as the library builds one
    /// when `dim` equals the codebook dimension.
    fn test_fsq() -> Result<ResidualFSQ> {
        let dev = &Device::Cpu;
        let levels = vec![8, 8, 8, 5, 5, 5];
        let codebook_dim = levels.len();
        let eye = Tensor::eye(codebook_dim, DType::F32, dev)?;
        let zeros = Tensor::zeros(codebook_dim, DType::F32, dev)?;
        let mut basis = vec![1i64];
        for i in 0..levels.len() - 1 {
            basis.push(basis[i] * levels[i] as i64);
        }
        Ok(ResidualFSQ {
            project_in: Linear::from_weights(eye.clone(), Some(zeros.clone())),
            project_out: Linear::from_weights(eye, Some(zeros)),
            basis,
            levels,
            model_dtype: DType::F32,
        })
    }

    fn close(got: &[f32], want: &[f32]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "dim {i}: got {g}, want {w}");
        }
    }

    /// Recorded from vector_quantize_pytorch 1.27.20:
    /// `FSQ([8,8,8,5,5,5], preserve_symmetry=True).indices_to_codes(...)`.
    #[test]
    fn indices_decode_to_symmetric_codes() -> Result<()> {
        let fsq = test_fsq()?;
        let cases: [(i64, [f32; 6]); 4] = [
            (0, [-1.0; 6]),
            (27605, [3.0 / 7.0, -3.0 / 7.0, 1.0, 0.5, -1.0, 0.0]),
            (32036, [1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 0.0, 0.0, 0.0]),
            (63999, [1.0; 6]),
        ];
        for (index, want) in cases {
            let indices = Tensor::new(&[[index]], &Device::Cpu)?.unsqueeze(2)?;
            let got = fsq
                .get_output_from_indices(&indices)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            close(&got, &want);
        }
        Ok(())
    }

    /// Recorded from vector_quantize_pytorch 1.27.20:
    /// `ResidualFSQ(dim=6, levels=[8,8,8,5,5,5], num_quantizers=1)(x)`.
    #[test]
    fn forward_matches_the_library() -> Result<()> {
        let fsq = test_fsq()?;
        let x = Tensor::new(
            &[[
                [0.3f32, -0.9, 2.0, 0.1, -0.05, 0.6],
                [-3.0, 0.0, 0.49, -0.6, 0.9, -0.2],
            ]],
            &Device::Cpu,
        )?;
        let (q, idx) = fsq.forward(&x)?;
        let q = q.flatten_all()?.to_vec1::<f32>()?;
        close(
            &q,
            &[
                3.0 / 7.0,
                -5.0 / 7.0,
                1.0,
                0.0,
                0.0,
                0.5,
                -1.0,
                1.0 / 7.0,
                3.0 / 7.0,
                -0.5,
                1.0,
                0.0,
            ],
        );
        assert_eq!(idx.flatten_all()?.to_vec1::<i64>()?, vec![45005, 36704]);
        // And the indices decode back to the same codes.
        let back = fsq
            .get_output_from_indices(&idx)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        close(&back, &q);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AudioTokenizer
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AudioTokenizer {
    audio_acoustic_proj: Linear,
    attention_pooler: AttentionPooler,
    quantizer: ResidualFSQ,
    pool_window_size: usize,
}

impl AudioTokenizer {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        Ok(Self {
            audio_acoustic_proj: linear(
                cfg.audio_acoustic_hidden_dim,
                enc_hs,
                vb.pp("audio_acoustic_proj"),
            )?,
            attention_pooler: AttentionPooler::new(cfg, vb.pp("attention_pooler"))?,
            quantizer: ResidualFSQ::new(cfg, vb.pp("quantizer"))?,
            pool_window_size: cfg.pool_window_size,
        })
    }

    pub fn pool_window_size(&self) -> usize {
        self.pool_window_size
    }

    /// Reconstruct from packed indices via the quantizer.
    /// Input: `(B, N, 1)` → Output: `(B, N, dim)`.
    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        self.quantizer.get_output_from_indices(indices)
    }

    /// Tokenize audio features into quantized tokens.
    /// Input: (B, T_patches, pool_window_size, acoustic_dim)
    /// Output: (quantized, indices)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.apply(&self.audio_acoustic_proj)?;
        let pooled = self.attention_pooler.forward(&x)?;
        self.quantizer.forward(&pooled)
    }
}

// ---------------------------------------------------------------------------
// AudioTokenDetokenizer
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AudioTokenDetokenizer {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    proj_out: Linear,
    special_tokens: Tensor,
    pool_window_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl AudioTokenDetokenizer {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        let num_layers = cfg.num_attention_pooler_hidden_layers;

        let mut layers = Vec::with_capacity(num_layers);
        let vb_l = vb.pp("layers");
        for i in 0..num_layers {
            layers.push(EncoderLayer::new(cfg, i, vb_l.pp(i))?);
        }
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        Ok(Self {
            embed_tokens: linear(enc_hs, enc_hs, vb.pp("embed_tokens"))?,
            layers,
            norm: RmsNorm::new(enc_hs, cfg.rms_norm_eps, vb.pp("norm"))?,
            proj_out: linear(enc_hs, cfg.audio_acoustic_hidden_dim, vb.pp("proj_out"))?,
            special_tokens: vb.get((1, cfg.pool_window_size, enc_hs), "special_tokens")?,
            pool_window_size: cfg.pool_window_size,
            rotary_emb,
        })
    }

    /// Detokenize quantized tokens back to continuous acoustic representations.
    /// Input: (B, T, D) → Output: (B, T * pool_window_size, acoustic_dim)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let x = x.apply(&self.embed_tokens)?;

        // Expand each token into pool_window_size patches
        let x = x
            .unsqueeze(2)?
            .broadcast_as((b, t, self.pool_window_size, x.dim(D::Minus1)?))?;
        let special = self.special_tokens.broadcast_as((
            b,
            t,
            self.pool_window_size,
            self.special_tokens.dim(D::Minus1)?,
        ))?;
        let x = (x.contiguous()? + special.contiguous()?)?;

        // Reshape for processing: (B*T, pool_window_size, hidden)
        let x = x.reshape((b * t, self.pool_window_size, ()))?;

        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None, &self.rotary_emb)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.apply(&self.proj_out)?;

        // Reshape back: (B*T, P, acoustic_dim) → (B, T*P, acoustic_dim)
        hidden.reshape((b, t * self.pool_window_size, ()))
    }
}
