//! Qwen3.6 vision tower.
//!
//! 27 pre-norm ViT blocks with **LayerNorm** (with biases — not the
//! `(1+w)·x` RmsNorm the language model uses), fused QKV attention,
//! GELU-tanh MLP. Followed by a `merger` that LayerNorms each
//! 1152-dim vision token, spatially 2×2-merges them into 4608-dim
//! groups, and projects to the LM's 5120-dim hidden via
//! `linear_fc1 → GELU → linear_fc2`.
//!
//! Architecture spec sourced from beast's cached Qwen3.6-27B
//! safetensors header (Stage A0, see
//! `doc/vision-qwen3_6-spec.md`). All weight shapes confirmed
//! from the live `.safetensors` headers, not inferred.
//!
//! **Conv3d wrinkle.** The published `patch_embed.proj.weight` is 5D
//! `[1152, 3, 2, 16, 16]` — a 3D conv with kernel
//! `(t=2, h=16, w=16)`. Candle 0.10 has no Conv3d. For static images
//! we get away with a trick: when the temporal patch size is 2 and we
//! duplicate the still image along the temporal axis (`T = 2`,
//! frame_0 == frame_1), the Conv3d output equals a Conv2d run with
//! the *sum* of the two temporal weight slices:
//!
//! ```text
//! output = W_0 · frame_0 + W_1 · frame_1 + bias
//!        = (W_0 + W_1) · frame + bias            (static image)
//! ```
//!
//! So at load we sum-collapse the temporal axis and use a 4D
//! `Conv2d` kernel. Video support would have to do the real Conv3d
//! (different frames mean the trick fails) — tracked alongside the
//! dynamic-resolution work in issue #14.
//!
//! Forward signature (Stage A — no LM splice yet):
//!
//! ```text
//! fn forward(&self, image: &Tensor) -> Result<Tensor>
//! ```
//!
//! `image` is `(3, H, W)` f32, normalised by `preprocess::preprocess`.
//! Returns `(N_lm_tokens, out_hidden_size)` post-merger tokens ready
//! to splice into the LM's input embeddings at `<|image_pad|>`
//! positions. For Qwen3.6 at 448×448 → 28×28 patches → 14×14 = 196
//! LM tokens of dim 5120.

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear};
use serde::Deserialize;

/// Qwen3.6 vision tower hyperparameters. Mirrors the `vision_config`
/// block of `config.json`. Only the fields we actually need are
/// captured; serde tolerates the rest.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Number of ViT blocks (`depth: 27` for Qwen3.6).
    pub depth: usize,
    /// Vision-token dimension throughout the tower (1152 for Qwen3.6).
    pub hidden_size: usize,
    /// MLP intermediate dim (4304).
    pub intermediate_size: usize,
    /// Attention head count (16). `head_dim = hidden_size / num_heads`.
    pub num_heads: usize,
    /// Number of slots in the learned position embedding (2304).
    /// Caps the maximum image patch count.
    pub num_position_embeddings: usize,
    /// Spatial patch edge in pixels (16).
    pub patch_size: usize,
    /// Temporal kernel depth in the patch embed (2 for Qwen3.6 — we
    /// collapse this into a single Conv2d for static-image inference;
    /// see the module-level Conv3d wrinkle).
    pub temporal_patch_size: usize,
    /// Patches grouped per LM token by the merger (2 → 2×2 = 4
    /// patches per LM token).
    pub spatial_merge_size: usize,
    /// Vision input channels (3, RGB).
    pub in_channels: usize,
    /// Merger output dim — matches the LM's `hidden_size` (5120 for
    /// Qwen3.6). The merger projects from vision dim → LM dim.
    pub out_hidden_size: usize,
}

const LAYER_NORM_EPS: f64 = 1e-6;
/// Number of LM tokens emitted by the merger per vision-token group.
const LM_TOKENS_PER_MERGE_GROUP: usize = 1;

/// One ViT block: pre-LN → attn → residual; pre-LN → MLP → residual.
struct VisionBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionBlock {
    fn load(cfg: &VisionConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = h / cfg.num_heads;
        let norm1 = layer_norm(vb.pp("norm1"), h)?;
        let qkv = linear(vb.pp("attn.qkv"), h, 3 * h)?;
        let proj = linear(vb.pp("attn.proj"), h, h)?;
        let norm2 = layer_norm(vb.pp("norm2"), h)?;
        let fc1 = linear(vb.pp("mlp.linear_fc1"), h, cfg.intermediate_size)?;
        let fc2 = linear(vb.pp("mlp.linear_fc2"), cfg.intermediate_size, h)?;
        Ok(Self {
            norm1,
            qkv,
            proj,
            norm2,
            fc1,
            fc2,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    /// `x`: `(N, hidden_size)` un-batched. Returns same shape.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn_in = self.norm1.forward(x)?;
        let attn_out = self.attention(&attn_in)?;
        let x = x.add(&attn_out)?;
        let mlp_in = self.norm2.forward(&x)?;
        let mlp_out = self.fc2.forward(&gelu_tanh(&self.fc1.forward(&mlp_in)?)?)?;
        x.add(&mlp_out).map_err(Into::into)
    }

    /// Multi-head self-attention over the patch sequence. No causal
    /// mask — every patch attends to every other patch.
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (n, hidden) = x.dims2()?;
        // qkv: (N, 3*hidden). Split into Q, K, V each (N, hidden).
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((n, 3, self.num_heads, self.head_dim))?;
        // Transpose to (3, num_heads, N, head_dim) for per-head views.
        let qkv = qkv.permute((1, 2, 0, 3))?.contiguous()?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // (num_heads, N, head_dim) @ (num_heads, head_dim, N) -> (num_heads, N, N)
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = (scores * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        // (num_heads, N, N) @ (num_heads, N, head_dim) -> (num_heads, N, head_dim)
        let out = probs.matmul(&v)?;
        // Merge heads back: (N, num_heads, head_dim) -> (N, hidden).
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((n, hidden))?;
        self.proj.forward(&out).map_err(Into::into)
    }
}

/// `merger`: LayerNorm per token → spatial 2×2 merge (concat 4
/// adjacent tokens into one 4608-dim vector) → fc1 → GELU-tanh →
/// fc2. Output dim is the LM's hidden_size.
struct VisionMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    merge_input_dim: usize,
    spatial_merge_size: usize,
}

impl VisionMerger {
    fn load(cfg: &VisionConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let merge = cfg.spatial_merge_size;
        let merge_input_dim = h * merge * merge;
        let norm = layer_norm(vb.pp("norm"), h)?;
        let fc1 = linear(vb.pp("linear_fc1"), merge_input_dim, merge_input_dim)?;
        let fc2 = linear(vb.pp("linear_fc2"), merge_input_dim, cfg.out_hidden_size)?;
        Ok(Self {
            norm,
            fc1,
            fc2,
            merge_input_dim,
            spatial_merge_size: merge,
        })
    }

    /// `tokens`: `(grid_h, grid_w, hidden_size)`. The merger reshapes
    /// each `merge×merge` block of adjacent patches into a single
    /// concatenated vector, then projects.
    ///
    /// `grid_h` and `grid_w` must both be multiples of
    /// `spatial_merge_size`. Returns
    /// `(grid_h/merge × grid_w/merge, out_hidden_size)`.
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (gh, gw, h) = tokens.dims3()?;
        let m = self.spatial_merge_size;
        anyhow::ensure!(
            gh.is_multiple_of(m) && gw.is_multiple_of(m),
            "merger expects spatial dims divisible by merge_size={m}; got ({gh}, {gw})"
        );
        let tokens = self.norm.forward(tokens)?;
        // (gh, gw, h) -> (gh/m, m, gw/m, m, h) -> (gh/m, gw/m, m, m, h)
        // -> flatten last three -> (gh/m, gw/m, m*m*h) -> (N_lm, merge_input_dim)
        let out_h = gh / m;
        let out_w = gw / m;
        let merged = tokens
            .reshape((out_h, m, out_w, m, h))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((out_h * out_w, self.merge_input_dim))?;
        let hidden = self.fc2.forward(&gelu_tanh(&self.fc1.forward(&merged)?)?)?;
        Ok(hidden)
    }
}

/// The vision tower itself.
pub struct VisionTower {
    /// Sum-collapsed temporal kernel (Conv2d, see module doc).
    patch_embed: Conv2d,
    pos_embed: Embedding,
    blocks: Vec<VisionBlock>,
    merger: VisionMerger,
    config: VisionConfig,
    dtype: DType,
    device: Device,
}

impl VisionTower {
    /// Load from a `ShardedVarBuilder` rooted at the safetensors
    /// `model.visual.` prefix. Caller is responsible for the `pp` —
    /// see `Qwen3_5ForCausalLM::new` (Stage A4).
    pub fn load(cfg: VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();

        // patch_embed.proj is published as 5D Conv3d weight; we
        // sum-collapse the temporal axis (size = temporal_patch_size)
        // to get a 4D Conv2d kernel. This is exact for the static-
        // image case where T = temporal_patch_size frames are
        // identical (i.e. the input was duplicated along T).
        let raw_weight = vb
            .pp("patch_embed.proj")
            .get(
                (
                    cfg.hidden_size,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ),
                "weight",
            )
            .context("load model.visual.patch_embed.proj.weight (5D Conv3d kernel)")?;
        // Sum along the temporal axis (dim 2) — see module doc-comment.
        let folded = raw_weight.sum(2)?; // -> (hidden, in_channels, patch, patch)
        let proj_bias = vb
            .pp("patch_embed.proj")
            .get(cfg.hidden_size, "bias")
            .context("load model.visual.patch_embed.proj.bias")?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embed = Conv2d::new(folded, Some(proj_bias), conv_cfg);

        let pos_embed_weight = vb
            .pp("pos_embed")
            .get((cfg.num_position_embeddings, cfg.hidden_size), "weight")
            .context("load model.visual.pos_embed.weight")?;
        let pos_embed = Embedding::new(pos_embed_weight, cfg.hidden_size);

        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(
                VisionBlock::load(&cfg, &blocks_vb.pp(i))
                    .with_context(|| format!("load vision block {i}"))?,
            );
        }
        let merger = VisionMerger::load(&cfg, &vb.pp("merger")).context("load vision merger")?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            config: cfg,
            dtype,
            device,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Number of LM tokens this tower emits for an `(H, W)` pixel
    /// image after the merger. Equal to
    /// `(H / patch_size / spatial_merge_size) * (W / patch_size / spatial_merge_size)`.
    pub fn lm_tokens_for(&self, h: u32, w: u32) -> usize {
        let m = self.config.spatial_merge_size;
        let patch = self.config.patch_size;
        let gh = (h as usize) / patch / m;
        let gw = (w as usize) / patch / m;
        gh * gw * LM_TOKENS_PER_MERGE_GROUP
    }

    /// Encode one image.
    ///
    /// `image`: row-major `(3, H, W)` f32 tensor on `self.device`,
    /// already normalised by `preprocess::preprocess`. Both `H` and
    /// `W` must be multiples of `patch_size * spatial_merge_size`.
    ///
    /// Returns `(N_lm, out_hidden_size)` — LM-side image tokens
    /// ready to splice into the language model's input embeddings.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let (c, h, w) = image.dims3()?;
        anyhow::ensure!(
            c == self.config.in_channels,
            "image must have {} channels, got {c}",
            self.config.in_channels
        );
        let patch = self.config.patch_size;
        anyhow::ensure!(
            h.is_multiple_of(patch) && w.is_multiple_of(patch),
            "image dims must be multiples of patch_size={patch}; got ({h}, {w})"
        );
        let gh = h / patch;
        let gw = w / patch;
        let n_patches = gh * gw;
        anyhow::ensure!(
            n_patches <= self.config.num_position_embeddings,
            "patch count {n_patches} exceeds pos_embed budget {}",
            self.config.num_position_embeddings
        );

        // Add batch axis for conv: (1, 3, H, W) → (1, hidden, gh, gw)
        // → (hidden, gh, gw) → permute to (gh, gw, hidden) → flatten to (N, hidden)
        let x = image.unsqueeze(0)?.to_dtype(self.dtype)?;
        let x = self.patch_embed.forward(&x)?;
        let x = x.squeeze(0)?;
        let x = x.permute((1, 2, 0))?.contiguous()?;
        let x = x.reshape((n_patches, self.config.hidden_size))?;

        // Add learned positional embeddings (sequential indices for
        // Stage A's fixed-resolution path; full 2D positional logic
        // lands with variable resolution, issue #14).
        let positions = Tensor::arange(0u32, n_patches as u32, &self.device)?;
        let pos = self.pos_embed.forward(&positions)?;
        let mut x = x.add(&pos)?;

        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x)
                .with_context(|| format!("vision block {i}"))?;
        }

        // (n_patches, hidden) → (gh, gw, hidden) for the merger.
        let x = x.reshape((gh, gw, self.config.hidden_size))?;
        self.merger.forward(&x)
    }
}

/// Manually load a candle_nn LayerNorm from a ShardedVarBuilder.
/// candle_nn's `layer_norm` builder takes `crate::VarBuilder`, not
/// `ShardedVarBuilder`, so the existing arch modules in this crate
/// uniformly do the manual load + struct construction pattern (see
/// `full_attn::load_linear_no_bias`). We follow suit here.
fn layer_norm(vb: ShardedVarBuilder, size: usize) -> Result<LayerNorm> {
    let weight = vb
        .get(size, "weight")
        .with_context(|| format!("load LayerNorm.weight at '{}'", vb.prefix()))?;
    let bias = vb
        .get(size, "bias")
        .with_context(|| format!("load LayerNorm.bias at '{}'", vb.prefix()))?;
    Ok(LayerNorm::new(weight, bias, LAYER_NORM_EPS))
}

/// Manually load a candle_nn Linear (with bias) from a
/// ShardedVarBuilder. Same rationale as `layer_norm` above.
fn linear(vb: ShardedVarBuilder, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load Linear.weight at '{}'", vb.prefix()))?;
    let bias = vb
        .get(out_dim, "bias")
        .with_context(|| format!("load Linear.bias at '{}'", vb.prefix()))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// PyTorch's `gelu_pytorch_tanh` approximation — what the Qwen3.6
/// vision tower's `hidden_act` specifies. candle's `Tensor::gelu`
/// uses the exact erf-based GELU, so we compute the tanh
/// approximation explicitly:
///
/// ```text
/// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// ```
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    // sqrt(2 / pi) = 0.7978845608028654
    const COEFF: f64 = 0.7978845608028654;
    const KAPPA: f64 = 0.044715;
    let x3 = x.powf(3.0)?;
    let inner = (x + (x3 * KAPPA)?)?;
    let inner = (inner * COEFF)?;
    let t = inner.tanh()?;
    let one_plus_t = (t + 1.0)?;
    let out = (x * 0.5)?;
    let out = out.broadcast_mul(&one_plus_t)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Build a tiny VisionConfig usable on CPU with random weights.
    /// Match the Qwen3.6 shape relations (depth-N stack, hidden mod
    /// num_heads, intermediate_size > hidden_size) but with small
    /// dims so tests run in milliseconds.
    fn tiny_config() -> VisionConfig {
        VisionConfig {
            depth: 2,
            hidden_size: 32,
            intermediate_size: 64,
            num_heads: 4,
            num_position_embeddings: 64,
            patch_size: 4,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            out_hidden_size: 48,
        }
    }

    /// Hand-construct a VisionTower with random weights. This is the
    /// same trick `linear_attn::tests::forward_smoke_with_tiny_dimensions`
    /// uses — bypass the safetensors-backed `ShardedVarBuilder` path
    /// (which can't be built from in-memory tensors) and assemble the
    /// struct fields directly. The real `VisionTower::load` is
    /// exercised by the cuda-integration smoke test in Stage A6.
    fn tiny_tower(cfg: &VisionConfig) -> VisionTower {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let zeros = |shape: &[usize]| Tensor::zeros(shape, dtype, &device).unwrap();
        let ones = |shape: &[usize]| Tensor::ones(shape, dtype, &device).unwrap();
        let randn = |shape: &[usize]| Tensor::randn(0_f32, 0.02, shape, &device).unwrap();

        let patch_embed = Conv2d::new(
            randn(&[
                cfg.hidden_size,
                cfg.in_channels,
                cfg.patch_size,
                cfg.patch_size,
            ]),
            Some(zeros(&[cfg.hidden_size])),
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
        );
        let pos_embed = Embedding::new(
            randn(&[cfg.num_position_embeddings, cfg.hidden_size]),
            cfg.hidden_size,
        );

        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            let head_dim = cfg.hidden_size / cfg.num_heads;
            blocks.push(VisionBlock {
                norm1: LayerNorm::new(
                    ones(&[cfg.hidden_size]),
                    zeros(&[cfg.hidden_size]),
                    LAYER_NORM_EPS,
                ),
                qkv: Linear::new(
                    randn(&[3 * cfg.hidden_size, cfg.hidden_size]),
                    Some(zeros(&[3 * cfg.hidden_size])),
                ),
                proj: Linear::new(
                    randn(&[cfg.hidden_size, cfg.hidden_size]),
                    Some(zeros(&[cfg.hidden_size])),
                ),
                norm2: LayerNorm::new(
                    ones(&[cfg.hidden_size]),
                    zeros(&[cfg.hidden_size]),
                    LAYER_NORM_EPS,
                ),
                fc1: Linear::new(
                    randn(&[cfg.intermediate_size, cfg.hidden_size]),
                    Some(zeros(&[cfg.intermediate_size])),
                ),
                fc2: Linear::new(
                    randn(&[cfg.hidden_size, cfg.intermediate_size]),
                    Some(zeros(&[cfg.hidden_size])),
                ),
                num_heads: cfg.num_heads,
                head_dim,
            });
        }

        let merge_input_dim = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merger = VisionMerger {
            norm: LayerNorm::new(
                ones(&[cfg.hidden_size]),
                zeros(&[cfg.hidden_size]),
                LAYER_NORM_EPS,
            ),
            fc1: Linear::new(
                randn(&[merge_input_dim, merge_input_dim]),
                Some(zeros(&[merge_input_dim])),
            ),
            fc2: Linear::new(
                randn(&[cfg.out_hidden_size, merge_input_dim]),
                Some(zeros(&[cfg.out_hidden_size])),
            ),
            merge_input_dim,
            spatial_merge_size: cfg.spatial_merge_size,
        };

        VisionTower {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            config: cfg.clone(),
            dtype,
            device,
        }
    }

    #[test]
    fn forward_with_random_weights_produces_finite_output() {
        let cfg = tiny_config();
        let tower = tiny_tower(&cfg);

        // 16×16 image at patch_size=4 → 4×4 patches → after 2×2
        // merge → 2×2 = 4 LM tokens of dim out_hidden_size.
        let image = Tensor::randn(0_f32, 1.0, (3, 16, 16), &Device::Cpu).unwrap();
        let out = tower.forward(&image).expect("forward");
        let (n_lm, hidden) = out.dims2().unwrap();
        assert_eq!(n_lm, 4);
        assert_eq!(hidden, cfg.out_hidden_size);

        // No NaN/Inf
        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "forward must produce finite values"
        );
    }

    #[test]
    fn lm_token_count_matches_grid() {
        let cfg = tiny_config();
        let tower = tiny_tower(&cfg);
        // 16x16 image → 4x4 patches → 2x2 = 4 LM tokens
        assert_eq!(tower.lm_tokens_for(16, 16), 4);
        // 32x32 image → 8x8 patches → 4x4 = 16 LM tokens
        assert_eq!(tower.lm_tokens_for(32, 32), 16);
    }

    #[test]
    fn rejects_image_with_dims_not_multiple_of_patch() {
        let cfg = tiny_config();
        let tower = tiny_tower(&cfg);
        let image = Tensor::randn(0_f32, 1.0, (3, 17, 17), &Device::Cpu).unwrap();
        let err = tower.forward(&image).unwrap_err();
        assert!(format!("{err:#}").contains("patch_size"));
    }

    #[test]
    fn rejects_image_with_wrong_channel_count() {
        let cfg = tiny_config();
        let tower = tiny_tower(&cfg);
        let image = Tensor::randn(0_f32, 1.0, (4, 16, 16), &Device::Cpu).unwrap();
        let err = tower.forward(&image).unwrap_err();
        assert!(format!("{err:#}").contains("channels"));
    }

    #[test]
    fn gelu_tanh_matches_known_values() {
        // Reference values for gelu_pytorch_tanh from PyTorch:
        //   gelu_tanh(0.0)  = 0.0
        //   gelu_tanh(1.0)  ≈ 0.8411920071
        //   gelu_tanh(-1.0) ≈ -0.1588079929
        let x = Tensor::new(&[0.0_f32, 1.0, -1.0], &Device::Cpu).unwrap();
        let y = gelu_tanh(&x).unwrap();
        let v: Vec<f32> = y.to_vec1().unwrap();
        assert!((v[0]).abs() < 1e-6, "gelu_tanh(0) ≈ 0, got {}", v[0]);
        assert!(
            (v[1] - 0.841_192_f32).abs() < 1e-5,
            "gelu_tanh(1) ≈ 0.84119, got {}",
            v[1]
        );
        assert!(
            (v[2] - -0.158_808_f32).abs() < 1e-5,
            "gelu_tanh(-1) ≈ -0.15881, got {}",
            v[2]
        );
    }
}
