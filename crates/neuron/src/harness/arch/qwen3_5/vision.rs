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

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Legacy escape hatch: when set, use the original Stage-A sequential
/// `pos_embed` lookup instead of the bilinear grid interpolation.
/// Default off (interpolation on) — for A/B comparison only.
fn vision_legacy_pos() -> bool {
    env_truthy("NEURON_VISION_LEGACY_POS")
}

/// Legacy escape hatch: when set, skip the 2D vision rotary in the ViT
/// attention (the original Stage-A behaviour). Default off (rotary on)
/// — for A/B comparison only.
fn vision_legacy_rope() -> bool {
    env_truthy("NEURON_VISION_LEGACY_ROPE")
}

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

    /// `x`: `(N, hidden_size)` un-batched. `rotary`: optional
    /// `(cos, sin)` each `(N, head_dim/2)` — the 2D vision rotary applied
    /// to q/k. Returns same shape.
    fn forward(&self, x: &Tensor, rotary: Option<&(Tensor, Tensor)>) -> Result<Tensor> {
        let attn_in = self.norm1.forward(x)?;
        let attn_out = self.attention(&attn_in, rotary)?;
        let x = x.add(&attn_out)?;
        let mlp_in = self.norm2.forward(&x)?;
        let mlp_out = self.fc2.forward(&gelu_tanh(&self.fc1.forward(&mlp_in)?)?)?;
        x.add(&mlp_out).map_err(Into::into)
    }

    /// Multi-head self-attention over the patch sequence. No causal
    /// mask — every patch attends to every other patch. When `rotary` is
    /// given, the 2D vision rotary (row/col position) is applied to q, k
    /// before the scores, matching HF `apply_rotary_pos_emb_vision`
    /// (`rope_slow` is the same rotate-half form).
    fn attention(&self, x: &Tensor, rotary: Option<&(Tensor, Tensor)>) -> Result<Tensor> {
        let (n, hidden) = x.dims2()?;
        // qkv: (N, 3*hidden). Split into Q, K, V each (N, hidden).
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((n, 3, self.num_heads, self.head_dim))?;
        // Transpose to (3, num_heads, N, head_dim) for per-head views.
        let qkv = qkv.permute((1, 2, 0, 3))?.contiguous()?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        // 2D vision rotary on q, k (full head_dim; rotate-half form).
        let (q, k) = match rotary {
            Some((cos, sin)) => {
                let q = candle_nn::rotary_emb::rope_slow(&q.unsqueeze(0)?, cos, sin)?.squeeze(0)?;
                let k = candle_nn::rotary_emb::rope_slow(&k.unsqueeze(0)?, cos, sin)?.squeeze(0)?;
                (q, k)
            }
            None => (q, k),
        };
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

/// 2D rotary position embedding for the vision tower. Each patch's
/// `head_dim` rotates by its `(row, col)` grid coordinates: the first
/// half of the rotary freqs are driven by the row position, the second
/// half by the column. Mirrors HF `Qwen3VLVisionRotaryEmbedding` +
/// `rot_pos_emb` (θ = 10000, `dim = head_dim/2`).
struct VisionRotaryEmbedding {
    /// `(half,)` f32, `half = head_dim/4` freqs per spatial axis.
    inv_freq: Vec<f32>,
}

impl VisionRotaryEmbedding {
    fn new(head_dim: usize) -> Self {
        // HF: Qwen3VLVisionRotaryEmbedding(head_dim // 2), theta 10000.
        let dim = head_dim / 2;
        let theta = 10000f32;
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect();
        Self { inv_freq }
    }

    /// cos/sin for a `gh×gw` patch grid in **row-major** order. Returns
    /// `(cos, sin)` each `(gh*gw, head_dim/2)`: per patch, the row-axis
    /// freqs `row·inv_freq` followed by the col-axis freqs `col·inv_freq`
    /// (then `rope_slow` duplicates them across the full head_dim).
    fn cos_sin(
        &self,
        gh: usize,
        gw: usize,
        dev: &Device,
        dtype: DType,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let n = gh * gw;
        let mut data = Vec::with_capacity(n * 2 * half);
        for hi in 0..gh {
            for wi in 0..gw {
                for &f in &self.inv_freq {
                    data.push(hi as f32 * f);
                }
                for &f in &self.inv_freq {
                    data.push(wi as f32 * f);
                }
            }
        }
        let freqs = Tensor::from_vec(data, (n, 2 * half), dev)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok((cos, sin))
    }
}

/// The vision tower itself.
pub struct VisionTower {
    /// Sum-collapsed temporal kernel (Conv2d, see module doc).
    patch_embed: Conv2d,
    pos_embed: Embedding,
    rotary: VisionRotaryEmbedding,
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
        let rotary = VisionRotaryEmbedding::new(cfg.hidden_size / cfg.num_heads);

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
            rotary,
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

    /// Bilinearly interpolate the learned `pos_embed` grid (a
    /// `num_grid_per_side × num_grid_per_side` table, 48×48 for Qwen3.6)
    /// onto the actual `gh × gw` patch grid, in **row-major** patch
    /// order. Port of the HF `fast_pos_embed_interpolate`: for each patch
    /// at fractional grid coord `(linspace(0, ngrid-1, gh)[hi],
    /// linspace(0, ngrid-1, gw)[wi])`, blend the 4 surrounding grid
    /// entries by bilinear weights. Returns `(gh*gw, hidden)` in
    /// `self.dtype`.
    fn interpolated_pos_embed(&self, gh: usize, gw: usize) -> Result<Tensor> {
        let ngrid = (self.config.num_position_embeddings as f64).sqrt().round() as usize;
        anyhow::ensure!(
            ngrid * ngrid == self.config.num_position_embeddings,
            "num_position_embeddings {} is not a perfect square",
            self.config.num_position_embeddings
        );
        // Evenly-spaced fractional indices into the [0, ngrid-1] grid.
        let lin = |n: usize| -> Vec<f64> {
            if n <= 1 {
                vec![0.0]
            } else {
                let step = (ngrid - 1) as f64 / (n - 1) as f64;
                (0..n).map(|i| i as f64 * step).collect()
            }
        };
        let hs = lin(gh);
        let ws = lin(gw);
        let n = gh * gw;

        // Four corner index sets + bilinear weight sets, row-major.
        let mut idx: [Vec<u32>; 4] = [
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        ];
        let mut wts: [Vec<f32>; 4] = [
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        ];
        for &hv in &hs {
            let hf = hv as usize; // floor (hv >= 0)
            let hc = (hf + 1).min(ngrid - 1);
            let dh = (hv - hf as f64) as f32;
            for &wv in &ws {
                let wf = wv as usize;
                let wc = (wf + 1).min(ngrid - 1);
                let dw = (wv - wf as f64) as f32;
                idx[0].push((hf * ngrid + wf) as u32);
                wts[0].push((1.0 - dh) * (1.0 - dw));
                idx[1].push((hf * ngrid + wc) as u32);
                wts[1].push((1.0 - dh) * dw);
                idx[2].push((hc * ngrid + wf) as u32);
                wts[2].push(dh * (1.0 - dw));
                idx[3].push((hc * ngrid + wc) as u32);
                wts[3].push(dh * dw);
            }
        }

        let mut acc: Option<Tensor> = None;
        for corner in 0..4 {
            let idx_t = Tensor::from_vec(std::mem::take(&mut idx[corner]), (n,), &self.device)?;
            let emb = self.pos_embed.forward(&idx_t)?; // (n, hidden), pos_embed dtype
            let wt = Tensor::from_vec(std::mem::take(&mut wts[corner]), (n, 1), &self.device)?
                .to_dtype(self.dtype)?;
            let term = emb.broadcast_mul(&wt)?;
            acc = Some(match acc {
                Some(a) => a.add(&term)?,
                None => term,
            });
        }
        Ok(acc.expect("4 corners accumulated"))
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

        // Learned absolute position embeddings. The `pos_embed` table is
        // a `num_position_embeddings = num_grid_per_side²` learned grid
        // (48×48 for Qwen3.6); for a `gh×gw` patch grid the reference
        // (`fast_pos_embed_interpolate`) bilinearly interpolates that
        // grid to `gh×gw`. The legacy path (a naive sequential lookup of
        // the first `n_patches` rows) mis-maps the grid stride and
        // scrambles spatial structure — kept only behind
        // `NEURON_VISION_LEGACY_POS=1` for A/B comparison.
        let pos = if vision_legacy_pos() {
            let positions = Tensor::arange(0u32, n_patches as u32, &self.device)?;
            self.pos_embed.forward(&positions)?
        } else {
            self.interpolated_pos_embed(gh, gw)?
        };
        let mut x = x.add(&pos)?;

        // 2D vision rotary (row/col per patch), computed once and applied
        // in every block's attention. Legacy escape hatch skips it.
        let rotary = if vision_legacy_rope() {
            None
        } else {
            Some(self.rotary.cos_sin(gh, gw, &self.device, self.dtype)?)
        };
        let rotary_ref = rotary.as_ref();

        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x, rotary_ref)
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

        let rotary = VisionRotaryEmbedding::new(cfg.hidden_size / cfg.num_heads);
        VisionTower {
            patch_embed,
            pos_embed,
            rotary,
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
    fn interpolated_pos_embed_reduces_to_sequential_at_native_grid() {
        // When the patch grid equals the pos_embed grid (gh=gw=ngrid),
        // linspace(0,ngrid-1,ngrid) is the integer ladder, so every patch
        // lands exactly on a grid node (dh=dw=0, corner-0 weight 1) and
        // the bilinear result is the raw pos_embed rows in row-major
        // order — i.e. identical to the legacy sequential lookup.
        let cfg = tiny_config();
        let tower = tiny_tower(&cfg);
        let ngrid = (cfg.num_position_embeddings as f64).sqrt() as usize; // 8
        let interp = tower.interpolated_pos_embed(ngrid, ngrid).unwrap();
        let seq = tower
            .pos_embed
            .forward(&Tensor::arange(0u32, (ngrid * ngrid) as u32, &Device::Cpu).unwrap())
            .unwrap();
        let a: Vec<f32> = interp.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = seq.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "interp {x} vs seq {y}");
        }
    }

    #[test]
    fn vision_rotary_row_col_structure() {
        // head_dim 8 → rotary dim 4 → inv_freq over [0,2] → 2 freqs/axis.
        let rot = VisionRotaryEmbedding::new(8);
        assert_eq!(rot.inv_freq.len(), 2);
        let (cos, sin) = rot.cos_sin(2, 2, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(cos.dims(), &[4, 4]); // 4 patches, head_dim/2 = 4 cols

        // Patch (0,0): all freqs 0 → cos 1, sin 0.
        let s0: Vec<f32> = sin.i(0).unwrap().to_vec1().unwrap();
        assert!(s0.iter().all(|&s| s.abs() < 1e-6));

        // Patch index 2 = grid (1,0): row=1 drives the first half, col=0
        // leaves the second half at zero.
        let s2: Vec<f32> = sin.i(2).unwrap().to_vec1().unwrap();
        assert!(s2[0].abs() > 1e-6, "row half must be non-zero");
        assert!(
            s2[2].abs() < 1e-6 && s2[3].abs() < 1e-6,
            "col half must be zero"
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
