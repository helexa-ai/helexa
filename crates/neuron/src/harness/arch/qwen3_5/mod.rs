//! Qwen3-Next (`model_type = "qwen3_5"`) architecture — Qwen3.6's
//! upstream architecture revision.
//!
//! ## Naming
//!
//! The model release this targets is `Qwen/Qwen3.6-*` but the
//! architecture name in HuggingFace's `config.json` is `qwen3_5`.
//! mistralrs calls the same architecture `qwen3_next`; that label
//! ages poorly the next time Qwen ship a new arch, so we key on the
//! canonical `qwen3_5` from the model's own config.
//!
//! ## Status
//!
//! **Single-GPU dense path is real**. Both attention flavours
//! (`full_attention` with the output-gated GQA causal attention and
//! `linear_attention` with the Gated DeltaNet recurrent block) are
//! implemented. The model loads from upstream safetensors via the
//! existing `load_arch_dense` dispatch and runs forward end to end.
//!
//! Numerical correctness vs the reference Python is **not yet
//! validated** — the structural code path is right, weight tensor
//! names match the upstream layout, shapes flow through cleanly, but
//! the Tbilisi probe (and any other downstream test) is the next
//! step. Likely places a bug would surface:
//! - Per-rank vs per-token-position offsets in the recurrent delta
//!   rule (`linear_attn.rs`).
//! - Off-by-one in the conv state continuation across decode steps.
//! - RoPE phase mismatch from MRoPE simplification (we treat the
//!   three position grids as collapsed, which is correct only for
//!   text-only inference).
//!
//! ## Submodules
//!
//! - [`rmsnorm`] — `Qwen3_5RmsNorm` (`(1+w)*x` variant), the
//!   `Qwen3_5RmsNormGated` used after the delta rule, and the
//!   `l2norm` helper.
//! - [`rope`] — text-side rotary embedding (mrope simplified, GLM
//!   rotate-half).
//! - [`mlp`] — SwiGLU MLP (gate/up/down, no bias).
//! - [`full_attn`] — `Qwen3_5Attention` with the output-gate
//!   widening on `q_proj`.
//! - [`linear_attn`] — `GatedDeltaNet` recurrent delta-rule block
//!   (causal depthwise Conv1d → silu → split → L2norm → per-token
//!   delta rule → RMSNormGated → out_proj).
//! - [`decoder`] — `Qwen3_5DecoderLayer` dispatching to one of the
//!   two attention flavours per layer index.
//!
//! ## Open work
//!
//! - **TP variant.** `harness/tp/tp_qwen3_5.rs` is the next step.
//!   Sharding strategy diverges by layer type:
//!   - Full-attention layers: column-parallel q/k/v (including the
//!     gate half of `q_proj`) + row-parallel `o_proj`, mirroring
//!     `tp_qwen3.rs`.
//!   - Linear-attention layers: the recurrent state is per-V-head, so
//!     V-head-dimension sharding works cleanly — split `num_v_heads`
//!     across ranks (`num_v_heads / world_size` per rank), shard
//!     `in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` along
//!     the V-head dim, and row-parallel `out_proj`. The `A_log` /
//!     `dt_bias` per-head params shard with the heads.
//!
//! - **Chunked delta-rule prefill.** `linear_attn.rs` runs the
//!   per-token recurrent path for prefill too — correct but O(L).
//!   Porting `torch_chunk_gated_delta_rule` (chunk_size=64) speeds
//!   prefill substantially with no surface change.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::Embedding;
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;
use serde::Deserialize;
use std::sync::Arc;

pub mod decoder;
pub mod full_attn;
pub mod linear_attn;
pub mod mlp;
pub mod moe;
pub mod rmsnorm;
pub mod rope;
pub mod snapshot;
pub mod vision;

use decoder::Qwen3_5DecoderLayer;
use rmsnorm::Qwen3_5RmsNorm;
use rope::RotaryEmbedding;

/// `model_type` we deserialise from `config.json`. Const so the
/// dispatch in `candle.rs::load_arch_dense` can pattern-match without
/// magic strings.
pub const MODEL_TYPE: &str = "qwen3_5";

/// `model_type` of the MoE sibling family (Qwen3-Next-80B-A3B /
/// Qwen3-Coder-Next): the same Gated DeltaNet hybrid layer stack with
/// a high-sparsity MoE FFN per layer. Served by this same arch module —
/// [`Config::from_config_json`] normalises the flat qwen3_next
/// `config.json` layout into the nested shape used here.
pub const MODEL_TYPE_NEXT: &str = "qwen3_next";

/// Top-level shape of Qwen3-Next's `config.json`. The real
/// hyperparameters live in `text_config`; the rest is multimodal /
/// tokeniser glue we don't need for the language-model forward.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Always `"qwen3_5"` for this architecture. Kept on the struct
    /// so the (eventual) dispatch / logging code can show it without
    /// re-parsing the JSON.
    pub model_type: String,
    /// The text-side hyperparameters. Everything we actually need.
    pub text_config: TextConfig,
    /// Vision tower hyperparameters. Present on multimodal
    /// checkpoints (e.g. Qwen/Qwen3.6-27B); absent on text-only
    /// variants. When present, `Qwen3_5ForCausalLM::new` loads the
    /// vision tower alongside the language model so vision-bearing
    /// requests can splice image embeddings at `<|image_pad|>` token
    /// positions.
    #[serde(default)]
    pub vision_config: Option<vision::VisionConfig>,
    /// Token id the chat template emits per image patch group.
    /// Mirrors the LM tokenizer's `<|image_pad|>` id (248056 for
    /// Qwen3.6). The runtime locates these in the prompt and splices
    /// in `VisionTower::forward` output. `None` for text-only models.
    #[serde(default)]
    pub image_token_id: Option<u32>,
}

impl Config {
    /// Parse a `config.json` for either family this arch serves,
    /// normalising layout differences (#92):
    ///
    /// - `model_type == "qwen3_5"` (Qwen3.6): hyperparameters nested
    ///   under `text_config`, RoPE nested under `rope_parameters` —
    ///   deserialises directly.
    /// - `model_type == "qwen3_next"` (Qwen3-Next-80B-A3B family):
    ///   **flat** layout — hyperparameters at the top level,
    ///   `rope_theta`/`partial_rotary_factor` flat, no vision block.
    ///   Wrapped into the nested shape here. The output gate on full
    ///   attention is unconditional in the upstream qwen3_next
    ///   implementation (the config carries no flag), so
    ///   `attn_output_gate` is forced on.
    ///
    /// Both variants may omit `layer_types` (qwen3_next always does);
    /// it is derived from `full_attention_interval` using the upstream
    /// convention: layer `i` is `full_attention` iff
    /// `(i + 1) % interval == 0`, else `linear_attention`.
    pub fn from_config_json(json: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).context("parse config.json as JSON")?;
        let model_type = v
            .get("model_type")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string();

        let mut cfg: Config = if model_type == MODEL_TYPE_NEXT {
            let mut text = v.clone();
            if text.get("rope_parameters").is_none() {
                let mut rope = serde_json::Map::new();
                for key in ["rope_theta", "partial_rotary_factor", "rope_type"] {
                    if let Some(val) = v.get(key) {
                        rope.insert(key.to_string(), val.clone());
                    }
                }
                text["rope_parameters"] = serde_json::Value::Object(rope);
            }
            let mut text_config: TextConfig = serde_json::from_value(text)
                .context("parse flat qwen3_next config.json hyperparameters")?;
            text_config.attn_output_gate = true;
            Config {
                model_type,
                text_config,
                vision_config: None,
                image_token_id: None,
            }
        } else {
            serde_json::from_str(json).context("parse nested qwen3_5 config.json")?
        };

        if cfg.text_config.layer_types.is_empty() {
            let interval = cfg.text_config.full_attention_interval.unwrap_or(4);
            anyhow::ensure!(
                interval > 0,
                "full_attention_interval must be >= 1 to derive layer_types"
            );
            cfg.text_config.layer_types = (0..cfg.text_config.num_hidden_layers)
                .map(|i| {
                    if (i + 1).is_multiple_of(interval) {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }

        Ok(cfg)
    }
}

/// Inner config (the `text_config` block). Mirrors the Qwen3 layout
/// but with the extras Qwen3-Next adds (`attn_output_gate`,
/// `layer_types`, `full_attention_interval`, larger `head_dim`).
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    /// Nested RoPE settings. Qwen3-Next puts `rope_theta` and
    /// `partial_rotary_factor` inside this block rather than at the
    /// top level — important because the partial rotary means only
    /// `head_dim * partial_rotary_factor` dims get RoPE applied (the
    /// rest pass through unchanged).
    pub rope_parameters: RopeParameters,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// New in Qwen3-Next: a sigmoid gate multiplied into the attention
    /// output before the o_proj. The Python reference applies it
    /// pointwise after softmax+matmul.
    #[serde(default)]
    pub attn_output_gate: bool,

    /// One entry per decoder layer; values are `"full_attention"` or
    /// `"linear_attention"`. Length must equal `num_hidden_layers`.
    /// `full_attention_interval` is a derived hint (every 4th layer
    /// by default) — `layer_types` is authoritative.
    #[serde(default)]
    pub layer_types: Vec<String>,

    /// Hint for the layer-type pattern (defaults to 4). Kept for
    /// logging / validation; the forward dispatches on `layer_types`.
    #[serde(default)]
    pub full_attention_interval: Option<usize>,

    /// Hidden activation (`"silu"` for Qwen3-Next). Used by the MLP
    /// and the linear-attention conv1d.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    // --- Gated DeltaNet (linear-attention) hyperparams -----------------
    /// Per-layer linear-attention V-head count (Qwen3.6-27B: 48).
    /// More V-heads than K-heads is fine — query/key get
    /// `repeat_interleave`'d to match before the delta rule.
    #[serde(default)]
    pub linear_num_value_heads: usize,
    /// Per-layer linear-attention K-head count (Qwen3.6-27B: 16).
    #[serde(default)]
    pub linear_num_key_heads: usize,
    /// Per-head key dimension for the linear-attention path
    /// (Qwen3.6-27B: 128). Separate from `head_dim` which the
    /// full-attention layers use.
    #[serde(default)]
    pub linear_key_head_dim: usize,
    /// Per-head value dimension for the linear-attention path
    /// (Qwen3.6-27B: 128).
    #[serde(default)]
    pub linear_value_head_dim: usize,
    /// Causal Conv1d kernel size used before the delta rule
    /// (Qwen3.6-27B: 4).
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,

    // --- High-sparsity MoE FFN (Qwen3-Next 80B-A3B family, #92) --------
    // All default to the dense case (0 experts) so existing dense
    // configs (Qwen3.6-27B) deserialise unchanged. A layer gets the MoE
    // FFN iff `layer_uses_moe` says so; otherwise the dense SwiGLU.
    /// Total routed experts per MoE layer (80B-A3B: 512). `0` → dense
    /// model, no MoE anywhere.
    #[serde(default)]
    pub num_experts: usize,
    /// Experts activated per token (80B-A3B: 10).
    #[serde(default)]
    pub num_experts_per_tok: usize,
    /// Per-expert FFN width (80B-A3B: 512). Distinct from the dense
    /// `intermediate_size`.
    #[serde(default)]
    pub moe_intermediate_size: usize,
    /// Width of the always-on shared expert (80B-A3B: 512). `0` → no
    /// shared expert (Qwen3-30B-A3B style).
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    /// Every `decoder_sparse_step`-th layer is MoE (1 → all layers,
    /// the 80B-A3B case). Follows the upstream `(i+1) % step == 0`
    /// convention.
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    /// Layer indices forced to the dense MLP even when MoE is on.
    /// Empty for 80B-A3B.
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    /// Renormalise the top-k routing weights to sum to 1 (80B-A3B:
    /// true). Upstream selects top-k *after* softmax over all experts.
    #[serde(default)]
    pub norm_topk_prob: bool,
}

impl TextConfig {
    /// Whether decoder layer `layer_idx` carries the MoE FFN (vs the
    /// dense SwiGLU). Mirrors upstream `Qwen3NextDecoderLayer`:
    /// experts configured, layer not in `mlp_only_layers`, and on the
    /// `decoder_sparse_step` grid.
    pub fn layer_uses_moe(&self, layer_idx: usize) -> bool {
        self.num_experts > 0
            && self.decoder_sparse_step > 0
            && !self.mlp_only_layers.contains(&layer_idx)
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

fn default_decoder_sparse_step() -> usize {
    1
}

fn default_hidden_act() -> String {
    "silu".into()
}

/// Nested `rope_parameters` block from a Qwen3-Next `config.json`.
///
/// For text-only inference the three MRoPE position grids carry
/// identical ids, so the interleave is a no-op and plain RoPE applies.
/// For vision inputs `mrope_section` + `mrope_interleaved` drive the
/// per-axis (text/height/width) rotary used by image tokens — see
/// `rope.rs`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// Base for the inverse-frequency computation. Qwen3.6: 10_000_000.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Fraction of `head_dim` that gets the rotation applied. The
    /// remaining `head_dim * (1 - partial_rotary_factor)` dims pass
    /// through unchanged. Qwen3.6 / Qwen3.5: 0.25.
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// `"default"` for the standard inv_freq RoPE; other values (e.g.
    /// `"linear"`, `"dynamic"`) are upstream-supported but not yet
    /// implemented here.
    #[serde(default)]
    pub rope_type: Option<String>,
    /// MRoPE per-axis section sizes `[text, height, width]` — e.g.
    /// `[11, 11, 10]` for Qwen3.6, summing to the rotary half-dim.
    /// Empty for models that don't declare MRoPE (→ plain RoPE).
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    /// Whether the three MRoPE axes are interleaved per-frequency
    /// (Qwen3-VL / Qwen3.6 style, `true`) rather than block-concatenated
    /// (Qwen2-VL style, `false`).
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

/// Splice rows from `img` into `h` at `positions`. Stage B helper.
///
/// `h`: `(1, L, hidden)` — the LM's input embedding tensor after
/// `embed_tokens.forward`.
/// `img`: `(N_img, hidden)` — image embeddings, one row per
/// `<|image_pad|>` token in the prompt. Must already be in `h.dtype()`.
/// `positions`: indices into the `L` axis where image rows go;
/// `positions.len() == N_img`.
///
/// Approach: group `positions` into contiguous runs (because the chat
/// template emits `<|vision_start|><|image_pad|>×N<|vision_end|>` —
/// the pad tokens for each image land in one contiguous span), then
/// `slice_assign` per run. For typical Qwen3.6 requests this is one
/// or two runs per image; `slice_assign` does one tensor copy per
/// run, which is cheap relative to the decoder forward pass.
pub(crate) fn splice_runs(
    h: &Tensor,
    img: &Tensor,
    positions: &[u32],
) -> candle_core::Result<Tensor> {
    debug_assert!(
        !positions.is_empty(),
        "splice_runs precondition: non-empty positions"
    );
    let hidden = h.dim(2)?;
    let mut out = h.clone();
    let mut img_offset = 0_usize;
    let mut run_start = positions[0] as usize;
    let mut run_end_exclusive = run_start + 1;
    for &p in &positions[1..] {
        let p = p as usize;
        if p == run_end_exclusive {
            run_end_exclusive = p + 1;
        } else {
            apply_run(
                &mut out,
                img,
                &mut img_offset,
                run_start,
                run_end_exclusive,
                hidden,
            )?;
            run_start = p;
            run_end_exclusive = p + 1;
        }
    }
    apply_run(
        &mut out,
        img,
        &mut img_offset,
        run_start,
        run_end_exclusive,
        hidden,
    )?;
    Ok(out)
}

fn apply_run(
    out: &mut Tensor,
    img: &Tensor,
    img_offset: &mut usize,
    run_start: usize,
    run_end_exclusive: usize,
    hidden: usize,
) -> candle_core::Result<()> {
    let run_len = run_end_exclusive - run_start;
    let slice = img
        .narrow(0, *img_offset, run_len)?
        .reshape((1, run_len, hidden))?;
    *out = out.slice_assign(&[0..1, run_start..run_end_exclusive, 0..hidden], &slice)?;
    *img_offset += run_len;
    Ok(())
}

/// Qwen3-Next base transformer (embedding + decoder stack + final
/// norm). Public so a TP variant in `harness/tp/tp_qwen3_5.rs` can
/// also build on it later — for now only `Qwen3_5ForCausalLM` is the
/// loaded handle.
pub struct Qwen3_5Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    /// Shared with every full-attention layer; the model uses it to
    /// build the per-forward cos/sin (interleaved M-RoPE for image
    /// tokens, plain for text) once, which the layers then apply.
    rotary: Arc<RotaryEmbedding>,
    /// `offset + rope_delta` is the text-axis position during decode.
    /// 0 for text-only; set from `get_rope_index` during a vision
    /// prefill (image tokens compress the position space, so text after
    /// the image resumes from a smaller counter than the sequence
    /// index). Reset in `clear_kv_cache`.
    rope_delta: i64,
    device: Device,
    dtype: DType,
}

impl Qwen3_5Model {
    /// `text_prefix` is where the text core lives in the checkpoint:
    /// - Qwen3.6 (multimodal, `model_type = "qwen3_5"`):
    ///   `model.language_model` — sibling to `model.visual.*` (the
    ///   vision tower) and top-level `lm_head` / `mtp.*`.
    /// - Qwen3-Next-80B-A3B (text-only, `model_type = "qwen3_next"`):
    ///   plain `model`.
    ///
    /// [`Qwen3_5ForCausalLM::new`] picks by `Config::model_type` via
    /// [`text_weight_prefix`].
    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder, text_prefix: &str) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();

        let text_vb = vb.pp(text_prefix);

        let embed_vb = text_vb.pp("embed_tokens");
        let embed_weight = embed_vb
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .with_context(|| format!("load '{}/weight'", embed_vb.prefix()))?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);

        if cfg.layer_types.len() != cfg.num_hidden_layers {
            anyhow::bail!(
                "config.text_config.layer_types must have num_hidden_layers ({}) entries; \
                 got {}",
                cfg.num_hidden_layers,
                cfg.layer_types.len()
            );
        }

        let vb_l = text_vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3_5DecoderLayer::load(
                cfg,
                rotary.clone(),
                i,
                &vb_l.pp(i),
            )?);
        }

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
        // New request → no image-compressed position offset until the
        // next vision prefill sets one.
        self.rope_delta = 0;
    }

    /// Capture every layer's cache state plus the rope position
    /// counter as one consistent prefix snapshot (#11). Only valid at
    /// a token boundary — i.e. between forward calls, which is the
    /// only time the caller can reach this anyway.
    pub fn snapshot_kv_cache(&self) -> candle_core::Result<snapshot::KvCacheSnapshot> {
        let layers = self
            .layers
            .iter()
            .map(|l| l.snapshot_kv())
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(snapshot::KvCacheSnapshot {
            layers,
            rope_delta: self.rope_delta,
        })
    }

    /// Replace the live cache state with a previously captured
    /// snapshot. The snapshot stays valid for further restores.
    pub fn restore_kv_cache(
        &mut self,
        snap: &snapshot::KvCacheSnapshot,
    ) -> candle_core::Result<()> {
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

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> candle_core::Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        self.forward_inner(input, offset, None, None, &[], None)
    }

    /// Lockstep batched decode step (#98): `input` is `(B, 1)` — one
    /// new token per batch row — with each row at its own sequence
    /// position `positions[i]` (typically `prefix_lens[i] + step`).
    /// The cache must hold batched state (see
    /// `snapshot::assemble_batch`); `attn_mask` is the padding mask
    /// from [`Self::batch_decode_mask`] (or `None` when no row is
    /// padded). Text-only: `rope_delta` is ignored — positions are
    /// explicit and vision requests never enter the batch path.
    pub fn forward_batch_decode(
        &mut self,
        input: &Tensor,
        positions: &[usize],
        attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        if l != 1 {
            candle_core::bail!("forward_batch_decode: expected (B, 1) input, got (B, {l})");
        }
        if positions.len() != b {
            candle_core::bail!(
                "forward_batch_decode: {} positions for batch of {b}",
                positions.len()
            );
        }
        let mut h = self.embed_tokens.forward(input)?;
        let (cos, sin) = self.rotary.batch_cos_sin(positions)?;
        for layer in &mut self.layers {
            h = layer.forward(&h, attn_mask, &cos, &sin)?;
        }
        self.norm.forward(&h)
    }

    /// Additive padding mask for a batched decode step: shape
    /// `(B, 1, 1, total_len)`, `-inf` on each row's padding gap
    /// `[prefix_lens[i], padded_len)`, zero elsewhere. `total_len` is
    /// the KV length *after* this step's append (`padded_len + step +
    /// 1`). Returns `None` when no row is padded (uniform prefix
    /// lengths) — the decode step then needs no mask at all, matching
    /// the single-sequence fast path.
    pub fn batch_decode_mask(
        &self,
        prefix_lens: &[usize],
        padded_len: usize,
        total_len: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        if prefix_lens.iter().all(|&len| len == padded_len) {
            return Ok(None);
        }
        let minf = f32::NEG_INFINITY;
        let b = prefix_lens.len();
        let mask: Vec<f32> = prefix_lens
            .iter()
            .flat_map(|&len| {
                (0..total_len).map(move |j| if j >= len && j < padded_len { minf } else { 0. })
            })
            .collect();
        Ok(Some(
            Tensor::from_vec(mask, (b, 1, 1, total_len), &self.device)?.to_dtype(self.dtype)?,
        ))
    }

    /// Forward for a vision-prefill chunk: optional image-embedding
    /// splice plus explicit interleaved-M-RoPE `position_ids` (the
    /// chunk's slice of the full prompt's 3D positions). Mirrors the TP
    /// `TpQwen3_5Model::forward_with_positions` — used by
    /// `Qwen3_5ForCausalLM::prefill_with_images_chunked`, which computes
    /// the positions once over the whole prompt and slices them per
    /// chunk so the position counters stay consistent across chunk
    /// boundaries (an image compresses the position space, so per-chunk
    /// offset arithmetic would be wrong).
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
            &[],
            Some(position_ids),
        )
    }

    /// Forward with image-embedding splice. Stage B of the vision plan.
    ///
    /// `input_ids`: `(1, L)` token ids — same shape the text-only
    /// `forward` accepts (single-batch; multi-batch vision is not in
    /// scope today).
    /// `image_embeds`: `(N_image_tokens, hidden_size)` — concatenation
    /// of every image's post-merger embedding (`VisionTower::forward`
    /// output), in the same order images appear in the input. The
    /// caller has already done the per-image patch-count expansion of
    /// `<|image_pad|>` tokens in `input_ids`, so `N_image_tokens`
    /// equals the number of `image_token_id` positions in `input_ids`.
    /// `image_token_id`: the sentinel token (e.g. 248056 for Qwen3.6).
    ///
    /// The splice replaces the LM's text-side embedding at each
    /// `image_token_id` position with the corresponding row from
    /// `image_embeds`. After the splice the decoder runs the interleaved
    /// M-RoPE path: `grids` carries each image's post-merge LM grid
    /// `(lm_gh, lm_gw)` so `get_rope_index` assigns image tokens their 2D
    /// coordinates (dynamic resolution, #14).
    pub fn forward_with_vision(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        image_embeds: &Tensor,
        image_token_id: u32,
        grids: &[(usize, usize)],
    ) -> candle_core::Result<Tensor> {
        self.forward_inner(
            input_ids,
            offset,
            Some(image_embeds),
            Some(image_token_id),
            grids,
            None,
        )
    }

    /// Shared forward. Splices image embeddings at `image_token_id`
    /// positions when present, then builds the rotary cos/sin, in
    /// precedence order: explicit `position_ids` (interleaved M-RoPE,
    /// the chunked-vision path that slices a once-computed position
    /// tensor) > internal M-RoPE from `grids` (single-shot vision) >
    /// plain positions at `offset + rope_delta` (text / decode).
    fn forward_inner(
        &mut self,
        input: &Tensor,
        offset: usize,
        image_embeds: Option<&Tensor>,
        image_token_id: Option<u32>,
        grids: &[(usize, usize)],
        position_ids: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        // Splice image embeddings at `image_token_id` positions, when
        // this forward carries any. Independent of how cos/sin is built.
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
                    "forward_with_vision: chunk has {} image-token positions but \
                     image_embeds carries {} tokens — per-image patch-count expansion \
                     / chunk slicing mismatch",
                    positions.len(),
                    n_img_tokens,
                );
            }
            if !positions.is_empty() {
                // Cast image_embeds to the LM's dtype, then splice the
                // contiguous `<|image_pad|>` runs in place.
                let img = img.to_dtype(self.dtype)?;
                h = splice_runs(&h, &img, &positions)?;
            }
        }

        // Build interleaved M-RoPE cos/sin so image tokens carry their
        // 2D (lm_gh × lm_gw) grid coordinates. Text / decode take the
        // plain-RoPE fast path — bit-for-bit the pre-M-RoPE behaviour
        // when `rope_delta == 0`.
        let (cos, sin) = if let Some(pos) = position_ids {
            // Pre-computed positions sliced for this chunk — the splice
            // above already advanced `rope_delta`'s effect into `pos`.
            self.rotary.mrope_cos_sin(pos)?
        } else if let Some(tok_id) = image_token_id {
            // Single-shot vision: compute the whole prompt's M-RoPE here
            // and stash `rope_delta` for the decode that follows.
            let ids: Vec<u32> = input.flatten_all()?.to_vec1()?;
            let (text, height, width, delta) = rope::get_rope_index(&ids, tok_id, grids)
                .map_err(|e| candle_core::Error::Msg(format!("get_rope_index: {e}")))?;
            self.rope_delta = delta;
            let pos = rope::mrope_position_tensor(&text, &height, &width, &self.device)?;
            self.rotary.mrope_cos_sin(&pos)?
        } else {
            let base = (offset as i64 + self.rope_delta).max(0) as usize;
            self.rotary.plain_cos_sin(base, l)?
        };

        // Causal mask only needed for L > 1 prefill; full-attention
        // layers consume it via broadcast_add. Linear-attention layers
        // ignore the mask.
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

pub struct Qwen3_5ForCausalLM {
    base: Qwen3_5Model,
    lm_head: Linear,
    /// Vision tower (Stage A4). `None` for text-only checkpoints or
    /// when the operator has opted out. When present, the harness's
    /// `Job::EncodeImage` dispatch path runs `vision.forward(image)`
    /// and the LM forward (Stage B) splices the result at
    /// `image_token_id` positions in the input embedding stream.
    vision: Option<vision::VisionTower>,
    /// Mirrors `Config::image_token_id`. Cached here so the runtime
    /// doesn't have to round-trip through the parsed config struct.
    image_token_id: Option<u32>,
}

/// Checkpoint prefix of the text core for a given `model_type` — see
/// [`Qwen3_5Model::load`].
pub fn text_weight_prefix(model_type: &str) -> &'static str {
    if model_type == MODEL_TYPE_NEXT {
        "model"
    } else {
        "model.language_model"
    }
}

impl Qwen3_5ForCausalLM {
    pub fn new(config: Config, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = &config.text_config;
        let base = Qwen3_5Model::load(cfg, &vb, text_weight_prefix(&config.model_type))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(base.embed_weight().clone(), None)
        } else {
            let weight = vb
                .pp("lm_head")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")
                .with_context(|| format!("load '{}/lm_head/weight'", vb.prefix()))?;
            Linear::new(weight, None)
        };
        // Stage A4: load the vision tower when the config carries a
        // `vision_config` block and the safetensors actually carry
        // `model.visual.*` weights. The `Option<VisionConfig>` on the
        // config makes this a single-source-of-truth decision —
        // text-only checkpoints just leave `vision_config` unset and
        // get `None` here without any extra plumbing.
        let vision = if let Some(vcfg) = config.vision_config.clone() {
            tracing::info!(
                depth = vcfg.depth,
                hidden_size = vcfg.hidden_size,
                "loading qwen3_5 vision tower"
            );
            Some(
                vision::VisionTower::load(vcfg, vb.pp("model.visual"))
                    .context("load qwen3_5 vision tower (model.visual.*)")?,
            )
        } else {
            None
        };
        Ok(Self {
            base,
            lm_head,
            vision,
            image_token_id: config.image_token_id,
        })
    }

    /// True when this checkpoint loaded a vision tower. Used by the
    /// HTTP layer to advertise vision capability in `/v1/models` and
    /// to reject image-bearing requests against text-only loads with
    /// a clean 400.
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Vision tower handle, if loaded. The device-worker
    /// `EncodeImage` job dispatches to `vision.forward(image)`.
    pub fn vision(&self) -> Option<&vision::VisionTower> {
        self.vision.as_ref()
    }

    /// `<|image_pad|>` token id from `config.json`, when known.
    /// The Stage B prompt-builder uses this to count expansion targets
    /// and the LM forward uses it to locate splice positions.
    pub fn image_token_id(&self) -> Option<u32> {
        self.image_token_id
    }

    /// `input`: token-id tensor of shape `(B, L)`. Returns logits at
    /// the last position, shape `(B, 1, vocab_size)` — same contract
    /// as `qwen3::ModelForCausalLM::forward` so the harness's
    /// `squeeze_to_vocab` helper handles both uniformly.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden = self.base.forward(input, offset)?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    /// Lockstep batched decode step (#98): `(B, 1)` input, per-row
    /// positions, padding mask from
    /// [`Qwen3_5Model::batch_decode_mask`]. Returns `(B, 1,
    /// vocab_size)` — one logits row per batch row.
    pub fn forward_batch_decode(
        &mut self,
        input: &Tensor,
        positions: &[usize],
        attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let hidden = self
            .base
            .forward_batch_decode(input, positions, attn_mask)?;
        hidden.apply(&self.lm_head)
    }

    /// Stage B: forward with image-embedding splice. Mirrors `forward`
    /// but routes through `Qwen3_5Model::forward_with_vision` so the
    /// LM's input embeddings get the image patches spliced in at
    /// `image_token_id` positions before the decoder stack runs.
    pub fn forward_with_vision(
        &mut self,
        input: &Tensor,
        offset: usize,
        image_embeds: &Tensor,
        image_token_id: u32,
        grids: &[(usize, usize)],
    ) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden =
            self.base
                .forward_with_vision(input, offset, image_embeds, image_token_id, grids)?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    /// Forward for a vision-prefill chunk: explicit M-RoPE positions +
    /// optional image splice. Mirrors `forward_with_vision` but routes
    /// through `Qwen3_5Model::forward_with_positions`. Used by
    /// [`Self::prefill_with_images_chunked`].
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

    /// Encode every preprocessed `(C, H, W)` image once through the
    /// vision tower and concatenate along the patch axis →
    /// `(sum_patches, hidden)`. Done once per prefill, not per chunk.
    fn encode_images_concat(&self, image_pixels: &[Tensor]) -> candle_core::Result<Tensor> {
        let tower = self.vision.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "encode_images_concat: loaded without a vision tower \
                 (config.json::vision_config absent or weights missing)"
                    .into(),
            )
        })?;
        let mut per_image = Vec::with_capacity(image_pixels.len());
        for (idx, img) in image_pixels.iter().enumerate() {
            let embed = tower
                .forward(img)
                .map_err(|e| candle_core::Error::Msg(format!("encode image[{idx}]: {e:#}")))?;
            per_image.push(embed);
        }
        Tensor::cat(&per_image.iter().collect::<Vec<_>>(), 0)
    }

    /// Chunked image prefill for the single-GPU path (#18) — parity with
    /// `TpQwen3_5ForCausalLM::prefill_with_images_chunked`. Encodes the
    /// image(s) once, then walks the (pre-expanded) prompt in
    /// `chunk_size`-token windows — exactly like the text
    /// `chunked_prefill_*` paths — splicing the patch embeddings into
    /// whichever chunk(s) carry `<|image_pad|>` positions. Activation
    /// memory is bounded by the chunk, not the full prompt, so a long
    /// vision context no longer single-shot-OOMs.
    ///
    /// The KV cache (and GDN recurrent state) accumulate across chunks
    /// via the growing offset — the same per-chunk associativity the
    /// text chunked prefill and prefix cache (#11/#23) rely on. Only the
    /// final chunk's last-position logits are returned; intermediate
    /// chunks just populate the cache. The caller is responsible for
    /// clearing the cache first.
    ///
    /// `base_offset` is the KV position the prefill starts at (0 for a
    /// fresh request). `image_pixels` are device-resident `(C, H, W)`
    /// tensors; grids and the interleaved-M-RoPE position ids are
    /// recomputed here so an image's position compression is consistent
    /// across chunk boundaries.
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
        let device = self.base.device.clone();

        let image_embeds = self.encode_images_concat(image_pixels)?;

        // Each image's LM grid (lm_gh, lm_gw) = (h/factor, w/factor),
        // factor = patch×merge — recomputed from the pixel tensors (#14
        // dynamic resolution).
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

        // Interleaved-M-RoPE 3D positions for the whole prompt, computed
        // once and sliced per chunk so image tokens get their grid
        // coordinates and text after an image resumes from the
        // compressed counter. `rope_delta` is stashed on the base model
        // for the decode that follows this prefill.
        let (text, height, width, delta) = rope::get_rope_index(tokens, image_token_id, &grids)
            .map_err(|e| candle_core::Error::Msg(format!("get_rope_index: {e}")))?;
        self.base.rope_delta = delta;
        let full_pos = rope::mrope_position_tensor(&text, &height, &width, &device)?;

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

    /// See [`Qwen3_5Model::snapshot_kv_cache`].
    pub fn snapshot_kv_cache(&self) -> candle_core::Result<snapshot::KvCacheSnapshot> {
        self.base.snapshot_kv_cache()
    }

    /// See [`Qwen3_5Model::restore_kv_cache`].
    pub fn restore_kv_cache(
        &mut self,
        snap: &snapshot::KvCacheSnapshot,
    ) -> candle_core::Result<()> {
        self.base.restore_kv_cache(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms we can deserialise the real upstream config shape.
    /// Sample taken from `Qwen/Qwen3.6-27B/config.json`, trimmed to
    /// the fields the architecture cares about. Note `rope_theta` and
    /// `partial_rotary_factor` are nested under `rope_parameters` —
    /// Qwen3-Next does NOT have a top-level `rope_theta`.
    #[test]
    fn config_deserialises_the_real_qwen3_6_shape() {
        let raw = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "language_model_only": false,
            "text_config": {
                "vocab_size": 248064,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "max_position_embeddings": 32768,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "layer_types": [
                    "linear_attention", "linear_attention",
                    "linear_attention", "full_attention"
                ]
            }
        }"#;
        let cfg: Config = serde_json::from_str(raw).expect("parse Qwen3.6 config");
        assert_eq!(cfg.model_type, "qwen3_5");
        assert_eq!(cfg.text_config.hidden_size, 5120);
        assert_eq!(cfg.text_config.head_dim, 256);
        assert!(cfg.text_config.attn_output_gate);
        assert_eq!(cfg.text_config.full_attention_interval, Some(4));
        assert_eq!(cfg.text_config.layer_types.len(), 4);
        assert_eq!(cfg.text_config.rope_parameters.rope_theta, 10_000_000.0);
        assert!((cfg.text_config.rope_parameters.partial_rotary_factor - 0.25).abs() < 1e-6);
        // Dense config: no MoE anywhere.
        assert_eq!(cfg.text_config.num_experts, 0);
        assert!(!cfg.text_config.layer_uses_moe(0));

        // The normalising entry point must agree with plain serde for
        // the nested shape (and leave the explicit layer_types alone).
        let via_norm = Config::from_config_json(raw).expect("normalised parse");
        assert_eq!(via_norm.text_config.layer_types.len(), 4);
        assert_eq!(via_norm.text_config.layer_types[3], "full_attention");
    }

    /// The flat qwen3_next layout (Qwen3-Next-80B-A3B family): all
    /// hyperparameters top-level, flat rope fields, no `layer_types`,
    /// MoE fields present. Sample mirrors
    /// `Qwen/Qwen3-Next-80B-A3B-Instruct/config.json`.
    #[test]
    fn config_normalises_the_flat_qwen3_next_shape() {
        let raw = r#"{
            "architectures": ["Qwen3NextForCausalLM"],
            "model_type": "qwen3_next",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 5120,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "moe_intermediate_size": 512,
            "norm_topk_prob": true,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "shared_expert_intermediate_size": 512
        }"#;
        let cfg = Config::from_config_json(raw).expect("parse qwen3_next config");
        assert_eq!(cfg.model_type, MODEL_TYPE_NEXT);
        assert!(cfg.vision_config.is_none());

        let t = &cfg.text_config;
        assert_eq!(t.hidden_size, 2048);
        // Flat rope fields normalised into the nested block.
        assert_eq!(t.rope_parameters.rope_theta, 10_000_000.0);
        assert!((t.rope_parameters.partial_rotary_factor - 0.25).abs() < 1e-6);
        // Output-gated attention is unconditional for qwen3_next.
        assert!(t.attn_output_gate);
        // layer_types derived from the interval: (i+1) % 4 == 0 → full.
        assert_eq!(t.layer_types.len(), 48);
        assert_eq!(t.layer_types[3], "full_attention");
        assert_eq!(t.layer_types[47], "full_attention");
        assert_eq!(t.layer_types[0], "linear_attention");
        assert_eq!(t.layer_types[46], "linear_attention");
        assert_eq!(
            t.layer_types
                .iter()
                .filter(|s| *s == "full_attention")
                .count(),
            12
        );
        // MoE hyperparameters land.
        assert_eq!(t.num_experts, 512);
        assert_eq!(t.num_experts_per_tok, 10);
        assert_eq!(t.moe_intermediate_size, 512);
        assert_eq!(t.shared_expert_intermediate_size, 512);
        assert!(t.norm_topk_prob);
        // decoder_sparse_step 1 + empty mlp_only_layers → every layer MoE.
        assert!(t.layer_uses_moe(0));
        assert!(t.layer_uses_moe(47));
    }

    /// End-to-end structural check for the qwen3_next path (#92): a
    /// tiny random-weight checkpoint in the **flat** layout (`model.*`
    /// prefix, fused `in_proj_qkvz`/`in_proj_ba`, per-expert MoE
    /// tensors, shared expert) loads through `Config::from_config_json`
    /// and `Qwen3_5ForCausalLM::new`, producing finite logits of the
    /// right shape. Numerical parity vs HF is pinned separately by the
    /// `qwen3_next_parity` fixture test.
    #[test]
    fn tiny_qwen3_next_checkpoint_loads_and_forwards() {
        use candle_core::Device;
        use std::collections::HashMap;

        let raw = r#"{
            "model_type": "qwen3_next",
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
            "full_attention_interval": 2,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "norm_topk_prob": true
        }"#;
        let cfg = Config::from_config_json(raw).expect("parse tiny qwen3_next config");
        assert_eq!(cfg.text_config.layer_types[0], "linear_attention");
        assert_eq!(cfg.text_config.layer_types[1], "full_attention");

        let dev = Device::Cpu;
        let randn = |shape: &[usize]| Tensor::randn(0f32, 0.1f32, shape, &dev).unwrap();
        let ones = |shape: &[usize]| Tensor::ones(shape, DType::F32, &dev).unwrap();
        let mut t: HashMap<String, Tensor> = HashMap::new();

        let (h, vocab) = (8usize, 32usize);
        t.insert("model.embed_tokens.weight".into(), randn(&[vocab, h]));
        t.insert("lm_head.weight".into(), randn(&[vocab, h]));
        t.insert("model.norm.weight".into(), ones(&[h]));

        let moe = |t: &mut HashMap<String, Tensor>, p: &str| {
            t.insert(format!("{p}.gate.weight"), randn(&[4, h]));
            for e in 0..4 {
                t.insert(format!("{p}.experts.{e}.gate_proj.weight"), randn(&[4, h]));
                t.insert(format!("{p}.experts.{e}.up_proj.weight"), randn(&[4, h]));
                t.insert(format!("{p}.experts.{e}.down_proj.weight"), randn(&[h, 4]));
            }
            t.insert(
                format!("{p}.shared_expert.gate_proj.weight"),
                randn(&[4, h]),
            );
            t.insert(format!("{p}.shared_expert.up_proj.weight"), randn(&[4, h]));
            t.insert(
                format!("{p}.shared_expert.down_proj.weight"),
                randn(&[h, 4]),
            );
            t.insert(format!("{p}.shared_expert_gate.weight"), randn(&[1, h]));
        };

        // Layer 0: linear_attention with the FUSED qwen3_next input
        // projections. key_dim = 2*4 = 8, value_dim = 4*4 = 16 →
        // qkvz rows = 2*8 + 2*16 = 48, ba rows = 2*4 = 8, conv_dim = 32.
        let l0 = "model.layers.0";
        t.insert(
            format!("{l0}.linear_attn.in_proj_qkvz.weight"),
            randn(&[48, h]),
        );
        t.insert(
            format!("{l0}.linear_attn.in_proj_ba.weight"),
            randn(&[8, h]),
        );
        t.insert(
            format!("{l0}.linear_attn.conv1d.weight"),
            randn(&[32, 1, 4]),
        );
        t.insert(format!("{l0}.linear_attn.dt_bias"), randn(&[4]));
        t.insert(format!("{l0}.linear_attn.A_log"), randn(&[4]));
        t.insert(format!("{l0}.linear_attn.norm.weight"), ones(&[4]));
        t.insert(format!("{l0}.linear_attn.out_proj.weight"), randn(&[h, 16]));
        t.insert(format!("{l0}.input_layernorm.weight"), ones(&[h]));
        t.insert(format!("{l0}.post_attention_layernorm.weight"), ones(&[h]));
        moe(&mut t, &format!("{l0}.mlp"));

        // Layer 1: full_attention (output-gated: q_proj is 2×).
        let l1 = "model.layers.1";
        t.insert(
            format!("{l1}.self_attn.q_proj.weight"),
            randn(&[2 * 2 * 4, h]),
        );
        t.insert(format!("{l1}.self_attn.k_proj.weight"), randn(&[4, h]));
        t.insert(format!("{l1}.self_attn.v_proj.weight"), randn(&[4, h]));
        t.insert(format!("{l1}.self_attn.o_proj.weight"), randn(&[h, 8]));
        t.insert(format!("{l1}.self_attn.q_norm.weight"), ones(&[4]));
        t.insert(format!("{l1}.self_attn.k_norm.weight"), ones(&[4]));
        t.insert(format!("{l1}.input_layernorm.weight"), ones(&[h]));
        t.insert(format!("{l1}.post_attention_layernorm.weight"), ones(&[h]));
        moe(&mut t, &format!("{l1}.mlp"));

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&t, &path).expect("save safetensors");
        // SAFETY: mmap of a file this test just wrote; nothing mutates it.
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path),
                DType::F32,
                &dev,
            )
            .expect("build ShardedVarBuilder")
        };

        let mut model = Qwen3_5ForCausalLM::new(cfg, vb).expect("load tiny qwen3_next checkpoint");
        let input = Tensor::new(&[1u32, 5, 9], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = model.forward(&input, 0).expect("forward");
        assert_eq!(logits.dims(), &[1, 1, vocab]);
        let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
    }

    /// `mlp_only_layers` and `decoder_sparse_step` gate `layer_uses_moe`
    /// per the upstream convention.
    #[test]
    fn layer_uses_moe_respects_step_and_exclusions() {
        let raw = r#"{
            "model_type": "qwen3_next",
            "vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
            "num_hidden_layers": 8, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
            "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [3]
        }"#;
        let cfg = Config::from_config_json(raw).expect("parse");
        let t = &cfg.text_config;
        // step 2 → layers 1, 3, 5, 7 are on the sparse grid…
        assert!(!t.layer_uses_moe(0));
        assert!(t.layer_uses_moe(1));
        // …but 3 is excluded by mlp_only_layers.
        assert!(!t.layer_uses_moe(3));
        assert!(t.layer_uses_moe(5));
    }

    /// `splice_runs` replaces (1, L, H) embedding rows at the given
    /// positions with rows from a (N_img, H) image-embedding tensor,
    /// in the order positions are supplied.
    #[test]
    fn splice_runs_replaces_at_contiguous_positions() {
        use candle_core::{DType, Device};

        let dev = Device::Cpu;
        // (1, L=5, H=2) text embeddings — encoded as floats so the
        // assertion can spot the change without dtype conversion.
        let h_vals: Vec<f32> = vec![
            10., 11., // pos 0
            20., 21., // pos 1
            30., 31., // pos 2
            40., 41., // pos 3
            50., 51., // pos 4
        ];
        let h = Tensor::from_vec(h_vals, (1, 5, 2), &dev).unwrap();

        // Two image embeddings to splice at positions 1 and 2 (a
        // contiguous run — single image emitting two patch tokens).
        let img_vals: Vec<f32> = vec![-1., -2., -3., -4.];
        let img = Tensor::from_vec(img_vals, (2, 2), &dev).unwrap();

        let out = splice_runs(&h, &img, &[1, 2]).unwrap();
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![10., 11., -1., -2., -3., -4., 40., 41., 50., 51.]);
        let _ = DType::F32;
    }

    /// Non-contiguous positions: two images at positions [1] and [3]
    /// each contributing one patch. `splice_runs` should iterate
    /// runs and place the corresponding image rows.
    #[test]
    fn splice_runs_handles_non_contiguous_runs() {
        use candle_core::Device;
        let dev = Device::Cpu;
        let h_vals: Vec<f32> = vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.];
        let h = Tensor::from_vec(h_vals, (1, 5, 2), &dev).unwrap();
        let img_vals: Vec<f32> = vec![-1., -2., -3., -4.];
        let img = Tensor::from_vec(img_vals, (2, 2), &dev).unwrap();
        let out = splice_runs(&h, &img, &[1, 3]).unwrap();
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1., 1., -1., -2., 3., 3., -3., -4., 5., 5.]);
    }
}
