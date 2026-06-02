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
pub mod rmsnorm;
pub mod rope;
pub mod vision;

use decoder::Qwen3_5DecoderLayer;
use rmsnorm::Qwen3_5RmsNorm;
use rope::RotaryEmbedding;

/// `model_type` we deserialise from `config.json`. Const so the
/// dispatch in `candle.rs::load_arch_dense` can pattern-match without
/// magic strings.
pub const MODEL_TYPE: &str = "qwen3_5";

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
}

fn default_hidden_act() -> String {
    "silu".into()
}

/// Nested `rope_parameters` block from a Qwen3-Next `config.json`.
/// `mrope_section` and `mrope_interleaved` are accepted via the
/// `#[serde(default)]` flatten-tolerance below but ignored — we treat
/// MRoPE as plain RoPE for text-only inference (the three position
/// grids carry identical ids when there's no vision input, so the
/// interleaving is a no-op).
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
fn splice_runs(h: &Tensor, img: &Tensor, positions: &[u32]) -> candle_core::Result<Tensor> {
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
    device: Device,
    dtype: DType,
}

impl Qwen3_5Model {
    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();

        // Qwen3-Next is a multimodal architecture whose text core lives
        // under `model.language_model.*` — sibling to `model.visual.*`
        // (the vision tower) and to top-level `lm_head` / `mtp.*`.
        // Every text-side tensor in the safetensors files is under
        // this prefix; we ignore the vision and MTP weights for
        // language-model inference.
        let text_vb = vb.pp("model.language_model");

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
        self.forward_inner(input, offset, None, None)
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
    /// `image_embeds`. After the splice the decoder runs unchanged.
    ///
    /// **MRoPE gap.** Qwen3.6's `rope_parameters` declares MRoPE
    /// (interleaved text/height/width axes); Stage B applies plain
    /// text-position RoPE to image tokens. The model still attends
    /// to image content but loses spatial structure that MRoPE-aware
    /// position encoding would preserve. Tracked under issue #15
    /// (numerical validation) — quality benchmark from Stage D should
    /// surface the impact, and the fix lives in `rope::RotaryEmbedding`.
    pub fn forward_with_vision(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        image_embeds: &Tensor,
        image_token_id: u32,
    ) -> candle_core::Result<Tensor> {
        self.forward_inner(input_ids, offset, Some(image_embeds), Some(image_token_id))
    }

    fn forward_inner(
        &mut self,
        input: &Tensor,
        offset: usize,
        image_embeds: Option<&Tensor>,
        image_token_id: Option<u32>,
    ) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Splice image embeddings at `image_token_id` positions. The
        // caller pre-expanded the prompt so every patch token in the
        // image_embeds tensor has a matching position in `input`. We
        // index_put the rows in place.
        if let (Some(img), Some(tok_id)) = (image_embeds, image_token_id) {
            // Locate image-token positions in input_ids. Operate on
            // CPU since the input ids are tiny (max ~10k entries
            // including the patch expansion) and the comparison is
            // not in the per-step hot path.
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
                    "forward_with_vision: prompt has {} image-token positions but \
                     image_embeds carries {} tokens — call build_prompt_for_request to \
                     ensure the per-image patch-count expansion has been applied",
                    positions.len(),
                    n_img_tokens,
                );
            }
            if !positions.is_empty() {
                // Cast image_embeds to the LM's dtype so the splice
                // produces a uniform tensor for the decoder stack.
                let img = img.to_dtype(self.dtype)?;
                // index_select would return the rows; we want to put.
                // candle's slice_assign with explicit positions ranges
                // doesn't exist; use scatter via index_select + an
                // accumulator: build a `(B, L, hidden)` zero tensor,
                // scatter the image rows in, then add to a masked
                // version of `h`. Simpler approach: walk positions
                // and use `slice_assign` for contiguous runs. Since
                // image_pad runs are contiguous (template emits
                // `<|vision_start|><|image_pad|>×N<|vision_end|>`),
                // we group positions and assign per run.
                h = splice_runs(&h, &img, &positions)?;
            }
        }
        // Causal mask only needed for L > 1 prefill; full-attention
        // layers consume it via broadcast_add. Linear-attention layers
        // ignore the mask.
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

impl Qwen3_5ForCausalLM {
    pub fn new(config: Config, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = &config.text_config;
        let base = Qwen3_5Model::load(cfg, &vb)?;
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
    ) -> candle_core::Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden = self
            .base
            .forward_with_vision(input, offset, image_embeds, image_token_id)?;
        hidden.i((.., l - 1.., ..))?.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
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
