//! `qwen4_exp`'s `config.json`, and the geometry derived from it.
//!
//! Read from the checkpoint's own file
//! (`Qwen/Qwen3.8-Flash-Next`, revision
//! `de4b8e4d43b917e7706784d8bb445c9af86a3540`), which nests everything
//! the text path needs under `text_config` alongside a `vision_config`
//! this module ignores until #314.
//!
//! Most of the vocabulary overlaps [`super::super::qwen3_5`] — the same
//! hybrid `layer_types`, the same GatedDeltaNet dimensions, the same
//! nested `rope_parameters` — so the reused blocks read their settings
//! from the same shapes. What is new is declared here: the four
//! residual streams, the hashed n-gram table, and the sparse-attention
//! indexer.
//!
//! Three of these fields say something other than what they appear to:
//!
//! 1. **`ple_layer_ids` is one-indexed.** The shipped `[2]` means
//!    zero-indexed layer **1**. Upstream computes
//!    `ple_layer_ids.index(layer_idx + 1)`; reading it as a plain layer
//!    number puts 28% of the model's parameters on the wrong layer, and
//!    the model still speaks.
//! 2. **`output_gate_type: sigmoid`.** Qwen3.6 leaves this unset and
//!    falls back to `hidden_act`, which is SiLU. Inheriting that here
//!    is wrong on all 36 linear-attention layers.
//! 3. **`mamba_ssm_dtype: float32`.** We default the equivalent
//!    GatedDeltaNet state to a bf16 round-trip (#284). This checkpoint
//!    makes an explicit choice and a port must not silently override
//!    it with ours.
//!
//! There is no `intermediate_size`: every layer's FFN is the MoE, so
//! there is no dense width to declare. See
//! `doc/qwen4_exp-port-spec.md`.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::harness::arch::qwen3_5::RopeParameters;
use crate::harness::arch::qwen3_5::rmsnorm::OutputGate;

/// `model_type` for the multimodal wrapper.
pub const MODEL_TYPE: &str = "qwen4_exp";
/// `model_type` of the nested `text_config`.
pub const TEXT_MODEL_TYPE: &str = "qwen4_exp_text";

/// The top-level config. `vision_config` is deliberately not modelled —
/// serde drops it, and #314 owns the tower.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub model_type: String,
    pub text_config: TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
}

impl Config {
    pub fn from_config_json(json: &str) -> Result<Self> {
        let mut cfg: Self = serde_json::from_str(json).context("parse qwen4_exp config.json")?;
        cfg.text_config.fill_layer_types()?;
        cfg.text_config.validate()?;
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    /// Reused verbatim from `qwen3_5` — the nesting and every field in
    /// it are identical, which is what makes `qwen3_5/rope.rs` reusable
    /// unchanged (spec §7).
    pub rope_parameters: RopeParameters,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// 248044 for this checkpoint — the id PLE's segment-aware shifts
    /// fill with, so it is load-bearing beyond stopping generation.
    pub eos_token_id: i64,

    /// Authoritative per-layer dispatch. Derived from
    /// `full_attention_interval` if absent.
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub full_attention_interval: Option<usize>,

    // --- Hyper-connections (§1) ----------------------------------------
    /// Parallel residual streams. The inter-layer residual is
    /// `hidden_size * hc_count` wide.
    pub hc_count: usize,
    /// Bottleneck width of the stream-mixing gate.
    pub hc_lowrank: usize,

    // --- PLE (§2) ------------------------------------------------------
    /// **One-indexed.** Use [`TextConfig::ple_layers`].
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel_size: usize,
    /// Also the short conv's dilation.
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: i64,
    pub make_ngram_vocab_size_divisible_by: usize,
    /// Shards the table is stored in; a flat split, concatenated in
    /// shard-index order.
    pub split_ngram_parts: usize,

    // --- QSA indexer (§3) ----------------------------------------------
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    /// In **tokens**, not blocks.
    pub indexer_budget: usize,
    /// Positions pooled into one scored block.
    pub indexer_compress_ratio: usize,

    // --- Linear attention (§5) -----------------------------------------
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    /// `"float32"` here; see the module note.
    #[serde(default)]
    pub mamba_ssm_dtype: Option<String>,
    /// `"sigmoid"` here, where Qwen3.6 falls back to `hidden_act`.
    #[serde(default)]
    pub output_gate_type: Option<String>,

    // --- MoE (§6) ------------------------------------------------------
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    /// Absent from this `config.json`; upstream's default is `true`.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    // --- MTP (§9, #313) ------------------------------------------------
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
}

impl TextConfig {
    /// Zero-indexed layers carrying a PLE block — `[2]` becomes `[1]`.
    pub fn ple_layers(&self) -> Vec<usize> {
        self.ple_layer_ids.iter().map(|id| id - 1).collect()
    }

    /// `(ngram_size - 1) * heads_per_ngram` — 16. There is no unigram
    /// head.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Dims each n-gram head contributes to the 2560-wide embedding.
    pub fn ngram_head_dim(&self) -> usize {
        self.ple_embed_dim / self.ngram_heads()
    }

    /// The inter-layer residual width, `hidden_size * hc_count`.
    pub fn stream_width(&self) -> usize {
        self.hidden_size * self.hc_count
    }

    /// Blocks the QSA indexer may select.
    pub fn block_topk(&self) -> usize {
        self.indexer_budget / self.indexer_compress_ratio
    }

    /// Which nonlinearity the GatedDeltaNet output gate applies.
    /// Falls back to `hidden_act` the way upstream does — but this
    /// checkpoint states it, and the two disagree.
    pub fn output_gate(&self) -> Result<OutputGate> {
        OutputGate::from_name(self.output_gate_type.as_deref().unwrap_or(&self.hidden_act))
    }

    /// Whether to carry the GatedDeltaNet recurrent state in f32.
    /// `mamba_ssm_dtype` is upstream's explicit choice; absent, we keep
    /// our own default rather than inventing one.
    pub fn ssm_state_is_f32(&self) -> Option<bool> {
        self.mamba_ssm_dtype
            .as_deref()
            .map(|d| d.eq_ignore_ascii_case("float32") || d.eq_ignore_ascii_case("f32"))
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .is_some_and(|t| t == "full_attention")
    }

    fn fill_layer_types(&mut self) -> Result<()> {
        if !self.layer_types.is_empty() {
            return Ok(());
        }
        let interval = self.full_attention_interval.unwrap_or(4);
        ensure!(
            interval > 0,
            "full_attention_interval must be >= 1 to derive layer_types"
        );
        // Every `interval`-th layer counting from one, so with
        // interval 4 the full-attention layers are 3, 7, … 47.
        self.layer_types = (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1).is_multiple_of(interval) {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.layer_types.len() == self.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            self.layer_types.len(),
            self.num_hidden_layers
        );
        ensure!(self.hc_count >= 1, "hc_count must be >= 1");
        ensure!(
            self.ngram_size >= 2,
            "ngram_size must be >= 2; {} leaves no n-gram heads",
            self.ngram_size
        );
        let heads = self.ngram_heads();
        ensure!(
            heads > 0 && self.ple_embed_dim.is_multiple_of(heads),
            "ple_embed_dim ({}) must divide evenly among {heads} n-gram heads",
            self.ple_embed_dim
        );
        for id in &self.ple_layer_ids {
            // One-indexed: 0 is not a layer, and N is the last one.
            ensure!(
                *id >= 1 && *id <= self.num_hidden_layers,
                "ple_layer_ids is one-indexed; {id} is outside 1..={}",
                self.num_hidden_layers
            );
        }
        ensure!(
            self.indexer_compress_ratio >= 1
                && self
                    .indexer_budget
                    .is_multiple_of(self.indexer_compress_ratio),
            "indexer_budget ({}) must be a whole number of blocks of {}",
            self.indexer_budget,
            self.indexer_compress_ratio
        );
        ensure!(
            self.num_experts_per_tok <= self.num_experts,
            "num_experts_per_tok ({}) exceeds num_experts ({})",
            self.num_experts_per_tok,
            self.num_experts
        );
        Ok(())
    }
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_true() -> bool {
    true
}

/// Whether a `config.json` declares this architecture, so the harness
/// can route it here rather than to the `qwen3_5` sibling.
pub fn is_qwen4_exp(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(|m| m == MODEL_TYPE)
        })
        .unwrap_or(false)
}

/// The shipped `config.json`, verbatim apart from the vision block
/// and the 48-entry `layer_types` (spelled out below in
/// `SHIPPED_LAYER_TYPES` so the derivation can be tested against
/// the real one). Values are from revision
/// `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
#[cfg(test)]
pub(crate) const SHIPPED: &str = r#"{
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "image_token_id": 248056,
    "language_model_only": false,
    "model_type": "qwen4_exp",
    "tie_word_embeddings": false,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "vision_config": {"depth": 27, "hidden_size": 1152},
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "heads_per_ngram": 8,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "initializer_range": 0.02,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 640,
        "mtp": {"hybrid": true, "num_hidden_layers": 1, "rope_theta": 10000000},
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "ngram_size": 3,
        "ngram_vocab_size_base": 20000000,
        "num_attention_heads": 24,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "num_hidden_layers": 48,
        "num_key_value_heads": 2,
        "output_gate_type": "sigmoid",
        "output_router_logits": false,
        "pad_token_id": null,
        "partial_rotary_factor": 0.25,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default"
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 640,
        "split_ngram_parts": 128,
        "tie_word_embeddings": false,
        "use_cache": true,
        "vocab_size": 248320
    }
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The real file's `layer_types`, as shipped.
    const SHIPPED_FULL_ATTENTION_LAYERS: [usize; 12] =
        [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];

    fn shipped() -> Config {
        Config::from_config_json(SHIPPED).unwrap()
    }

    #[test]
    fn parses_the_shipped_config() {
        let cfg = shipped();
        assert_eq!(cfg.model_type, MODEL_TYPE);
        assert!(!cfg.tie_word_embeddings);
        let t = &cfg.text_config;
        assert_eq!(t.hidden_size, 2560);
        assert_eq!(t.num_hidden_layers, 48);
        assert_eq!(t.vocab_size, 248_320);
        assert_eq!(t.max_position_embeddings, 262_144);
        assert_eq!(t.eos_token_id, 248_044);
        // Four streams of 2560 — the residual every decoder block sees.
        assert_eq!(t.stream_width(), 10_240);
        assert_eq!(t.hc_lowrank, 320);
        assert!(is_qwen4_exp(SHIPPED));
    }

    /// The single most expensive off-by-one available here: reading
    /// `[2]` as a layer number rather than a one-indexed position puts
    /// PLE on layer 2 instead of layer 1.
    #[test]
    fn ple_layer_ids_are_one_indexed() {
        let t = &shipped().text_config;
        assert_eq!(t.ple_layer_ids, vec![2], "as shipped, one-indexed");
        assert_eq!(t.ple_layers(), vec![1], "zero-indexed, where it runs");
    }

    /// Sixteen heads of 160, and no unigram head — `ngram_size 3` runs
    /// the bigram and trigram orders only.
    #[test]
    fn ngram_geometry_is_sixteen_heads_of_one_sixty() {
        let t = &shipped().text_config;
        assert_eq!(t.ngram_heads(), 16);
        assert_eq!(t.ngram_head_dim(), 160);
        assert_eq!(t.ngram_heads() * t.ngram_head_dim(), t.ple_embed_dim);
        // The conv's dilation is the n-gram size, not its kernel.
        assert_eq!(t.ple_conv_kernel_size, 4);
        assert_eq!(t.ngram_size, 3);
    }

    /// The derivation must reproduce the file's own `layer_types`:
    /// every fourth layer counting from one, so 3, 7, … 47.
    #[test]
    fn derived_layer_types_match_the_shipped_ones() {
        let t = &shipped().text_config;
        let full: Vec<usize> = (0..t.num_hidden_layers)
            .filter(|i| t.is_full_attention(*i))
            .collect();
        assert_eq!(full, SHIPPED_FULL_ATTENTION_LAYERS);
        assert_eq!(full.len(), 12);
        assert_eq!(t.num_hidden_layers - full.len(), 36);
    }

    /// `output_gate_type` overrides `hidden_act`, and the two disagree:
    /// inheriting SiLU here is wrong on all 36 linear-attention layers.
    #[test]
    fn the_output_gate_is_sigmoid_not_the_hidden_act() {
        let t = &shipped().text_config;
        assert_eq!(t.hidden_act, "silu");
        assert_eq!(t.output_gate().unwrap(), OutputGate::Sigmoid);
        assert_eq!(t.ssm_state_is_f32(), Some(true));
    }

    /// The reused rope module is only reusable if these four values are
    /// what its masks were pinned against (spec §7).
    #[test]
    fn rope_parameters_are_the_ones_qwen3_5_already_implements() {
        let t = &shipped().text_config;
        let r = &t.rope_parameters;
        assert_eq!(r.rope_theta, 10_000_000.0);
        assert_eq!(r.partial_rotary_factor, 0.25);
        assert_eq!(r.mrope_section, vec![11, 11, 10]);
        assert!(r.mrope_interleaved);
        // 64 of 256 head dims rotate; 32 frequencies, which is what
        // mrope_section sums to.
        assert_eq!((t.head_dim as f32 * r.partial_rotary_factor) as usize, 64);
        assert_eq!(r.mrope_section.iter().sum::<usize>(), 32);
    }

    #[test]
    fn qsa_geometry_selects_512_blocks_of_four() {
        let t = &shipped().text_config;
        assert_eq!(t.block_topk(), 512);
        assert_eq!(t.indexer_compress_ratio, 4);
        assert_eq!(t.indexer_n_heads, 4);
        assert_eq!(t.indexer_kv_heads, 1);
        assert_eq!(t.indexer_head_dim, 128);
    }

    /// A one-indexed field that reads 0 is a config written against the
    /// wrong convention, not a layer.
    #[test]
    fn rejects_a_zero_ple_layer_id() {
        let bad = SHIPPED.replace("\"ple_layer_ids\": [2]", "\"ple_layer_ids\": [0]");
        let err = Config::from_config_json(&bad).unwrap_err().to_string();
        assert!(err.contains("one-indexed"), "got: {err}");
    }

    #[test]
    fn rejects_layer_types_that_do_not_cover_every_layer() {
        let bad = SHIPPED.replace(
            "\"num_hidden_layers\": 48",
            "\"num_hidden_layers\": 48, \"layer_types\": [\"full_attention\"]",
        );
        let err = Config::from_config_json(&bad).unwrap_err().to_string();
        assert!(err.contains("layer_types"), "got: {err}");
    }

    /// The harness dispatches on `model_type`, and the sibling archs
    /// must not claim this checkpoint or be claimed by it.
    #[test]
    fn architecture_detection_is_exact() {
        assert!(is_qwen4_exp(SHIPPED));
        assert!(!is_qwen4_exp(r#"{"model_type": "qwen3_5"}"#));
        assert!(!is_qwen4_exp(r#"{"model_type": "qwen3_next"}"#));
        // The nested text_config's own model_type is a different string
        // and must not be what we match on.
        assert_ne!(MODEL_TYPE, TEXT_MODEL_TYPE);
        assert!(!is_qwen4_exp(r#"{"model_type": "qwen4_exp_text"}"#));
        assert!(!is_qwen4_exp("not json"));
    }
}
