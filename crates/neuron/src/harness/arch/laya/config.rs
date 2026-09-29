//! The two config files a Laya checkpoint ships, and what they mean.
//!
//! A checkpoint directory holds `rl_agent_config.json` (the decision
//! model: head depth, token budgets, calibration temperatures) and
//! `encoder/config.json` (the ModernBERT backbone, written by
//! transformers 5). There is no top-level `config.json`.
//!
//! The encoder config is parsed here rather than through
//! `candle_transformers::models::modernbert::Config` because
//! transformers 5 moved the rotary bases into a nested
//! `rope_parameters.{full_attention,sliding_attention}.rope_theta`
//! block, which that struct cannot read; the older flat
//! `global_rope_theta` / `local_rope_theta` fields are still honoured
//! when a checkpoint carries them. mmBERT (the multilingual
//! checkpoint's backbone) is the case that matters: both of its bases
//! are 160,000, where the flat-field defaults would silently give the
//! sliding layers 10,000.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// `rl_agent_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct LayaConfig {
    /// The backbone the head was trained on, e.g.
    /// `answerdotai/ModernBERT-large`. Informational: the weights come
    /// from the checkpoint, the shape from `encoder/config.json`.
    #[serde(default)]
    pub encoder: Option<String>,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    /// One act head output per entry, plus one for "answer". The act
    /// head's width is `act_costs.len() + 1`.
    #[serde(default)]
    pub act_costs: BTreeMap<String, f64>,
    /// Per-question-type temperatures, indexed choice / score / noul.
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f64>,
    /// Temperatures fitted per `"<type>:<cardinality bucket>"`.
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f64>,
    /// Autocast dtype the reference runs on GPU (`"bf16"` or `"fp16"`).
    #[serde(default)]
    pub amp_dtype: Option<String>,
}

fn default_head_layers() -> usize {
    2
}
fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temperature() -> Vec<f64> {
    vec![1.0, 1.0, 1.0]
}

impl LayaConfig {
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

/// Whether an attention layer sees the whole sequence or a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Global,
    Sliding,
}

/// The ModernBERT backbone shape, resolved from `encoder/config.json`.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub pad_token_id: u32,
    /// Full sliding-window width. A token attends to keys at distance
    /// `<= local_attention / 2` on either side.
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    pub layer_kinds: Vec<LayerKind>,
}

#[derive(Deserialize)]
struct RawEncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    norm_eps: Option<f64>,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
    pad_token_id: u32,
    #[serde(default = "default_global_every")]
    global_attn_every_n_layers: usize,
    local_attention: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    hidden_activation: Option<String>,
}

fn default_global_every() -> usize {
    3
}

/// transformers' ModernBERT defaults for checkpoints that predate
/// `rope_parameters` and also omit the flat fields.
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 160_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;

impl EncoderConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        let raw: RawEncoderConfig =
            serde_json::from_str(text).context("parse ModernBERT encoder config")?;

        // Variants this port does not implement are refused, not
        // approximated: each would load without error and answer wrong.
        ensure!(
            !raw.norm_bias && !raw.attention_bias && !raw.mlp_bias,
            "ModernBERT with norm/attention/mlp biases is not supported"
        );
        if let Some(act) = raw.hidden_activation.as_deref() {
            ensure!(
                act == "gelu",
                "ModernBERT hidden_activation {act:?} is not supported (only exact \"gelu\")"
            );
        }
        ensure!(
            raw.hidden_size.is_multiple_of(raw.num_attention_heads),
            "hidden_size {} is not divisible by num_attention_heads {}",
            raw.hidden_size,
            raw.num_attention_heads
        );

        let (global_rope_theta, local_rope_theta) = rope_thetas(&raw)?;

        let layer_kinds = match &raw.layer_types {
            Some(types) => {
                ensure!(
                    types.len() == raw.num_hidden_layers,
                    "layer_types has {} entries for {} layers",
                    types.len(),
                    raw.num_hidden_layers
                );
                types
                    .iter()
                    .map(|t| match t.as_str() {
                        "full_attention" => Ok(LayerKind::Global),
                        "sliding_attention" => Ok(LayerKind::Sliding),
                        other => bail!("unknown ModernBERT layer type {other:?}"),
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            None => (0..raw.num_hidden_layers)
                .map(|i| {
                    if i % raw.global_attn_every_n_layers == 0 {
                        LayerKind::Global
                    } else {
                        LayerKind::Sliding
                    }
                })
                .collect(),
        };

        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            intermediate_size: raw.intermediate_size,
            max_position_embeddings: raw.max_position_embeddings,
            norm_eps: raw.norm_eps.or(raw.layer_norm_eps).unwrap_or(1e-5),
            pad_token_id: raw.pad_token_id,
            local_attention: raw.local_attention,
            global_rope_theta,
            local_rope_theta,
            layer_kinds,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Resolve the two rotary bases. `rope_parameters` (transformers 5)
/// wins over the flat fields; within it, a per-layer-type block wins
/// over a shared top-level `rope_theta`.
fn rope_thetas(raw: &RawEncoderConfig) -> Result<(f64, f64)> {
    let mut global = raw.global_rope_theta.unwrap_or(DEFAULT_GLOBAL_ROPE_THETA);
    let mut local = raw.local_rope_theta.unwrap_or(DEFAULT_LOCAL_ROPE_THETA);
    if let Some(rope) = &raw.rope_parameters {
        let shared = rope.get("rope_theta").and_then(|v| v.as_f64());
        for (key, slot) in [
            ("full_attention", &mut global),
            ("sliding_attention", &mut local),
        ] {
            let block = rope.get(key);
            if let Some(rope_type) = block
                .and_then(|b| b.get("rope_type"))
                .and_then(|v| v.as_str())
            {
                ensure!(
                    rope_type == "default",
                    "rope_type {rope_type:?} for {key} is not supported"
                );
            }
            let theta = block
                .and_then(|b| b.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .or(shared);
            if let Some(theta) = theta {
                *slot = theta;
            }
        }
    }
    Ok((global, local))
}

/// Both configs of one checkpoint directory.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub laya: LayaConfig,
    pub encoder: EncoderConfig,
}

impl CheckpointConfig {
    /// Read `rl_agent_config.json` and `encoder/config.json` from a
    /// checkpoint directory (the repo root, or one of its subfolders).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let laya_path = dir.join("rl_agent_config.json");
        let laya: LayaConfig = serde_json::from_str(
            &std::fs::read_to_string(&laya_path)
                .with_context(|| format!("read {}", laya_path.display()))?,
        )
        .with_context(|| format!("parse {}", laya_path.display()))?;
        let enc_path = dir.join("encoder").join("config.json");
        let encoder = EncoderConfig::from_json(
            &std::fs::read_to_string(&enc_path)
                .with_context(|| format!("read {}", enc_path.display()))?,
        )
        .with_context(|| format!("parse {}", enc_path.display()))?;
        ensure!(
            laya.temperature.len() == 3,
            "rl_agent_config temperature must have 3 entries (choice, score, noul), got {}",
            laya.temperature.len()
        );
        Ok(Self { laya, encoder })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "intermediate_size": 96,
            "max_position_embeddings": 512, "norm_eps": 1e-5, "pad_token_id": 0,
            "global_attn_every_n_layers": 3, "local_attention": 16,
        })
    }

    #[test]
    fn nested_rope_parameters_override_flat_defaults() {
        let mut v = base();
        v["rope_parameters"] = serde_json::json!({
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"},
        });
        let c = EncoderConfig::from_json(&v.to_string()).unwrap();
        assert_eq!(c.global_rope_theta, 160_000.0);
        // The mmBERT case: the flat default would have been 10,000.
        assert_eq!(c.local_rope_theta, 160_000.0);
    }

    #[test]
    fn flat_fields_are_read_without_rope_parameters() {
        let mut v = base();
        v["global_rope_theta"] = serde_json::json!(80000.0);
        v["local_rope_theta"] = serde_json::json!(5000.0);
        let c = EncoderConfig::from_json(&v.to_string()).unwrap();
        assert_eq!((c.global_rope_theta, c.local_rope_theta), (80000.0, 5000.0));
    }

    #[test]
    fn layer_kinds_follow_every_n_when_layer_types_absent() {
        let c = EncoderConfig::from_json(&base().to_string()).unwrap();
        assert_eq!(
            c.layer_kinds,
            vec![
                LayerKind::Global,
                LayerKind::Sliding,
                LayerKind::Sliding,
                LayerKind::Global
            ]
        );
    }

    #[test]
    fn explicit_layer_types_win() {
        let mut v = base();
        v["layer_types"] = serde_json::json!([
            "sliding_attention",
            "full_attention",
            "full_attention",
            "sliding_attention"
        ]);
        let c = EncoderConfig::from_json(&v.to_string()).unwrap();
        assert_eq!(c.layer_kinds[0], LayerKind::Sliding);
        assert_eq!(c.layer_kinds[1], LayerKind::Global);
    }

    #[test]
    fn unsupported_variants_are_refused() {
        let mut v = base();
        v["attention_bias"] = serde_json::json!(true);
        assert!(EncoderConfig::from_json(&v.to_string()).is_err());
        let mut v = base();
        v["hidden_activation"] = serde_json::json!("gelu_pytorch_tanh");
        assert!(EncoderConfig::from_json(&v.to_string()).is_err());
        let mut v = base();
        v["rope_parameters"] =
            serde_json::json!({"full_attention": {"rope_type": "yarn", "rope_theta": 1.0}});
        assert!(EncoderConfig::from_json(&v.to_string()).is_err());
    }
}
