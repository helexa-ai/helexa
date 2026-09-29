//! The tokenizer a decision checkpoint ships, and its special tokens.

use std::path::Path;

/// The special tokens the sequence layout uses, resolved to ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    /// The option marker's text (`[MASK]`, `<mask>` for mmBERT). Literal
    /// occurrences in caller text are replaced with a space before
    /// tokenizing, so no caller can forge a marker.
    pub mask_token: String,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    /// Padding for batched rows; not part of any sequence.
    pub pad_id: u32,
}

/// Text → token ids, with no special tokens added.
pub trait TextEncoder {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
    fn special(&self) -> &SpecialTokens;
}

/// A checkpoint's `tokenizer/` directory: `tokenizer.json` plus the special
/// token names in `tokenizer_config.json`.
pub struct DecisionTokenizer {
    tok: tokenizers::Tokenizer,
    special: SpecialTokens,
}

impl DecisionTokenizer {
    pub fn from_dir(dir: &Path) -> anyhow::Result<Self> {
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{}: {e}", dir.join("tokenizer.json").display()))?;
        // The Python tokenizer call the reference makes pads and truncates
        // nothing unless asked; a tokenizer.json that carries either setting
        // would otherwise apply it here.
        tok.with_truncation(None)
            .map_err(|e| anyhow::anyhow!("disabling truncation: {e}"))?;
        tok.with_padding(None);
        let cfg_path = dir.join("tokenizer_config.json");
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", cfg_path.display()))?;
        let name = |key: &str| -> anyhow::Result<String> {
            cfg.get(key)
                .and_then(|v| {
                    v.as_str()
                        .or_else(|| v.get("content").and_then(|c| c.as_str()))
                })
                .map(str::to_owned)
                .ok_or_else(|| anyhow::anyhow!("{}: no {key}", cfg_path.display()))
        };
        let id = |token: &str| -> anyhow::Result<u32> {
            tok.token_to_id(token)
                .ok_or_else(|| anyhow::anyhow!("special token {token:?} is not in the vocabulary"))
        };
        let mask_token = name("mask_token")?;
        let special = SpecialTokens {
            cls_id: id(&name("cls_token")?)?,
            sep_id: id(&name("sep_token")?)?,
            mask_id: id(&mask_token)?,
            pad_id: id(&name("pad_token")?)?,
            mask_token,
        };
        Ok(Self { tok, special })
    }
}

impl TextEncoder for DecisionTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }

    fn special(&self) -> &SpecialTokens {
        &self.special
    }
}
