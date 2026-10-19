//! Language Model pipeline for ACE-Step audio code generation.
//!
//! Implements two-phase autoregressive generation using a fine-tuned Qwen3 model,
//! as upstream's `generate_with_stop_condition` does (#384):
//! 1. **CoT phase**: the plan (BPM, caption, duration, key, language, time
//!    signature) under [`MetaFsm`]'s constrained decoding, without CFG, stopped
//!    where `</think>` would begin.
//! 2. **Codes phase**: a fresh prompt carrying the plan as upstream re-dumps it
//!    ([`cot::format_plan`]), then audio code tokens `<|audio_code_N|>` under
//!    classifier-free guidance against an unconditional `NO USER INPUT` prompt,
//!    with EOS forced at exactly five codes per second of audio.
//!
//! The generated audio codes are converted to latents via
//! `ResidualFSQ::get_output_from_indices → AudioTokenDetokenizer` and fed to the
//! DiT as `precomputed_lm_hints_25hz`.
//!
//! Reference: `acestep/llm_inference.py`, `acestep/constrained_logits_processor.py`

use std::collections::BTreeMap;

use super::cot;
use super::fsm::{MetaFsm, Vocab};
use super::qwen3;
use candle_core::{Device, Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

/// Default system instruction for the LM.
pub const DEFAULT_LM_INSTRUCTION: &str =
    "Generate audio semantic tokens based on the given conditions:";

/// Number of audio codes generated per second of audio (5 Hz).
pub const CODES_PER_SECOND: f64 = 5.0;

/// Maximum valid audio code value (codebook size = 64000).
pub const MAX_AUDIO_CODE: u32 = 63999;

/// Generation parameters for the LM pipeline.
pub struct LmConfig {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    /// Maximum tokens for CoT (metadata) generation.
    pub max_cot_tokens: usize,
    /// Maximum tokens for audio code generation.
    pub max_code_tokens: usize,
    pub repetition_penalty: f64,
    pub seed: u64,
    /// Classifier-free guidance in the codes phase (upstream's default 2.0;
    /// 1.0 turns it off). The CoT phase never uses CFG.
    pub cfg_scale: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            temperature: 0.85,
            top_p: None,
            top_k: None,
            max_cot_tokens: 512,
            max_code_tokens: 3072,
            repetition_penalty: 1.0,
            seed: 42,
            cfg_scale: 2.0,
        }
    }
}

/// Result of LM generation.
pub struct LmOutput {
    /// Parsed metadata from CoT (bpm, caption, duration, genres, keyscale, language, timesignature).
    pub metadata: BTreeMap<String, String>,
    /// Raw audio code values in `[0, 63999]`.
    pub audio_codes: Vec<i64>,
    /// Full generated text (CoT + codes) for debugging.
    pub raw_text: String,
}

/// Token IDs discovered from the tokenizer at runtime.
pub struct TokenIds {
    pub eos: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub audio_code_start: u32,
    pub audio_code_end: u32,
}

impl TokenIds {
    /// Discover special token IDs by encoding known strings.
    ///
    /// `encode_fn` should encode a string without special tokens and return
    /// the token IDs. It must produce exactly one token for each special string.
    pub fn discover(encode_fn: impl Fn(&str) -> Result<Vec<u32>>) -> Result<Self> {
        let get_single = |s: &str| -> Result<u32> {
            let ids = encode_fn(s)?;
            if ids.len() != 1 {
                candle_core::bail!("expected single token for {s:?}, got {ids:?}");
            }
            Ok(ids[0])
        };

        let eos = get_single("<|im_end|>")?;
        let think_start = get_single("<think>")?;
        let think_end = get_single("</think>")?;
        let audio_code_start = get_single("<|audio_code_0|>")?;
        let audio_code_end = audio_code_start + MAX_AUDIO_CODE;

        Ok(Self {
            eos,
            think_start,
            think_end,
            audio_code_start,
            audio_code_end,
        })
    }
}

/// The codes phase's unconditional user turn: upstream's default negative
/// prompt, which training used for CFG dropout.
pub const NEGATIVE_PROMPT: &str = "NO USER INPUT";

/// Upstream's `DURATION_MIN..=DURATION_MAX`, which the code budget is
/// clamped to.
const DURATION_RANGE: (f64, f64) = (10.0, 600.0);

/// A tokenizer reached through the caller's closures.
struct FnVocab<'a> {
    encode: &'a dyn Fn(&str) -> Result<Vec<u32>>,
    decode: &'a dyn Fn(&[u32]) -> Result<String>,
}

impl Vocab for FnVocab<'_> {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        (self.encode)(text)
    }
    fn decode(&self, token: u32) -> Result<String> {
        (self.decode)(&[token])
    }
}

/// One codes-phase step's logits: CFG over the tokens that may appear (the
/// audio codes and EOS), everything else `-inf`, and EOS blocked until
/// `target` codes have been generated and forced from then on.
fn guided_code_logits(
    cond: &[f32],
    uncond: Option<&[f32]>,
    cfg_scale: f32,
    ids: &TokenIds,
    count: usize,
    target: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; cond.len()];
    let guide = |i: usize| match uncond {
        Some(u) => u[i] + cfg_scale * (cond[i] - u[i]),
        None => cond[i],
    };
    let eos = ids.eos as usize;
    if count >= target {
        if eos < out.len() {
            out[eos] = guide(eos);
        }
        return out;
    }
    let hi = (ids.audio_code_end as usize).min(cond.len().saturating_sub(1));
    for (i, v) in out
        .iter_mut()
        .enumerate()
        .take(hi + 1)
        .skip(ids.audio_code_start as usize)
    {
        *v = guide(i);
    }
    out
}

/// LM pipeline wrapping a Qwen3 causal LM for audio code generation.
pub struct LmPipeline {
    model: qwen3::ModelForCausalLM,
    token_ids: TokenIds,
    device: Device,
}

impl LmPipeline {
    /// Create from a pre-loaded model and discovered token IDs.
    pub fn new(model: qwen3::ModelForCausalLM, token_ids: TokenIds, device: Device) -> Self {
        Self {
            model,
            token_ids,
            device,
        }
    }

    /// The chat frame every planner prompt shares, with the assistant turn
    /// left open.
    fn chat(user: &str) -> String {
        let system = format!("# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n");
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    fn user_turn(caption: &str, lyrics: &str) -> String {
        let lyrics = if lyrics.is_empty() {
            "[Instrumental]"
        } else {
            lyrics
        };
        format!("# Caption\n{caption}\n\n# Lyric\n{lyrics}\n")
    }

    /// The CoT-phase prompt (upstream's `build_formatted_prompt`).
    pub fn format_prompt(caption: &str, lyrics: &str) -> String {
        Self::chat(&Self::user_turn(caption, lyrics))
    }

    /// The codes-phase prompt: the CoT prompt followed by the plan block
    /// (`<think>…</think>`) and the blank line training put before the
    /// first code (upstream's `build_formatted_prompt_with_cot`).
    pub fn format_codes_prompt(caption: &str, lyrics: &str, plan_block: &str) -> String {
        format!("{}{plan_block}\n\n", Self::format_prompt(caption, lyrics))
    }

    /// The codes phase's unconditional prompt: the negative prompt as the
    /// whole user turn, and empty reasoning.
    pub fn format_unconditional_codes_prompt() -> String {
        format!("{}<think>\n\n</think>\n\n", Self::chat(NEGATIVE_PROMPT))
    }

    /// Feed `tokens` from position `offset` and return the last position's
    /// logits on the CPU, with non-finite values made `-inf`.
    fn step(
        model: &mut qwen3::ModelForCausalLM,
        device: &Device,
        tokens: &[u32],
        offset: usize,
    ) -> Result<Vec<f32>> {
        let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset)?;
        let mut data = logits
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        for v in data.iter_mut() {
            if !v.is_finite() {
                *v = f32::NEG_INFINITY;
            }
        }
        Ok(data)
    }

    /// Run two-phase generation: CoT (metadata) + audio codes.
    ///
    /// `decode_fn` decodes a token sequence (multi-byte characters can span
    /// tokens, so text is only ever decoded whole); `encode_fn` tokenizes
    /// without adding special tokens.
    pub fn generate(
        &mut self,
        caption: &str,
        lyrics: &str,
        target_duration_secs: f64,
        config: &LmConfig,
        decode_fn: &dyn Fn(&[u32]) -> Result<String>,
        encode_fn: &dyn Fn(&str) -> Result<Vec<u32>>,
    ) -> Result<LmOutput> {
        let vocab = FnVocab {
            encode: encode_fn,
            decode: decode_fn,
        };
        let prompt_tokens = encode_fn(&Self::format_prompt(caption, lyrics))?;
        let cot_tokens = self.generate_plan(&prompt_tokens, config, &vocab)?;
        let cot_text = decode_fn(&cot_tokens)?;
        let plan = cot::parse_plan(&cot_text);
        let plan_block = cot::format_plan(&plan);
        let metadata = plan.iter().map(|(k, v)| (k.clone(), v.as_text())).collect();

        let codes_prompt = Self::format_codes_prompt(caption, lyrics, &plan_block);
        let codes = self.generate_codes(&codes_prompt, target_duration_secs, config, &vocab)?;

        let audio_codes: Vec<i64> = codes
            .iter()
            .filter(|&&t| {
                t >= self.token_ids.audio_code_start && t <= self.token_ids.audio_code_end
            })
            .map(|&t| (t - self.token_ids.audio_code_start) as i64)
            .collect();
        let codes_text = decode_fn(&codes)?;
        Ok(LmOutput {
            metadata,
            audio_codes,
            raw_text: format!("{cot_text}\n{plan_block}\n\n{codes_text}"),
        })
    }

    /// The CoT phase: the plan, under the metadata grammar, no CFG.
    /// Returns the generated tokens, ending in `<|im_end|>`.
    fn generate_plan(
        &mut self,
        prompt_tokens: &[u32],
        config: &LmConfig,
        vocab: &FnVocab,
    ) -> Result<Vec<u32>> {
        let mut fsm = MetaFsm::new(
            vocab,
            self.token_ids.eos,
            (
                self.token_ids.audio_code_start,
                self.token_ids.audio_code_end,
            ),
        )?;
        let mut sampler = self.make_logits_processor(config);
        self.model.clear_kv_cache();
        let mut logits = Self::step(&mut self.model, &self.device, prompt_tokens, 0)?;
        let mut tokens = Vec::new();
        for pos in (prompt_tokens.len()..).take(config.max_cot_tokens) {
            fsm.constrain(vocab, &mut logits)?;
            let n = logits.len();
            let token = sampler.sample(&Tensor::from_vec(logits, n, &Device::Cpu)?)?;
            fsm.update(vocab, token)?;
            tokens.push(token);
            if token == self.token_ids.eos || fsm.is_completed() {
                break;
            }
            logits = Self::step(&mut self.model, &self.device, &[token], pos)?;
        }
        Ok(tokens)
    }

    /// The codes phase: audio codes under CFG, the same token fed to the
    /// conditional and unconditional sequences, which keep separate caches.
    fn generate_codes(
        &mut self,
        codes_prompt: &str,
        target_duration_secs: f64,
        config: &LmConfig,
        vocab: &FnVocab,
    ) -> Result<Vec<u32>> {
        let duration = target_duration_secs.clamp(DURATION_RANGE.0, DURATION_RANGE.1);
        let target = (duration * CODES_PER_SECOND) as usize;
        let max_new = (target + 10).min(config.max_code_tokens.max(target + 1));
        let cfg = config.cfg_scale as f32;

        let cond_ids = (vocab.encode)(codes_prompt)?;
        self.model.clear_kv_cache();
        // The weights are shared; only the KV cache is the clone's own.
        let mut uncond_model = (cfg > 1.0).then(|| self.model.clone());
        let mut cond = Self::step(&mut self.model, &self.device, &cond_ids, 0)?;
        let mut uncond_len = 0;
        let mut uncond = match uncond_model.as_mut() {
            Some(m) => {
                let ids = (vocab.encode)(&Self::format_unconditional_codes_prompt())?;
                uncond_len = ids.len();
                Some(Self::step(m, &self.device, &ids, 0)?)
            }
            None => None,
        };

        let mut sampler = self.make_logits_processor(config);
        let mut out = Vec::with_capacity(target + 1);
        for (i, cond_pos) in (cond_ids.len()..).take(max_new).enumerate() {
            let guided = guided_code_logits(
                &cond,
                uncond.as_deref(),
                cfg,
                &self.token_ids,
                out.len(),
                target,
            );
            let n = guided.len();
            let token = sampler.sample(&Tensor::from_vec(guided, n, &Device::Cpu)?)?;
            out.push(token);
            if token == self.token_ids.eos {
                break;
            }
            cond = Self::step(&mut self.model, &self.device, &[token], cond_pos)?;
            if let Some(m) = uncond_model.as_mut() {
                uncond = Some(Self::step(m, &self.device, &[token], uncond_len + i)?);
            }
        }
        Ok(out)
    }

    /// Read the plan out of CoT text (upstream's `parse_lm_output`): the
    /// caption unfolded to one line, every value as text.
    pub fn parse_metadata(text: &str) -> BTreeMap<String, String> {
        cot::parse_plan(text)
            .into_iter()
            .map(|(k, v)| (k, v.as_text()))
            .collect()
    }

    fn make_logits_processor(&self, config: &LmConfig) -> LogitsProcessor {
        let sampling = if config.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (config.top_k, config.top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: config.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: config.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: config.temperature,
                },
                (None, None) => Sampling::All {
                    temperature: config.temperature,
                },
            }
        };
        LogitsProcessor::from_sampling(config.seed, sampling)
    }
}

/// Parse audio code values from a text string containing `<|audio_code_N|>` tokens.
///
/// Returns code values in `[0, 63999]`, clamped to valid range.
pub fn parse_audio_codes_from_text(text: &str) -> Vec<i64> {
    let mut codes = Vec::new();
    let mut remaining = text;
    while let Some(start) = remaining.find("<|audio_code_") {
        let after_prefix = &remaining[start + 13..]; // skip "<|audio_code_"
        if let Some(end) = after_prefix.find("|>") {
            let num_str = &after_prefix[..end];
            if let Ok(val) = num_str.parse::<i64>() {
                codes.push(val.clamp(0, MAX_AUDIO_CODE as i64));
            }
            remaining = &after_prefix[end + 2..];
        } else {
            break;
        }
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metadata() {
        let text = "<think>\nbpm: 120\ncaption: A calm piano melody\nduration: 30\ngenres: Classical\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n</think>";
        let meta = LmPipeline::parse_metadata(text);
        assert_eq!(meta["bpm"], "120");
        assert_eq!(meta["caption"], "A calm piano melody");
        assert_eq!(meta["duration"], "30");
        assert_eq!(meta["keyscale"], "C major");
        assert_eq!(meta["language"], "en");
        assert_eq!(meta["timesignature"], "4");
    }

    #[test]
    fn test_parse_metadata_multiline_caption() {
        let text = "<think>\nbpm: 90\ncaption: A long description\n  that spans multiple lines\nduration: 60\n</think>";
        let meta = LmPipeline::parse_metadata(text);
        assert_eq!(
            meta["caption"],
            "A long description that spans multiple lines"
        );
        assert_eq!(meta["bpm"], "90");
        assert_eq!(meta["duration"], "60");
    }

    #[test]
    fn test_parse_audio_codes_from_text() {
        let text = "<|audio_code_100|><|audio_code_200|><|audio_code_63999|>";
        let codes = parse_audio_codes_from_text(text);
        assert_eq!(codes, vec![100, 200, 63999]);
    }

    #[test]
    fn test_parse_audio_codes_clamp() {
        let text = "<|audio_code_99999|><|audio_code_0|>";
        let codes = parse_audio_codes_from_text(text);
        assert_eq!(codes, vec![63999, 0]);
    }

    #[test]
    fn test_parse_audio_codes_empty() {
        assert!(parse_audio_codes_from_text("no codes here").is_empty());
        assert!(parse_audio_codes_from_text("").is_empty());
    }

    fn fixture_planner() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/harness/testdata/acestep/vocal_en_30s_plan__cpu_f32.json");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        meta
    }

    /// Both phases' prompts, byte for byte as upstream built them.
    #[test]
    fn prompts_match_upstream() {
        let f = fixture_planner();
        let caption = f["case"]["caption"].as_str().unwrap();
        let lyrics = f["case"]["lyrics"].as_str().unwrap();
        let planner = &f["planner"];
        assert_eq!(
            LmPipeline::format_prompt(caption, lyrics),
            planner[0]["formatted_prompt"].as_str().unwrap()
        );
        let plan = cot::parse_plan(planner[0]["output_text"].as_str().unwrap());
        assert_eq!(
            LmPipeline::format_codes_prompt(caption, lyrics, &cot::format_plan(&plan)),
            planner[1]["formatted_prompt"].as_str().unwrap()
        );
    }

    #[test]
    fn test_format_prompt_no_lyrics() {
        let prompt = LmPipeline::format_prompt("calm piano", "");
        assert!(prompt.contains("[Instrumental]"));
    }

    #[test]
    fn the_unconditional_codes_prompt_is_the_negative_prompt_with_empty_reasoning() {
        assert_eq!(
            LmPipeline::format_unconditional_codes_prompt(),
            "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\nNO USER INPUT<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn guidance_covers_codes_only_and_eos_lands_on_target() {
        let ids = TokenIds {
            eos: 10,
            think_start: 20,
            think_end: 21,
            audio_code_start: 100,
            audio_code_end: 199,
        };
        let cond = vec![1.0f32; 200];
        let mut uncond = vec![0.5f32; 200];
        uncond[150] = 2.0;

        let g = guided_code_logits(&cond, Some(&uncond), 2.0, &ids, 0, 3);
        assert_eq!(g[0], f32::NEG_INFINITY, "text is never a code");
        assert_eq!(g[10], f32::NEG_INFINITY, "EOS is blocked before the target");
        assert_eq!(g[100], 0.5 + 2.0 * (1.0 - 0.5));
        assert_eq!(g[150], 2.0 + 2.0 * (1.0 - 2.0));

        let g = guided_code_logits(&cond, Some(&uncond), 2.0, &ids, 3, 3);
        assert!(
            g.iter()
                .enumerate()
                .all(|(i, v)| (i == 10) == v.is_finite()),
            "only EOS at the target"
        );

        let g = guided_code_logits(&cond, None, 2.0, &ids, 0, 3);
        assert_eq!(
            g[100], 1.0,
            "without an unconditional pass, the conditional logits"
        );
    }
}
