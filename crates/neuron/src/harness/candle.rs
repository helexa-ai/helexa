//! Candle harness — in-process inference using huggingface/candle.
//!
//! This is the sole `Harness` implementation. Inference runs inside
//! the neuron process; there is no external subprocess.
//!
//! - Stage 2 wired GGUF (Qwen3 only) load/unload via `quantized_qwen3`.
//! - Stage 3 (this) adds `chat_completion` — a non-streaming OpenAI
//!   compatible chat completion routed to the loaded model's forward
//!   pass on a per-model serialised generation loop.

use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Weights;
use cortex_core::harness::{Harness, HarnessHealth, ModelInfo, ModelSpec};
use cortex_core::openai::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    MessageContent, Usage,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, RwLock};

/// In-process candle harness. Owns the loaded model registry.
pub struct CandleHarness {
    models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    hf_cache: Option<PathBuf>,
    bind_url: String,
}

/// A loaded model with its tokenizer, device placement, and architecture-
/// specific weights. The `arch` is `Arc<Mutex<>>` so the lock guard can be
/// moved into `spawn_blocking` for synchronous candle forward passes.
pub struct LoadedModel {
    pub model_id: String,
    pub arch: Arc<Mutex<ModelArch>>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub quant: Option<String>,
    pub devices: Vec<u32>,
}

/// Architecture-specific weights. Stage 3 still supports only Qwen3
/// quantized; Stage 8 broadens this to additional families and
/// non-quantized variants.
pub enum ModelArch {
    Qwen3Quantized(QuantizedQwen3Weights),
}

impl CandleHarness {
    pub fn new(bind_url: String, hf_cache: Option<PathBuf>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            hf_cache,
            bind_url,
        }
    }

    /// Pick a candle `Device` for the requested indices. Without the
    /// `cuda` feature, or if CUDA initialisation fails, falls back to CPU.
    fn pick_device(devices: &[u32]) -> Result<Device> {
        let _idx = devices.first().copied().unwrap_or(0) as usize;
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(_idx) {
                Ok(d) => return Ok(d),
                Err(e) => tracing::warn!(
                    device = _idx,
                    error = %e,
                    "CUDA device unavailable, falling back to CPU"
                ),
            }
        }
        Ok(Device::Cpu)
    }

    /// Resolve a model spec to local GGUF and tokenizer file paths via
    /// hf-hub. Downloads on first use; subsequent calls are cached.
    async fn resolve_files(&self, spec: &ModelSpec) -> Result<(PathBuf, PathBuf)> {
        let mut builder = hf_hub::api::tokio::ApiBuilder::new();
        if let Some(cache) = &self.hf_cache {
            builder = builder.with_cache_dir(cache.clone());
        }
        let api = builder.build().context("build hf-hub API")?;
        let repo = api.model(spec.model_id.clone());

        let info = repo
            .info()
            .await
            .with_context(|| format!("fetch HF repo info for {}", spec.model_id))?;

        let quant = spec.quant.as_deref().unwrap_or("");
        let quant_lc = quant.to_lowercase();
        let gguf_filename = info
            .siblings
            .iter()
            .map(|s| s.rfilename.as_str())
            .filter(|name| name.to_lowercase().ends_with(".gguf"))
            .find(|name| quant_lc.is_empty() || name.to_lowercase().contains(&quant_lc))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no GGUF file matching quant {:?} in repo {}",
                    spec.quant,
                    spec.model_id
                )
            })?
            .to_string();

        tracing::info!(
            model = %spec.model_id,
            file = %gguf_filename,
            "resolving GGUF (may be cached)"
        );
        let gguf_path = repo
            .get(&gguf_filename)
            .await
            .with_context(|| format!("fetch GGUF {gguf_filename}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .context("fetch tokenizer.json")?;
        Ok((gguf_path, tokenizer_path))
    }

    /// Run a non-streaming chat completion against a loaded model.
    ///
    /// Returns a typed `InferenceError` when the model isn't loaded so the
    /// handler can map to an appropriate HTTP status without string-matching.
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, InferenceError> {
        let loaded = {
            let models = self.models.read().await;
            models.get(&request.model).cloned()
        };
        let loaded = loaded.ok_or_else(|| InferenceError::ModelNotLoaded(request.model.clone()))?;

        let prompt = format_qwen3_prompt(&request.messages);

        let encoding = loaded
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p;
        let max_new = request.max_tokens.unwrap_or(512) as usize;
        let seed = unix_subsec_nanos();

        let eos_id = loaded
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| loaded.tokenizer.token_to_id("<|endoftext|>"));

        let arch_arc = Arc::clone(&loaded.arch);
        let device = loaded.device.clone();
        let model_id = request.model.clone();

        let (generated_ids, finish_reason) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<u32>, String)> {
                let mut guard = arch_arc.blocking_lock();
                run_inference(
                    &mut guard,
                    &device,
                    &prompt_tokens,
                    max_new,
                    temperature,
                    top_p,
                    seed,
                    eos_id,
                )
            })
            .await
            .map_err(|e| InferenceError::Other(anyhow::anyhow!("inference task panicked: {e}")))?
            .map_err(InferenceError::Other)?;

        let completion_text = loaded
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| InferenceError::Other(anyhow::anyhow!("detokenize: {e}")))?;

        let usage = Usage {
            prompt_tokens: prompt_len as u64,
            completion_tokens: generated_ids.len() as u64,
            total_tokens: (prompt_len + generated_ids.len()) as u64,
        };

        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{:x}", unix_subsec_nanos()),
            object: "chat.completion".into(),
            created: unix_now_secs(),
            model: model_id,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: MessageContent::Text(completion_text),
                    extra: serde_json::Value::Object(Default::default()),
                },
                finish_reason: Some(finish_reason),
                extra: serde_json::Value::Object(Default::default()),
            }],
            usage: Some(usage),
            extra: serde_json::Value::Object(Default::default()),
        })
    }
}

#[async_trait]
impl Harness for CandleHarness {
    fn name(&self) -> &str {
        "candle"
    }

    async fn health(&self) -> HarnessHealth {
        HarnessHealth {
            name: "candle".into(),
            running: true,
            uptime_secs: None,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let models = self.models.read().await;
        Ok(models
            .values()
            .map(|m| ModelInfo {
                id: m.model_id.clone(),
                harness: "candle".into(),
                status: "loaded".into(),
                devices: m.devices.clone(),
                vram_used_mb: None,
            })
            .collect())
    }

    async fn load_model(&self, spec: &ModelSpec) -> Result<()> {
        if spec.harness != "candle" {
            anyhow::bail!("expected harness=candle, got harness={}", spec.harness);
        }

        {
            let models = self.models.read().await;
            if models.contains_key(&spec.model_id) {
                anyhow::bail!("model '{}' already loaded", spec.model_id);
            }
        }

        let devices = spec.devices.clone().unwrap_or_else(|| vec![0]);
        let device = Self::pick_device(&devices)?;

        let (gguf_path, tokenizer_path) = self.resolve_files(spec).await?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        // File I/O + GGUF parsing + tensor materialisation are CPU-bound,
        // so run them on a blocking task to avoid stalling the runtime.
        let device_for_load = device.clone();
        let gguf_path_for_load = gguf_path.clone();
        let model_id_for_log = spec.model_id.clone();
        let arch = tokio::task::spawn_blocking(move || -> Result<ModelArch> {
            tracing::info!(model = %model_id_for_log, path = ?gguf_path_for_load, "loading GGUF");
            let mut file = std::fs::File::open(&gguf_path_for_load).context("open GGUF file")?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|e| anyhow::anyhow!("parse GGUF: {e}"))?;

            let architecture = content
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_default();
            tracing::info!(architecture = %architecture, "GGUF architecture");

            match architecture.as_str() {
                "qwen3" => {
                    let weights =
                        QuantizedQwen3Weights::from_gguf(content, &mut file, &device_for_load)
                            .map_err(|e| anyhow::anyhow!("from_gguf qwen3: {e}"))?;
                    Ok(ModelArch::Qwen3Quantized(weights))
                }
                other => anyhow::bail!(
                    "unsupported GGUF architecture '{other}'; Stage 3 only supports qwen3"
                ),
            }
        })
        .await
        .context("blocking load task panicked")??;

        let loaded = Arc::new(LoadedModel {
            model_id: spec.model_id.clone(),
            arch: Arc::new(Mutex::new(arch)),
            tokenizer,
            device,
            quant: spec.quant.clone(),
            devices,
        });

        let mut models = self.models.write().await;
        models.insert(spec.model_id.clone(), loaded);
        tracing::info!(model = %spec.model_id, "model loaded");
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        if models.remove(model_id).is_none() {
            anyhow::bail!("model '{model_id}' not loaded");
        }
        tracing::info!(model = %model_id, "model unloaded");
        Ok(())
    }

    async fn inference_endpoint(&self, model_id: &str) -> Option<String> {
        let models = self.models.read().await;
        models.contains_key(model_id).then(|| self.bind_url.clone())
    }
}

/// Errors returned by `CandleHarness::chat_completion`. The
/// `ModelNotLoaded` variant lets the HTTP handler map cleanly to 404
/// without string-matching on anyhow messages.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("model '{0}' not loaded on this neuron")]
    ModelNotLoaded(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Apply the Qwen3 chat template:
///
/// ```text
/// <|im_start|>{role}\n{content}<|im_end|>\n
/// ...
/// <|im_start|>assistant\n
/// ```
///
/// The trailing `<|im_start|>assistant\n` cues the model to begin a turn.
/// Non-text content parts (vision blocks) are joined as text only; full
/// multimodal handling is out of scope for Stage 3.
fn format_qwen3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let content = match &msg.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join(""),
        };
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[allow(clippy::too_many_arguments)]
fn run_inference(
    arch: &mut ModelArch,
    device: &Device,
    prompt_tokens: &[u32],
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    eos_id: Option<u32>,
) -> Result<(Vec<u32>, String)> {
    let mut logits_processor = {
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match top_p {
                Some(p) => Sampling::TopP { p, temperature },
                None => Sampling::All { temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let mut generated: Vec<u32> = Vec::new();

    let mut next_token = match arch {
        ModelArch::Qwen3Quantized(model) => {
            model.clear_kv_cache();
            let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        }
    };

    if Some(next_token) == eos_id {
        return Ok((generated, "stop".into()));
    }
    generated.push(next_token);

    for index in 0..max_new.saturating_sub(1) {
        next_token = match arch {
            ModelArch::Qwen3Quantized(model) => {
                let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                let logits = model.forward(&input, prompt_tokens.len() + index)?;
                let logits = logits.squeeze(0)?;
                logits_processor.sample(&logits)?
            }
        };
        if Some(next_token) == eos_id {
            return Ok((generated, "stop".into()));
        }
        generated.push(next_token);
    }

    Ok((generated, "length".into()))
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_subsec_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
