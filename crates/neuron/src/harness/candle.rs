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
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Weights;
use candle_transformers::models::qwen3 as qwen3_dense;
use cortex_core::harness::{Harness, HarnessHealth, ModelInfo, ModelSpec};
use cortex_core::openai::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, ChunkChoice, MessageContent, Usage,
};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, RwLock, mpsc};

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

/// Architecture-specific weights.
///
/// - `Qwen3Quantized` — GGUF source, pre-quantized. Single-GPU only;
///   TP sharding pre-quantized super-blocks is intractable. Stays the
///   default for small models loaded via `Qwen/Qwen3-*-GGUF` and
///   `unsloth/Qwen3-*-GGUF` repos.
/// - `Qwen3Dense` — bf16 safetensors source. The path that supports
///   TP (Stage 7b-ii+) because slicing dense weights by row/column
///   under safetensors is mechanical. Used when `ModelSpec.quant` is
///   None; intended target for Qwen3.6-27B etc.
///
/// Stage 8 broadens this to additional families.
pub enum ModelArch {
    Qwen3Quantized(QuantizedQwen3Weights),
    Qwen3Dense(qwen3_dense::ModelForCausalLM),
}

/// Repetition penalty applied to recently-generated tokens before
/// sampling. 1.0 disables it; >1.0 makes recently-emitted tokens less
/// likely. mistral.rs and llama.cpp default to 1.1, which is enough to
/// stop small quantized models from degenerating into "Wait, no, no..."
/// loops without distorting normal output.
const REPEAT_PENALTY: f32 = 1.1;

/// Number of recently-generated tokens to feed into the repetition
/// penalty. Matches the candle quantized-qwen3 example default.
const REPEAT_LAST_N: usize = 64;

/// Apply the repetition penalty (if any) to the prediction logits and
/// then sample. Centralises the prefill / generation-loop call sites
/// so they share identical sampling behaviour.
fn sample_with_penalty(
    logits: &Tensor,
    history: &[u32],
    logits_processor: &mut LogitsProcessor,
) -> Result<u32> {
    let penalised = if (REPEAT_PENALTY - 1.0).abs() < f32::EPSILON || history.is_empty() {
        logits.clone()
    } else {
        let start = history.len().saturating_sub(REPEAT_LAST_N);
        candle_transformers::utils::apply_repeat_penalty(logits, REPEAT_PENALTY, &history[start..])?
    };
    Ok(logits_processor.sample(&penalised)?)
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

    /// Build an hf-hub API client pre-configured with the harness's
    /// `hf_cache` (when one is set).
    fn hf_api(&self) -> Result<hf_hub::api::tokio::Api> {
        let mut builder = hf_hub::api::tokio::ApiBuilder::new();
        if let Some(cache) = &self.hf_cache {
            builder = builder.with_cache_dir(cache.clone());
        }
        builder.build().context("build hf-hub API")
    }

    /// Resolve a dense (bf16/fp16 safetensors) model to its local file
    /// paths.
    ///
    /// Handles both sharded repos (`model.safetensors.index.json` plus
    /// several `model-*.safetensors`) and the single-file layout
    /// (`model.safetensors`). Returns the safetensors paths in
    /// arbitrary order — `VarBuilder` unifies them into one tensor view.
    async fn resolve_dense_files(
        &self,
        spec: &ModelSpec,
    ) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
        let api = self.hf_api()?;
        let repo = api.model(spec.model_id.clone());

        let config_path = repo
            .get("config.json")
            .await
            .with_context(|| format!("fetch config.json from {}", spec.model_id))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .with_context(|| format!("fetch tokenizer.json from {}", spec.model_id))?;

        // Prefer the sharded layout (most HF dense models > 5B ship it).
        let safetensors_paths = match repo.get("model.safetensors.index.json").await {
            Ok(index_path) => {
                let index_text = std::fs::read_to_string(&index_path)
                    .context("read model.safetensors.index.json")?;
                let index: serde_json::Value = serde_json::from_str(&index_text)
                    .context("parse model.safetensors.index.json")?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| {
                        anyhow::anyhow!("safetensors index missing weight_map object")
                    })?;
                let unique: std::collections::BTreeSet<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                let mut paths = Vec::with_capacity(unique.len());
                for fname in unique {
                    let p = repo
                        .get(&fname)
                        .await
                        .with_context(|| format!("fetch sharded safetensors {fname}"))?;
                    paths.push(p);
                }
                paths
            }
            Err(_) => {
                // Single-file fallback.
                let p = repo
                    .get("model.safetensors")
                    .await
                    .context("fetch model.safetensors (single-file layout)")?;
                vec![p]
            }
        };
        Ok((config_path, tokenizer_path, safetensors_paths))
    }

    /// Resolve + load a GGUF (pre-quantized) Qwen3. Returns the
    /// tokenizer.json path so the caller can construct the Tokenizer
    /// uniformly across source formats.
    async fn load_arch_gguf(
        &self,
        spec: &ModelSpec,
        device: &Device,
    ) -> Result<(PathBuf, ModelArch)> {
        let (gguf_path, tokenizer_path) = self.resolve_files(spec).await?;
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
                    "unsupported GGUF architecture '{other}'; quantized path only supports qwen3"
                ),
            }
        })
        .await
        .context("blocking GGUF load task panicked")??;
        Ok((tokenizer_path, arch))
    }

    /// Resolve + load a dense Qwen3 from safetensors. Uses
    /// `candle-transformers::models::qwen3::ModelForCausalLM` and
    /// builds a VarBuilder over the mmap'd safetensors files. dtype
    /// is bf16 by default to match the HF distribution dtype for
    /// recent Qwen3 family models; fall back to f16 if the device
    /// doesn't support bf16.
    async fn load_arch_dense(
        &self,
        spec: &ModelSpec,
        device: &Device,
    ) -> Result<(PathBuf, ModelArch)> {
        let (config_path, tokenizer_path, safetensors_paths) =
            self.resolve_dense_files(spec).await?;
        let device_for_load = device.clone();
        let model_id_for_log = spec.model_id.clone();

        let arch = tokio::task::spawn_blocking(move || -> Result<ModelArch> {
            tracing::info!(
                model = %model_id_for_log,
                shards = safetensors_paths.len(),
                "loading dense Qwen3 from safetensors"
            );
            let cfg_text = std::fs::read_to_string(&config_path).context("read config.json")?;
            let cfg: qwen3_dense::Config =
                serde_json::from_str(&cfg_text).context("parse Qwen3 config.json")?;

            // bf16 is the canonical Qwen3 distribution dtype. CUDA
            // devices on Ada+ support it; Ampere also supports bf16
            // natively. CPU candle handles bf16 via emulation.
            let dtype = DType::BF16;
            // SAFETY: VarBuilder::from_mmaped_safetensors mmaps the files;
            // mutation of the underlying files by another process while
            // we hold the mapping is UB. We trust that nothing else on
            // the host modifies the HF cache files during a model's
            // lifetime (hf-hub itself is immutable-by-design).
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&safetensors_paths, dtype, &device_for_load)
                    .context("build VarBuilder over safetensors")?
            };
            let model = qwen3_dense::ModelForCausalLM::new(&cfg, vb)
                .map_err(|e| anyhow::anyhow!("build Qwen3 dense model: {e}"))?;
            Ok(ModelArch::Qwen3Dense(model))
        })
        .await
        .context("blocking dense load task panicked")??;
        Ok((tokenizer_path, arch))
    }

    /// Resolve a model spec to local GGUF and tokenizer file paths via
    /// hf-hub. Downloads on first use; subsequent calls are cached.
    async fn resolve_files(&self, spec: &ModelSpec) -> Result<(PathBuf, PathBuf)> {
        let api = self.hf_api()?;
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

        // GGUF-only HF repos (unsloth/Qwen3-*-GGUF, Qwen/Qwen3-*-GGUF,
        // etc.) ship the .gguf file but not tokenizer.json — the
        // tokenizer.json lives in the base non-GGUF repo. Derive the
        // base repo id by stripping a `-GGUF` / `-gguf` suffix; if
        // there's no such suffix the same repo is used (works for
        // non-GGUF model_ids).
        let tokenizer_repo_id = spec
            .model_id
            .strip_suffix("-GGUF")
            .or_else(|| spec.model_id.strip_suffix("-gguf"))
            .unwrap_or(spec.model_id.as_str())
            .to_string();
        let tokenizer_repo = if tokenizer_repo_id == spec.model_id {
            repo
        } else {
            tracing::debug!(
                from = %spec.model_id,
                to = %tokenizer_repo_id,
                "tokenizer.json sourced from base repo (GGUF suffix stripped)"
            );
            api.model(tokenizer_repo_id.clone())
        };
        let tokenizer_path = tokenizer_repo
            .get("tokenizer.json")
            .await
            .with_context(|| format!("fetch tokenizer.json from {tokenizer_repo_id}"))?;
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

    /// Run a streaming chat completion against a loaded model.
    ///
    /// Returns an `mpsc::Receiver` that yields `ChatCompletionChunk`s in
    /// OpenAI SSE format. The first chunk carries the assistant role;
    /// subsequent chunks carry incremental `content` deltas; the final
    /// chunk carries `finish_reason`. The handler is responsible for
    /// wrapping these into an SSE response and appending the `[DONE]`
    /// terminator.
    ///
    /// Token-by-token decoding tracks the cumulative decoded prefix so
    /// BPE byte-fallback boundaries don't split a UTF-8 char across
    /// chunks.
    pub async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<mpsc::Receiver<ChatCompletionChunk>, InferenceError> {
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
        let tokenizer = loaded.tokenizer.clone();
        let model_id = request.model.clone();
        let id = format!("chatcmpl-{:x}", unix_subsec_nanos());
        let created = unix_now_secs();

        // Bounded channel so the producer (blocking inference) is back-
        // pressured by the consumer (SSE writer). 32 is generous —
        // tokens arrive one at a time and the SSE writer is async.
        let (tx, rx) = mpsc::channel::<ChatCompletionChunk>(32);

        // Lead chunk: announce the assistant role per OpenAI streaming
        // conventions. Tools that auto-detect a streaming reply expect
        // this before any content delta.
        let role_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: json!({"role": "assistant"}),
                finish_reason: None,
                extra: serde_json::Value::Object(Default::default()),
            }],
            usage: None,
            extra: serde_json::Value::Object(Default::default()),
        };
        // If sending the role chunk fails the receiver is already gone;
        // bail before kicking off the heavy blocking work.
        tx.send(role_chunk)
            .await
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("client disconnected")))?;

        tokio::task::spawn_blocking(move || {
            let mut guard = arch_arc.blocking_lock();
            if let Err(e) = run_inference_streaming(
                &mut guard,
                &device,
                &tokenizer,
                &prompt_tokens,
                max_new,
                temperature,
                top_p,
                seed,
                eos_id,
                &id,
                created,
                &model_id,
                &tx,
            ) {
                tracing::warn!(model = %model_id, error = %e, "streaming inference failed");
            }
        });

        Ok(rx)
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

        // Stage 7a-i scaffolds tensor-parallel worker subprocesses but
        // does not yet route inference through them. Refuse TP loads
        // for now with a clear marker so the request surface is honest;
        // Stage 7b-iv replaces this bail with the TP dispatch.
        let tp_size = spec.tensor_parallel.unwrap_or(1);
        if tp_size > 1 {
            anyhow::bail!(
                "tensor_parallel={tp_size} requested for '{}': TP worker \
                 lifecycle + NCCL handshake are in place (Stage 7a) but \
                 TP-aware Qwen3 inference orchestration lands in Stage \
                 7b-iv; single-GPU loads only for now",
                spec.model_id
            );
        }

        let devices = spec.devices.clone().unwrap_or_else(|| vec![0]);
        let device = Self::pick_device(&devices)?;

        // Dispatch by source format: GGUF (pre-quantized, single-GPU
        // only path) vs safetensors dense (bf16/fp16; the path that
        // grows TP support). `spec.quant` is the signal — Some means
        // the operator picked a quantized GGUF; None means dense.
        let (tokenizer_path, arch) = if spec.quant.is_some() {
            self.load_arch_gguf(spec, &device).await?
        } else {
            self.load_arch_dense(spec, &device).await?
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

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
            sample_with_penalty(&logits, &generated, &mut logits_processor)?
        }
        ModelArch::Qwen3Dense(model) => {
            model.clear_kv_cache();
            let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            sample_with_penalty(&logits, &generated, &mut logits_processor)?
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
                sample_with_penalty(&logits, &generated, &mut logits_processor)?
            }
            ModelArch::Qwen3Dense(model) => {
                let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                let logits = model.forward(&input, prompt_tokens.len() + index)?;
                let logits = logits.squeeze(0)?;
                sample_with_penalty(&logits, &generated, &mut logits_processor)?
            }
        };
        if Some(next_token) == eos_id {
            return Ok((generated, "stop".into()));
        }
        generated.push(next_token);
    }

    Ok((generated, "length".into()))
}

/// Streaming counterpart to `run_inference`. Emits chunks via `tx` as
/// tokens are generated and exits on EOS, max_new, or receiver drop.
///
/// Detokenization tracks the cumulative decoded prefix so each chunk's
/// `content` delta is the substring appended since the last chunk —
/// safe across BPE byte-fallback boundaries.
#[allow(clippy::too_many_arguments)]
fn run_inference_streaming(
    arch: &mut ModelArch,
    device: &Device,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    eos_id: Option<u32>,
    id: &str,
    created: u64,
    model_id: &str,
    tx: &mpsc::Sender<ChatCompletionChunk>,
) -> Result<()> {
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

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut decoded_prefix = String::new();
    let mut finish_reason = "length".to_string();

    let mut next_token = match arch {
        ModelArch::Qwen3Quantized(model) => {
            model.clear_kv_cache();
            let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?
        }
        ModelArch::Qwen3Dense(model) => {
            model.clear_kv_cache();
            let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?
        }
    };

    let emit_token = |all_tokens: &[u32], decoded_prefix: &mut String| -> Result<bool> {
        let full = tokenizer
            .decode(all_tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        if full.len() > decoded_prefix.len() {
            let delta = full[decoded_prefix.len()..].to_string();
            *decoded_prefix = full;
            let chunk = ChatCompletionChunk {
                id: id.into(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_id.into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: json!({ "content": delta }),
                    finish_reason: None,
                    extra: serde_json::Value::Object(Default::default()),
                }],
                usage: None,
                extra: serde_json::Value::Object(Default::default()),
            };
            // blocking_send returns Err if the consumer hung up — signal
            // the caller to stop generating.
            if tx.blocking_send(chunk).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    };

    if Some(next_token) == eos_id {
        finish_reason = "stop".into();
    } else {
        all_tokens.push(next_token);
        if !emit_token(&all_tokens, &mut decoded_prefix)? {
            return Ok(());
        }

        for index in 0..max_new.saturating_sub(1) {
            next_token = match arch {
                ModelArch::Qwen3Quantized(model) => {
                    let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, prompt_tokens.len() + index)?;
                    let logits = logits.squeeze(0)?;
                    sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?
                }
                ModelArch::Qwen3Dense(model) => {
                    let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, prompt_tokens.len() + index)?;
                    let logits = logits.squeeze(0)?;
                    sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?
                }
            };
            if Some(next_token) == eos_id {
                finish_reason = "stop".into();
                break;
            }
            all_tokens.push(next_token);
            if !emit_token(&all_tokens, &mut decoded_prefix)? {
                return Ok(());
            }
        }
    }

    let final_chunk = ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: serde_json::Value::Object(Default::default()),
            finish_reason: Some(finish_reason),
            extra: serde_json::Value::Object(Default::default()),
        }],
        usage: None,
        extra: serde_json::Value::Object(Default::default()),
    };
    let _ = tx.blocking_send(final_chunk);
    Ok(())
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
