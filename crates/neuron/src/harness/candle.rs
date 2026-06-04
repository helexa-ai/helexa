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
use candle_transformers::models::llama as llama_dense;
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaWeights;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Weights;
use candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE;
use candle_transformers::models::qwen3 as qwen3_dense;
use candle_transformers::models::qwen3_moe as qwen3_moe_dense;
use cortex_core::harness::{Harness, HarnessHealth, ModelInfo, ModelSpec};
use cortex_core::openai::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, MessageContent, Usage,
};

use crate::wire::{
    FinishReason, InferenceEvent, ReasoningTokenPair, ToolCallTokenPair,
    detect_reasoning_token_pair, detect_tool_call_token_pair, openai_chat as wire_chat,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda")]
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::Instrument;

/// In-process candle harness. Owns the loaded model registry.
pub struct CandleHarness {
    models: Arc<RwLock<HashMap<String, LoadedHandle>>>,
    /// Post-resolution source map: scheme → endpoint/token/cache. Built
    /// in `new()` from the operator's `CandleHarnessConfig`; auth tokens
    /// are read from their configured env vars at startup so secrets
    /// don't leak through the config file.
    sources: HashMap<String, ResolvedSource>,
    /// Scheme to substitute for bare `org/name` model ids.
    default_source: String,
    bind_url: String,
    /// One worker thread per CUDA device index that owns its
    /// `CudaContext` for the daemon's lifetime. Populated lazily by
    /// `ensure_device_worker()` when a model is loaded onto a CUDA
    /// device. CPU `Device::Cpu` loads don't get an entry; they have
    /// no context to own. Unused on the no-cuda build (the harness
    /// can still load on CPU for tests, just without worker threads).
    #[allow(dead_code)]
    device_workers: Arc<RwLock<HashMap<u32, Arc<super::device_worker::DeviceWorkerHandle>>>>,
}

/// One entry in the harness's loaded-model registry. Single-GPU loads
/// land in `Single`; loads with `tensor_parallel > 1` land in `Tp`.
/// The two variants share the same `model_id` key in the map, so
/// `list_models`, `unload_model`, and `inference_endpoint` can walk
/// them uniformly without branching the storage layout.
///
/// `Clone` is cheap: both variants hold `Arc<_>` and cloning just bumps
/// the refcount.
#[derive(Clone)]
pub enum LoadedHandle {
    Single(Arc<LoadedModel>),
    #[cfg(feature = "cuda")]
    Tp(Arc<TpLoadedModel>),
}

impl LoadedHandle {
    pub fn model_id(&self) -> &str {
        match self {
            LoadedHandle::Single(m) => &m.model_id,
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => &m.model_id,
        }
    }

    pub fn devices(&self) -> Vec<u32> {
        match self {
            LoadedHandle::Single(m) => m.devices.clone(),
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => m.devices.clone(),
        }
    }

    /// True if an earlier inference left the device context in an
    /// unrecoverable state. Surfaced in `/models` so cortex (and an
    /// operator running `curl beast:13131/models`) can see at a glance
    /// that the model needs unload+reload.
    pub fn is_poisoned(&self) -> bool {
        match self {
            LoadedHandle::Single(m) => m.poisoned.load(Ordering::Acquire),
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => m.poisoned.load(Ordering::Acquire),
        }
    }

    /// Modalities the loaded model supports. Stage B7 (single-GPU) +
    /// TP-vision (#12) — both single-GPU and TP loads advertise
    /// `"vision"` when a replicated vision tower materialised.
    pub fn capabilities(&self) -> Vec<String> {
        let mut caps = vec!["text".to_string()];
        let has_vision = match self {
            LoadedHandle::Single(m) => m.has_vision,
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => m.has_vision,
        };
        if has_vision {
            caps.push("vision".to_string());
        }
        caps
    }
}

/// A loaded model with its tokenizer, device placement, and architecture-
/// specific weights. The `arch` is `Arc<Mutex<>>` so the lock guard can be
/// moved into `spawn_blocking` for synchronous candle forward passes.
pub struct LoadedModel {
    pub model_id: String,
    /// Local (async-side) handle to the model architecture. `Some`
    /// only when the model loaded onto the CPU device (no CUDA
    /// available); the inference path then takes this mutex via
    /// `spawn_blocking` and runs candle ops on the CPU backend.
    /// `None` when the model loaded onto a CUDA device — in that case
    /// the architecture lives in the worker thread's slab and is
    /// addressed via [`Self::arch_handle`].
    pub arch: Option<Arc<Mutex<ModelArch>>>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub quant: Option<String>,
    pub devices: Vec<u32>,
    /// Set to `true` after any forward / kv-cache call fails. A CUDA
    /// driver error (OOM, illegal address) leaves the device's context
    /// in an unrecoverable state — subsequent kernels can hang, return
    /// garbage, or hit another illegal address. The harness refuses
    /// further inference against a poisoned model and reports a clear
    /// error so an operator knows to unload+reload to recover. See
    /// the 2026-05-26 beast incident where a 14k-token prefill OOM
    /// silently turned every subsequent request into a stuck wait.
    pub poisoned: AtomicBool,
    /// Handle to the per-device CUDA worker thread for this model's
    /// device. `None` for CPU loads (no context to own). VRAM queries
    /// and — for CUDA loads — forward / kv-cache / drop ops route
    /// through this handle so the device's CUDA context stays bound
    /// to one OS thread for the daemon's lifetime.
    pub worker: Option<Arc<super::device_worker::DeviceWorkerHandle>>,
    /// Index into the worker's `ModelArch` slab. `Some` iff the model
    /// loaded onto a CUDA device and was successfully transferred to
    /// the worker; in that case [`Self::arch`] is `None`. The two
    /// fields are mutually exclusive.
    pub arch_handle: Option<super::device_worker::ArchHandle>,
    /// Serialises chat-completion requests against this model. Held
    /// from the start of `clear_kv_cache` through the last decode
    /// step, so concurrent requests can't interleave their KV-cache
    /// mutations. Without this, two requests' chunked-prefill
    /// `clear → forward(chunk0) → forward(chunk1) → ...` sequences
    /// could end up sharing a cache between them — the device worker
    /// channel serialises individual jobs, but not the sequence
    /// boundary. Observed on benjy 2026-05-27 18:41 when agent-zero's
    /// memorize extensions fired in parallel and produced a
    /// shape-mismatch failure mid-prefill. Mirrors TpLoadedModel.pool
    /// for the TP path (which already had this invariant by accident
    /// because the pool lock covered the same window).
    pub inference_lock: tokio::sync::Mutex<()>,
    /// Open/close token IDs for the reasoning marker this model
    /// emits, populated once at load time by probing the tokenizer's
    /// added-tokens table. `None` for non-reasoning models or
    /// reasoning models whose markers aren't single tokens. When
    /// `Some`, the streaming inference loop splits output into
    /// [`InferenceEvent::TextDelta`] and
    /// [`InferenceEvent::ReasoningDelta`] at the token boundary;
    /// when `None` everything is `TextDelta`.
    pub reasoning_tokens: Option<ReasoningTokenPair>,
    /// Open/close token IDs for the model's tool-call marker
    /// pair (`<tool_call>` / `</tool_call>` on Qwen3-Coder / Hermes
    /// / DeepSeek / gpt-oss). `None` for models that don't emit
    /// structured tool calls in this convention; output passes
    /// through as plain text in that case and the consumer parses
    /// the markers itself if it knows how.
    pub tool_call_tokens: Option<ToolCallTokenPair>,
    /// Raw Jinja `chat_template` string loaded from this model's
    /// `tokenizer_config.json` at load time. `None` when the file
    /// is absent / unparseable / lacks the field. When `Some`,
    /// the prompt-build path renders it through `minijinja` with
    /// `chat_template_kwargs` from the request body; when `None`,
    /// the hardcoded Qwen3 ChatML fallback (`format_qwen3_prompt`)
    /// is used. The `NEURON_USE_CHAT_TEMPLATE=false` env var
    /// forces the fallback path even when `Some`.
    pub chat_template: Option<String>,
    /// Vision capability flag derived at load time. `true` iff the
    /// loaded `ModelArch` exposes a vision tower (Stage A4 wires this
    /// from `Qwen3_5ForCausalLM::has_vision`). Used by the chat
    /// completion handler to reject image content on non-vision
    /// models with a structured 400 (Stage B6) and by `/v1/models`
    /// to advertise `capabilities: ["text", "vision"]` (Stage B7).
    pub has_vision: bool,
    /// `<|image_pad|>` token id from `config.json::image_token_id`.
    /// The Stage B prompt-builder uses this to compute expansion
    /// targets and the worker forward uses it to locate splice
    /// positions in the LM input embeddings.
    pub image_token_id: Option<u32>,
    /// `patch_size × spatial_merge_size` — divides a resized pixel
    /// dimension into LM-grid units. Per-image LM token count is
    /// `(h/factor) × (w/factor)` (#14 dynamic resolution). `None` for
    /// text-only models. Set at load time.
    pub image_grid_factor: Option<usize>,
}

impl LoadedModel {
    /// Free / total VRAM on this model's device in MiB. Routes the
    /// query through the device worker thread (where the CUDA context
    /// is already bound) rather than rebinding on whatever tokio
    /// thread the caller happens to be on. Returns `(0, 0)` on CPU
    /// loads, or if the worker is gone / poisoned / the cudarc call
    /// itself failed — same sentinel the previous `device_vram_mb`
    /// helper returned, so log field values stay comparable.
    pub async fn query_vram(&self) -> (u64, u64) {
        match &self.worker {
            Some(w) => w.query_vram().await.unwrap_or((0, 0)),
            None => (0, 0),
        }
    }
}

/// Tensor-parallel loaded model. Holds the leader's rank-0 shard
/// (which the inference loop drives via spawn_blocking) and the
/// `WorkerPool` (which drives every non-zero rank over the RPC
/// channel). Both are behind tokio Mutexes so concurrent inference
/// requests against the same model are serialised; concurrent loads
/// for *different* models would each have their own pool.
#[cfg(feature = "cuda")]
pub struct TpLoadedModel {
    pub model_id: String,
    pub tokenizer: Tokenizer,
    pub devices: Vec<u32>,
    /// One end-to-end gate: the pool's RPC stream to the subprocess
    /// workers isn't safe to use concurrently. After Phase 3 the
    /// leader's `TpLeaderModel` lives in the worker thread's slab,
    /// so this Mutex no longer covers the leader's KV cache; it just
    /// serialises subprocess RPC traffic on the pool's
    /// `Vec<Worker>` channels.
    pub pool: tokio::sync::Mutex<super::tp::WorkerPool>,
    /// Handle into the leader device worker's TP slab. The boxed
    /// `TpLeaderModel` (with its embedded `Arc<Comm>` clones and
    /// per-rank CUDA tensors) lives on the worker thread; we hold an
    /// opaque index. Forward / clear_kv / unload all route through
    /// `Job::Tp*` against this handle.
    pub leader_handle: super::device_worker::TpHandle,
    /// Candle device for rank 0. Mirrors what
    /// `TpLeaderModel::device()` would return, kept on the struct so
    /// the request path can name the device without an RPC.
    pub leader_device: Device,
    /// Same poisoning gate as [`LoadedModel::poisoned`]. A TP forward
    /// failure (CUDA OOM on any rank, NCCL desync, illegal address) is
    /// terminal: the leader's and workers' CUDA contexts cannot be
    /// reliably reset without restarting the worker subprocesses.
    pub poisoned: AtomicBool,
    /// Worker thread for the leader's CUDA device. Owns the leader's
    /// `CudaContext`, `NcclState`, and the boxed `TpLeaderModel`
    /// referenced by `leader_handle`.
    pub worker: Arc<super::device_worker::DeviceWorkerHandle>,
    /// Same shape as [`LoadedModel::reasoning_tokens`] — open/close
    /// reasoning marker token IDs probed from the tokenizer at
    /// load time. `None` when the model declares no reasoning
    /// markers.
    pub reasoning_tokens: Option<ReasoningTokenPair>,
    /// Same shape as [`LoadedModel::tool_call_tokens`].
    pub tool_call_tokens: Option<ToolCallTokenPair>,
    /// Same shape as [`LoadedModel::chat_template`].
    pub chat_template: Option<String>,
    /// Vision capability flag (TP-vision). `true` iff every rank
    /// materialised a replicated vision tower. Mirrors
    /// [`LoadedModel::has_vision`]; drives capability advertising and
    /// the TP vision dispatch.
    pub has_vision: bool,
    /// `<|image_pad|>` token id — same as [`LoadedModel::image_token_id`].
    pub image_token_id: Option<u32>,
    /// Pixel→LM-grid divisor — same as
    /// [`LoadedModel::image_grid_factor`].
    pub image_grid_factor: Option<usize>,
}

#[cfg(feature = "cuda")]
impl TpLoadedModel {
    /// Free / total VRAM on the leader's device in MiB. See
    /// [`LoadedModel::query_vram`] for rationale and sentinel
    /// semantics — same pattern, TP just always has a worker because
    /// the harness rejects TP without CUDA at load time.
    pub async fn query_vram(&self) -> (u64, u64) {
        self.worker.query_vram().await.unwrap_or((0, 0))
    }
}

/// Architecture-specific weights. Each variant covers one (family,
/// source-format) pair; the dense variants take the safetensors path
/// and the `Quantized*` variants take the GGUF path.
///
/// TP currently only works through `Qwen3Dense` (see `tp_qwen3.rs`);
/// every other variant is single-GPU. Quantized variants can't shard
/// across GPUs at all — slicing GGUF super-blocks is intractable —
/// and the new dense families (Llama, Qwen3 MoE) lack their own
/// TP-aware modules yet.
pub enum ModelArch {
    // Qwen3 family
    Qwen3Quantized(QuantizedQwen3Weights),
    Qwen3Dense(qwen3_dense::ModelForCausalLM),
    Qwen3MoeQuantized(GGUFQWenMoE),
    Qwen3MoeDense(qwen3_moe_dense::ModelForCausalLM),

    // Llama family (covers Llama 1/2/3/3.1/3.3). Boxed because the
    // wrapper carries an inline Cache + Config — without indirection
    // the enum's `LlamaDense` variant is several hundred bytes larger
    // than the others (clippy::large_enum_variant).
    LlamaQuantized(QuantizedLlamaWeights),
    LlamaDense(Box<LlamaDense>),

    // Qwen3-Next family (model_type "qwen3_5") — Qwen3.6's
    // architecture. Stage 8c scaffolding only: dispatch + config parse
    // are real; forward bails "not implemented yet". See
    // `arch/qwen3_5.rs` for the open architecture work.
    Qwen3_5Dense(super::arch::qwen3_5::Qwen3_5ForCausalLM),
}

impl ModelArch {
    /// One forward step on this arch with the rank-1 vocab logits
    /// extracted. Hides per-family shape differences (some return
    /// `[B, V]`, others `[B, 1, V]`) — every caller gets a `[V]`
    /// tensor ready for sampling.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let raw = match self {
            ModelArch::Qwen3Quantized(m) => m.forward(input, offset)?,
            ModelArch::Qwen3Dense(m) => m.forward(input, offset)?,
            ModelArch::Qwen3MoeQuantized(m) => m.forward(input, offset)?,
            ModelArch::Qwen3MoeDense(m) => m.forward(input, offset)?,
            ModelArch::LlamaQuantized(m) => m.forward(input, offset)?,
            ModelArch::LlamaDense(m) => m.forward(input, offset)?,
            ModelArch::Qwen3_5Dense(m) => m.forward(input, offset)?,
        };
        squeeze_to_vocab(&raw)
    }

    /// Reset the KV cache before each new request so we don't attend
    /// over a previous request's tokens. Some architectures have an
    /// in-place reset; Llama needs a Cache rebuild (held inline in
    /// the wrapper).
    pub fn clear_kv_cache(&mut self) -> Result<()> {
        match self {
            ModelArch::Qwen3Quantized(_) => Ok(()), /* keeps cache by design;
            * forward() handles offset */
            ModelArch::Qwen3Dense(m) => {
                m.clear_kv_cache();
                Ok(())
            }
            ModelArch::Qwen3MoeQuantized(_) => Ok(()),
            ModelArch::Qwen3MoeDense(m) => {
                m.clear_kv_cache();
                Ok(())
            }
            ModelArch::LlamaQuantized(_) => Ok(()),
            ModelArch::LlamaDense(m) => m.clear_kv_cache(),
            ModelArch::Qwen3_5Dense(m) => {
                m.clear_kv_cache();
                Ok(())
            }
        }
    }

    /// Forward step that splices vision-tower output at
    /// `<|image_pad|>` token positions. Stage B2.
    ///
    /// Only `Qwen3_5Dense` supports this — other architectures error
    /// because they don't have a vision tower. The HTTP layer is
    /// expected to have rejected image content for non-vision models
    /// already (Stage B6); this is a defence-in-depth error path.
    ///
    /// Returns rank-1 `[vocab_size]` logits, same shape contract as
    /// `forward`.
    pub fn forward_with_vision(
        &mut self,
        input: &Tensor,
        offset: usize,
        image_embeds: &Tensor,
        image_token_id: u32,
        grids: &[(usize, usize)],
    ) -> Result<Tensor> {
        let raw = match self {
            ModelArch::Qwen3_5Dense(m) => {
                m.forward_with_vision(input, offset, image_embeds, image_token_id, grids)?
            }
            other => anyhow::bail!(
                "forward_with_vision: architecture {} has no vision tower",
                std::any::type_name_of_val(other)
            ),
        };
        squeeze_to_vocab(&raw)
    }

    /// `patch_size × spatial_merge_size` for the loaded vision tower —
    /// divides a resized pixel dim into LM-grid units (an image of
    /// resized `(h, w)` yields the LM grid `(h/factor, w/factor)`).
    /// `None` for architectures/checkpoints without a vision tower.
    pub fn vision_grid_factor(&self) -> Option<usize> {
        match self {
            ModelArch::Qwen3_5Dense(m) => m.vision().map(|v| {
                let c = v.config();
                c.patch_size * c.spatial_merge_size
            }),
            _ => None,
        }
    }

    /// Encode a preprocessed image into LM-side token embeddings via
    /// the loaded vision tower. Stage A5.
    ///
    /// `image`: device-resident `(C, H, W)` f32 tensor — caller has
    /// already preprocessed via `harness::preprocess::preprocess` and
    /// uploaded to the worker's device. Returns
    /// `(N_lm_tokens, hidden_size)`.
    ///
    /// Errors when the loaded architecture has no vision tower
    /// (text-only checkpoint, or architecture that doesn't support
    /// vision at all). The HTTP layer maps this to a 400 with
    /// `vision_unsupported` so clients see a clean rejection rather
    /// than a confident text-only hallucination.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        match self {
            ModelArch::Qwen3_5Dense(m) => m
                .vision()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "encode_image: this Qwen3.6 checkpoint was loaded without a vision \
                         tower (config.json::vision_config absent or weights missing)"
                    )
                })?
                .forward(image),
            other => anyhow::bail!(
                "encode_image: architecture {} has no vision tower",
                std::any::type_name_of_val(other)
            ),
        }
    }
}

/// Squeeze any leading singleton dims off the logits tensor so the
/// caller gets a rank-1 `[vocab_size]` slice ready for sampling. Bails
/// on a non-singleton leading dim (would mean a batched forward, which
/// no caller emits today).
fn squeeze_to_vocab(t: &Tensor) -> Result<Tensor> {
    let mut t = t.clone();
    while t.dims().len() > 1 {
        if t.dims()[0] != 1 {
            anyhow::bail!(
                "logits expected to start with a singleton dim, got shape {:?}",
                t.dims()
            );
        }
        t = t.squeeze(0)?;
    }
    Ok(t)
}

/// Llama dense wrapper. Bundles candle's `Llama` model with its
/// externally-managed `Cache` plus enough config to rebuild the
/// cache on `clear_kv_cache` (Llama's Cache doesn't expose a reset).
pub struct LlamaDense {
    model: llama_dense::Llama,
    cache: llama_dense::Cache,
    config: llama_dense::Config,
    dtype: DType,
    device: Device,
}

impl LlamaDense {
    /// Constructor used by the dispatch-side loader. Keeps the field
    /// names private while letting the worker thread build a
    /// `LlamaDense` from already-loaded weights without going through
    /// async candle code.
    pub(crate) fn from_parts(
        model: llama_dense::Llama,
        cache: llama_dense::Cache,
        config: llama_dense::Config,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            model,
            cache,
            config,
            dtype,
            device,
        }
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        Ok(self.model.forward(input, offset, &mut self.cache)?)
    }

    pub fn clear_kv_cache(&mut self) -> Result<()> {
        self.cache = llama_dense::Cache::new(true, self.dtype, &self.config, &self.device)
            .context("rebuild Llama Cache for new request")?;
        Ok(())
    }
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

/// Architectures the dense safetensors path can construct. Keep
/// alphabetical; one entry per supported `config.json#/model_type`
/// value. New entries land alongside a new `ModelArch` variant + a
/// dispatch branch in `load_arch_dense` (plus, for TP, a parallel
/// pattern in `tp_qwen3.rs`).
const DENSE_SUPPORTED_MODEL_TYPES: &[&str] = &["llama", "qwen3", "qwen3_5", "qwen3_moe"];

/// Pre-flight check the operator's `config.json` against the set of
/// architectures the dense path actually knows how to build. Surfaces
/// architecture mismatches as a single clean error before the serde
/// deserializer trips on missing fields — the latter happens because
/// every architecture has different hyperparameter names, so when the
/// JSON is e.g. Qwen3.6 wrapped under `text_config: {...}`, candle's
/// `qwen3::Config` finds none of its expected top-level fields and
/// fails with a cryptic `missing field 'vocab_size' at line N col 1`.
///
/// The result message names the model_type we saw, the supported set,
/// and points at the files an operator (or future contributor) needs
/// to touch to grow the supported set.
pub(crate) fn check_dense_config_supported(config_json: &str, model_id: &str) -> Result<()> {
    let v: serde_json::Value = serde_json::from_str(config_json)
        .with_context(|| format!("parse config.json for '{model_id}' as JSON"))?;
    let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    if model_type.is_empty() {
        anyhow::bail!(
            "config.json for '{model_id}' is missing `model_type`; the dense \
             path needs it to gate architecture support (supported: {:?})",
            DENSE_SUPPORTED_MODEL_TYPES
        );
    }
    if DENSE_SUPPORTED_MODEL_TYPES.contains(&model_type) {
        return Ok(());
    }
    // Bonus context: the model usually also lists architectures, which
    // is what `transformers` keys on. Including it makes the error
    // self-contained.
    let architectures = v
        .get("architectures")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::bail!(
        "unsupported model_type '{model_type}' for '{model_id}' \
         (architectures={architectures:?}); the dense path supports {:?}. \
         Add a `ModelArch` variant + load/forward branches in \
         crates/neuron/src/harness/candle.rs (and the TP analogue in \
         tp_qwen3.rs) to extend coverage.",
        DENSE_SUPPORTED_MODEL_TYPES
    );
}

/// Architectures the TP path can actually load and run. A subset of
/// `DENSE_SUPPORTED_MODEL_TYPES` — the single-GPU path supports more
/// families than the TP path because each TP-aware module is a real
/// chunk of work (`tp_qwen3.rs` is the only one shipped today).
#[cfg(feature = "cuda")]
const TP_SUPPORTED_MODEL_TYPES: &[&str] = &["qwen3", "qwen3_5"];

/// TP-side counterpart to `check_dense_config_supported`. Gates the
/// `load_tp` path on a narrower architecture set: even though the
/// single-GPU dense path knows how to build a Llama model, the worker
/// pool's `load_dense_shard` reconstructs the config as Qwen3 — there
/// is no `tp_llama.rs` yet. Surfacing this as a config-time error
/// (before we spawn workers and burn NCCL handshake cost) is much
/// kinder than the inevitable per-rank deserialise failure.
#[cfg(feature = "cuda")]
fn check_tp_arch_supported(config_json: &str, model_id: &str) -> Result<()> {
    let v: serde_json::Value = serde_json::from_str(config_json)
        .with_context(|| format!("parse config.json for '{model_id}' as JSON"))?;
    let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    if TP_SUPPORTED_MODEL_TYPES.contains(&model_type) {
        return Ok(());
    }
    anyhow::bail!(
        "tensor_parallel requested for '{model_id}' (model_type='{model_type}') but \
         the TP path supports only {TP_SUPPORTED_MODEL_TYPES:?}. Adding a new \
         TP-aware architecture needs a `harness/tp/tp_<family>.rs` module mirroring \
         `tp_qwen3.rs` (sharded linears, AllReduce, per-rank head counts) and a \
         dispatch in `WorkerPool::load_dense_shard`. For models that fit on one \
         GPU, drop `tensor_parallel` to use the single-GPU dense path."
    )
}

/// Resolve the effective HuggingFace cache directory for the candle
/// harness. Precedence (first hit wins):
///
/// 1. Explicit `hf_cache` from `[harness.candle]` in `neuron.toml`.
///    Operator's wishes always win.
/// 2. `HF_HUB_CACHE` env var. The Python `huggingface_hub` library
///    points at the cache root directly with this var; the Rust
///    `hf-hub` crate doesn't read it natively, so we bridge here.
///    Honouring it lets a neuron host share a cache directory with
///    Python tooling and other harnesses without per-tool config.
/// 3. `HF_HOME` env var. Canonical HuggingFace base directory; the
///    cache lives at `$HF_HOME/hub`. Hf-hub respects this on its own,
///    but we resolve it here too so the resulting path shows up in
///    logs alongside the explicit/HF_HUB_CACHE cases.
/// 4. `None`. Falls through to `hf-hub`'s default
///    (`~/.cache/huggingface/hub`).
fn resolve_hf_cache(explicit: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p);
    }
    if let Ok(v) = std::env::var("HF_HUB_CACHE")
        && !v.is_empty()
    {
        return Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("HF_HOME")
        && !v.is_empty()
    {
        return Some(PathBuf::from(v).join("hub"));
    }
    None
}

/// Summary stats over a 1-D logits tensor, used for the failure log
/// when sampling rejects the distribution. Gathers nan/inf/negative
/// counts and finite min/max/mean — enough to distinguish a NaN
/// cascade (all-NaN, typical of softmax overflow propagating) from
/// an Inf at a single position (numerical edge case) from negative
/// weights (different bug entirely).
///
/// Computed only on the failure path, so the to_vec1 copy cost is
/// paid at most once per poisoned model.
#[derive(Debug)]
#[allow(dead_code)]
struct LogitsHealth {
    len: usize,
    nan: usize,
    pos_inf: usize,
    neg_inf: usize,
    neg: usize,
    finite_min: Option<f32>,
    finite_max: Option<f32>,
    finite_mean: Option<f32>,
}

#[allow(dead_code)]
fn logits_health(t: &Tensor) -> LogitsHealth {
    let values: Vec<f32> = match t
        .to_dtype(candle_core::DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => v,
        Err(_) => {
            return LogitsHealth {
                len: 0,
                nan: 0,
                pos_inf: 0,
                neg_inf: 0,
                neg: 0,
                finite_min: None,
                finite_max: None,
                finite_mean: None,
            };
        }
    };
    logits_health_slice(&values)
}

/// Same diagnostic as [`logits_health`] but operates directly on a
/// `[f32]` slice. Used by the worker-routed inference paths where the
/// device → host copy has already happened on the worker thread and
/// the async caller has the values in hand. Avoids the round-trip of
/// rebuilding a Tensor just to call to_vec1 again.
#[allow(dead_code)]
fn logits_health_slice(values: &[f32]) -> LogitsHealth {
    let mut nan = 0usize;
    let mut pos_inf = 0usize;
    let mut neg_inf = 0usize;
    let mut neg = 0usize;
    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    let mut finite_sum = 0.0_f64;
    let mut finite_count = 0usize;
    for &v in values {
        if v.is_nan() {
            nan += 1;
        } else if v == f32::INFINITY {
            pos_inf += 1;
        } else if v == f32::NEG_INFINITY {
            neg_inf += 1;
        } else {
            if v < 0.0 {
                neg += 1;
            }
            if v < finite_min {
                finite_min = v;
            }
            if v > finite_max {
                finite_max = v;
            }
            finite_sum += v as f64;
            finite_count += 1;
        }
    }
    let finite_mean = if finite_count > 0 {
        Some((finite_sum / finite_count as f64) as f32)
    } else {
        None
    };
    LogitsHealth {
        len: values.len(),
        nan,
        pos_inf,
        neg_inf,
        neg,
        finite_min: (finite_count > 0).then_some(finite_min),
        finite_max: (finite_count > 0).then_some(finite_max),
        finite_mean,
    }
}

/// Classify an inference-failure error string: should we mark the
/// model poisoned, or is this a logic / numerical / tokenizer failure
/// that leaves the device context healthy? Default is "yes, poison" —
/// the cost of failing to poison a genuinely-corrupt context (next
/// request hangs or returns garbage) outweighs the cost of
/// over-poisoning (operator unload+reloads). The opt-out list covers
/// errors we know don't touch device state.
///
/// Pass the `format!("{err:#}")` rendering of an anyhow::Error (or the
/// already-stringified error in paths that stringify failures, like
/// the TP streaming task). Matching against the full chain lets the
/// classification survive `.context("…")` and `format!("…: {e}")`
/// wrappers in the call sites.
fn is_device_fault(chain_text: &str) -> bool {
    let chain = chain_text.to_lowercase();
    // Non-device patterns: shape errors are pre-kernel and don't touch
    // GPU state; NaN-logits failures happen on the CPU side after the
    // forward; tokenize/detokenize is pure CPU; missing-handle lookups
    // are pre-dispatch. Everything else we treat conservatively as a
    // potential device fault.
    let non_device_markers = [
        "shape mismatch",
        "broadcast",
        "cannot broadcast",
        "logits unhealthy",
        "tokenize",
        "detokenize",
        "decode_stream",
        "no model for handle",
        "no tp model for handle",
        "empty prompt",
    ];
    !non_device_markers.iter().any(|m| chain.contains(m))
}

/// Build the InferenceError reported to a client when their request
/// hits a model that's been marked poisoned by an earlier driver
/// failure. The message names the model and the recovery procedure so
/// the operator doesn't have to chase the original failure to know
/// what to do.
fn poisoned_error(model_id: &str) -> InferenceError {
    InferenceError::Other(anyhow::anyhow!(
        "model '{model_id}' is in a poisoned state \
         (an earlier inference hit a CUDA driver error and the device \
          context cannot be safely reused); unload and reload the model \
          to recover"
    ))
}

/// Free/total VRAM on the candle `Device` in MiB. Returns `(0, 0)` if
/// the query fails or the device is the CPU fallback so logging never
/// crashes the request path. Mirrors the existing helper in
/// `tp_qwen3_5.rs`; kept separate to avoid coupling the inference path
/// to the TP-specific module.
#[cfg(feature = "cuda")]
fn device_vram_mb(device: &Device) -> (u64, u64) {
    use candle_core::cuda::cudarc::driver::result;
    use candle_core::cuda_backend::WrapErr;
    let Device::Cuda(dev) = device else {
        return (0, 0);
    };
    let Ok(()) = dev.cuda_stream().context().bind_to_thread().w() else {
        return (0, 0);
    };
    match result::mem_get_info() {
        Ok((free, total)) => (
            (free / (1024 * 1024)) as u64,
            (total / (1024 * 1024)) as u64,
        ),
        Err(_) => (0, 0),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn device_vram_mb(_device: &Device) -> (u64, u64) {
    (0, 0)
}

/// A short hex tag used to group every log line emitted on behalf of
/// one chat-completion request. Six hex digits is unique enough across
/// a 4-hour journal window (24 bits ≈ 16M values, while a busy neuron
/// sees ~10³ requests/hour) and fits cleanly inside `req_id=…` in the
/// fmt subscriber's span-prefix output.
fn new_req_id() -> String {
    format!("{:06x}", unix_subsec_nanos() & 0xFFFFFF)
}

/// Read a positive `usize` from `name` in the process env, falling back
/// to `default` if unset or unparseable. Used for runtime tuning knobs
/// that we want operators to be able to adjust without a recompile.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v: &usize| *v > 0)
        .unwrap_or(default)
}

/// Same as [`env_usize`] but for `u64`.
fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Prefill chunk size in tokens. The initial forward over a long prompt
/// is split into windows of this many tokens, each with a monotonically
/// growing offset, so activation memory is bounded by chunk × layers ×
/// hidden instead of prompt × layers × hidden. The default (512) keeps
/// activation peaks under ~1 GiB on a 27B Qwen-class model while
/// keeping the per-step overhead negligible vs. one big prefill.
fn prefill_chunk_tokens() -> usize {
    env_usize("NEURON_PREFILL_CHUNK_TOKENS", 512)
}

/// Maximum allowed prompt length, in tokens. Requests above this are
/// rejected with [`InferenceError::PromptTooLong`] before any device
/// work — this is the explicit upper bound on context size, separate
/// from the model's `max_position_embeddings` (which can be much
/// larger than what fits in VRAM in practice).
fn max_prompt_tokens() -> usize {
    env_usize("NEURON_MAX_PROMPT_TOKENS", 16384)
}

/// Minimum free VRAM (MiB) required to even attempt a prefill. Requests
/// below this are rejected with [`InferenceError::InsufficientVram`]
/// before any device work. Acts as a backstop when concurrent requests
/// have eaten the headroom; intentionally conservative — a request
/// that gets past this can still OOM, but the rejection is a clean 503
/// rather than a poisoned context.
fn min_free_vram_mb() -> u64 {
    env_u64("NEURON_MIN_FREE_VRAM_MB", 1500)
}

/// Pre-flight check: reject the request if the prompt exceeds the
/// configured max, or if there isn't enough free VRAM to safely start a
/// prefill. Called from every chat_completion entry point right after
/// the VRAM query. A `prompt_len == 0` is accepted (some clients send
/// empty inputs to probe the endpoint); the prefill loop handles it.
/// Rough MiB of VRAM a vision prefill needs per 1000 prompt tokens
/// (accumulating KV cache + per-chunk activation headroom). Tunable;
/// the default is deliberately permissive so the guard rejects only
/// clearly-too-large requests, not ones the chunked prefill handles.
fn vision_prefill_mb_per_1k_tokens() -> u64 {
    env_u64("NEURON_VISION_PREFILL_MB_PER_1K_TOKENS", 500)
}

/// Fixed VRAM overhead (MiB) a vision prefill reserves on top of the
/// per-token estimate — image encode buffers + one chunk's activations.
fn vision_prefill_base_mb() -> u64 {
    env_u64("NEURON_VISION_PREFILL_BASE_MB", 2000)
}

/// Pre-flight check specific to vision prefills. Even with the chunked
/// prefill bounding per-step activation, the accumulating KV cache for
/// a long prompt can exhaust VRAM mid-forward — and on the TP path a
/// mid-forward OOM strands the NCCL collective (one rank dies, the other
/// hangs on the all-reduce, holding the pool lock). Reject up front with
/// a clean `InsufficientVram` when the estimated footprint exceeds free
/// VRAM, so a doomed request fails fast instead of hanging the daemon.
///
/// Heuristic and tunable (`NEURON_VISION_PREFILL_*`); the default errs
/// permissive. Skipped on the CPU sentinel (`vram_free_mb == 0`).
fn validate_vision_prefill(prompt_len: usize, vram_free_mb: u64) -> Result<(), InferenceError> {
    if vram_free_mb == 0 {
        return Ok(());
    }
    let required_mb = vision_prefill_base_mb()
        + (prompt_len as u64).saturating_mul(vision_prefill_mb_per_1k_tokens()) / 1000;
    if required_mb > vram_free_mb {
        return Err(InferenceError::InsufficientVram {
            free_mb: vram_free_mb,
            required_mb,
        });
    }
    Ok(())
}

fn validate_request(prompt_len: usize, vram_free_mb: u64) -> Result<(), InferenceError> {
    let max = max_prompt_tokens();
    if prompt_len > max {
        return Err(InferenceError::PromptTooLong { prompt_len, max });
    }
    // VRAM check is skipped on CPU loads (vram_free_mb == 0 sentinel)
    // because the (0, 0) reply from `query_vram` is also what a missing
    // worker returns. The CPU path has no per-GPU memory limit anyway —
    // host RAM is bounded by the OOM killer, not this check.
    let min = min_free_vram_mb();
    if vram_free_mb != 0 && vram_free_mb < min {
        return Err(InferenceError::InsufficientVram {
            free_mb: vram_free_mb,
            required_mb: min,
        });
    }
    Ok(())
}

/// Threshold above which `pool.lock().await` blocking is interesting
/// enough to warn about. Healthy concurrent requests serialise behind
/// the pool in single-digit ms — anything past 2 seconds is either a
/// huge in-flight prompt or, more often, a stuck request holding the
/// lock against a poisoned CUDA context. See the 2026-05-26 4-hour
/// silence on beast where dozens of requests piled up invisibly here.
#[cfg(feature = "cuda")]
const POOL_LOCK_WARN_THRESHOLD: Duration = Duration::from_secs(2);

/// Acquire the TP pool lock, emitting a warn-level breadcrumb if the
/// wait exceeds [`POOL_LOCK_WARN_THRESHOLD`]. Wrapped in a helper so
/// the warn happens at the call site — the request whose lock-wait is
/// slow is the one that knows its prompt_len and other context.
#[cfg(feature = "cuda")]
async fn acquire_pool_lock<'a>(
    pool: &'a tokio::sync::Mutex<super::tp::WorkerPool>,
    model_id: &str,
) -> tokio::sync::MutexGuard<'a, super::tp::WorkerPool> {
    let start = std::time::Instant::now();
    // Tick once at the threshold so a stuck request shows up in
    // journalctl even while it's still waiting. Without this the wait
    // looks like silence in the log right up until the lock is freed.
    tokio::pin! {
        let lock = pool.lock();
    }
    loop {
        tokio::select! {
            guard = &mut lock => {
                let elapsed = start.elapsed();
                if elapsed >= POOL_LOCK_WARN_THRESHOLD {
                    tracing::warn!(
                        model = %model_id,
                        waited_ms = elapsed.as_millis(),
                        "TP chat_completion: pool lock acquired after long wait"
                    );
                }
                return guard;
            }
            _ = tokio::time::sleep(POOL_LOCK_WARN_THRESHOLD) => {
                tracing::warn!(
                    model = %model_id,
                    waited_ms = start.elapsed().as_millis(),
                    "TP chat_completion: still waiting on pool lock"
                );
            }
        }
    }
}

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

/// Chunked prefill against an in-process [`ModelArch`]. Splits
/// `prompt_tokens` into [`prefill_chunk_tokens()`]-sized windows, runs
/// each through `arch.forward(chunk, offset)` with a monotonically
/// growing offset, and returns the last chunk's logits ready for
/// sampling. Bounds activation memory to O(chunk × layers × hidden)
/// instead of O(prompt × layers × hidden); the KV cache grows
/// monotonically so the model sees the full prompt at the final chunk.
fn chunked_prefill_local(
    arch: &mut ModelArch,
    device: &Device,
    prompt_tokens: &[u32],
) -> Result<Tensor> {
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        anyhow::bail!("chunked_prefill_local: empty prompt");
    }
    let chunk_size = prefill_chunk_tokens();
    let mut offset = 0;
    let mut last_logits: Option<Tensor> = None;
    while offset < prompt_len {
        let end = (offset + chunk_size).min(prompt_len);
        let chunk = &prompt_tokens[offset..end];
        let input = Tensor::new(chunk, device)?.unsqueeze(0)?;
        let logits = arch.forward(&input, offset)?;
        if end == prompt_len {
            last_logits = Some(logits);
        }
        offset = end;
    }
    last_logits.ok_or_else(|| anyhow::anyhow!("chunked_prefill_local: no chunks produced"))
}

/// Chunked prefill via the per-device worker. Same shape as
/// [`chunked_prefill_local`] but the forward runs on the worker thread
/// and replies with a CPU-side `Vec<f32>` of logits at the final
/// chunk's last position. Tensors never escape the worker.
#[cfg(feature = "cuda")]
async fn chunked_prefill_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
    prompt_tokens: &[u32],
) -> Result<Vec<f32>> {
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        anyhow::bail!("chunked_prefill_via_worker: empty prompt");
    }
    let chunk_size = prefill_chunk_tokens();
    let mut offset = 0;
    let mut last_logits: Option<Vec<f32>> = None;
    let total_chunks = prompt_len.div_ceil(chunk_size);
    let mut chunk_idx = 0_usize;
    while offset < prompt_len {
        let end = (offset + chunk_size).min(prompt_len);
        let chunk = prompt_tokens[offset..end].to_vec();
        let chunk_len = chunk.len();
        let step_start = std::time::Instant::now();
        let logits = worker
            .forward_logits(handle, chunk, offset)
            .await
            .map_err(|e| anyhow::anyhow!("prefill chunk {chunk_idx}/{total_chunks}: {e}"))?;
        tracing::debug!(
            chunk_idx,
            total_chunks,
            chunk_len,
            offset,
            elapsed_ms = step_start.elapsed().as_millis(),
            "chunked prefill (worker): chunk done"
        );
        if end == prompt_len {
            last_logits = Some(logits);
        }
        offset = end;
        chunk_idx += 1;
    }
    last_logits.ok_or_else(|| anyhow::anyhow!("chunked_prefill_via_worker: no chunks produced"))
}

/// Chunked prefill via the TP `WorkerPool`. Same shape as
/// [`chunked_prefill_via_worker`] but the forward fans out to every
/// rank via `pool.generate_step`. Returns the leader's CPU-side
/// `Vec<f32>` of logits at the final chunk's last position.
#[cfg(feature = "cuda")]
async fn chunked_prefill_tp(
    pool: &mut super::tp::WorkerPool,
    model_id: &str,
    leader_handle: super::device_worker::TpHandle,
    prompt_tokens: &[u32],
) -> Result<Vec<f32>> {
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        anyhow::bail!("chunked_prefill_tp: empty prompt");
    }
    let chunk_size = prefill_chunk_tokens();
    let mut offset = 0;
    let mut last_logits: Option<Vec<f32>> = None;
    let total_chunks = prompt_len.div_ceil(chunk_size);
    let mut chunk_idx = 0_usize;
    while offset < prompt_len {
        let end = (offset + chunk_size).min(prompt_len);
        let chunk = prompt_tokens[offset..end].to_vec();
        let chunk_len = chunk.len();
        let step_start = std::time::Instant::now();
        let logits = pool
            .generate_step(model_id, leader_handle, chunk, offset)
            .await
            .map_err(|e| anyhow::anyhow!("TP prefill chunk {chunk_idx}/{total_chunks}: {e}"))?;
        tracing::debug!(
            chunk_idx,
            total_chunks,
            chunk_len,
            offset,
            elapsed_ms = step_start.elapsed().as_millis(),
            "chunked prefill (TP): chunk done"
        );
        if end == prompt_len {
            last_logits = Some(logits);
        }
        offset = end;
        chunk_idx += 1;
    }
    last_logits.ok_or_else(|| anyhow::anyhow!("chunked_prefill_tp: no chunks produced"))
}

/// Per-scheme source after env-var resolution. The auth token is the
/// already-read env-var value (or None for anonymous access), and the
/// cache dir is the post-`resolve_hf_cache` path for the huggingface
/// scheme and the operator's literal value for everything else.
#[derive(Debug, Clone)]
struct ResolvedSource {
    endpoint: String,
    auth_token: Option<String>,
    cache_dir: Option<PathBuf>,
}

impl CandleHarness {
    /// Construct a new harness for `bind_url` using `config`. Resolves
    /// every configured source's auth env var and cache dir up front so
    /// the hot load path (`hf_api_for`) is a pure HashMap lookup.
    pub fn new(bind_url: String, config: &crate::config::CandleHarnessConfig) -> Self {
        let raw_sources = config.effective_sources();
        let default_source = config.effective_default_source().to_string();
        let mut sources = HashMap::with_capacity(raw_sources.len());
        for (scheme, src) in raw_sources.into_iter() {
            // Only the huggingface source gets the legacy
            // HF_HUB_CACHE/HF_HOME env-var fallback chain — other
            // schemes resolve to whatever the operator typed.
            let cache_dir = if scheme == crate::config::DEFAULT_SOURCE_SCHEME {
                resolve_hf_cache(src.cache_dir.clone())
            } else {
                src.cache_dir.clone()
            };
            let auth_token = src
                .auth_env
                .as_deref()
                .and_then(|var| std::env::var(var).ok())
                .filter(|v| !v.is_empty());
            if let Some(p) = &cache_dir {
                tracing::info!(
                    scheme = %scheme,
                    endpoint = %src.endpoint,
                    cache = %p.display(),
                    auth = auth_token.is_some(),
                    "candle harness source resolved"
                );
            } else {
                tracing::info!(
                    scheme = %scheme,
                    endpoint = %src.endpoint,
                    auth = auth_token.is_some(),
                    "candle harness source resolved (no cache dir; using hf-hub default)"
                );
            }
            sources.insert(
                scheme,
                ResolvedSource {
                    endpoint: src.endpoint,
                    auth_token,
                    cache_dir,
                },
            );
        }
        if !sources.contains_key(&default_source) {
            tracing::warn!(
                default_source,
                "configured default_source has no matching [harness.candle.sources.*] entry; \
                 bare model ids will fail to resolve until this is fixed"
            );
        }
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            sources,
            default_source,
            bind_url,
            device_workers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Scheme to substitute for bare `org/name` model ids. Mirrors the
    /// effective default from the operator's config, exposed for the
    /// load path's `ModelSourceId::with_default_scheme`.
    pub(crate) fn default_source_scheme(&self) -> &str {
        &self.default_source
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

    /// Return the worker handle for `device_index`, spawning it on
    /// first request. The handle is cached on `self` so subsequent
    /// loads against the same device share the same thread. Used to
    /// populate `LoadedModel::worker` and `TpLoadedModel::worker` at
    /// load time; in later refactor phases the worker also owns the
    /// `ModelArch` and `TpLeaderModel` slabs.
    #[allow(dead_code)]
    async fn ensure_device_worker(
        &self,
        device_index: u32,
    ) -> Result<Arc<super::device_worker::DeviceWorkerHandle>> {
        {
            let workers = self.device_workers.read().await;
            if let Some(w) = workers.get(&device_index) {
                return Ok(Arc::clone(w));
            }
        }
        // Write-lock acquired separately so the read path stays cheap.
        // The `get` is repeated under the write lock to handle the
        // race where two loads against a fresh device land here at
        // once — the second caller sees the first's insertion and
        // skips the second spawn.
        let mut workers = self.device_workers.write().await;
        if let Some(w) = workers.get(&device_index) {
            return Ok(Arc::clone(w));
        }
        let handle = super::device_worker::DeviceWorkerHandle::spawn(device_index)
            .with_context(|| format!("spawn device worker for cuda:{device_index}"))?;
        workers.insert(device_index, Arc::clone(&handle));
        tracing::info!(device_index, "spawned device worker");
        Ok(handle)
    }

    /// Build an hf-hub API client for the given scheme. The scheme
    /// must be present in the operator's configured `sources` table
    /// (the synth `huggingface` entry counts). Each source carries its
    /// own endpoint, optional bearer token, and cache directory, so
    /// the same `org/name` served by two registries cannot collide on
    /// disk.
    pub(crate) fn hf_api_for(&self, scheme: &str) -> Result<hf_hub::api::tokio::Api> {
        let src = self.sources.get(scheme).ok_or_else(|| {
            let mut configured: Vec<&str> = self.sources.keys().map(String::as_str).collect();
            configured.sort();
            anyhow::anyhow!(
                "no source configured for scheme '{scheme}'; \
                 configured: {configured:?}. Add a \
                 [harness.candle.sources.{scheme}] block to neuron.toml \
                 with endpoint = '...'."
            )
        })?;
        let mut builder = hf_hub::api::tokio::ApiBuilder::new().with_endpoint(src.endpoint.clone());
        if let Some(cache) = &src.cache_dir {
            builder = builder.with_cache_dir(cache.clone());
        }
        if let Some(token) = &src.auth_token {
            builder = builder.with_token(Some(token.clone()));
        }
        builder
            .build()
            .with_context(|| format!("build hf-hub API for scheme '{scheme}'"))
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
        source_id: &cortex_core::source::ModelSourceId,
    ) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
        let api = self.hf_api_for(&source_id.scheme)?;
        let repo = api.model(source_id.repo_path());
        let display_id = source_id.to_string();
        let _ = spec; // reserved for future use (quant-aware filtering)

        let config_path = repo
            .get("config.json")
            .await
            .with_context(|| format!("fetch config.json from {display_id}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .with_context(|| format!("fetch tokenizer.json from {display_id}"))?;

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
        source_id: &cortex_core::source::ModelSourceId,
        device: &Device,
    ) -> Result<(PathBuf, ModelArch)> {
        let (gguf_path, tokenizer_path) = self.resolve_files(spec, source_id).await?;
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

            // The `general.architecture` GGUF metadata key follows
            // llama.cpp conventions (lowercase, no underscores in some
            // cases) — `qwen3moe`, not `qwen3_moe`.
            match architecture.as_str() {
                "qwen3" => {
                    let weights =
                        QuantizedQwen3Weights::from_gguf(content, &mut file, &device_for_load)
                            .map_err(|e| anyhow::anyhow!("from_gguf qwen3: {e}"))?;
                    Ok(ModelArch::Qwen3Quantized(weights))
                }
                "qwen3moe" => {
                    // GGUFQWenMoE takes an explicit compute dtype
                    // alongside the device — F16 matches the GGUF
                    // weights' typical accumulation precision and
                    // gives the best tokens/sec on consumer cards.
                    let weights =
                        GGUFQWenMoE::from_gguf(content, &mut file, &device_for_load, DType::F16)
                            .map_err(|e| anyhow::anyhow!("from_gguf qwen3_moe: {e}"))?;
                    Ok(ModelArch::Qwen3MoeQuantized(weights))
                }
                "llama" => {
                    let weights =
                        QuantizedLlamaWeights::from_gguf(content, &mut file, &device_for_load)
                            .map_err(|e| anyhow::anyhow!("from_gguf llama: {e}"))?;
                    Ok(ModelArch::LlamaQuantized(weights))
                }
                other => anyhow::bail!(
                    "unsupported GGUF architecture '{other}'; quantized path supports \
                     qwen3, qwen3moe, llama"
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
        source_id: &cortex_core::source::ModelSourceId,
        device: &Device,
    ) -> Result<(PathBuf, ModelArch)> {
        let (config_path, tokenizer_path, safetensors_paths) =
            self.resolve_dense_files(spec, source_id).await?;
        let device_for_load = device.clone();
        let model_id_for_log = spec.model_id.clone();

        let arch = tokio::task::spawn_blocking(move || -> Result<ModelArch> {
            let cfg_text = std::fs::read_to_string(&config_path).context("read config.json")?;
            check_dense_config_supported(&cfg_text, &model_id_for_log)?;
            // Peek at model_type to choose the family before the
            // typed deserialize — each family has its own Config.
            let model_type = serde_json::from_str::<serde_json::Value>(&cfg_text)
                .ok()
                .as_ref()
                .and_then(|v| v.get("model_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            tracing::info!(
                model = %model_id_for_log,
                model_type = %model_type,
                shards = safetensors_paths.len(),
                "loading dense model from safetensors"
            );

            // bf16 is the canonical distribution dtype for Qwen3 /
            // Llama 3 / Qwen3 MoE. CUDA on Ada+ has hardware bf16;
            // Ampere has it too. CPU emulates.
            let dtype = DType::BF16;
            // SAFETY: VarBuilder::from_mmaped_safetensors mmaps the files;
            // mutation by another process while we hold the mapping is
            // UB. We trust the HF cache is immutable-by-design.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&safetensors_paths, dtype, &device_for_load)
                    .context("build VarBuilder over safetensors")?
            };

            match model_type.as_str() {
                "qwen3" => {
                    let cfg: qwen3_dense::Config =
                        serde_json::from_str(&cfg_text).context("parse Qwen3 config.json")?;
                    let model = qwen3_dense::ModelForCausalLM::new(&cfg, vb)
                        .map_err(|e| anyhow::anyhow!("build Qwen3 dense model: {e}"))?;
                    Ok(ModelArch::Qwen3Dense(model))
                }
                "qwen3_moe" => {
                    let cfg: qwen3_moe_dense::Config =
                        serde_json::from_str(&cfg_text).context("parse Qwen3 MoE config.json")?;
                    let model = qwen3_moe_dense::ModelForCausalLM::new(&cfg, vb)
                        .map_err(|e| anyhow::anyhow!("build Qwen3 MoE dense model: {e}"))?;
                    Ok(ModelArch::Qwen3MoeDense(model))
                }
                "llama" => {
                    let cfg: llama_dense::LlamaConfig =
                        serde_json::from_str(&cfg_text).context("parse Llama config.json")?;
                    // Llama has multiple sub-variants (Llama 1 has no
                    // GQA; Llama 3 does). `LlamaConfig::into_config`
                    // resolves the right shape; the `use_flash_attn`
                    // arg defaults to false — the flash kernel is a
                    // separate feature flag and uses extra VRAM.
                    let config = cfg.into_config(false);
                    let cache = llama_dense::Cache::new(true, dtype, &config, &device_for_load)
                        .context("build Llama Cache")?;
                    let model = llama_dense::Llama::load(vb, &config)
                        .map_err(|e| anyhow::anyhow!("build Llama dense model: {e}"))?;
                    Ok(ModelArch::LlamaDense(Box::new(LlamaDense {
                        model,
                        cache,
                        config,
                        dtype,
                        device: device_for_load,
                    })))
                }
                "qwen3_5" => {
                    // Qwen3-Next needs a ShardedVarBuilder because its
                    // load functions use the sharded backend (so they
                    // can be reused unchanged by the future TP variant).
                    // With world_size=1 the backend falls through to
                    // the unsharded path, so there is no per-load cost.
                    let cfg: super::arch::qwen3_5::Config = serde_json::from_str(&cfg_text)
                        .context("parse Qwen3-Next (qwen3_5) config.json")?;
                    let sharded_vb = unsafe {
                        candle_nn::var_builder::ShardedSafeTensors::var_builder(
                            &safetensors_paths,
                            dtype,
                            &device_for_load,
                        )
                        .context("build ShardedVarBuilder for Qwen3-Next")?
                    };
                    let model = super::arch::qwen3_5::Qwen3_5ForCausalLM::new(cfg, sharded_vb)
                        .context("build Qwen3-Next dense model")?;
                    Ok(ModelArch::Qwen3_5Dense(model))
                }
                other => {
                    // Defensive: `check_dense_config_supported` already
                    // gated on the supported set, so this branch is
                    // unreachable unless that list and the match here
                    // drift apart.
                    anyhow::bail!(
                        "unrouted supported model_type '{other}' — \
                         DENSE_SUPPORTED_MODEL_TYPES and load_arch_dense \
                         must stay in sync"
                    )
                }
            }
        })
        .await
        .context("blocking dense load task panicked")??;
        Ok((tokenizer_path, arch))
    }

    /// Resolve a model spec to local GGUF and tokenizer file paths via
    /// hf-hub. Downloads on first use; subsequent calls are cached.
    async fn resolve_files(
        &self,
        spec: &ModelSpec,
        source_id: &cortex_core::source::ModelSourceId,
    ) -> Result<(PathBuf, PathBuf)> {
        let api = self.hf_api_for(&source_id.scheme)?;
        let repo_path = source_id.repo_path();
        let repo = api.model(repo_path.clone());
        let display_id = source_id.to_string();

        let info = repo
            .info()
            .await
            .with_context(|| format!("fetch HF repo info for {display_id}"))?;

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
                    "no GGUF file matching quant {:?} in repo {display_id}",
                    spec.quant,
                )
            })?
            .to_string();

        tracing::info!(
            model = %display_id,
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
        // non-GGUF model_ids). Stripping happens on the repo_path
        // (scheme already accounted for) so this composes cleanly with
        // helexa-scheme GGUF repos too.
        let tokenizer_repo_path = repo_path
            .strip_suffix("-GGUF")
            .or_else(|| repo_path.strip_suffix("-gguf"))
            .unwrap_or(&repo_path)
            .to_string();
        let tokenizer_repo = if tokenizer_repo_path == repo_path {
            repo
        } else {
            tracing::debug!(
                from = %repo_path,
                to = %tokenizer_repo_path,
                "tokenizer.json sourced from base repo (GGUF suffix stripped)"
            );
            api.model(tokenizer_repo_path.clone())
        };
        let tokenizer_path = tokenizer_repo
            .get("tokenizer.json")
            .await
            .with_context(|| format!("fetch tokenizer.json from {tokenizer_repo_path}"))?;
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
        let handle = {
            let models = self.models.read().await;
            models.get(&request.model).cloned()
        };
        let handle = handle.ok_or_else(|| InferenceError::ModelNotLoaded(request.model.clone()))?;
        // The match is technically infallible without `cuda` (only Single
        // exists), but the cfg-gated Tp arm makes this the right shape
        // under both feature flags.
        #[allow(clippy::infallible_destructuring_match)]
        let loaded = match handle {
            LoadedHandle::Single(m) => m,
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => {
                return self.chat_completion_tp(m, request).await;
            }
        };

        // Span every line of this request with a short req_id +
        // model so `grep req_id=…` over the journal can reconstruct
        // one request even when dozens overlap. Add a terminal log
        // line on both success and failure — the single-GPU path
        // used to log nothing on either side, so a failing request
        // looked exactly like an idle neuron.
        let req_id = new_req_id();
        let model_id = request.model.clone();
        let span = tracing::info_span!("chat", req_id = %req_id, model = %model_id);
        let req_start = std::time::Instant::now();

        // Refuse the request up front if a prior inference poisoned
        // the device context — otherwise we hand the doomed forward
        // off to spawn_blocking and stall waiting for CUDA to fail.
        if loaded.poisoned.load(Ordering::Acquire) {
            let _g = span.enter();
            tracing::warn!("chat_completion: refusing request, model poisoned");
            return Err(poisoned_error(&model_id));
        }

        // Serialise concurrent requests against this model. Holds for
        // the duration of clear_kv_cache → prefill → decode so two
        // requests' chunked-prefill sequences can't interleave on the
        // shared KV cache (see `LoadedModel.inference_lock` for the
        // observed failure mode).
        let _inference_guard = loaded.inference_lock.lock().await;

        let result = async {
            let prompt = build_prompt_for_request(loaded.chat_template.as_deref(), &request);

            let encoding = loaded
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
            let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

            // Stage B: when the request carries images, preprocess
            // them, expand each `<|image_pad|>` sentinel to N copies
            // matching the per-image patch count, and route to the
            // vision-aware worker path. Non-image requests skip all
            // of this and follow the existing text-only flow.
            let vision_route = if request_has_images(&request) {
                // Stage B6: surface a structured `vision_unsupported`
                // rejection when the request asks for vision against a
                // text-only model. Cheap and stops the issue-#3 silent-
                // drop pattern.
                if !loaded.has_vision {
                    return Err(InferenceError::VisionUnsupported {
                        model_id: request.model.clone(),
                    });
                }
                let image_token_id = loaded
                    .image_token_id
                    .ok_or_else(|| InferenceError::VisionUnsupported {
                        model_id: request.model.clone(),
                    })?;
                let factor = loaded.image_grid_factor.ok_or_else(|| {
                    InferenceError::VisionUnsupported {
                        model_id: request.model.clone(),
                    }
                })?;
                let profile = super::preprocess::PreprocessProfile::qwen3_6();
                let images = extract_images_from_request(&request, &profile).map_err(|e| {
                    InferenceError::Other(anyhow::anyhow!("extract_images: {e}"))
                })?;
                if images.is_empty() {
                    // request_has_images said true but extract returned
                    // empty — defensive bail rather than silently dropping.
                    return Err(InferenceError::Other(anyhow::anyhow!(
                        "request has image content but extractor produced zero images"
                    )));
                }
                // Per-image LM token count from each image's resized grid
                // (#14 dynamic resolution; was a constant 196).
                let per_image_counts: Vec<usize> = images
                    .iter()
                    .map(|im| (im.h / factor) * (im.w / factor))
                    .collect();
                prompt_tokens =
                    expand_image_pad_tokens(&prompt_tokens, image_token_id, &per_image_counts)
                        .map_err(InferenceError::Other)?;
                Some((images, image_token_id))
            } else {
                None
            };

            let prompt_len = prompt_tokens.len();
            let temperature = request.temperature.unwrap_or(0.7);
            let top_p = request.top_p;
            let max_new = request.max_tokens.unwrap_or(8192) as usize;
            let seed = unix_subsec_nanos();

            let eos_id = loaded
                .tokenizer
                .token_to_id("<|im_end|>")
                .or_else(|| loaded.tokenizer.token_to_id("<|endoftext|>"));

            let (vram_free_mb, vram_total_mb) = loaded.query_vram().await;
            tracing::info!(
                prompt_len,
                max_new,
                temperature,
                ?top_p,
                ?eos_id,
                vram_free_mb,
                vram_total_mb,
                vision = vision_route.is_some(),
                "chat_completion: starting"
            );

            validate_request(prompt_len, vram_free_mb)?;
            if vision_route.is_some() {
                validate_vision_prefill(prompt_len, vram_free_mb)?;
            }
        if vision_route.is_some() {
            validate_vision_prefill(prompt_len, vram_free_mb)?;
        }

            // Routing: CUDA loads go through the per-device worker
            // thread (introduced in Phase 1; forward/clear added in
            // Phase 2). CPU loads keep the existing spawn_blocking
            // path because there's no context to own and the channel
            // round-trip would only add latency. The two arms produce
            // the same `(Vec<u32>, String)` shape so the rest of the
            // path is shared.
            let (generated_ids, finish_reason) = if let (Some(worker), Some(handle)) =
                (loaded.worker.as_ref(), loaded.arch_handle)
            {
                // Worker path (CUDA).
                #[cfg(feature = "cuda")]
                {
                    let result = match &vision_route {
                        Some((images, image_token_id)) => {
                            run_inference_with_images_via_worker(
                                worker,
                                handle,
                                &prompt_tokens,
                                images.clone(),
                                *image_token_id,
                                max_new,
                                temperature,
                                top_p,
                                seed,
                                eos_id,
                            )
                            .await
                        }
                        None => {
                            run_inference_via_worker(
                                worker,
                                handle,
                                &prompt_tokens,
                                max_new,
                                temperature,
                                top_p,
                                seed,
                                eos_id,
                            )
                            .await
                        }
                    };
                    match result {
                        Ok(v) => v,
                        Err(e) => {
                            let chain = format!("{e:#}");
                            if is_device_fault(&chain) {
                                loaded.poisoned.store(true, Ordering::Release);
                                tracing::warn!(
                                    error = %chain,
                                    "chat_completion: failed with device fault, model marked poisoned"
                                );
                            } else {
                                tracing::warn!(
                                    error = %chain,
                                    "chat_completion: failed (non-device fault); model NOT marked poisoned"
                                );
                            }
                            return Err(InferenceError::Other(e));
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    // Can't happen: `loaded.worker` is only Some on
                    // CUDA builds. The dead branch keeps the no-cuda
                    // build well-typed.
                    let _ = (worker, handle);
                    unreachable!("worker handle present without cuda feature");
                }
            } else if let Some(arch_arc) = loaded.arch.clone() {
                // CPU path: existing spawn_blocking on the local
                // Arc<Mutex<ModelArch>>.
                let device = loaded.device.clone();
                let inference_result =
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
                    .await;

                // Distinguish "inference returned Err" (almost always a
                // candle/CUDA failure that propagated through `?`, e.g.
                // an OOM or driver error — the context is unreliable,
                // poison the model) from "spawn_blocking task panicked
                // or was cancelled" (a Rust-level panic in the closure,
                // not a device fault; failing the one request without
                // tearing down the model for everyone else is correct).
                match inference_result {
                    Ok(Ok(v)) => v,
                    Ok(Err(e)) => {
                        let chain = format!("{e:#}");
                        if is_device_fault(&chain) {
                            loaded.poisoned.store(true, Ordering::Release);
                            tracing::warn!(
                                error = %chain,
                                "chat_completion: failed with device fault, model marked poisoned"
                            );
                        } else {
                            tracing::warn!(
                                error = %chain,
                                "chat_completion: failed (non-device fault); model NOT marked poisoned"
                            );
                        }
                        return Err(InferenceError::Other(e));
                    }
                    Err(join_err) => {
                        let cause = if join_err.is_panic() {
                            "panicked"
                        } else if join_err.is_cancelled() {
                            "was cancelled"
                        } else {
                            "ended abnormally"
                        };
                        tracing::error!(
                            cause,
                            error = %join_err,
                            "chat_completion: inference task {cause}; model NOT marked poisoned"
                        );
                        return Err(InferenceError::Other(anyhow::anyhow!(
                            "inference task {cause}: {join_err}"
                        )));
                    }
                }
            } else {
                // LoadedModel invariant: exactly one of `worker` /
                // `arch` is Some. Reaching here is a construction bug.
                return Err(InferenceError::Other(anyhow::anyhow!(
                    "LoadedModel has neither worker handle nor local arch — load-path bug"
                )));
            };

            let completion_text = loaded
                .tokenizer
                .decode(&generated_ids, true)
                .map_err(|e| InferenceError::Other(anyhow::anyhow!("detokenize: {e}")))?;

            let usage = Usage {
                prompt_tokens: prompt_len as u64,
                completion_tokens: generated_ids.len() as u64,
                total_tokens: (prompt_len + generated_ids.len()) as u64,
            };

            tracing::info!(
                prompt_tokens = prompt_len,
                completion_tokens = generated_ids.len(),
                finish_reason = %finish_reason,
                total_ms = req_start.elapsed().as_millis(),
                "chat_completion: done"
            );

            Ok::<_, InferenceError>(ChatCompletionResponse {
                id: format!("chatcmpl-{:x}", unix_subsec_nanos()),
                object: "chat.completion".into(),
                created: unix_now_secs(),
                model: request.model.clone(),
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
        .instrument(span.clone())
        .await;

        if let Err(ref e) = result {
            let _g = span.enter();
            tracing::error!(
                error = %format!("{e:#}"),
                total_ms = req_start.elapsed().as_millis(),
                "chat_completion: failed"
            );
        }
        result
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
        self.chat_completion_stream_with(request, wire_chat::ChatProjectionConfig::default())
            .await
    }

    /// Same as [`Self::chat_completion_stream`] but lets the caller
    /// pick the projection config — currently used by the HTTP
    /// handler to thread `x-include-thinking` from the request
    /// headers into [`wire_chat::ChatProjectionConfig::include_thinking`].
    pub async fn chat_completion_stream_with(
        &self,
        request: ChatCompletionRequest,
        mut config: wire_chat::ChatProjectionConfig,
    ) -> Result<mpsc::Receiver<ChatCompletionChunk>, InferenceError> {
        let stream = self.inference_stream(request).await?;
        // Fill in the model's reasoning markers if the caller
        // didn't pre-populate them — they're a property of the
        // loaded model (which the HTTP handler doesn't reach into
        // directly), not of the request.
        if config.reasoning_markers.is_none() {
            config.reasoning_markers = stream.reasoning_markers.clone();
        }
        Ok(wire_chat::project_chat_stream_with(
            stream.events,
            stream.id,
            stream.created,
            stream.model_id,
            config,
        ))
    }

    /// Streaming OpenAI Responses API entry point. Same harness
    /// output as [`Self::chat_completion_stream`], projected into
    /// the named-event SSE frames the Responses API client wants.
    /// `response_id` and `message_item_id` are stamped into every
    /// frame so the consumer can correlate.
    pub async fn responses_stream(
        &self,
        request: ChatCompletionRequest,
        response_id: String,
        message_item_id: String,
    ) -> Result<mpsc::Receiver<crate::wire::openai_responses::ResponseStreamFrame>, InferenceError>
    {
        let stream = self.inference_stream(request).await?;
        let meta = crate::wire::openai_responses::ResponseMeta {
            response_id,
            created_at: stream.created,
            model_id: stream.model_id,
            message_item_id,
        };
        Ok(crate::wire::openai_responses::project_responses_stream(
            stream.events,
            meta,
        ))
    }

    /// Format-agnostic streaming inference. Returns the raw
    /// [`InferenceEvent`] receiver plus the per-request metadata
    /// wire projectors stamp onto their frames. Lets every wire
    /// format land on the same harness output without duplicating
    /// setup / dispatch / spawn logic.
    async fn inference_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<InferenceStream, InferenceError> {
        let handle = {
            let models = self.models.read().await;
            models.get(&request.model).cloned()
        };
        let handle = handle.ok_or_else(|| InferenceError::ModelNotLoaded(request.model.clone()))?;
        // The match is technically infallible without `cuda` (only Single
        // exists), but the cfg-gated Tp arm makes this the right shape
        // under both feature flags.
        #[allow(clippy::infallible_destructuring_match)]
        let loaded = match handle {
            LoadedHandle::Single(m) => m,
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => {
                return self.inference_tp_stream(m, request).await;
            }
        };

        let prompt = build_prompt_for_request(loaded.chat_template.as_deref(), &request);
        let encoding = loaded
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Stage C1: vision routing for the streaming path. Mirrors the
        // non-streaming `chat_completion` block — detect image content,
        // reject it against text-only models, preprocess each image and
        // expand its `<|image_pad|>` sentinel to the per-image patch
        // count, then carry the payload through to a single-shot
        // image-spliced prefill. Non-image requests skip all of this.
        // Returning early here (before the `Start` event below) keeps a
        // rejected vision request from opening a half-formed SSE stream.
        let vision_route: Option<(Vec<super::device_worker::jobs::ImageInput>, u32)> =
            if request_has_images(&request) {
                if !loaded.has_vision {
                    return Err(InferenceError::VisionUnsupported {
                        model_id: request.model.clone(),
                    });
                }
                let image_token_id =
                    loaded
                        .image_token_id
                        .ok_or_else(|| InferenceError::VisionUnsupported {
                            model_id: request.model.clone(),
                        })?;
                let factor =
                    loaded
                        .image_grid_factor
                        .ok_or_else(|| InferenceError::VisionUnsupported {
                            model_id: request.model.clone(),
                        })?;
                let profile = super::preprocess::PreprocessProfile::qwen3_6();
                let images = extract_images_from_request(&request, &profile)
                    .map_err(|e| InferenceError::Other(anyhow::anyhow!("extract_images: {e}")))?;
                if images.is_empty() {
                    return Err(InferenceError::Other(anyhow::anyhow!(
                        "request has image content but extractor produced zero images"
                    )));
                }
                // Per-image LM token count from each image's resized grid (#14).
                let per_image_counts: Vec<usize> = images
                    .iter()
                    .map(|im| (im.h / factor) * (im.w / factor))
                    .collect();
                prompt_tokens =
                    expand_image_pad_tokens(&prompt_tokens, image_token_id, &per_image_counts)
                        .map_err(InferenceError::Other)?;
                Some((images, image_token_id))
            } else {
                None
            };

        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p;
        let max_new = request.max_tokens.unwrap_or(8192) as usize;
        let seed = unix_subsec_nanos();

        let eos_id = loaded
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| loaded.tokenizer.token_to_id("<|endoftext|>"));

        let device = loaded.device.clone();
        let tokenizer = loaded.tokenizer.clone();
        let model_id = request.model.clone();
        let id = format!("chatcmpl-{:x}", unix_subsec_nanos());
        let created = unix_now_secs();

        // Bounded channel so the producer (blocking inference) is back-
        // pressured by the consumer (SSE writer, via the wire
        // projector). 32 is generous — tokens arrive one at a time
        // and downstream consumption is async.
        let (tx, event_rx) = mpsc::channel::<InferenceEvent>(32);

        // Refuse if the model is already poisoned. No point opening
        // an SSE stream just to send the Start event and then bail.
        if loaded.poisoned.load(Ordering::Acquire) {
            return Err(poisoned_error(&model_id));
        }

        // Start event: tells the wire projector to emit its
        // format-specific "the assistant is about to speak" frame
        // (an OpenAI `delta: {role: "assistant"}` chunk here; a
        // `response.created` + `response.output_item.added` pair on
        // the Responses path). If sending fails the receiver is
        // already gone; bail before kicking off the heavy work.
        tx.send(InferenceEvent::Start)
            .await
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("client disconnected")))?;

        // Span context — spawn_blocking detaches from the async
        // executor so we capture the span explicitly and re-enter it
        // inside the closure to keep the req_id on every emitted line.
        let req_id = new_req_id();
        let span = tracing::info_span!("chat_stream", req_id = %req_id, model = %model_id);
        let prompt_len = prompt_tokens.len();
        let req_start = std::time::Instant::now();
        // Cloned `Arc<LoadedModel>` so the spawned task can mark the
        // model poisoned if its forward fails.
        let loaded_for_task = Arc::clone(&loaded);
        let span_for_starting = span.clone();
        let span_for_task = span.clone();
        // Query VRAM before entering the span so we don't await inside
        // an entered guard (Span::enter creates a synchronous guard
        // that can't span await points). The span gets entered in a
        // separate scope below purely for the log emission.
        let (vram_free_mb, vram_total_mb) = loaded.query_vram().await;
        {
            let _g = span_for_starting.enter();
            tracing::info!(
                prompt_len,
                max_new,
                temperature,
                ?top_p,
                ?eos_id,
                vram_free_mb,
                vram_total_mb,
                vision = vision_route.is_some(),
                "chat_completion (stream): starting"
            );
        }

        validate_request(prompt_len, vram_free_mb)?;
        if vision_route.is_some() {
            validate_vision_prefill(prompt_len, vram_free_mb)?;
        }

        // Routing parallel to the non-streaming chat_completion: CUDA
        // goes through the worker (async task), CPU keeps the
        // spawn_blocking + Arc<Mutex<ModelArch>> path. Both branches
        // acquire `loaded.inference_lock` from inside the spawned
        // task so concurrent stream requests against the same model
        // serialise at the request boundary (preventing the
        // chunked-prefill KV-cache interleave failure mode). The
        // role chunk was already sent above, so the client sees
        // immediate "stream open" feedback even when this request
        // queues behind another for the lock.
        if let (Some(worker), Some(handle)) = (loaded.worker.clone(), loaded.arch_handle) {
            #[cfg(feature = "cuda")]
            {
                let prompt_tokens = prompt_tokens.clone();
                let reasoning_tokens_inner = loaded.reasoning_tokens.clone();
                let tool_call_tokens_inner = loaded.tool_call_tokens.clone();
                tokio::spawn(
                    async move {
                        let _inference_guard = loaded_for_task.inference_lock.lock().await;
                        match stream_inference_via_worker(
                            worker,
                            handle,
                            tokenizer,
                            prompt_tokens,
                            vision_route,
                            max_new,
                            temperature,
                            top_p,
                            seed,
                            eos_id,
                            reasoning_tokens_inner,
                            tool_call_tokens_inner,
                            tx,
                        )
                        .await
                        {
                            Ok(_finish_reason) => tracing::info!(
                                prompt_tokens = prompt_len,
                                total_ms = req_start.elapsed().as_millis(),
                                "chat_completion (stream): done"
                            ),
                            Err(e) => {
                                let chain = format!("{e:#}");
                                if is_device_fault(&chain) {
                                    loaded_for_task.poisoned.store(true, Ordering::Release);
                                    tracing::error!(
                                        error = %chain,
                                        prompt_tokens = prompt_len,
                                        total_ms = req_start.elapsed().as_millis(),
                                        "chat_completion (stream): failed with device fault, model marked poisoned"
                                    );
                                } else {
                                    tracing::error!(
                                        error = %chain,
                                        prompt_tokens = prompt_len,
                                        total_ms = req_start.elapsed().as_millis(),
                                        "chat_completion (stream): failed (non-device fault); model NOT marked poisoned"
                                    );
                                }
                            }
                        }
                    }
                    .instrument(span_for_task),
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (worker, handle, span_for_task);
                unreachable!("worker handle present without cuda feature");
            }
        } else if let Some(arch_arc) = loaded.arch.clone() {
            let reasoning_tokens_inner = loaded.reasoning_tokens.clone();
            let tool_call_tokens_inner = loaded.tool_call_tokens.clone();
            tokio::task::spawn_blocking(move || {
                let _g = span_for_task.enter();
                // `blocking_lock` is safe here: spawn_blocking runs on
                // a dedicated thread, not on the async runtime, so
                // there's no executor to stall.
                let _inference_guard = loaded_for_task.inference_lock.blocking_lock();
                let mut guard = arch_arc.blocking_lock();
                match run_inference_streaming(
                    &mut guard,
                    &device,
                    &tokenizer,
                    &prompt_tokens,
                    max_new,
                    temperature,
                    top_p,
                    seed,
                    eos_id,
                    reasoning_tokens_inner.as_ref(),
                    tool_call_tokens_inner.as_ref(),
                    &tx,
                ) {
                    Ok(()) => tracing::info!(
                        prompt_tokens = prompt_len,
                        total_ms = req_start.elapsed().as_millis(),
                        "chat_completion (stream): done"
                    ),
                    Err(e) => {
                        let chain = format!("{e:#}");
                        if is_device_fault(&chain) {
                            loaded_for_task.poisoned.store(true, Ordering::Release);
                            tracing::error!(
                                error = %chain,
                                prompt_tokens = prompt_len,
                                total_ms = req_start.elapsed().as_millis(),
                                "chat_completion (stream): failed with device fault, model marked poisoned"
                            );
                        } else {
                            tracing::error!(
                                error = %chain,
                                prompt_tokens = prompt_len,
                                total_ms = req_start.elapsed().as_millis(),
                                "chat_completion (stream): failed (non-device fault); model NOT marked poisoned"
                            );
                        }
                    }
                }
            });
        } else {
            return Err(InferenceError::Other(anyhow::anyhow!(
                "LoadedModel has neither worker handle nor local arch — load-path bug"
            )));
        }

        // Hand the raw event channel back to the public entry
        // points (chat_completion_stream / responses_stream); they
        // pick the wire projection.
        let reasoning_markers = loaded.reasoning_tokens.clone();
        Ok(InferenceStream {
            events: event_rx,
            id,
            created,
            model_id,
            reasoning_markers,
        })
    }
}

/// The seam between inference (one shape, always) and wire formats
/// (many shapes, projector-per-format). Public so the format
/// projectors live outside the harness and the harness's
/// streaming-inference internals stay encapsulated.
pub struct InferenceStream {
    /// Stream of model-output events. Producers (the various
    /// inference loops) emit on this; consumers (wire projectors)
    /// read from it.
    pub events: mpsc::Receiver<InferenceEvent>,
    /// Request id stamped into every wire-format frame
    /// (`chatcmpl-…` for chat completions; the Responses path
    /// makes its own `resp_…` id separately and ignores this one).
    pub id: String,
    /// Unix seconds when inference began. Same field threads into
    /// every wire format's `created` / `created_at` slot.
    pub created: u64,
    /// Local model id (no endpoint prefix). Stamped into every
    /// wire-format frame so consumers can correlate.
    pub model_id: String,
    /// Open/close reasoning marker text (and token ids) for the
    /// loaded model, or `None` for non-reasoning models. Used by
    /// the chat-completions projector when `include_thinking` is
    /// set — the projector re-wraps reasoning content with the
    /// literal markers so client-side parsers (helexa-acp's
    /// `ThinkParser`) see the original on-the-wire shape.
    pub reasoning_markers: Option<ReasoningTokenPair>,
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
            .map(|h| ModelInfo {
                id: h.model_id().into(),
                harness: "candle".into(),
                status: if h.is_poisoned() {
                    "poisoned".into()
                } else {
                    "loaded".into()
                },
                devices: h.devices(),
                vram_used_mb: None,
                capabilities: h.capabilities(),
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

        // Parse the model id, substituting the harness's default
        // source for bare `org/name` entries so existing operator
        // configs keep working unchanged. Stored on the request-local
        // path so downstream resolve_* can ask the right registry.
        let source_id = spec
            .model_id
            .parse::<cortex_core::source::ModelSourceId>()
            .with_context(|| format!("parse model id '{}' as scheme:org/name", spec.model_id))?
            .with_default_scheme(self.default_source_scheme());

        // Preflight: classify the source repo and apply the
        // tp/quant/source feasibility table before any device
        // allocation, NCCL handshake, or weight fetch. Failures bubble
        // up as `super::preflight::PreflightError` wrapped in anyhow;
        // the api.rs handler downcasts to produce a 422 with structured
        // JSON. The plan it returns is not yet threaded through the
        // dispatch — downstream `resolve_files` / `resolve_dense_files`
        // re-run their own substring match — but the structured error
        // surface is the main payoff.
        let api = self.hf_api_for(&source_id.scheme)?;
        super::preflight::preflight(&api, &source_id, spec)
            .await
            .map_err(anyhow::Error::new)?;

        let tp_size = spec.tensor_parallel.unwrap_or(1);
        if tp_size > 1 {
            #[cfg(feature = "cuda")]
            {
                return self.load_tp(spec, &source_id, tp_size).await;
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!(
                    "tensor_parallel={tp_size} requested for '{}': this neuron \
                     binary was built without --features cuda; TP requires CUDA + NCCL",
                    spec.model_id
                );
            }
        }

        let devices = spec.devices.clone().unwrap_or_else(|| vec![0]);
        let device = Self::pick_device(&devices)?;

        // Phase 4: load directly on the worker thread for CUDA;
        // legacy spawn_blocking + Arc<Mutex<>> only for CPU. Resolve
        // hf-hub paths up front (always async), then either dispatch
        // a load Job (CUDA) or call the legacy local loader (CPU).
        let worker: Option<Arc<super::device_worker::DeviceWorkerHandle>> = match &device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Some(self.ensure_device_worker(devices[0]).await?),
            _ => None,
        };

        let (tokenizer_path, arch_local, arch_handle, vision_meta) = if let Some(w) = &worker {
            // CUDA path: resolve, then load in the worker.
            if spec.quant.is_some() {
                let (gguf_path, tokenizer_path) = self.resolve_files(spec, &source_id).await?;
                let handle = w
                    .load_gguf(gguf_path, spec.model_id.clone())
                    .await
                    .map_err(|e| anyhow::anyhow!("worker load_gguf: {e}"))?;
                // GGUF Qwen3.6 releases don't ship the vision tower
                // (Qwen-VL weights are in the dense safetensors only),
                // so a GGUF load is text-only by construction.
                (tokenizer_path, None, Some(handle), VisionMeta::default())
            } else {
                let (config_path, tokenizer_path, safetensors_paths) =
                    self.resolve_dense_files(spec, &source_id).await?;
                let meta = VisionMeta::from_config_path(&config_path);
                let handle = w
                    .load_dense(config_path, safetensors_paths, spec.model_id.clone())
                    .await
                    .map_err(|e| anyhow::anyhow!("worker load_dense: {e}"))?;
                (tokenizer_path, None, Some(handle), meta)
            }
        } else {
            // CPU path: legacy spawn_blocking + Arc<Mutex<ModelArch>>.
            let (tokenizer_path, arch) = if spec.quant.is_some() {
                self.load_arch_gguf(spec, &source_id, &device).await?
            } else {
                self.load_arch_dense(spec, &source_id, &device).await?
            };
            // CPU Qwen3.6 isn't a supported deployment target — the
            // 27B doesn't fit any reasonable CPU memory budget — so
            // we don't attempt to reach into the arch for vision
            // metadata. Stays text-only.
            (
                tokenizer_path,
                Some(Arc::new(Mutex::new(arch))),
                None,
                VisionMeta::default(),
            )
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        // Probe for reasoning markers in the tokenizer's
        // added-tokens table — `<think>` / `</think>` on Qwen3 +
        // DeepSeek-R1 + gpt-oss, `[THINK]` / `[/THINK]` on
        // Mistral Magistral, etc. `None` for non-reasoning models.
        // The streaming loop uses this to route between TextDelta
        // and ReasoningDelta without any hardcoded model
        // knowledge; wire projectors decide what to do with the
        // split.
        let reasoning_tokens = detect_reasoning_token_pair(|s| tokenizer.token_to_id(s));
        if let Some(ref pair) = reasoning_tokens {
            tracing::info!(
                model = %spec.model_id,
                open = %pair.open_text,
                close = %pair.close_text,
                open_id = pair.open_id,
                close_id = pair.close_id,
                "reasoning markers detected — streaming will route ReasoningDelta separately"
            );
        }
        let tool_call_tokens = detect_tool_call_token_pair(|s| tokenizer.token_to_id(s));
        if let Some(ref pair) = tool_call_tokens {
            tracing::info!(
                model = %spec.model_id,
                open = %pair.open_text,
                close = %pair.close_text,
                open_id = pair.open_id,
                close_id = pair.close_id,
                "tool-call markers detected — streaming will emit structured ToolCall events"
            );
        }
        // Probe `tokenizer_config.json` in the same snapshot dir.
        // When present and non-empty, the inference path renders
        // this Jinja template with the request's
        // `chat_template_kwargs` instead of using the hardcoded
        // ChatML formatter. Best-effort: missing or unparseable
        // configs silently fall through to the legacy path.
        let chat_template = super::chat_template::load_chat_template_alongside(&tokenizer_path);
        if chat_template.is_some() {
            tracing::info!(
                model = %spec.model_id,
                "chat_template loaded from tokenizer_config.json — prompt assembly will use the model's own template"
            );
        }

        let loaded = Arc::new(LoadedModel {
            model_id: spec.model_id.clone(),
            arch: arch_local,
            tokenizer,
            device,
            quant: spec.quant.clone(),
            devices,
            poisoned: AtomicBool::new(false),
            worker,
            arch_handle,
            inference_lock: tokio::sync::Mutex::new(()),
            reasoning_tokens,
            tool_call_tokens,
            chat_template,
            has_vision: vision_meta.has_vision,
            image_token_id: vision_meta.image_token_id,
            image_grid_factor: vision_meta.image_grid_factor,
        });

        let mut models = self.models.write().await;
        models.insert(spec.model_id.clone(), LoadedHandle::Single(loaded));
        tracing::info!(model = %spec.model_id, "model loaded");
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let removed = {
            let mut models = self.models.write().await;
            models.remove(model_id)
        };
        let Some(handle) = removed else {
            anyhow::bail!("model '{model_id}' not loaded");
        };
        // Single-GPU drops are immediate — the LoadedModel goes out of
        // scope with the Arc and candle frees VRAM. CUDA loads also
        // ship a `Job::DropArch` to the device worker so the boxed
        // `ModelArch` releases its CUDA allocations on the right
        // thread (with the bound context); without that, the Drop
        // would run on whatever tokio thread happens to be holding
        // the last `Arc<LoadedModel>` clone when this fn returns.
        // TP unloads further coordinate the subprocess pool below.
        match handle {
            LoadedHandle::Single(single) => {
                if let (Some(worker), Some(arch_handle)) =
                    (single.worker.as_ref(), single.arch_handle)
                    && let Err(e) = worker.drop_arch(arch_handle).await
                {
                    tracing::warn!(
                        model = %model_id,
                        error = %e,
                        "single-GPU unload: DropArch RPC failed (model state may leak in worker slab)"
                    );
                }
            }
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(tp) => {
                // Try to recover the inner TpLoadedModel so we can move
                // the pool and shut it down. If anyone else still holds
                // a clone of the Arc (shouldn't happen — the only owners
                // are the registry and any in-flight chat_completion),
                // bail with a clear marker rather than silently leaking.
                let tp = match Arc::try_unwrap(tp) {
                    Ok(t) => t,
                    Err(arc) => {
                        // Reinsert so we don't leave the registry in an
                        // inconsistent state.
                        let mut models = self.models.write().await;
                        models.insert(model_id.into(), LoadedHandle::Tp(arc));
                        anyhow::bail!("cannot unload '{model_id}': inference still in flight");
                    }
                };
                // Drop the leader's TpLeaderModel on the device worker
                // thread (CUDA tensors and Arc<Comm> clones release on
                // the same OS thread that allocated them).
                if let Err(e) = tp.worker.drop_tp(tp.leader_handle).await {
                    tracing::warn!(
                        model = %model_id,
                        error = %e,
                        "TP unload: DropTp RPC failed (leader model may leak in worker slab)"
                    );
                }
                let mut pool = tp.pool.into_inner();
                if let Err(e) = pool.unload_model(model_id).await {
                    tracing::warn!(model = %model_id, error = %e, "TP unload RPC failed");
                }
                if let Err(e) = pool.shutdown().await {
                    tracing::warn!(model = %model_id, error = %e, "TP pool shutdown failed");
                }
            }
        }
        tracing::info!(model = %model_id, "model unloaded");
        Ok(())
    }

    async fn inference_endpoint(&self, model_id: &str) -> Option<String> {
        let models = self.models.read().await;
        models.contains_key(model_id).then(|| self.bind_url.clone())
    }
}

impl CandleHarness {
    /// Tensor-parallel load. Resolves dense safetensors via hf-hub the
    /// same way the single-GPU dense path does, spins up a TP worker
    /// pool sized to `tp_size`, runs the NCCL handshake, then has
    /// every rank load its shard of the model.
    ///
    /// `spec.devices` carries the per-rank CUDA device indices (one
    /// entry per rank, in rank order); defaults to `0..tp_size`.
    #[cfg(feature = "cuda")]
    async fn load_tp(
        &self,
        spec: &ModelSpec,
        source_id: &cortex_core::source::ModelSourceId,
        tp_size: u32,
    ) -> Result<()> {
        use std::sync::Arc as StdArc;
        use tokio::sync::Mutex as TMutex;

        // Default per-rank device assignment: 0, 1, ..., tp_size - 1.
        let devices = spec
            .devices
            .clone()
            .unwrap_or_else(|| (0..tp_size).collect());
        if devices.len() as u32 != tp_size {
            anyhow::bail!(
                "tensor_parallel={tp_size} requires {tp_size} entries in devices, got {}",
                devices.len()
            );
        }
        // `quant` on the TP path now means in-situ quantization (ISQ):
        // load safetensors, quantize the per-rank shard to the named
        // GgmlDType at load time. The worker's parse_quant_string
        // accepts the same names (q5k, q8_0, etc.) as the single-GPU
        // path. GGUF-source-file models still aren't TP-loadable, but
        // resolve_dense_files only looks for safetensors so that path
        // errors out cleanly later if no safetensors are present.

        // 1. Resolve config + tokenizer + safetensors via hf-hub.
        let (config_path, tokenizer_path, safetensors_paths) =
            self.resolve_dense_files(spec, source_id).await?;
        let config_json = std::fs::read_to_string(&config_path).context("read config.json")?;
        // Reject unsupported architectures *before* spawning the worker
        // pool and fanning out NCCL — otherwise we'd burn the pool
        // lifecycle on a load that's guaranteed to fail at deserialise
        // time inside every rank.
        check_dense_config_supported(&config_json, &spec.model_id)?;
        // The TP path knows how to ship and reconstruct a Qwen3 dense
        // shard (`tp_qwen3.rs`). Other architectures may pass the
        // single-GPU `check_dense_config_supported` check above but
        // have no TP-aware module — bail with a clear marker pointing
        // at the file the implementer needs to add. This keeps an
        // operator who sets `tensor_parallel=2` on a Llama model from
        // silently routing through `pool.load_dense_shard` (which
        // assumes Qwen3 config shape on the worker side) and producing
        // a confusing config-parse failure inside every rank.
        check_tp_arch_supported(&config_json, &spec.model_id)?;

        // 2. Spawn the worker pool. Rank 0 stays in-process; ranks
        //    1..tp_size are subprocesses, one per device after the
        //    leader's own. The leader's device worker thread is
        //    spawned (or reused) here and passed into the pool so
        //    `init_nccl`, the load, every TP forward, and KV-cache
        //    clears all dispatch from the same OS thread.
        let exe = std::env::current_exe().context("resolve current_exe for worker spawn")?;
        let leader_worker = self.ensure_device_worker(devices[0]).await?;
        let mut pool =
            super::tp::WorkerPool::spawn(&exe, tp_size, &devices, leader_worker.clone()).await?;

        // 3. NCCL handshake across all ranks.
        let leader_device_idx = devices[0];
        pool.init_nccl(leader_device_idx).await?;

        // 4. Pick the leader's candle Device (same index as init_nccl).
        let leader_device = candle_core::Device::new_cuda(leader_device_idx as usize)
            .context("Device::new_cuda for TP leader")?;

        // 5. Load this rank's shard on every rank. After Phase 3
        //    `load_dense_shard` transfers the freshly-built
        //    `TpLeaderModel` into the device worker's TP slab and
        //    returns the resulting handle.
        let leader_handle = pool
            .load_dense_shard(
                &spec.model_id,
                &config_json,
                &safetensors_paths,
                &leader_device,
                candle_core::DType::BF16,
                spec.quant.clone(),
            )
            .await?;

        // 6. Tokenizer (same as single-GPU path).
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        // Reasoning + tool-call marker probes — identical to the
        // single-GPU path. See LoadedModel's matching fields for
        // the why.
        let reasoning_tokens = detect_reasoning_token_pair(|s| tokenizer.token_to_id(s));
        if let Some(ref pair) = reasoning_tokens {
            tracing::info!(
                model = %spec.model_id,
                open = %pair.open_text,
                close = %pair.close_text,
                "TP load: reasoning markers detected"
            );
        }
        let tool_call_tokens = detect_tool_call_token_pair(|s| tokenizer.token_to_id(s));
        if let Some(ref pair) = tool_call_tokens {
            tracing::info!(
                model = %spec.model_id,
                open = %pair.open_text,
                close = %pair.close_text,
                "TP load: tool-call markers detected"
            );
        }
        let chat_template = super::chat_template::load_chat_template_alongside(&tokenizer_path);
        if chat_template.is_some() {
            tracing::info!(
                model = %spec.model_id,
                "TP load: chat_template loaded from tokenizer_config.json"
            );
        }

        // Vision metadata from the same config.json the shards loaded
        // from. The TP model builder (Stage 1) materialises a replicated
        // vision tower on every rank when `vision_config` is present, so
        // `has_vision` here is consistent with what each rank loaded.
        let vision_meta = VisionMeta::from_config_path(&config_path);
        if vision_meta.has_vision {
            tracing::info!(
                model = %spec.model_id,
                image_token_id = ?vision_meta.image_token_id,
                image_grid_factor = ?vision_meta.image_grid_factor,
                "TP load: vision tower present, advertising vision capability"
            );
        }

        let tp_loaded = StdArc::new(TpLoadedModel {
            model_id: spec.model_id.clone(),
            tokenizer,
            devices: devices.clone(),
            pool: TMutex::new(pool),
            leader_handle,
            leader_device: leader_device.clone(),
            poisoned: AtomicBool::new(false),
            // Same `leader_worker` we passed into the pool above —
            // single `Arc` shared between WorkerPool and
            // TpLoadedModel so they reference the same thread.
            worker: leader_worker,
            reasoning_tokens,
            tool_call_tokens,
            chat_template,
            has_vision: vision_meta.has_vision,
            image_token_id: vision_meta.image_token_id,
            image_grid_factor: vision_meta.image_grid_factor,
        });

        let mut models = self.models.write().await;
        models.insert(spec.model_id.clone(), LoadedHandle::Tp(tp_loaded));
        tracing::info!(
            model = %spec.model_id,
            tp_size,
            ?devices,
            "TP model loaded"
        );
        Ok(())
    }

    /// Non-streaming chat completion against a TP model.
    ///
    /// The actual work runs inside a `tokio::spawn`'d task so the HTTP
    /// client disconnecting (curl timeout, browser nav-away, etc.)
    /// can't cancel the future mid-`pool.generate_step` and leave the
    /// worker subprocesses mid-RPC. If the spawned task is dropped,
    /// it still runs to completion and finishes draining the pool —
    /// the next inference request finds a clean pool. The HTTP layer
    /// just gives up on the response.
    ///
    /// Every step also emits `info`/`debug` tracing so journalctl
    /// shows where time went without needing to surface internals in
    /// the HTTP error response.
    #[cfg(feature = "cuda")]
    async fn chat_completion_tp(
        &self,
        tp: Arc<TpLoadedModel>,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, InferenceError> {
        // Tag every line of this request with a short req_id so a
        // grep over journalctl reconstructs one request even when
        // dozens are queued and interleaved. The span prefix is added
        // by the fmt subscriber to every event emitted within the
        // instrumented future, including events from `WorkerPool::*`
        // since those run on the leader's task.
        let req_id = new_req_id();
        let model_id = request.model.clone();
        let span = tracing::info_span!("tp_chat", req_id = %req_id, model = %model_id);
        let req_start = std::time::Instant::now();

        if tp.poisoned.load(Ordering::Acquire) {
            let _g = span.enter();
            tracing::warn!("TP chat_completion: refusing request, model poisoned");
            return Err(poisoned_error(&model_id));
        }

        // Reject image-bearing requests against a TP model with no
        // vision tower, cleanly (`vision_unsupported`) rather than
        // silently dropping the image. Vision-capable TP loads fall
        // through to the image-aware prefill in chat_completion_tp_inner.
        if request_has_images(&request) && !tp.has_vision {
            let _g = span.enter();
            tracing::warn!(
                "TP chat_completion: rejecting image request, model has no vision tower"
            );
            return Err(InferenceError::VisionUnsupported { model_id });
        }

        let tp_for_marker = Arc::clone(&tp);
        let handle = tokio::spawn(chat_completion_tp_inner(tp, request).instrument(span.clone()));
        match handle.await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => {
                // The inner task returned Err. Only poison when the
                // failure indicates a CUDA / NCCL driver fault — shape
                // mismatches, NaN logits, tokenizer errors etc. don't
                // touch the device context and shouldn't take the
                // model down for everyone else.
                let chain = format!("{e:#}");
                let _g = span.enter();
                if matches!(&e, InferenceError::Other(inner) if is_device_fault(&format!("{inner:#}")))
                {
                    tp_for_marker.poisoned.store(true, Ordering::Release);
                    tracing::error!(
                        error = %chain,
                        total_ms = req_start.elapsed().as_millis(),
                        "TP chat_completion: failed with device fault, model marked poisoned"
                    );
                } else {
                    tracing::error!(
                        error = %chain,
                        total_ms = req_start.elapsed().as_millis(),
                        "TP chat_completion: failed (non-device fault); model NOT marked poisoned"
                    );
                }
                Err(e)
            }
            Err(join_err) => {
                // JoinError: the spawned task panicked or was cancelled.
                // Tokenizer / sampling / serialisation panics don't touch
                // the device, so don't poison the model — failing this
                // one request is enough. (CUDA failures arrive as Err
                // through `?`, not as panics, and are handled above.)
                let cause = if join_err.is_panic() {
                    "panicked"
                } else if join_err.is_cancelled() {
                    "was cancelled"
                } else {
                    "ended abnormally"
                };
                let _g = span.enter();
                tracing::error!(
                    cause,
                    error = %join_err,
                    total_ms = req_start.elapsed().as_millis(),
                    "TP chat_completion: inference task {cause}; model NOT marked poisoned"
                );
                Err(InferenceError::Other(anyhow::anyhow!(
                    "TP inference task {cause}: {join_err}"
                )))
            }
        }
    }

    /// Streaming counterpart to `chat_completion_tp`. Same per-step
    /// orchestration (clear cache, prefill, sample, decode loop) but
    /// emits one `ChatCompletionChunk` per token over an mpsc channel
    /// so the handler can write an SSE stream.
    ///
    /// Unlike the single-GPU streaming path (which runs the candle
    /// forward inside `spawn_blocking` and uses `blocking_send`), the
    /// TP loop is itself async — every `pool.generate_step` awaits the
    /// leader's spawn_blocking forward plus every worker's recv_only.
    /// So we `tokio::spawn` the orchestration task and use plain
    /// `Sender::send`.
    #[cfg(feature = "cuda")]
    async fn inference_tp_stream(
        &self,
        tp: Arc<TpLoadedModel>,
        request: ChatCompletionRequest,
    ) -> Result<InferenceStream, InferenceError> {
        if tp.poisoned.load(Ordering::Acquire) {
            return Err(poisoned_error(&request.model));
        }

        // Reject image requests against a non-vision TP model before
        // opening the SSE stream. Vision-capable TP loads fall through
        // to the image-aware prefill in the orchestration task below.
        if request_has_images(&request) && !tp.has_vision {
            tracing::warn!(
                "TP chat_completion (stream): rejecting image request, model has no vision tower"
            );
            return Err(InferenceError::VisionUnsupported {
                model_id: request.model.clone(),
            });
        }

        let prompt = build_prompt_for_request(tp.chat_template.as_deref(), &request);
        let encoding = tp
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        // TP-vision (streaming): same detection + pad expansion as the
        // non-streaming path. The resulting `vision_route` moves into
        // the orchestration task, which runs a single-shot image prefill
        // when present. Returning early here keeps a rejected request
        // from opening the SSE stream.
        let vision_route: Option<(Vec<String>, u32)> = if request_has_images(&request) {
            if !tp.has_vision {
                return Err(InferenceError::VisionUnsupported {
                    model_id: request.model.clone(),
                });
            }
            let image_token_id =
                tp.image_token_id
                    .ok_or_else(|| InferenceError::VisionUnsupported {
                        model_id: request.model.clone(),
                    })?;
            let factor = tp
                .image_grid_factor
                .ok_or_else(|| InferenceError::VisionUnsupported {
                    model_id: request.model.clone(),
                })?;
            let data_uris = extract_image_data_uris(&request);
            if data_uris.is_empty() {
                return Err(InferenceError::Other(anyhow::anyhow!(
                    "request has image content but extractor produced zero data URIs"
                )));
            }
            // Per-image LM token count from each image's resized grid (#14).
            // Decode header + smart_resize only; the workers re-derive the
            // same dims when they preprocess for the replicated tower.
            let profile = super::preprocess::PreprocessProfile::qwen3_6();
            let per_image_counts: Vec<usize> = data_uris
                .iter()
                .enumerate()
                .map(|(i, uri)| {
                    let (h, w) =
                        super::preprocess::resized_dims_for_uri(uri, &profile).map_err(|e| {
                            InferenceError::Other(anyhow::anyhow!("resized_dims image #{i}: {e}"))
                        })?;
                    Ok::<usize, InferenceError>((h as usize / factor) * (w as usize / factor))
                })
                .collect::<Result<Vec<_>, _>>()?;
            prompt_tokens =
                expand_image_pad_tokens(&prompt_tokens, image_token_id, &per_image_counts)
                    .map_err(InferenceError::Other)?;
            Some((data_uris, image_token_id))
        } else {
            None
        };

        let prompt_len = prompt_tokens.len();

        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p;
        let max_new = request.max_tokens.unwrap_or(8192) as usize;
        let seed = unix_subsec_nanos();

        let eos_id = tp
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tp.tokenizer.token_to_id("<|endoftext|>"));

        let model_id = request.model.clone();
        let id = format!("chatcmpl-{:x}", unix_subsec_nanos());
        let created = unix_now_secs();
        let tokenizer = tp.tokenizer.clone();
        let reasoning_tokens = tp.reasoning_tokens.clone();
        let tool_call_tokens = tp.tool_call_tokens.clone();
        // The spawned orchestration task below consumes both `id`
        // and `model_id` (tracing, pool lookups, NCCL ops use them
        // heavily). The wire projector at the bottom of this fn
        // also needs them to stamp request metadata onto every
        // chunk. Clone here so each side owns its copy.
        let projector_id = id.clone();
        let projector_model_id = model_id.clone();

        // Bounded channel — back-pressures the producer when
        // downstream consumption (wire projector → SSE writer) is
        // slow.
        let (tx, event_rx) = mpsc::channel::<InferenceEvent>(32);

        // Start event first, before kicking off the heavy work — if
        // the receiver is gone by now there's no point starting
        // inference. The wire projector materialises this as the
        // OpenAI `delta: {role: "assistant"}` chunk.
        tx.send(InferenceEvent::Start)
            .await
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("client disconnected")))?;

        // The orchestration task. Holds the pool lock for the lifetime
        // of this inference; concurrent requests against the same TP
        // model serialise behind it.
        //
        // Tagged with the same req_id span as the non-streaming path
        // so the journal can be reconstructed regardless of which API
        // surface the client hit.
        let req_id = new_req_id();
        let span = tracing::info_span!(
            "tp_chat_stream",
            req_id = %req_id,
            model = %model_id
        );
        let req_start = std::time::Instant::now();
        let (vram_free_mb, vram_total_mb) = tp.query_vram().await;
        tracing::info!(
            parent: &span,
            prompt_len,
            max_new,
            temperature,
            ?top_p,
            ?eos_id,
            vram_free_mb,
            vram_total_mb,
            "TP chat_completion (stream): starting"
        );

        validate_request(prompt_len, vram_free_mb)?;
        if vision_route.is_some() {
            validate_vision_prefill(prompt_len, vram_free_mb)?;
        }

        let tp_for_task = Arc::clone(&tp);
        tokio::spawn(
            async move {
                let mut failure: Option<String> = None;
                let mut pool = acquire_pool_lock(&tp_for_task.pool, &model_id).await;
                let leader_handle = tp_for_task.leader_handle;

                let mut all_tokens: Vec<u32> = Vec::new();
                // Incremental detokenizer. See the equivalent in
                // `stream_inference_via_worker` for the why: the old
                // "full decode + byte-slice delta" pattern panicked on
                // UTF-8 mid-codepoint boundaries when BPE byte-fallback
                // split a multi-byte char across tokens.
                let mut decode_stream = tokenizer.decode_stream(true);
                let mut finish_reason = FinishReason::Length;
                // Reasoning + tool-call state machines — same as
                // the single-GPU path. The TP path needs its own
                // copies because the spawn closure owns them.
                let mut in_reasoning = false;
                let mut in_tool_call = false;
                let mut tool_call_buf = String::new();
                let mut tool_call_idx: usize = 0;

                'work: {
                    if let Err(e) = pool.clear_kv_cache(&model_id, leader_handle).await {
                        failure = Some(format!("clear_kv_cache: {e:#}"));
                        break 'work;
                    }

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

                    // Chunked prefill — see `chunked_prefill_tp`. Each
                    // chunk fans out to every rank with a growing
                    // offset; only the final chunk's logits are kept
                    // for the first sample.
                    // Vision requests do a chunked image prefill (encode
                    // once, splice per chunk); text requests chunk it the
                    // same way. `vision_route` was moved into this task
                    // from the synchronous setup above.
                    let prefill_result = match &vision_route {
                        Some((data_uris, image_token_id)) => {
                            pool.generate_step_with_images(
                                &model_id,
                                leader_handle,
                                prompt_tokens.clone(),
                                0,
                                *image_token_id,
                                data_uris.clone(),
                                prefill_chunk_tokens(),
                            )
                            .await
                        }
                        None => {
                            chunked_prefill_tp(&mut pool, &model_id, leader_handle, &prompt_tokens)
                                .await
                        }
                    };
                    let logits_vec = match prefill_result {
                        Ok(l) => l,
                        Err(e) => {
                            failure = Some(format!("prefill: {e:#}"));
                            break 'work;
                        }
                    };
                    let (post_prefill_vram_free_mb, _) = tp_for_task.query_vram().await;
                    tracing::info!(
                        model = %model_id,
                        prompt_len,
                        vram_free_mb = post_prefill_vram_free_mb,
                        "TP chat_completion (stream): prefill complete"
                    );
                    let logits = match Tensor::new(logits_vec.as_slice(), &Device::Cpu) {
                        Ok(t) => t,
                        Err(e) => {
                            failure = Some(format!("prefill build cpu logits: {e:#}"));
                            break 'work;
                        }
                    };
                    let mut next_token =
                        match sample_with_penalty(&logits, &all_tokens, &mut logits_processor) {
                            Ok(t) => t,
                            Err(e) => {
                                let health = logits_health_slice(&logits_vec);
                                tracing::warn!(
                                    model = %model_id,
                                    ?health,
                                    "TP chat_completion (stream): prefill sample failed; logits unhealthy"
                                );
                                failure = Some(format!("prefill sample: {e:#}"));
                                break 'work;
                            }
                        };

                    if Some(next_token) == eos_id {
                        finish_reason = FinishReason::Stop;
                    } else {
                        all_tokens.push(next_token);
                        match handle_tool_call_marker(
                            next_token,
                            tool_call_tokens.as_ref(),
                            &mut in_tool_call,
                            &mut tool_call_buf,
                        ) {
                            ToolCallMarker::Enter => {}
                            ToolCallMarker::Exit { buffer } => {
                                let idx = tool_call_idx;
                                tool_call_idx += 1;
                                match parse_tool_call_body(&buffer, idx) {
                                    Some((id, name, arguments)) => {
                                        if tx
                                            .send(InferenceEvent::ToolCall {
                                                index: idx,
                                                id,
                                                name,
                                                arguments,
                                            })
                                            .await
                                            .is_err()
                                        {
                                            break 'work;
                                        }
                                    }
                                    None => {
                                        let open = tool_call_tokens
                                            .as_ref()
                                            .map(|p| p.open_text.as_str())
                                            .unwrap_or("<tool_call>");
                                        let close = tool_call_tokens
                                            .as_ref()
                                            .map(|p| p.close_text.as_str())
                                            .unwrap_or("</tool_call>");
                                        let raw = format!("{open}{buffer}{close}");
                                        if !emit_delta(&raw, &tx, in_reasoning).await {
                                            break 'work;
                                        }
                                    }
                                }
                            }
                            ToolCallMarker::None => {
                                if in_tool_call {
                                    match decode_stream.step(next_token) {
                                        Ok(Some(s)) => tool_call_buf.push_str(&s),
                                        Ok(None) => {}
                                        Err(e) => tracing::warn!(
                                            model = %model_id,
                                            error = %e,
                                            "TP stream: decode_stream step failed (in tool_call)"
                                        ),
                                    }
                                } else if handle_reasoning_marker(
                                    next_token,
                                    reasoning_tokens.as_ref(),
                                    &mut in_reasoning,
                                ) {
                                    // marker — nothing to emit
                                } else {
                                    match decode_stream.step(next_token) {
                                        Ok(Some(delta)) => {
                                            if !emit_delta(&delta, &tx, in_reasoning).await {
                                                break 'work;
                                            }
                                        }
                                        Ok(None) => {}
                                        Err(e) => tracing::warn!(
                                            model = %model_id,
                                            error = %e,
                                            "TP stream: decode_stream step failed"
                                        ),
                                    }
                                }
                            }
                        }

                        for index in 0..max_new.saturating_sub(1) {
                            let logits_vec = match pool
                                .generate_step(
                                    &model_id,
                                    leader_handle,
                                    vec![next_token],
                                    prompt_len + index,
                                )
                                .await
                            {
                                Ok(l) => l,
                                Err(e) => {
                                    failure = Some(format!("decode step {index}: {e:#}"));
                                    break 'work;
                                }
                            };
                            let logits = match Tensor::new(logits_vec.as_slice(), &Device::Cpu) {
                                Ok(t) => t,
                                Err(e) => {
                                    failure =
                                        Some(format!("decode build cpu logits {index}: {e:#}"));
                                    break 'work;
                                }
                            };
                            next_token = match sample_with_penalty(
                                &logits,
                                &all_tokens,
                                &mut logits_processor,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    let health = logits_health_slice(&logits_vec);
                                    tracing::warn!(
                                        model = %model_id,
                                        step = index,
                                        ?health,
                                        "TP chat_completion (stream): decode sample failed; logits unhealthy"
                                    );
                                    failure = Some(format!("decode sample {index}: {e:#}"));
                                    break 'work;
                                }
                            };
                            // Always await the query (even when the
                            // trace! is filtered out by RUST_LOG): the
                            // channel hop is ~tens of µs, comparable to
                            // the previous in-line bind+query cost, and
                            // making the call conditional adds complexity
                            // for negligible win. Revisit if it shows up
                            // in a hot-path profile.
                            let step_vram_free_mb = tp_for_task.query_vram().await.0;
                            tracing::trace!(
                                model = %model_id,
                                step = index,
                                next_token,
                                vram_free_mb = step_vram_free_mb,
                                "TP chat_completion (stream): decode step"
                            );
                            if Some(next_token) == eos_id {
                                finish_reason = FinishReason::Stop;
                                break;
                            }
                            all_tokens.push(next_token);
                            match handle_tool_call_marker(
                                next_token,
                                tool_call_tokens.as_ref(),
                                &mut in_tool_call,
                                &mut tool_call_buf,
                            ) {
                                ToolCallMarker::Enter => continue,
                                ToolCallMarker::Exit { buffer } => {
                                    let idx = tool_call_idx;
                                    tool_call_idx += 1;
                                    match parse_tool_call_body(&buffer, idx) {
                                        Some((id, name, arguments)) => {
                                            if tx
                                                .send(InferenceEvent::ToolCall {
                                                    index: idx,
                                                    id,
                                                    name,
                                                    arguments,
                                                })
                                                .await
                                                .is_err()
                                            {
                                                break 'work;
                                            }
                                        }
                                        None => {
                                            let open = tool_call_tokens
                                                .as_ref()
                                                .map(|p| p.open_text.as_str())
                                                .unwrap_or("<tool_call>");
                                            let close = tool_call_tokens
                                                .as_ref()
                                                .map(|p| p.close_text.as_str())
                                                .unwrap_or("</tool_call>");
                                            let raw = format!("{open}{buffer}{close}");
                                            if !emit_delta(&raw, &tx, in_reasoning).await {
                                                break 'work;
                                            }
                                        }
                                    }
                                    continue;
                                }
                                ToolCallMarker::None => {}
                            }
                            if in_tool_call {
                                match decode_stream.step(next_token) {
                                    Ok(Some(s)) => tool_call_buf.push_str(&s),
                                    Ok(None) => {}
                                    Err(e) => tracing::warn!(
                                        model = %model_id,
                                        error = %e,
                                        "TP stream: decode_stream step failed (in tool_call)"
                                    ),
                                }
                                continue;
                            }
                            if handle_reasoning_marker(
                                next_token,
                                reasoning_tokens.as_ref(),
                                &mut in_reasoning,
                            ) {
                                continue;
                            }
                            match decode_stream.step(next_token) {
                                Ok(Some(delta)) => {
                                    if !emit_delta(&delta, &tx, in_reasoning).await {
                                        break 'work;
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => tracing::warn!(
                                    model = %model_id,
                                    error = %e,
                                    "TP stream: decode_stream step failed"
                                ),
                            }
                        }
                    }
                }

                // One terminal line per request, success or failure. The
                // success branch was previously implicit (the SSE final
                // chunk went out and the spawned task just ended); now
                // there's always a log line for the operator.
                if let Some(err) = &failure {
                    if is_device_fault(err) {
                        tp_for_task.poisoned.store(true, Ordering::Release);
                        tracing::error!(
                            error = %err,
                            completion_tokens = all_tokens.len(),
                            total_ms = req_start.elapsed().as_millis(),
                            "TP chat_completion (stream): failed with device fault, model marked poisoned"
                        );
                    } else {
                        tracing::error!(
                            error = %err,
                            completion_tokens = all_tokens.len(),
                            total_ms = req_start.elapsed().as_millis(),
                            "TP chat_completion (stream): failed (non-device fault); model NOT marked poisoned"
                        );
                    }
                } else {
                    tracing::info!(
                        prompt_tokens = prompt_len,
                        completion_tokens = all_tokens.len(),
                        finish_reason = finish_reason.as_openai_str(),
                        total_ms = req_start.elapsed().as_millis(),
                        "TP chat_completion (stream): done"
                    );
                }

                // Finish event — only on the success path. On
                // failure we drop the channel so the client sees the
                // SSE stream end abruptly (matches the pre-refactor
                // behaviour when the failed-path early-returned
                // without a final chunk).
                if failure.is_none() {
                    let _ = tx
                        .send(InferenceEvent::Finish {
                            reason: finish_reason,
                        })
                        .await;
                }
            }
            .instrument(span),
        );

        // Hand the raw event channel back to the public entry
        // points; they pick the wire projection. Uses the clones
        // we stashed before the spawn — the originals were moved
        // into the orchestration task above.
        let reasoning_markers = tp.reasoning_tokens.clone();
        Ok(InferenceStream {
            events: event_rx,
            id: projector_id,
            created,
            model_id: projector_model_id,
            reasoning_markers,
        })
    }
}

/// Body of the TP non-streaming chat completion, hoisted out of
/// `CandleHarness::chat_completion_tp` so it can run inside
/// `tokio::spawn` (which requires a `'static` future) and survive
/// HTTP-layer cancellation.
///
/// Tracing strategy: `info` for request entry/exit so journalctl
/// always shows when an inference started and finished; `debug` for
/// per-step timing so an operator running with `RUST_LOG=debug` sees
/// where the request actually spends its time without needing to
/// instrument the model code.
#[cfg(feature = "cuda")]
async fn chat_completion_tp_inner(
    tp: Arc<TpLoadedModel>,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, InferenceError> {
    let req_start = std::time::Instant::now();
    let model_id = request.model.clone();

    let prompt = build_prompt_for_request(tp.chat_template.as_deref(), &request);
    let encoding = tp
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
    let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    // TP-vision: when the request carries images (and the model has a
    // replicated tower — enforced by the caller's guard), expand each
    // `<|image_pad|>` sentinel to the per-image patch count and carry
    // the source data URIs through to the single-shot image prefill.
    // Mirrors the single-GPU `chat_completion` vision_route block.
    let vision_route: Option<(Vec<String>, u32)> = if request_has_images(&request) {
        if !tp.has_vision {
            return Err(InferenceError::VisionUnsupported {
                model_id: request.model.clone(),
            });
        }
        let image_token_id =
            tp.image_token_id
                .ok_or_else(|| InferenceError::VisionUnsupported {
                    model_id: request.model.clone(),
                })?;
        let factor = tp
            .image_grid_factor
            .ok_or_else(|| InferenceError::VisionUnsupported {
                model_id: request.model.clone(),
            })?;
        let data_uris = extract_image_data_uris(&request);
        if data_uris.is_empty() {
            return Err(InferenceError::Other(anyhow::anyhow!(
                "request has image content but extractor produced zero data URIs"
            )));
        }
        // Per-image LM token count from each image's resized grid (#14).
        let profile = super::preprocess::PreprocessProfile::qwen3_6();
        let per_image_counts: Vec<usize> = data_uris
            .iter()
            .enumerate()
            .map(|(i, uri)| {
                let (h, w) =
                    super::preprocess::resized_dims_for_uri(uri, &profile).map_err(|e| {
                        InferenceError::Other(anyhow::anyhow!("resized_dims image #{i}: {e}"))
                    })?;
                Ok::<usize, InferenceError>((h as usize / factor) * (w as usize / factor))
            })
            .collect::<Result<Vec<_>, _>>()?;
        prompt_tokens = expand_image_pad_tokens(&prompt_tokens, image_token_id, &per_image_counts)
            .map_err(InferenceError::Other)?;
        Some((data_uris, image_token_id))
    } else {
        None
    };

    let prompt_len = prompt_tokens.len();

    let temperature = request.temperature.unwrap_or(0.7);
    let top_p = request.top_p;
    let max_new = request.max_tokens.unwrap_or(8192) as usize;
    let seed = unix_subsec_nanos();

    let eos_id = tp
        .tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tp.tokenizer.token_to_id("<|endoftext|>"));

    let (vram_free_mb, vram_total_mb) = tp.query_vram().await;
    tracing::info!(
        model = %model_id,
        prompt_len,
        max_new,
        temperature,
        ?top_p,
        ?eos_id,
        vram_free_mb,
        vram_total_mb,
        "TP chat_completion: starting"
    );

    validate_request(prompt_len, vram_free_mb)?;
    if vision_route.is_some() {
        validate_vision_prefill(prompt_len, vram_free_mb)?;
    }

    // Acquire the pool lock for the duration of the request. After
    // Phase 3 the leader's TpLeaderModel lives in the device worker
    // thread, so the pool lock now serialises only subprocess RPC
    // traffic — but holding it for the whole request still keeps
    // concurrent chat_completions against the same TP model from
    // interleaving prefill/decode jobs.
    let mut pool = acquire_pool_lock(&tp.pool, &model_id).await;
    let leader_handle = tp.leader_handle;

    // Reset every rank's KV cache so this request doesn't attend
    // over the previous request's tokens.
    let clear_start = std::time::Instant::now();
    pool.clear_kv_cache(&model_id, leader_handle)
        .await
        .map_err(InferenceError::Other)?;
    tracing::debug!(
        model = %model_id,
        elapsed_ms = clear_start.elapsed().as_millis(),
        "TP chat_completion: kv cache cleared"
    );

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
    let mut finish_reason = "length".to_string();

    // Prefill: chunk the prompt through `chunked_prefill_tp` so
    // activation memory is bounded by chunk size rather than the full
    // prompt length. Every rank still sees the prompt in order, just
    // spread across multiple `generate_step` calls with monotonically
    // growing offsets.
    let prefill_start = std::time::Instant::now();
    // Vision requests do a chunked image prefill (every rank encodes its
    // replicated tower once, then splices per chunk); text requests
    // chunk the prefill the same way.
    let logits_vec = match &vision_route {
        Some((data_uris, image_token_id)) => pool
            .generate_step_with_images(
                &model_id,
                leader_handle,
                prompt_tokens.clone(),
                0,
                *image_token_id,
                data_uris.clone(),
                prefill_chunk_tokens(),
            )
            .await
            .map_err(InferenceError::Other)?,
        None => chunked_prefill_tp(&mut pool, &model_id, leader_handle, &prompt_tokens)
            .await
            .map_err(InferenceError::Other)?,
    };
    let (post_prefill_vram_free_mb, _) = tp.query_vram().await;
    tracing::info!(
        model = %model_id,
        prompt_len,
        elapsed_ms = prefill_start.elapsed().as_millis(),
        vram_free_mb = post_prefill_vram_free_mb,
        "TP chat_completion: prefill complete"
    );
    // Wrap the CPU-side logits in a CPU candle Tensor for sampling.
    // No device touch on the async caller's thread — sampling reads
    // from CPU memory only.
    let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)
        .map_err(|e| InferenceError::Other(anyhow::anyhow!("build cpu logits: {e}")))?;
    let mut next_token = match sample_with_penalty(&logits, &generated, &mut logits_processor) {
        Ok(t) => t,
        Err(e) => {
            // Logits health snapshot — the surrounding wrapper logs
            // "failed, model marked poisoned" with the error chain;
            // this WARN sits just above that and carries the actual
            // numerical state so an operator can tell at a glance
            // whether it was a NaN cascade, an Inf, or something else.
            let health = logits_health_slice(&logits_vec);
            tracing::warn!(
                model = %model_id,
                ?health,
                "TP chat_completion: prefill sample failed; logits unhealthy"
            );
            return Err(InferenceError::Other(e));
        }
    };

    if Some(next_token) == eos_id {
        finish_reason = "stop".into();
    } else {
        generated.push(next_token);
        let decode_start = std::time::Instant::now();
        for index in 0..max_new.saturating_sub(1) {
            let step_start = std::time::Instant::now();
            let logits_vec = pool
                .generate_step(
                    &model_id,
                    leader_handle,
                    vec![next_token],
                    prompt_len + index,
                )
                .await
                .map_err(InferenceError::Other)?;
            let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu).map_err(|e| {
                InferenceError::Other(anyhow::anyhow!("build cpu logits step {index}: {e}"))
            })?;
            next_token = match sample_with_penalty(&logits, &generated, &mut logits_processor) {
                Ok(t) => t,
                Err(e) => {
                    let health = logits_health_slice(&logits_vec);
                    tracing::warn!(
                        model = %model_id,
                        step = index,
                        ?health,
                        "TP chat_completion: decode sample failed; logits unhealthy"
                    );
                    return Err(InferenceError::Other(e));
                }
            };
            let step_vram_free_mb = tp.query_vram().await.0;
            tracing::trace!(
                model = %model_id,
                step = index,
                next_token,
                step_ms = step_start.elapsed().as_millis(),
                vram_free_mb = step_vram_free_mb,
                "TP chat_completion: decode step"
            );
            if Some(next_token) == eos_id {
                finish_reason = "stop".into();
                break;
            }
            generated.push(next_token);
        }
        tracing::info!(
            model = %model_id,
            generated = generated.len(),
            elapsed_ms = decode_start.elapsed().as_millis(),
            "TP chat_completion: decode complete"
        );
    }
    drop(pool);

    let completion_text = tp
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| InferenceError::Other(anyhow::anyhow!("detokenize: {e}")))?;

    let usage = Usage {
        prompt_tokens: prompt_len as u64,
        completion_tokens: generated.len() as u64,
        total_tokens: (prompt_len + generated.len()) as u64,
    };

    tracing::info!(
        model = %model_id,
        prompt_tokens = prompt_len,
        completion_tokens = generated.len(),
        finish_reason = %finish_reason,
        total_ms = req_start.elapsed().as_millis(),
        "TP chat_completion: done"
    );

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

/// Send `delta` as an [`InferenceEvent::TextDelta`]. Returns `false`
/// if the receiver has hung up — the caller should bail. Empty
/// deltas (the DecodeStream is buffering an incomplete UTF-8
/// sequence) are a no-op return-true so the caller can treat "no
/// delta yet" and "tx still live" uniformly.
///
/// Wire-format-specific metadata (chunk id, created, model_id)
/// stays out of this function — the wire projector in
/// [`crate::wire::openai_chat`] stamps it onto every chunk
/// downstream.
#[cfg(feature = "cuda")]
async fn emit_delta(delta: &str, tx: &mpsc::Sender<InferenceEvent>, in_reasoning: bool) -> bool {
    if delta.is_empty() {
        return true;
    }
    let event = if in_reasoning {
        InferenceEvent::ReasoningDelta(delta.into())
    } else {
        InferenceEvent::TextDelta(delta.into())
    };
    tx.send(event).await.is_ok()
}

/// Sync counterpart of [`emit_delta`] for the CPU path's
/// `spawn_blocking` closure. Same shape, `blocking_send` instead of
/// `send`. Kept as a separate fn so the async / blocking-send choice
/// is local to one place per path.
fn emit_delta_blocking(delta: &str, tx: &mpsc::Sender<InferenceEvent>, in_reasoning: bool) -> bool {
    if delta.is_empty() {
        return true;
    }
    let event = if in_reasoning {
        InferenceEvent::ReasoningDelta(delta.into())
    } else {
        InferenceEvent::TextDelta(delta.into())
    };
    tx.blocking_send(event).is_ok()
}

/// If `next_token` is one of the loaded model's reasoning markers,
/// flip `in_reasoning` and return `true` to tell the caller to
/// skip detokenisation + emission for this token. The markers
/// themselves never appear in the streamed output — they exist
/// purely to transition state.
///
/// `pair = None` short-circuits to `false` (no reasoning markers
/// configured for this model → pass-through).
fn handle_reasoning_marker(
    next_token: u32,
    pair: Option<&ReasoningTokenPair>,
    in_reasoning: &mut bool,
) -> bool {
    let Some(pair) = pair else { return false };
    if next_token == pair.open_id {
        *in_reasoning = true;
        return true;
    }
    if next_token == pair.close_id {
        *in_reasoning = false;
        return true;
    }
    false
}

/// Outcome of checking a sampled token against the model's
/// tool-call markers.
enum ToolCallMarker {
    /// Not a tool-call marker — caller proceeds with the normal
    /// detokenize-and-emit path.
    None,
    /// `<tool_call>` token — caller starts buffering subsequent
    /// detokenized text into the tool-call buffer instead of
    /// emitting it. The token itself produces no output.
    Enter,
    /// `</tool_call>` token — caller takes ownership of the
    /// buffered JSON, parses it, and emits either a structured
    /// `InferenceEvent::ToolCall` or (on parse failure) the
    /// original `<tool_call>{buf}</tool_call>` as text. The
    /// returned buffer is `std::mem::take`-d out of the inner
    /// state.
    Exit { buffer: String },
}

fn handle_tool_call_marker(
    next_token: u32,
    pair: Option<&ToolCallTokenPair>,
    in_tool_call: &mut bool,
    buffer: &mut String,
) -> ToolCallMarker {
    let Some(pair) = pair else {
        return ToolCallMarker::None;
    };
    if next_token == pair.open_id {
        *in_tool_call = true;
        buffer.clear();
        return ToolCallMarker::Enter;
    }
    if next_token == pair.close_id {
        *in_tool_call = false;
        return ToolCallMarker::Exit {
            buffer: std::mem::take(buffer),
        };
    }
    ToolCallMarker::None
}

/// Parse a `<tool_call>{json}</tool_call>` body into the fields the
/// `InferenceEvent::ToolCall` variant carries. Returns `None` when
/// the body isn't valid JSON or doesn't carry a `name`. The caller
/// falls back to passing the original text through on `None` so
/// downstream consumers (helexa-acp's existing `ToolCallParser`,
/// which has its own repair passes) can take another swing.
///
/// Generates a fresh `call_<hex>` id per parsed call; the model
/// itself doesn't include ids in the wire convention we model.
fn parse_tool_call_body(body: &str, index: usize) -> Option<(String, String, String)> {
    let value: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "{}".into());
    let id = format!("call_{:x}_{}", unix_subsec_nanos(), index);
    Some((id, name, arguments))
}

/// Errors returned by `CandleHarness::chat_completion`. The
/// `ModelNotLoaded`, `PromptTooLong`, and `InsufficientVram` variants
/// let the HTTP handler map cleanly to 404 / 400 / 503 without
/// string-matching on anyhow messages.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("model '{0}' not loaded on this neuron")]
    ModelNotLoaded(String),
    #[error("prompt has {prompt_len} tokens but max is {max}")]
    PromptTooLong { prompt_len: usize, max: usize },
    #[error(
        "insufficient free VRAM for prefill: {free_mb} MiB free, need at least {required_mb} MiB"
    )]
    InsufficientVram { free_mb: u64, required_mb: u64 },
    /// Request carried `image_url` content but the loaded model has
    /// no vision tower. Stage B6 — replaces the silent-drop pattern
    /// from issue #3 with an explicit 400 + `vision_unsupported`
    /// error body that clients (litellm, agent0, …) can act on.
    #[error(
        "model '{model_id}' does not support image input; \
         load a vision-capable model (e.g. Qwen/Qwen3.6-27B) or \
         remove the image_url content parts from the request"
    )]
    VisionUnsupported { model_id: String },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Build the model's prompt from a [`ChatCompletionRequest`].
///
/// Prefers the model's own `chat_template` when one was loaded
/// from `tokenizer_config.json` at startup and the
/// `NEURON_USE_CHAT_TEMPLATE` kill switch isn't tripped. The
/// request's `chat_template_kwargs` (e.g.
/// `{"enable_thinking": false}` on Qwen3) and `tools` array flow
/// into the template's Jinja context so model-specific behaviour
/// like reasoning-suppression-at-generation works.
///
/// Falls back to [`format_qwen3_prompt`] (the legacy hardcoded
/// ChatML glue) on any of:
///
/// - no `chat_template` loaded for this model (older quantised
///   variants, fallback-only models)
/// - the env kill switch is set to a falsy value
/// - the template rendered to an error (caller can flip the env
///   var to force fallback while debugging the template)
///
/// Failures are logged at `warn` so an operator running with
/// `RUST_LOG=neuron=debug` sees which path each request took.
fn build_prompt_for_request(
    chat_template: Option<&str>,
    request: &ChatCompletionRequest,
) -> String {
    if !super::chat_template::chat_templates_enabled() {
        return format_qwen3_prompt(&request.messages);
    }
    let Some(tmpl) = chat_template else {
        return format_qwen3_prompt(&request.messages);
    };

    // Pull `chat_template_kwargs` and `tools` from the request's
    // catch-all `extra` field. Both are optional; absent fields
    // become `Value::Null`, which the renderer skips inserting
    // into the Jinja context.
    let kwargs = request
        .extra
        .get("chat_template_kwargs")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let tools = request
        .extra
        .get("tools")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    match super::chat_template::render_chat_template(tmpl, &request.messages, &tools, &kwargs) {
        Ok(prompt) => prompt,
        Err(e) => {
            tracing::warn!(
                model = %request.model,
                error = %format!("{e:#}"),
                "chat_template render failed; falling back to format_qwen3_prompt"
            );
            format_qwen3_prompt(&request.messages)
        }
    }
}

/// Vision metadata derived at model-load time. Stashed on
/// `LoadedModel` so the chat-completion hot path doesn't have to
/// re-parse `config.json` or reach across the worker thread to peek
/// at the loaded `ModelArch`.
#[derive(Debug, Default, Clone, Copy)]
struct VisionMeta {
    has_vision: bool,
    image_token_id: Option<u32>,
    /// `patch_size × spatial_merge_size` — the divisor that turns a
    /// resized pixel dimension into an LM-grid dimension. An image of
    /// resized `(h, w)` emits `(h/factor) × (w/factor)` LM tokens (#14
    /// dynamic resolution; was a constant 196 at the old fixed 448²).
    /// `None` for text-only models.
    image_grid_factor: Option<usize>,
}

impl VisionMeta {
    /// Peek at `config.json` for vision-related fields. Returns the
    /// default (no-vision) struct on any read/parse error — vision is
    /// best-effort metadata; load can still succeed for text usage.
    fn from_config_path(config_path: &std::path::Path) -> Self {
        let Ok(text) = std::fs::read_to_string(config_path) else {
            return Self::default();
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Self::default();
        };
        let Some(vision_config) = v.get("vision_config") else {
            return Self::default();
        };
        let patch_size = vision_config
            .get("patch_size")
            .and_then(|x| x.as_u64())
            .unwrap_or(16) as usize;
        let spatial_merge_size = vision_config
            .get("spatial_merge_size")
            .and_then(|x| x.as_u64())
            .unwrap_or(2) as usize;
        let image_token_id = v
            .get("image_token_id")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32);
        // The pixel→LM-grid divisor. An image resized to (h, w) emits
        // (h/factor) × (w/factor) LM tokens — computed per image at
        // request time now that resolution is dynamic (#14).
        let image_grid_factor = if patch_size > 0 && spatial_merge_size > 0 {
            Some(patch_size * spatial_merge_size)
        } else {
            None
        };
        Self {
            has_vision: true,
            image_token_id,
            image_grid_factor,
        }
    }
}

/// True iff any message in the request carries an `image_url`
/// content part. The Stage B routing decision in `chat_completion`
/// dispatches to the vision-aware worker job when this is true.
fn request_has_images(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(|m| {
        matches!(&m.content, MessageContent::Parts(parts)
        if parts.iter().any(|p|
            p.get("type").and_then(|v| v.as_str()) == Some("image_url")))
    })
}

/// Extract `image_url` content parts from a chat request and turn
/// each one into a preprocessed `ImageInput` ready for the device
/// worker. Stage B4.
///
/// Walks `request.messages`, looking for `MessageContent::Parts` and
/// pulling out entries whose `type == "image_url"`. Each is run
/// through `harness::preprocess::decode_data_uri` + `preprocess` with
/// the supplied `profile` (Stage B always uses
/// `PreprocessProfile::qwen3_6()` — fixed 448×448 — so every image
/// produces the same patch count; dynamic resolution per issue #14
/// would parameterise this).
///
/// Returns images in the order they appear in the request, which
/// matches the order the chat template emits `<|image_pad|>` tokens.
fn extract_images_from_request(
    request: &ChatCompletionRequest,
    profile: &super::preprocess::PreprocessProfile,
) -> anyhow::Result<Vec<super::device_worker::jobs::ImageInput>> {
    let mut out = Vec::new();
    for msg in &request.messages {
        if let MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                let kind = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if kind != "image_url" {
                    continue;
                }
                let url = part
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("image_url part missing url field"))?;
                let (pixels, h, w) = super::preprocess::preprocess_data_uri(url, profile)
                    .with_context(|| format!("preprocess image #{}", out.len()))?;
                out.push(super::device_worker::jobs::ImageInput {
                    pixels,
                    c: 3,
                    h: h as usize,
                    w: w as usize,
                });
            }
        }
    }
    Ok(out)
}

/// Collect the raw `image_url.url` strings (data URIs) from a chat
/// request, in prompt order. The TP vision path (Stage C / TP-vision)
/// ships these verbatim to every rank, which each preprocess + encode
/// identically — so unlike `extract_images_from_request` (which
/// preprocesses on the leader for the single-GPU worker job) this
/// keeps the source form for replicated per-rank encoding.
///
/// Cuda-gated: the only callers are the TP entry points, which compile
/// only under the `cuda` feature.
#[cfg(feature = "cuda")]
fn extract_image_data_uris(request: &ChatCompletionRequest) -> Vec<String> {
    let mut out = Vec::new();
    for msg in &request.messages {
        if let MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                if part.get("type").and_then(|v| v.as_str()) != Some("image_url") {
                    continue;
                }
                if let Some(url) = part
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                {
                    out.push(url.to_string());
                }
            }
        }
    }
    out
}

/// Expand each occurrence of `image_token_id` in `input_ids` into
/// `patches_per_image[i]` copies (one expansion per image, in order).
/// Stage B4 helper.
///
/// The chat template emits a single `<|image_pad|>` per image; this
/// is what fits Qwen3-VL's template-then-runtime-expansion convention.
/// The runtime (us) is responsible for replacing each one with N
/// copies based on the corresponding image's patch count.
///
/// For Stage B fixed resolution every entry of `patches_per_image`
/// is the same constant (196 at 448×448). For dynamic resolution
/// (issue #14) each image gets its own value.
///
/// Errors if the number of `image_token_id` occurrences in `input_ids`
/// doesn't equal `patches_per_image.len()` — a mismatch means the
/// template emitted the wrong number of pad tokens (operator-visible
/// template bug, not a runtime error).
fn expand_image_pad_tokens(
    input_ids: &[u32],
    image_token_id: u32,
    patches_per_image: &[usize],
) -> anyhow::Result<Vec<u32>> {
    let occurrences = input_ids.iter().filter(|&&t| t == image_token_id).count();
    if occurrences != patches_per_image.len() {
        anyhow::bail!(
            "expand_image_pad_tokens: prompt has {occurrences} image_token_id occurrences but \
             {} images were preprocessed — chat template emitted the wrong number of pad tokens",
            patches_per_image.len()
        );
    }
    let total_extra: usize = patches_per_image.iter().map(|n| n.saturating_sub(1)).sum();
    let mut out = Vec::with_capacity(input_ids.len() + total_extra);
    let mut img_idx = 0;
    for &t in input_ids {
        if t == image_token_id {
            let n = patches_per_image[img_idx];
            for _ in 0..n {
                out.push(image_token_id);
            }
            img_idx += 1;
        } else {
            out.push(t);
        }
    }
    Ok(out)
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
/// Run the full single-GPU inference loop via the device worker.
///
/// Mirrors `run_inference`'s logic but routes each forward step
/// through `worker.forward_logits()` (returns CPU-side `Vec<f32>`)
/// and runs `apply_repeat_penalty` + sampling on a CPU candle tensor.
/// The device-resident logits tensor never escapes the worker thread.
///
/// Used by the CUDA path of `chat_completion`. The CPU path keeps
/// `run_inference` (spawn_blocking against `Arc<Mutex<ModelArch>>`)
/// because there's no CUDA context to own and the worker indirection
/// would only add channel overhead with no diagnostic benefit.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
/// Vision-aware analogue of `run_inference_via_worker`. Stage B5.
///
/// Single-shot prefill carrying the pre-expanded prompt + the image
/// payloads. The worker encodes each image through the vision tower,
/// splices the resulting embeddings at `image_token_id` positions,
/// and returns the last-position logits. Decode steps thereafter
/// follow the existing text-only `forward_logits` path — the KV
/// cache holds the image-conditioned hidden states from prefill, so
/// no further splicing is needed.
///
/// Stage B skips chunked prefill for vision (the fixed-resolution
/// budget — 196 image tokens at 448×448 + typical text — stays well
/// under the activation-memory threshold). Long-prompt-with-images
/// chunking is a Stage D follow-up.
#[allow(clippy::too_many_arguments)]
async fn run_inference_with_images_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
    prompt_tokens: &[u32],
    images: Vec<super::device_worker::jobs::ImageInput>,
    image_token_id: u32,
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
    let prompt_len = prompt_tokens.len();

    worker
        .clear_kv_cache(handle)
        .await
        .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e}"))?;

    // Single-shot prefill with image splicing.
    let logits_vec = worker
        .forward_logits_with_images(handle, prompt_tokens.to_vec(), 0, images, image_token_id)
        .await
        .map_err(|e| anyhow::anyhow!("forward_logits_with_images: {e}"))?;
    let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
    let mut next_token = match sample_with_penalty(&logits, &generated, &mut logits_processor) {
        Ok(t) => t,
        Err(e) => {
            let health = logits_health_slice(&logits_vec);
            tracing::warn!(
                ?health,
                "chat_completion (worker, vision): prefill sample failed; logits unhealthy"
            );
            return Err(e);
        }
    };

    if Some(next_token) == eos_id {
        return Ok((generated, "stop".into()));
    }
    generated.push(next_token);

    for index in 0..max_new.saturating_sub(1) {
        let logits_vec = worker
            .forward_logits(handle, vec![next_token], prompt_len + index)
            .await
            .map_err(|e| anyhow::anyhow!("decode step {index}: {e}"))?;
        let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
        next_token = match sample_with_penalty(&logits, &generated, &mut logits_processor) {
            Ok(t) => t,
            Err(e) => {
                let health = logits_health_slice(&logits_vec);
                tracing::warn!(
                    step = index,
                    ?health,
                    "chat_completion (worker, vision): decode sample failed; logits unhealthy"
                );
                return Err(e);
            }
        };
        if Some(next_token) == eos_id {
            return Ok((generated, "stop".into()));
        }
        generated.push(next_token);
    }
    Ok((generated, "length".into()))
}

#[cfg(feature = "cuda")]
async fn run_inference_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
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
    let prompt_len = prompt_tokens.len();

    worker
        .clear_kv_cache(handle)
        .await
        .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e}"))?;

    // Prefill the prompt in `prefill_chunk_tokens()`-sized chunks so
    // activation memory is bounded per step rather than scaling with
    // prompt length. The KV cache accumulates across chunks; we keep
    // only the final chunk's logits for sampling the first generated
    // token.
    let logits_vec = chunked_prefill_via_worker(worker, handle, prompt_tokens).await?;
    let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
    let mut next_token = match sample_with_penalty(&logits, &generated, &mut logits_processor) {
        Ok(t) => t,
        Err(e) => {
            let health = logits_health_slice(&logits_vec);
            tracing::warn!(
                ?health,
                "chat_completion (worker): prefill sample failed; logits unhealthy"
            );
            return Err(e);
        }
    };

    if Some(next_token) == eos_id {
        return Ok((generated, "stop".into()));
    }
    generated.push(next_token);

    for index in 0..max_new.saturating_sub(1) {
        let logits_vec = worker
            .forward_logits(handle, vec![next_token], prompt_len + index)
            .await
            .map_err(|e| anyhow::anyhow!("decode step {index}: {e}"))?;
        let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
        next_token = match sample_with_penalty(&logits, &generated, &mut logits_processor) {
            Ok(t) => t,
            Err(e) => {
                let health = logits_health_slice(&logits_vec);
                tracing::warn!(
                    step = index,
                    ?health,
                    "chat_completion (worker): decode sample failed; logits unhealthy"
                );
                return Err(e);
            }
        };
        if Some(next_token) == eos_id {
            return Ok((generated, "stop".into()));
        }
        generated.push(next_token);
    }

    Ok((generated, "length".into()))
}

/// Streaming counterpart of [`run_inference_via_worker`]. Emits one
/// `ChatCompletionChunk` per generated token via `tx`; routes every
/// forward step through `worker.forward_logits()`. Same per-step
/// CPU-side sampling discipline — no device tensor escapes the
/// worker thread.
///
/// `images` carries the Stage C vision payload. When `Some`, prefill
/// is a single-shot `forward_logits_with_images` that splices image
/// embeddings at `image_token_id` positions (same contract as the
/// non-streaming [`run_inference_with_images_via_worker`]); image
/// embeddings are prefill-only, so every decode step below takes the
/// plain `forward_logits` path regardless. When `None`, prefill is
/// chunked (`chunked_prefill_via_worker`) to bound activation memory
/// — the original text-only behaviour, unchanged. The decode loop and
/// the `route_token!` reasoning/tool-call state machine are shared
/// across both prefill shapes, so there's exactly one copy to maintain.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn stream_inference_via_worker(
    worker: Arc<super::device_worker::DeviceWorkerHandle>,
    handle: super::device_worker::ArchHandle,
    tokenizer: Tokenizer,
    prompt_tokens: Vec<u32>,
    images: Option<(Vec<super::device_worker::jobs::ImageInput>, u32)>,
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    eos_id: Option<u32>,
    reasoning_tokens: Option<ReasoningTokenPair>,
    tool_call_tokens: Option<ToolCallTokenPair>,
    tx: mpsc::Sender<InferenceEvent>,
) -> Result<String> {
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
    // Incremental detokenizer. Replaces the old "decode cumulative
    // tokens, byte-slice the delta against a stored prefix" pattern
    // that panicked when BPE byte-fallback split a multi-byte UTF-8
    // sequence (e.g. an emoji) across tokens. `step` returns
    // `Ok(Some(delta))` only when the trailing bytes form a complete
    // codepoint; `Ok(None)` while it's buffering an incomplete one.
    let mut decode_stream = tokenizer.decode_stream(true);
    let prompt_len = prompt_tokens.len();
    let mut finish_reason = FinishReason::Length;
    // Reasoning + tool-call state machines — see
    // `run_inference_streaming` for the why. Markers never reach
    // `decode_stream`; they toggle state. Tool-call content
    // accumulates into `tool_call_buf` until the close marker.
    let mut in_reasoning = false;
    let mut in_tool_call = false;
    let mut tool_call_buf = String::new();
    let mut tool_call_idx: usize = 0;

    worker
        .clear_kv_cache(handle)
        .await
        .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e}"))?;

    // Prefill. Vision-bearing requests (`images = Some`) do a
    // single-shot prefill that splices the image embeddings; text-only
    // requests use chunked prefill (see `chunked_prefill_via_worker`)
    // to bound activation memory. Either way the owning
    // `prompt_tokens: Vec<u32>` outlives this step; we use `prompt_len`
    // (already extracted above) for the decode-step offset arithmetic.
    let logits_vec = match images {
        Some((imgs, image_token_id)) => worker
            .forward_logits_with_images(handle, prompt_tokens.clone(), 0, imgs, image_token_id)
            .await
            .map_err(|e| anyhow::anyhow!("forward_logits_with_images: {e}"))?,
        None => chunked_prefill_via_worker(&*worker, handle, &prompt_tokens).await?,
    };
    let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
    let mut next_token = match sample_with_penalty(&logits, &all_tokens, &mut logits_processor) {
        Ok(t) => t,
        Err(e) => {
            let health = logits_health_slice(&logits_vec);
            tracing::warn!(
                ?health,
                "chat_completion (stream/worker): prefill sample failed; logits unhealthy"
            );
            return Err(e);
        }
    };

    // Per-token routing. `tokenizers::DecodeStream` carries five
    // generic parameters (`M, N, PT, PP, D`) which makes naming
    // its type from a helper signature painful. Use a macro
    // instead — the body expands inline with `decode_stream`'s
    // concrete type inferred from the call site. The macro
    // contains `.await` calls, so it can only expand inside an
    // `async` context (which both call sites below are).
    //
    // The macro takes a single `$next_token` expression and
    // returns control to the enclosing scope via `break 'work_step`
    // (success path) — labels are needed because Rust macros can't
    // emit naked `return` from the caller when the caller's return
    // type isn't `()`. Instead the macro `break`s out of a
    // labelled block, and the surrounding `if !routed { ... }`
    // checks whether the consumer hung up via a captured `routed`
    // flag.
    macro_rules! route_token {
        ($next_token:expr) => {{
            let nt = $next_token;
            all_tokens.push(nt);
            let mut consumer_alive = true;
            'route: {
                match handle_tool_call_marker(
                    nt,
                    tool_call_tokens.as_ref(),
                    &mut in_tool_call,
                    &mut tool_call_buf,
                ) {
                    ToolCallMarker::Enter => break 'route,
                    ToolCallMarker::Exit { buffer } => {
                        let idx = tool_call_idx;
                        tool_call_idx += 1;
                        match parse_tool_call_body(&buffer, idx) {
                            Some((id, name, arguments)) => {
                                if tx
                                    .send(InferenceEvent::ToolCall {
                                        index: idx,
                                        id,
                                        name,
                                        arguments,
                                    })
                                    .await
                                    .is_err()
                                {
                                    consumer_alive = false;
                                }
                            }
                            None => {
                                let open = tool_call_tokens
                                    .as_ref()
                                    .map(|p| p.open_text.as_str())
                                    .unwrap_or("<tool_call>");
                                let close = tool_call_tokens
                                    .as_ref()
                                    .map(|p| p.close_text.as_str())
                                    .unwrap_or("</tool_call>");
                                let raw = format!("{open}{buffer}{close}");
                                if !emit_delta(&raw, &tx, in_reasoning).await {
                                    consumer_alive = false;
                                }
                            }
                        }
                        break 'route;
                    }
                    ToolCallMarker::None => {}
                }
                if in_tool_call {
                    match decode_stream.step(nt) {
                        Ok(Some(s)) => tool_call_buf.push_str(&s),
                        Ok(None) => {}
                        Err(e) => tracing::warn!(
                            error = %e,
                            "decode_stream step failed (in tool_call)"
                        ),
                    }
                    break 'route;
                }
                if handle_reasoning_marker(nt, reasoning_tokens.as_ref(), &mut in_reasoning) {
                    break 'route;
                }
                match decode_stream.step(nt) {
                    Ok(Some(delta)) => {
                        if !emit_delta(&delta, &tx, in_reasoning).await {
                            consumer_alive = false;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!(error = %e, "decode_stream step failed"),
                }
            }
            consumer_alive
        }};
    }

    if Some(next_token) == eos_id {
        finish_reason = FinishReason::Stop;
    } else if !route_token!(next_token) {
        return Ok(finish_reason.as_openai_str().to_string());
    }

    for index in 0..max_new.saturating_sub(1) {
        let logits_vec = worker
            .forward_logits(handle, vec![next_token], prompt_len + index)
            .await
            .map_err(|e| anyhow::anyhow!("decode step {index}: {e}"))?;
        let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
        next_token = match sample_with_penalty(&logits, &all_tokens, &mut logits_processor) {
            Ok(t) => t,
            Err(e) => {
                let health = logits_health_slice(&logits_vec);
                tracing::warn!(
                    step = index,
                    ?health,
                    "chat_completion (stream/worker): decode sample failed; logits unhealthy"
                );
                return Err(e);
            }
        };
        if Some(next_token) == eos_id {
            finish_reason = FinishReason::Stop;
            break;
        }
        if !route_token!(next_token) {
            return Ok(finish_reason.as_openai_str().to_string());
        }
    }

    // Terminal Finish event. The wire projector turns this into a
    // format-specific final chunk (`finish_reason: "stop"` on
    // OpenAI chat, `response.completed` on Responses).
    let _ = tx
        .send(InferenceEvent::Finish {
            reason: finish_reason,
        })
        .await;

    Ok(finish_reason.as_openai_str().to_string())
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

    arch.clear_kv_cache()?;
    let logits = chunked_prefill_local(arch, device, prompt_tokens)?;
    let mut next_token = sample_with_penalty(&logits, &generated, &mut logits_processor)?;

    if Some(next_token) == eos_id {
        return Ok((generated, "stop".into()));
    }
    generated.push(next_token);

    for index in 0..max_new.saturating_sub(1) {
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = arch.forward(&input, prompt_tokens.len() + index)?;
        next_token = sample_with_penalty(&logits, &generated, &mut logits_processor)?;
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
    reasoning_tokens: Option<&ReasoningTokenPair>,
    tool_call_tokens: Option<&ToolCallTokenPair>,
    tx: &mpsc::Sender<InferenceEvent>,
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
    // Incremental detokenizer. See `stream_inference_via_worker` for
    // the same reasoning — `tokenizer.decode_stream(true).step(id)`
    // buffers incomplete multi-byte UTF-8 sequences across token
    // boundaries and only emits when a clean codepoint completes.
    let mut decode_stream = tokenizer.decode_stream(true);
    let mut finish_reason = FinishReason::Length;
    // Reasoning marker state machine. Flips on
    // `next_token == reasoning_tokens.open_id`, off on
    // `.close_id`. The marker tokens themselves never feed into
    // `decode_stream` — they aren't part of any visible output,
    // they exist purely as state transitions.
    let mut in_reasoning = false;
    // Tool-call state. While `in_tool_call`, content tokens get
    // accumulated into `tool_call_buf` instead of emitted; on the
    // close marker we parse the buffer and emit a structured
    // ToolCall event (or fall back to passing the raw text
    // through if the buffer doesn't parse).
    let mut in_tool_call = false;
    let mut tool_call_buf = String::new();
    let mut tool_call_idx: usize = 0;

    arch.clear_kv_cache()?;
    let logits = chunked_prefill_local(arch, device, prompt_tokens)?;
    let mut next_token = sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?;

    // Per-token routing block, used at both the prefill-sample
    // tail and the decode loop. Macros are ugly but Rust's
    // closure inference fights `&mut DecodeStream<'_>` capture +
    // mutable borrows of the surrounding `tool_call_buf` /
    // `in_reasoning` / etc. Inline the body via a macro and live
    // with the duplication of the call sites instead.
    macro_rules! route_token {
        ($next_token:expr) => {{
            let nt = $next_token;
            all_tokens.push(nt);
            match handle_tool_call_marker(nt, tool_call_tokens, &mut in_tool_call, &mut tool_call_buf) {
                ToolCallMarker::Enter => {}
                ToolCallMarker::Exit { buffer } => {
                    let idx = tool_call_idx;
                    tool_call_idx += 1;
                    match parse_tool_call_body(&buffer, idx) {
                        Some((id, name, arguments)) => {
                            if tx
                                .blocking_send(InferenceEvent::ToolCall {
                                    index: idx,
                                    id,
                                    name,
                                    arguments,
                                })
                                .is_err()
                            {
                                return Ok(());
                            }
                        }
                        None => {
                            // Malformed JSON — pass the block
                            // through as text so consumer parsers
                            // can try their own repair.
                            let open = tool_call_tokens
                                .map(|p| p.open_text.as_str())
                                .unwrap_or("<tool_call>");
                            let close = tool_call_tokens
                                .map(|p| p.close_text.as_str())
                                .unwrap_or("</tool_call>");
                            let raw = format!("{open}{buffer}{close}");
                            if !emit_delta_blocking(&raw, tx, in_reasoning) {
                                return Ok(());
                            }
                        }
                    }
                }
                ToolCallMarker::None => {
                    if in_tool_call {
                        // Buffer JSON content without emitting.
                        match decode_stream.step(nt) {
                            Ok(Some(s)) => tool_call_buf.push_str(&s),
                            Ok(None) => {}
                            Err(e) => tracing::warn!(
                                error = %e,
                                "stream: decode_stream step failed (in tool_call)"
                            ),
                        }
                    } else if handle_reasoning_marker(nt, reasoning_tokens, &mut in_reasoning) {
                        // marker — nothing to emit
                    } else {
                        match decode_stream.step(nt) {
                            Ok(Some(delta)) => {
                                if !emit_delta_blocking(&delta, tx, in_reasoning) {
                                    return Ok(());
                                }
                            }
                            Ok(None) => {}
                            Err(e) => tracing::warn!(
                                error = %e,
                                "stream: decode_stream step failed"
                            ),
                        }
                    }
                }
            }
        }};
    }

    if Some(next_token) == eos_id {
        finish_reason = FinishReason::Stop;
    } else {
        route_token!(next_token);
    }

    for index in 0..max_new.saturating_sub(1) {
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = arch.forward(&input, prompt_tokens.len() + index)?;
        next_token = sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?;
        if Some(next_token) == eos_id {
            finish_reason = FinishReason::Stop;
            break;
        }
        route_token!(next_token);
    }

    let _ = tx.blocking_send(InferenceEvent::Finish {
        reason: finish_reason,
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_dense_config_accepts_qwen3() {
        let cfg = r#"{
            "model_type": "qwen3",
            "vocab_size": 151936,
            "architectures": ["Qwen3ForCausalLM"]
        }"#;
        check_dense_config_supported(cfg, "Qwen/Qwen3-1.7B").expect("qwen3 should pass");
    }

    #[test]
    fn check_dense_config_rejects_unsupported_arch_with_clear_message() {
        // Use a deliberately-fake model_type so this test stays
        // meaningful as the supported set grows. (qwen3_5 was the
        // motivating real example but now lives in the supported set
        // as a Stage 8c scaffold.)
        let cfg = r#"{
            "model_type": "fictional_arch_99",
            "architectures": ["FictionalArch99ForCausalLM"]
        }"#;
        let err = check_dense_config_supported(cfg, "Fake/Model-99")
            .expect_err("fictional_arch_99 should be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported model_type 'fictional_arch_99'"),
            "message should name the rejected type: {msg}"
        );
        assert!(
            msg.contains("Fake/Model-99"),
            "message should echo the model id: {msg}"
        );
        assert!(
            msg.contains("qwen3"),
            "message should list the supported set: {msg}"
        );
    }

    #[test]
    fn check_dense_config_accepts_qwen3_5() {
        // Sanity: Stage 8c scaffold means qwen3_5 deserialises into the
        // supported set. Forward still bails (covered by tests on the
        // architecture module itself), but the dispatch gate must let
        // it through.
        let cfg = r#"{
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"hidden_size": 5120}
        }"#;
        check_dense_config_supported(cfg, "Qwen/Qwen3.6-27B")
            .expect("qwen3_5 should be in the supported set as of Stage 8c scaffold");
    }

    #[test]
    fn check_dense_config_rejects_missing_model_type() {
        let cfg = r#"{ "vocab_size": 1234 }"#;
        let err = check_dense_config_supported(cfg, "anon/no-type")
            .expect_err("missing model_type should be rejected");
        assert!(
            format!("{err}").contains("missing `model_type`"),
            "message should call out the missing field"
        );
    }

    #[test]
    fn check_dense_config_rejects_invalid_json() {
        let err = check_dense_config_supported("not json", "anon/bad-json")
            .expect_err("malformed JSON should be rejected");
        assert!(
            format!("{err:#}").contains("config.json"),
            "message should mention config.json"
        );
    }

    #[test]
    fn is_device_fault_rejects_known_non_device_errors() {
        // Shape mismatches happen pre-kernel; device is healthy.
        assert!(!is_device_fault(
            "prefill chunk 0/9: shape mismatch in broadcast_add, lhs: [1, 32, 512, 1024], rhs: [1, 1, 512, 512]"
        ));
        // NaN logits are CPU-side numerical, not driver.
        assert!(!is_device_fault(
            "prefill sample failed; logits unhealthy nan: 248320/248320"
        ));
        // Tokenizer/detokenizer errors are pure host.
        assert!(!is_device_fault("tokenize: invalid utf-8 sequence"));
        assert!(!is_device_fault("detokenize: byte fallback failed"));
        // Missing handle is a dispatch-side bug, not a device fault.
        assert!(!is_device_fault("ForwardLogits: no model for handle 42"));
        // DecodeStream errors during SSE are not device faults.
        assert!(!is_device_fault(
            "decode_stream step failed: invalid prefix"
        ));
    }

    #[test]
    fn is_device_fault_defaults_to_poisoning() {
        // Unknown errors default to "poison" — better to over-reject
        // than to keep serving from a corrupted context.
        assert!(is_device_fault("some unrecognised candle error"));
        // Real driver faults — these strings come from cudarc's
        // DriverError Display impl and we want them to poison.
        assert!(is_device_fault(
            "leader forward failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(is_device_fault(
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")"
        ));
    }

    /// Phase 1 of plan-source-aware-loader: harness must resolve each
    /// configured scheme to its own endpoint+cache, and reject schemes
    /// the operator hasn't configured with a useful error.
    #[test]
    fn hf_api_for_routes_per_scheme() {
        use crate::config::{CandleHarnessConfig, SourceConfig};
        use std::collections::HashMap;

        let mut sources = HashMap::new();
        sources.insert(
            "huggingface".to_string(),
            SourceConfig {
                endpoint: "https://huggingface.example.org".into(),
                auth_env: None,
                cache_dir: Some(std::path::PathBuf::from("/tmp/hf-cache")),
            },
        );
        sources.insert(
            "helexa".to_string(),
            SourceConfig {
                endpoint: "https://registry.helexa.example.ai".into(),
                auth_env: None,
                cache_dir: Some(std::path::PathBuf::from("/tmp/helexa-cache")),
            },
        );
        let cfg = CandleHarnessConfig {
            sources,
            default_source: Some("huggingface".into()),
            ..Default::default()
        };
        let harness = CandleHarness::new("http://localhost:13131".into(), &cfg);

        // Both configured schemes build cleanly.
        harness
            .hf_api_for("huggingface")
            .expect("huggingface scheme should build");
        harness
            .hf_api_for("helexa")
            .expect("helexa scheme should build");

        // Unknown scheme errors with a message that names the configured
        // set so the operator can act on it.
        let err = harness
            .hf_api_for("does-not-exist")
            .expect_err("unknown scheme should error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("does-not-exist") && msg.contains("huggingface") && msg.contains("helexa"),
            "error must list configured schemes: {msg}"
        );

        assert_eq!(harness.default_source_scheme(), "huggingface");
    }

    /// Operator with only `hf_cache` set (no `sources` table) still
    /// gets a working `huggingface` source pointed at HF.
    #[test]
    fn hf_api_for_synthesises_huggingface_from_legacy_hf_cache() {
        use crate::config::CandleHarnessConfig;

        let cfg = CandleHarnessConfig {
            hf_cache: Some(std::path::PathBuf::from("/archive3/llm-cache")),
            ..Default::default()
        };
        let harness = CandleHarness::new("http://localhost:13131".into(), &cfg);
        harness
            .hf_api_for("huggingface")
            .expect("synth huggingface source should build");
        assert_eq!(harness.default_source_scheme(), "huggingface");
    }

    #[test]
    fn expand_image_pad_replaces_single_token_with_n_copies() {
        // Mimics the chat template's output: each image emits
        // [vision_start, image_pad, vision_end]. After expansion
        // with 3 patches/image we want
        // [vision_start, pad×3, vision_end].
        let pad = 248056_u32;
        let vstart = 248053_u32;
        let vend = 248054_u32;
        let input = vec![1, vstart, pad, vend, 2];
        let out = expand_image_pad_tokens(&input, pad, &[3]).unwrap();
        assert_eq!(out, vec![1, vstart, pad, pad, pad, vend, 2]);
    }

    #[test]
    fn expand_image_pad_handles_multiple_images() {
        let pad = 248056_u32;
        // Two images in one prompt; first gets 2 patches, second 3.
        let input = vec![pad, 99, pad];
        let out = expand_image_pad_tokens(&input, pad, &[2, 3]).unwrap();
        assert_eq!(out, vec![pad, pad, 99, pad, pad, pad]);
    }

    #[test]
    fn expand_image_pad_errors_on_count_mismatch() {
        let pad = 248056_u32;
        // Prompt has 2 pad tokens but caller supplied 3 images.
        let input = vec![pad, 99, pad];
        let err = expand_image_pad_tokens(&input, pad, &[2, 3, 4]).unwrap_err();
        assert!(format!("{err:#}").contains("emitted the wrong number"));
    }

    #[test]
    fn expand_image_pad_passes_through_when_no_images() {
        let pad = 248056_u32;
        let input = vec![1, 2, 3];
        let out = expand_image_pad_tokens(&input, pad, &[]).unwrap();
        assert_eq!(out, input);
    }

    /// `request_has_images` is the gate that routes both the
    /// non-streaming (`chat_completion`) and streaming
    /// (`inference_stream`, Stage C1) paths to the vision-aware
    /// prefill. Exercise the three shapes it must distinguish: plain
    /// text, a text-only content-parts array, and a parts array
    /// carrying an `image_url`.
    #[test]
    fn request_has_images_detects_image_url_parts() {
        let text_only: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();
        assert!(!request_has_images(&text_only));

        let parts_text_only: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello"}
            ]}],
        }))
        .unwrap();
        assert!(!request_has_images(&parts_text_only));

        let with_image: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}}
            ]}],
        }))
        .unwrap();
        assert!(request_has_images(&with_image));
    }

    /// The vision pre-flight guard rejects a prefill whose estimated
    /// footprint exceeds free VRAM (so a doomed request fails clean
    /// instead of OOM-hanging the TP collective), passes one that fits,
    /// and is skipped on the CPU sentinel.
    #[test]
    fn vision_prefill_guard_behaviour() {
        // CPU sentinel (vram_free_mb == 0) is always allowed.
        assert!(validate_vision_prefill(10_000_000, 0).is_ok());

        // A clearly-oversized prompt against tiny free VRAM is rejected
        // for any non-degenerate config (default: 2000 base + 500/1k).
        assert!(matches!(
            validate_vision_prefill(10_000_000, 50),
            Err(InferenceError::InsufficientVram { .. })
        ));

        // With defaults, the agent-0-sized 12,960-token prompt that
        // OOM'd single-shot fits the estimate at ~12 GB free (2000 +
        // 12960*500/1000 = 8480 MiB) — the chunked prefill handles it,
        // so the guard must NOT reject it.
        assert!(validate_vision_prefill(12_960, 12_445).is_ok());
    }
}
