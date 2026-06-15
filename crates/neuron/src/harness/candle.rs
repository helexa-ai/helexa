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
    /// Auto-recovery (#17): model ids whose poisoned context is being
    /// rebuilt via unload+reload, mapped to a devices/capabilities
    /// snapshot taken at trigger time. Insert is the single-flight gate
    /// (one recovery per model in flight); membership lets the request
    /// path answer "recovering, retry shortly" during the reload gap
    /// rather than a bare "not loaded", and the snapshot keeps the
    /// model listed (status `recovering`) in `list_models` while its
    /// registry slot is briefly absent — so cortex holds the route
    /// instead of treating the model as evicted/unknown (#20).
    recovering: Arc<RwLock<HashMap<String, RecoveringSnapshot>>>,
    /// Sender to the background recovery task. The request path enqueues
    /// a poisoned model id here; the task (holding a `Weak<Self>`) runs
    /// the unload→reload→health-gate. Unbounded + tiny (model ids), and
    /// the `recovering` set dedupes, so it can't back up.
    recovery_tx: tokio::sync::mpsc::UnboundedSender<String>,
    /// Prefix-cache settings (#11), applied per loaded model at load
    /// time (snapshot-capable archs only).
    prefix_cache_cfg: crate::config::PrefixCacheConfig,
}

/// Devices/capabilities snapshot of a model entering auto-recovery
/// (#20). Captured while the registry slot still exists; `list_models`
/// uses it to keep advertising the model during the unload→reload
/// window, when the slot itself is gone.
#[derive(Debug, Clone, Default)]
struct RecoveringSnapshot {
    devices: Vec<u32>,
    capabilities: Vec<String>,
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

    /// The spec this model was loaded from (for auto-recovery #17).
    pub fn spec(&self) -> &ModelSpec {
        match self {
            LoadedHandle::Single(m) => &m.spec,
            #[cfg(feature = "cuda")]
            LoadedHandle::Tp(m) => &m.spec,
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

/// Reference to one stored prefix snapshot (#11). CUDA loads keep the
/// snapshot tensors in the device worker and hold only the opaque id
/// here; CPU loads hold the snapshot itself (no context-affinity
/// constraint on CPU tensors). TP loads hold the pool-minted id every
/// rank stored its shard snapshot under.
#[derive(Clone)]
pub enum KvSnapshotRef {
    Worker(super::device_worker::KvSnapshotId),
    Local(Arc<super::arch::qwen3_5::snapshot::KvCacheSnapshot>),
    #[cfg(feature = "cuda")]
    Tp(u64),
}

/// Per-model prefix-cache registry: matching/eviction bookkeeping in
/// `prefix_cache::PrefixCache`, snapshot refs as the payload. Guarded
/// by a std Mutex — every access is short and already serialised by
/// `LoadedModel::inference_lock`.
pub type ModelPrefixCache = std::sync::Mutex<super::prefix_cache::PrefixCache<KvSnapshotRef>>;

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
    /// The spec this model was loaded from — retained so auto-recovery
    /// (#17) can `unload_model` + `load_model(spec)` a poisoned model
    /// without an operator reconstructing it.
    pub spec: ModelSpec,
    /// Prefix-cache registry (#11). `None` when the arch has no
    /// snapshot support or the operator disabled the cache — the
    /// request path then clears the KV cache every request, exactly
    /// the pre-#11 behaviour. Dropped with the model, so unload and
    /// auto-recovery invalidate every entry for free (the worker-side
    /// snapshots go with `Job::DropArch`).
    pub prefix_cache: Option<ModelPrefixCache>,
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
    /// Loading spec, retained for auto-recovery (#17) — see
    /// [`LoadedModel::spec`].
    pub spec: ModelSpec,
    /// Prefix-cache registry (#11) — see [`LoadedModel::prefix_cache`].
    /// Entries hold [`KvSnapshotRef::Tp`] ids; the per-rank snapshot
    /// tensors live on the leader's device worker and in each
    /// subprocess rank, all keyed by the same pool-minted id.
    pub prefix_cache: Option<ModelPrefixCache>,
    /// Mint for pool-wide snapshot ids. Plain counter; uniqueness only
    /// needs to hold per model lifetime (snapshots die with the model).
    pub next_snapshot_id: std::sync::atomic::AtomicU64,
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

    /// Whether this arch can capture/restore prefix snapshots (#11).
    /// Only the in-tree qwen3_5 arch exposes its cache state; the
    /// candle-transformers archs keep theirs private, so they stay on
    /// the clear-every-request path.
    pub fn supports_kv_snapshot(&self) -> bool {
        matches!(self, ModelArch::Qwen3_5Dense(_))
    }

    /// Capture the live cache state as a prefix snapshot. See
    /// `arch/qwen3_5/snapshot.rs` for what a snapshot contains and the
    /// copy-semantics constraints.
    pub fn snapshot_kv_cache(&self) -> Result<super::arch::qwen3_5::snapshot::KvCacheSnapshot> {
        match self {
            ModelArch::Qwen3_5Dense(m) => Ok(m.snapshot_kv_cache()?),
            _ => anyhow::bail!("snapshot_kv_cache: architecture has no snapshot support"),
        }
    }

    /// Replace the live cache state with a stored snapshot — the
    /// restore-instead-of-clear half of prefix caching.
    pub fn restore_kv_cache(
        &mut self,
        snap: &super::arch::qwen3_5::snapshot::KvCacheSnapshot,
    ) -> Result<()> {
        match self {
            ModelArch::Qwen3_5Dense(m) => Ok(m.restore_kv_cache(snap)?),
            _ => anyhow::bail!("restore_kv_cache: architecture has no snapshot support"),
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

    /// Chunked image prefill (#18) — encode once, walk the prompt in
    /// `chunk_size` windows splicing per-chunk image-pad rows, return
    /// the final chunk's `[vocab]` logits. Bounds activation memory to
    /// one chunk so a long single-GPU vision context serves instead of
    /// single-shot OOMing. Only `Qwen3_5Dense` has a vision tower.
    pub fn prefill_with_images_chunked(
        &mut self,
        tokens: &[u32],
        offset: usize,
        image_pixels: &[Tensor],
        image_token_id: u32,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let raw = match self {
            ModelArch::Qwen3_5Dense(m) => m.prefill_with_images_chunked(
                tokens,
                offset,
                image_pixels,
                image_token_id,
                chunk_size,
            )?,
            other => anyhow::bail!(
                "prefill_with_images_chunked: architecture {} has no vision tower",
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

/// Reported while auto-recovery (#17) is rebuilding a poisoned model's
/// context. Unlike [`poisoned_error`] this is a *transient* state — the
/// model is being reloaded automatically; the client should retry.
fn recovering_error(model_id: &str) -> InferenceError {
    InferenceError::Other(anyhow::anyhow!(
        "model '{model_id}' is recovering (its device context was poisoned \
         by an earlier failure and is being automatically rebuilt); retry \
         shortly"
    ))
}

/// Verification hook for #17 auto-recovery. When `NEURON_DEBUG_POISON`
/// names a model, the **first** request for it (process-wide) returns
/// true, so the request path can trigger recovery as if a device fault
/// had occurred — exercising the unload→reload→healthy cycle without
/// corrupting the GPU. One-shot (a `swap` latch) so it can't loop the
/// model through endless recoveries. No-op unless the env var is set.
fn debug_poison_armed(model_id: &str) -> bool {
    static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let armed = std::env::var("NEURON_DEBUG_POISON").ok().as_deref() == Some(model_id);
    armed && !FIRED.swap(true, Ordering::Relaxed)
}

/// Background auto-recovery task (#17). Drains poisoned model ids and
/// rebuilds each via [`CandleHarness::recover_one`]. Holds a `Weak` so a
/// shutting-down harness lets the task exit; processes one id at a time,
/// which (with the `recovering` set deduping enqueues) keeps recovery
/// single-flight per model.
async fn recovery_loop(
    weak: std::sync::Weak<CandleHarness>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
) {
    while let Some(model_id) = rx.recv().await {
        let Some(this) = weak.upgrade() else {
            break;
        };
        this.recover_one(&model_id).await;
    }
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
/// `pub(crate)` so the device-worker dispatch handler can chunk the
/// single-GPU vision prefill (#18) with the same policy.
pub(crate) fn prefill_chunk_tokens() -> usize {
    env_usize("NEURON_PREFILL_CHUNK_TOKENS", 512)
}

/// Maximum allowed prompt length, in tokens. Requests above this are
/// rejected with [`InferenceError::PromptTooLong`] before any device
/// work — this is the explicit upper bound on context size, separate
/// from the model's `max_position_embeddings` (which can be much
/// larger than what fits in VRAM in practice).
pub(crate) fn max_prompt_tokens() -> usize {
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
///
/// `start_offset` is the number of leading prompt tokens already in
/// the cache (a restored prefix snapshot, #11); prefill begins there.
/// 0 = the classic full prefill after a cache clear.
fn chunked_prefill_local(
    arch: &mut ModelArch,
    device: &Device,
    prompt_tokens: &[u32],
    start_offset: usize,
) -> Result<Tensor> {
    let prompt_len = prompt_tokens.len();
    if start_offset >= prompt_len {
        anyhow::bail!(
            "chunked_prefill_local: start_offset {start_offset} leaves no tokens of {prompt_len} to prefill"
        );
    }
    let chunk_size = prefill_chunk_tokens();
    let mut offset = start_offset;
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
/// `start_offset` skips a restored cached prefix, as in
/// [`chunked_prefill_local`].
#[cfg(feature = "cuda")]
async fn chunked_prefill_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
    prompt_tokens: &[u32],
    start_offset: usize,
) -> Result<Vec<f32>> {
    let prompt_len = prompt_tokens.len();
    if start_offset >= prompt_len {
        anyhow::bail!(
            "chunked_prefill_via_worker: start_offset {start_offset} leaves no tokens of {prompt_len} to prefill"
        );
    }
    let chunk_size = prefill_chunk_tokens();
    let mut offset = start_offset;
    let mut last_logits: Option<Vec<f32>> = None;
    let total_chunks = (prompt_len - start_offset).div_ceil(chunk_size);
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
/// `start_offset` skips a restored cached prefix, as in
/// [`chunked_prefill_local`].
#[cfg(feature = "cuda")]
async fn chunked_prefill_tp(
    pool: &mut super::tp::WorkerPool,
    model_id: &str,
    leader_handle: super::device_worker::TpHandle,
    prompt_tokens: &[u32],
    start_offset: usize,
) -> Result<Vec<f32>> {
    let prompt_len = prompt_tokens.len();
    if start_offset >= prompt_len {
        anyhow::bail!(
            "chunked_prefill_tp: start_offset {start_offset} leaves no tokens of {prompt_len} to prefill"
        );
    }
    let chunk_size = prefill_chunk_tokens();
    let mut offset = start_offset;
    let mut last_logits: Option<Vec<f32>> = None;
    let total_chunks = (prompt_len - start_offset).div_ceil(chunk_size);
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
    pub fn new(bind_url: String, config: &crate::config::CandleHarnessConfig) -> Arc<Self> {
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
        let (recovery_tx, recovery_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let this = Arc::new(Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            sources,
            default_source,
            bind_url,
            device_workers: Arc::new(RwLock::new(HashMap::new())),
            recovering: Arc::new(RwLock::new(HashMap::new())),
            recovery_tx,
            prefix_cache_cfg: config.prefix_cache.clone(),
        });
        // Background auto-recovery task (#17). Holds a `Weak` so it can't
        // keep the harness alive. Spawned only when a tokio runtime is
        // present — sync unit tests that build a harness without one
        // simply skip it (they don't exercise recovery).
        if tokio::runtime::Handle::try_current().is_ok() {
            let weak = Arc::downgrade(&this);
            tokio::spawn(recovery_loop(weak, recovery_rx));
        }
        this
    }

    /// Scheme to substitute for bare `org/name` model ids. Mirrors the
    /// effective default from the operator's config, exposed for the
    /// load path's `ModelSourceId::with_default_scheme`.
    pub(crate) fn default_source_scheme(&self) -> &str {
        &self.default_source
    }

    /// Fresh per-model prefix-cache registry, or `None` when the arch
    /// can't snapshot or the operator disabled/zeroed the cache.
    fn new_prefix_cache(&self, arch_supported: bool) -> Option<ModelPrefixCache> {
        let cfg = &self.prefix_cache_cfg;
        (arch_supported && cfg.enabled && cfg.budget_mb > 0 && cfg.max_entries > 0).then(|| {
            std::sync::Mutex::new(super::prefix_cache::PrefixCache::new(
                cfg.budget_mb.saturating_mul(1024 * 1024),
                cfg.max_entries,
            ))
        })
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
        let handle = match handle {
            Some(h) => h,
            // Absent from the registry: distinguish a genuinely unloaded
            // model from one whose slot is briefly gone mid auto-recovery
            // (#17), so the client gets a transient "retry shortly" instead
            // of a misleading "not loaded".
            None if self.is_recovering(&request.model).await => {
                return Err(recovering_error(&request.model));
            }
            None => return Err(InferenceError::ModelNotLoaded(request.model.clone())),
        };
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
            return Err(self.trigger_recovery(&model_id).await);
        }
        if debug_poison_armed(&model_id) {
            let _g = span.enter();
            tracing::warn!("NEURON_DEBUG_POISON: forcing auto-recovery (#17 verification)");
            return Err(self.trigger_recovery(&model_id).await);
        }

        // Serialise concurrent requests against this model. Holds for
        // the duration of clear_kv_cache → prefill → decode so two
        // requests' chunked-prefill sequences can't interleave on the
        // shared KV cache (see `LoadedModel.inference_lock` for the
        // observed failure mode).
        let _inference_guard = loaded.inference_lock.lock().await;

        let result = async {
            let prompt = build_prompt_for_request(loaded.chat_template.as_deref(), &request)?;

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
                                loaded.prefix_cache.as_ref(),
                                loaded.tokenizer.token_to_id("<|im_start|>"),
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
                let loaded_for_cache = Arc::clone(&loaded);
                let im_start_id = loaded.tokenizer.token_to_id("<|im_start|>");
                let inference_result =
                    tokio::task::spawn_blocking(move || -> Result<(Vec<u32>, String)> {
                        let mut guard = arch_arc.blocking_lock();
                        run_inference(
                            &mut guard,
                            &device,
                            &prompt_tokens,
                            loaded_for_cache.prefix_cache.as_ref(),
                            im_start_id,
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
        let handle = match handle {
            Some(h) => h,
            // Absent from the registry: distinguish a genuinely unloaded
            // model from one whose slot is briefly gone mid auto-recovery
            // (#17), so the client gets a transient "retry shortly" instead
            // of a misleading "not loaded".
            None if self.is_recovering(&request.model).await => {
                return Err(recovering_error(&request.model));
            }
            None => return Err(InferenceError::ModelNotLoaded(request.model.clone())),
        };
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

        let prompt = build_prompt_for_request(loaded.chat_template.as_deref(), &request)?;
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
            return Err(self.trigger_recovery(&model_id).await);
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
        let tool_schemas = build_tool_schemas(&request);
        if let (Some(worker), Some(handle)) = (loaded.worker.clone(), loaded.arch_handle) {
            #[cfg(feature = "cuda")]
            {
                let prompt_tokens = prompt_tokens.clone();
                let reasoning_tokens_inner = loaded.reasoning_tokens.clone();
                let tool_call_tokens_inner = loaded.tool_call_tokens.clone();
                let tool_schemas_inner = tool_schemas.clone();
                tokio::spawn(
                    async move {
                        let _inference_guard = loaded_for_task.inference_lock.lock().await;
                        match stream_inference_via_worker(
                            worker,
                            handle,
                            tokenizer,
                            prompt_tokens,
                            vision_route,
                            loaded_for_task.prefix_cache.as_ref(),
                            max_new,
                            temperature,
                            top_p,
                            seed,
                            eos_id,
                            reasoning_tokens_inner,
                            tool_call_tokens_inner,
                            tool_schemas_inner,
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
            let tool_schemas_inner = tool_schemas.clone();
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
                    loaded_for_task.prefix_cache.as_ref(),
                    max_new,
                    temperature,
                    top_p,
                    seed,
                    eos_id,
                    reasoning_tokens_inner.as_ref(),
                    tool_call_tokens_inner.as_ref(),
                    tool_schemas_inner,
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

/// Auto-recovery (#17) — rebuild a poisoned model's device context
/// automatically instead of leaving it bricked until a human reloads.
impl CandleHarness {
    /// True while `model_id` is being auto-recovered (its slot is briefly
    /// absent from the registry during the reload).
    pub async fn is_recovering(&self, model_id: &str) -> bool {
        self.recovering.read().await.contains_key(model_id)
    }

    /// Single-flight trigger from the request path: enqueue a rebuild for a
    /// poisoned model (only the first caller per model enqueues) and return
    /// the transient "recovering" error to hand back to the client.
    async fn trigger_recovery(&self, model_id: &str) -> InferenceError {
        // Snapshot the model's shape while its registry slot still
        // exists — it disappears during the unload→reload window, and
        // list_models needs it to keep advertising the model (#20).
        let snapshot = {
            let models = self.models.read().await;
            models
                .get(model_id)
                .map(|h| RecoveringSnapshot {
                    devices: h.devices(),
                    capabilities: h.capabilities(),
                })
                .unwrap_or_default()
        };
        let newly = self
            .recovering
            .write()
            .await
            .insert(model_id.to_string(), snapshot)
            .is_none();
        if newly {
            tracing::warn!(model = %model_id, "auto-recovery: poisoned, enqueueing rebuild");
            if self.recovery_tx.send(model_id.to_string()).is_err() {
                // Background task gone (harness shutting down). Drop the
                // marker and fall back to the manual-reload message.
                self.recovering.write().await.remove(model_id);
                tracing::error!(model = %model_id, "auto-recovery: task unavailable");
                return poisoned_error(model_id);
            }
        }
        recovering_error(model_id)
    }

    /// Rebuild a poisoned model: `unload_model` (drops it → cudarc aborts
    /// NCCL + releases the context) then `load_model` from the retained
    /// spec. A successful reload re-runs NCCL init + sanity inside the load
    /// path, so it returns a fresh, healthy model; a failed reload leaves
    /// the model unloaded (recoverable by the next load), never poisoned
    /// forever. Runs on the background task — never inline on the request
    /// path (would deadlock on the `models` write lock).
    async fn recover_one(&self, model_id: &str) {
        let spec = {
            let models = self.models.read().await;
            models.get(model_id).map(|h| h.spec().clone())
        };
        let Some(spec) = spec else {
            self.recovering.write().await.remove(model_id);
            return;
        };
        tracing::warn!(model = %model_id, "auto-recovery: unload+reload starting");
        if let Err(e) = self.unload_model(model_id).await {
            tracing::error!(
                model = %model_id,
                error = %format!("{e:#}"),
                "auto-recovery: unload failed (continuing to reload)"
            );
        }
        match self.load_model(&spec).await {
            Ok(()) => tracing::info!(model = %model_id, "auto-recovery: reloaded; model healthy"),
            Err(e) => tracing::error!(
                model = %model_id,
                error = %format!("{e:#}"),
                "auto-recovery: reload failed; model left unloaded"
            ),
        }
        self.recovering.write().await.remove(model_id);
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
        let recovering = self.recovering.read().await;
        let mut out: Vec<ModelInfo> = models
            .values()
            .map(|h| ModelInfo {
                id: h.model_id().into(),
                harness: "candle".into(),
                // A poisoned model with recovery in flight reports
                // `recovering` (the operator-actionable state); bare
                // `poisoned` only appears if the recovery task is gone.
                status: if recovering.contains_key(h.model_id()) {
                    "recovering".into()
                } else if h.is_poisoned() {
                    "poisoned".into()
                } else {
                    "loaded".into()
                },
                devices: h.devices(),
                vram_used_mb: None,
                capabilities: h.capabilities(),
            })
            .collect();
        // Models mid-recovery whose registry slot is absent (the
        // unload→reload window, ~minutes for a large TP model) stay
        // listed from their trigger-time snapshot so cortex holds the
        // route instead of reporting them evicted/unknown (#20).
        for (id, snap) in recovering.iter() {
            if !models.contains_key(id) {
                out.push(ModelInfo {
                    id: id.clone(),
                    harness: "candle".into(),
                    status: "recovering".into(),
                    devices: snap.devices.clone(),
                    vram_used_mb: None,
                    capabilities: snap.capabilities.clone(),
                });
            }
        }
        Ok(out)
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

        let (tokenizer_path, arch_local, arch_handle, vision_meta, snapshot_capable) =
            if let Some(w) = &worker {
                // CUDA path: resolve, then load in the worker.
                if spec.quant.is_some() {
                    let (gguf_path, tokenizer_path) = self.resolve_files(spec, &source_id).await?;
                    let handle = w
                        .load_gguf(gguf_path, spec.model_id.clone())
                        .await
                        .map_err(|e| anyhow::anyhow!("worker load_gguf: {e}"))?;
                    // GGUF Qwen3.6 releases don't ship the vision tower
                    // (Qwen-VL weights are in the dense safetensors only),
                    // so a GGUF load is text-only by construction. GGUF
                    // archs are candle-transformers types — no snapshot
                    // support either.
                    (
                        tokenizer_path,
                        None,
                        Some(handle),
                        VisionMeta::default(),
                        false,
                    )
                } else {
                    let (config_path, tokenizer_path, safetensors_paths) =
                        self.resolve_dense_files(spec, &source_id).await?;
                    let meta = VisionMeta::from_config_path(&config_path);
                    // Prefix snapshots (#11) exist only for the in-tree
                    // qwen3_5 arch; the worker holds the ModelArch so
                    // the async side decides from config.json instead.
                    let snapshot_capable = config_model_type(&config_path).as_deref()
                        == Some(super::arch::qwen3_5::MODEL_TYPE);
                    let handle = w
                        .load_dense(config_path, safetensors_paths, spec.model_id.clone())
                        .await
                        .map_err(|e| anyhow::anyhow!("worker load_dense: {e}"))?;
                    (tokenizer_path, None, Some(handle), meta, snapshot_capable)
                }
            } else {
                // CPU path: legacy spawn_blocking + Arc<Mutex<ModelArch>>.
                let (tokenizer_path, arch) = if spec.quant.is_some() {
                    self.load_arch_gguf(spec, &source_id, &device).await?
                } else {
                    self.load_arch_dense(spec, &source_id, &device).await?
                };
                let snapshot_capable = arch.supports_kv_snapshot();
                // CPU Qwen3.6 isn't a supported deployment target — the
                // 27B doesn't fit any reasonable CPU memory budget — so
                // we don't attempt to reach into the arch for vision
                // metadata. Stays text-only.
                (
                    tokenizer_path,
                    Some(Arc::new(Mutex::new(arch))),
                    None,
                    VisionMeta::default(),
                    snapshot_capable,
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
            spec: spec.clone(),
            prefix_cache: self.new_prefix_cache(snapshot_capable),
        });
        if loaded.prefix_cache.is_some() {
            tracing::info!(
                model = %spec.model_id,
                budget_mb = self.prefix_cache_cfg.budget_mb,
                max_entries = self.prefix_cache_cfg.max_entries,
                "prefix cache enabled for this model"
            );
        }

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
            spec: spec.clone(),
            prefix_cache: self.new_prefix_cache(
                config_model_type(&config_path).as_deref()
                    == Some(super::arch::qwen3_5::MODEL_TYPE),
            ),
            next_snapshot_id: std::sync::atomic::AtomicU64::new(1),
        });
        if tp_loaded.prefix_cache.is_some() {
            tracing::info!(
                model = %spec.model_id,
                budget_mb = self.prefix_cache_cfg.budget_mb,
                max_entries = self.prefix_cache_cfg.max_entries,
                "prefix cache enabled for this TP model"
            );
        }

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
            return Err(self.trigger_recovery(&model_id).await);
        }
        if debug_poison_armed(&model_id) {
            let _g = span.enter();
            tracing::warn!("NEURON_DEBUG_POISON: forcing auto-recovery (#17 verification)");
            return Err(self.trigger_recovery(&model_id).await);
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
            return Err(self.trigger_recovery(&request.model).await);
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

        let prompt = build_prompt_for_request(tp.chat_template.as_deref(), &request)?;
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

        let tool_schemas = build_tool_schemas(&request);
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
                // copies because the spawn closure owns them. Start in
                // reasoning mode when the prompt itself opens a <think>
                // block (Qwen3.6 injects it into the generation prompt).
                let mut in_reasoning =
                    prompt_opens_reasoning(&prompt_tokens, reasoning_tokens.as_ref());
                let mut in_tool_call = false;
                let mut tool_call_buf = String::new();
                let mut tool_call_idx: usize = 0;
                // Set when a `<tool_call>` block parses into a structured
                // call — promotes the terminal finish_reason to ToolCalls
                // so Anthropic clients see stop_reason: tool_use.
                let mut emitted_tool_call = false;

                'work: {
                    // Prefix-cache decision (#11): vision requests
                    // clear as before; text requests restore the
                    // longest matching snapshot on every rank.
                    let reused = if vision_route.is_some() {
                        match pool.clear_kv_cache(&model_id, leader_handle).await {
                            Ok(()) => 0,
                            Err(e) => {
                                failure = Some(format!("clear_kv_cache: {e:#}"));
                                break 'work;
                            }
                        }
                    } else {
                        match restore_or_clear_tp(&mut pool, &tp_for_task, &prompt_tokens).await {
                            Ok(reused) => reused,
                            Err(e) => {
                                failure = Some(format!("restore_or_clear: {e:#}"));
                                break 'work;
                            }
                        }
                    };

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
                            // Two-stage prefill around the
                            // retokenization-stable snapshot boundary
                            // — see `run_inference_via_worker`.
                            let cut = if tp_for_task.prefix_cache.is_some() {
                                stable_snapshot_cut(
                                    &prompt_tokens,
                                    tokenizer.token_to_id("<|im_start|>"),
                                )
                                .filter(|&c| c > reused)
                            } else {
                                None
                            };
                            match cut {
                                Some(c) => {
                                    match chunked_prefill_tp(
                                        &mut pool,
                                        &model_id,
                                        leader_handle,
                                        &prompt_tokens[..c],
                                        reused,
                                    )
                                    .await
                                    {
                                        Ok(_) => {
                                            store_prefix_snapshot_tp(
                                                &mut pool,
                                                &tp_for_task,
                                                prompt_tokens[..c].to_vec(),
                                            )
                                            .await;
                                            chunked_prefill_tp(
                                                &mut pool,
                                                &model_id,
                                                leader_handle,
                                                &prompt_tokens,
                                                c,
                                            )
                                            .await
                                        }
                                        Err(e) => Err(e),
                                    }
                                }
                                None => {
                                    chunked_prefill_tp(
                                        &mut pool,
                                        &model_id,
                                        leader_handle,
                                        &prompt_tokens,
                                        reused,
                                    )
                                    .await
                                }
                            }
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
                                match parse_tool_call_body(&buffer, idx, &tool_schemas) {
                                    Some((id, name, arguments)) => {
                                        emitted_tool_call = true;
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
                                    match parse_tool_call_body(&buffer, idx, &tool_schemas) {
                                        Some((id, name, arguments)) => {
                                            emitted_tool_call = true;
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
                if emitted_tool_call && finish_reason == FinishReason::Stop {
                    finish_reason = FinishReason::ToolCalls;
                }
                if failure.is_none() {
                    let _ = tx
                        .send(InferenceEvent::Finish {
                            reason: finish_reason,
                            prompt_tokens: prompt_len as u32,
                            completion_tokens: all_tokens.len() as u32,
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

    let prompt = build_prompt_for_request(tp.chat_template.as_deref(), &request)?;
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

    // Prefix-cache decision (#11): restore the longest matching
    // snapshot on every rank, or reset every rank's KV cache as
    // before. Vision requests bypass the cache (image content is not
    // in the token sequence).
    let clear_start = std::time::Instant::now();
    let reused = if vision_route.is_some() {
        pool.clear_kv_cache(&model_id, leader_handle)
            .await
            .map_err(InferenceError::Other)?;
        0
    } else {
        restore_or_clear_tp(&mut pool, &tp, &prompt_tokens)
            .await
            .map_err(InferenceError::Other)?
    };
    tracing::debug!(
        model = %model_id,
        reused,
        elapsed_ms = clear_start.elapsed().as_millis(),
        "TP chat_completion: kv cache ready"
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
        None => {
            // Two-stage prefill around the retokenization-stable
            // snapshot boundary — see `run_inference_via_worker`.
            let cut = if tp.prefix_cache.is_some() {
                stable_snapshot_cut(&prompt_tokens, tp.tokenizer.token_to_id("<|im_start|>"))
                    .filter(|&c| c > reused)
            } else {
                None
            };
            match cut {
                Some(c) => {
                    chunked_prefill_tp(
                        &mut pool,
                        &model_id,
                        leader_handle,
                        &prompt_tokens[..c],
                        reused,
                    )
                    .await
                    .map_err(InferenceError::Other)?;
                    store_prefix_snapshot_tp(&mut pool, &tp, prompt_tokens[..c].to_vec()).await;
                    chunked_prefill_tp(&mut pool, &model_id, leader_handle, &prompt_tokens, c)
                        .await
                        .map_err(InferenceError::Other)?
                }
                None => {
                    chunked_prefill_tp(&mut pool, &model_id, leader_handle, &prompt_tokens, reused)
                        .await
                        .map_err(InferenceError::Other)?
                }
            }
        }
    };
    let (post_prefill_vram_free_mb, _) = tp.query_vram().await;
    tracing::info!(
        model = %model_id,
        prompt_len,
        reused,
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

/// Whether a rendered prompt leaves the model *inside* an open
/// reasoning block. Some chat templates (Qwen3.6) inject the opening
/// `<think>` into the generation prompt, so generation begins
/// mid-thought and the open marker is never *sampled* — leaving
/// `in_reasoning` false would stream the model's thinking out as
/// visible text. Replaying the prompt's reasoning markers and starting
/// the loop in whatever state the prompt ends in fixes that without
/// disabling thinking. `None` pair (non-reasoning model) → false.
fn prompt_opens_reasoning(prompt_tokens: &[u32], pair: Option<&ReasoningTokenPair>) -> bool {
    let Some(pair) = pair else { return false };
    let mut open = false;
    for &t in prompt_tokens {
        if t == pair.open_id {
            open = true;
        } else if t == pair.close_id {
            open = false;
        }
    }
    open
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
/// Per-request tool parameter types, keyed by function name then
/// parameter name (the JSON-schema `type` string, e.g. `"string"`,
/// `"integer"`). Built from the request's OpenAI-shape `tools` array so
/// the Qwen-XML tool-call parser can coerce each `<parameter>` string
/// to its declared JSON type. An empty map (no tools, or untyped
/// params) makes the parser fall back to value-sniffing.
type ToolSchemas = std::collections::HashMap<String, std::collections::HashMap<String, String>>;

/// Extract [`ToolSchemas`] from a request's `tools` (OpenAI shape:
/// `{type:"function", function:{name, parameters:{properties:{p:{type}}}}}`).
/// cortex normalises Anthropic tools into exactly this shape before the
/// request reaches neuron, so one extractor covers both client APIs.
fn build_tool_schemas(request: &ChatCompletionRequest) -> ToolSchemas {
    let mut schemas = ToolSchemas::new();
    let Some(tools) = request.extra.get("tools").and_then(|t| t.as_array()) else {
        return schemas;
    };
    for tool in tools {
        // Tolerate both the wrapped (`{function:{…}}`) and bare shapes.
        let func = tool.get("function").unwrap_or(tool);
        let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let mut params = std::collections::HashMap::new();
        if let Some(props) = func
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (pname, pschema) in props {
                if let Some(ty) = pschema.get("type").and_then(|t| t.as_str()) {
                    params.insert(pname.clone(), ty.to_string());
                }
            }
        }
        schemas.insert(name.to_string(), params);
    }
    schemas
}

/// Parse a buffered `<tool_call>…</tool_call>` body into
/// `(id, name, arguments_json)`. Two on-the-wire forms are accepted:
///
/// 1. **JSON** (Qwen3-Instruct / Hermes): `{"name":…,"arguments":{…}}`.
/// 2. **Qwen-XML** (Qwen3-Coder / Qwen3.6):
///    `<function=NAME><parameter=KEY>VALUE</parameter>…</function>`,
///    with each VALUE coerced to its declared JSON type from `schemas`.
///
/// Returns `None` only when neither form yields a usable name, so the
/// caller can re-emit the raw block as text instead of swallowing it.
fn parse_tool_call_body(
    body: &str,
    index: usize,
    schemas: &ToolSchemas,
) -> Option<(String, String, String)> {
    let trimmed = body.trim();
    // Form 1: a JSON object with a `name`.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(name) = value.get("name").and_then(|n| n.as_str())
    {
        let arguments = value
            .get("arguments")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "{}".into());
        let id = format!("call_{:x}_{}", unix_subsec_nanos(), index);
        return Some((id, name.to_string(), arguments));
    }
    // Form 2: Qwen-XML.
    parse_qwen_xml_tool_call(trimmed, index, schemas)
}

/// Parse the Qwen-XML tool-call body form. See [`parse_tool_call_body`].
fn parse_qwen_xml_tool_call(
    body: &str,
    index: usize,
    schemas: &ToolSchemas,
) -> Option<(String, String, String)> {
    // `<function=NAME>` — name runs to the closing `>` or end of line.
    let after_fn = body.split("<function=").nth(1)?;
    let name = after_fn.split(['>', '\n']).next()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let param_types = schemas.get(&name);

    // `<parameter=KEY>VALUE</parameter>`, repeated. Walk a cursor
    // forward so multiple parameters each get parsed exactly once.
    // `find` (not `split`) so `seg` keeps the *full* remainder — a
    // `split("<parameter=")` would truncate it at the next parameter
    // and only the first would ever parse.
    let mut args = serde_json::Map::new();
    let mut rest = after_fn;
    while let Some(pos) = rest.find("<parameter=") {
        let seg = &rest[pos + "<parameter=".len()..];
        let key = seg
            .split(['>', '\n'])
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let after_gt = seg.split_once('>').map(|(_, after)| after).unwrap_or("");
        let value_raw = after_gt.split("</parameter>").next().unwrap_or("").trim();
        if !key.is_empty() {
            let declared = param_types.and_then(|m| m.get(&key)).map(String::as_str);
            args.insert(key, coerce_param_value(value_raw, declared));
        }
        match after_gt.split_once("</parameter>") {
            Some((_, tail)) => rest = tail,
            None => break,
        }
    }

    let arguments = serde_json::Value::Object(args).to_string();
    let id = format!("call_{:x}_{}", unix_subsec_nanos(), index);
    Some((id, name, arguments))
}

/// Coerce a raw Qwen-XML parameter string to a JSON value using the
/// declared schema type. Unknown/absent type → sniff (JSON-parse, else
/// string). A coercion that fails for the declared type falls back to a
/// string so a mistyped schema never drops the argument.
fn coerce_param_value(raw: &str, declared: Option<&str>) -> serde_json::Value {
    use serde_json::Value;
    let sniff = || serde_json::from_str::<Value>(raw).ok();
    let as_string = || Value::String(raw.to_string());
    match declared {
        Some("string") => as_string(),
        Some("integer") | Some("number") => {
            sniff().filter(Value::is_number).unwrap_or_else(as_string)
        }
        Some("boolean") => match raw.trim() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => as_string(),
        },
        Some("object") | Some("array") => sniff().unwrap_or_else(as_string),
        // No declared type (untyped schema / model invented a param):
        // best-effort sniff, then string.
        _ => sniff().unwrap_or_else(as_string),
    }
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
    /// The loaded model's chat template could not render the request
    /// (e.g. a message / tool-call structure it rejects). Returned only
    /// when the request carried tools — silently degrading to a
    /// tool-less prompt breaks tool calling invisibly, which is the
    /// failure mode that hid several client-compat bugs. Maps to 422.
    #[error("chat template could not render this request: {detail}")]
    TemplateRenderFailed { detail: String },
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
) -> Result<String, InferenceError> {
    if !super::chat_template::chat_templates_enabled() {
        return Ok(format_qwen3_prompt(&request.messages));
    }
    let Some(tmpl) = chat_template else {
        return Ok(format_qwen3_prompt(&request.messages));
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
        Ok(prompt) => {
            // Ground-truth visibility for tool-format debugging: the
            // fully rendered prompt the model actually sees, including
            // whether the chat template's `<tool_call>` format
            // instruction survived a large system prompt + tool list.
            // trace! so it's opt-in (it can be tens of KB):
            // RUST_LOG=neuron::harness::candle=trace.
            tracing::trace!(
                model = %request.model,
                prompt_chars = prompt.len(),
                n_tools = tools.as_array().map(|a| a.len()).unwrap_or(0),
                has_kwargs = !kwargs.is_null(),
                prompt = %prompt,
                "chat_template: rendered prompt"
            );
            Ok(prompt)
        }
        Err(e) => {
            let detail = format!("{e:#}");
            // A tools-bearing request the template can't render must NOT
            // silently degrade to a tool-less fallback prompt — that
            // strips every tool and breaks tool calling invisibly (the
            // failure mode behind the system-message, arguments-format,
            // and tool-render bugs). Surface it as an error instead.
            let has_tools = tools.as_array().is_some_and(|a| !a.is_empty());
            if has_tools {
                tracing::warn!(
                    model = %request.model,
                    error = %detail,
                    "chat_template render failed on a tools-bearing request — returning 422 (refusing silent tool-less fallback)"
                );
                return Err(InferenceError::TemplateRenderFailed { detail });
            }
            tracing::warn!(
                model = %request.model,
                error = %detail,
                "chat_template render failed; falling back to format_qwen3_prompt (no tools to drop)"
            );
            Ok(format_qwen3_prompt(&request.messages))
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

/// Peek at `config.json` for its `model_type`. Best-effort — `None`
/// on any read/parse error. The load path uses this to decide prefix-
/// snapshot capability for worker-held models the async side can't
/// inspect directly.
fn config_model_type(config_path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(config_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("model_type")
        .and_then(|x| x.as_str())
        .map(str::to_string)
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

/// The retokenization-stable snapshot boundary for a prompt: one past
/// the last `<|im_start|>` token.
///
/// A snapshot covering the *whole* prompt is unreliable: the prompt
/// ends with `…<|im_start|>assistant\n`, and when the next turn
/// re-tokenizes that same text followed by the assistant's reply, BPE
/// merges the trailing `\n` with the reply's first characters — the
/// final token(s) differ and the exact-prefix match never fires
/// (observed on Qwen3.6-27B; a reply starting with an atomic special
/// token like `<think>` masks the problem, which is why the 0.8B
/// validation initially passed). Special tokens are hard segmentation
/// points for the tokenizer, so ids up to and including the last
/// `<|im_start|>` are provably identical between the two renders —
/// snapshotting there leaves only the ~2-token `assistant\n` tail to
/// re-prefill on a hit.
///
/// Returns `None` (run the request without storing a snapshot) when
/// the marker id is unknown or the prompt has no usable boundary.
fn stable_snapshot_cut(prompt_tokens: &[u32], im_start_id: Option<u32>) -> Option<usize> {
    let id = im_start_id?;
    let cut = prompt_tokens.iter().rposition(|&t| t == id)? + 1;
    (cut < prompt_tokens.len()).then_some(cut)
}

/// Lock a per-model prefix-cache registry, recovering from a poisoned
/// mutex (a panic mid-bookkeeping leaves the registry consistent
/// enough — worst case a stale entry that a failed restore later
/// drops).
fn lock_prefix_cache(
    cache: &ModelPrefixCache,
) -> std::sync::MutexGuard<'_, super::prefix_cache::PrefixCache<KvSnapshotRef>> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Prefix-cache decision at the start of a text-only request (#11):
/// restore the longest snapshot whose tokens strictly prefix the
/// prompt, or clear the KV cache as before. Returns the number of
/// prompt tokens already in the cache after this call — prefill
/// resumes at that offset. A failed restore drops the entry and falls
/// back to clear + full prefill.
#[cfg(feature = "cuda")]
async fn restore_or_clear_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
    prefix_cache: Option<&ModelPrefixCache>,
    prompt_tokens: &[u32],
) -> Result<usize> {
    // Bind the match outside the lock so the registry guard is
    // released before any await — the guard must never be live across
    // a suspension point.
    let hit = prefix_cache.and_then(|cache| lock_prefix_cache(cache).longest_match(prompt_tokens));
    if let (Some(cache), Some(m)) = (prefix_cache, hit) {
        let KvSnapshotRef::Worker(id) = m.snapshot else {
            anyhow::bail!(
                "prefix cache: local snapshot ref on a worker-loaded model — load-path bug"
            );
        };
        match worker.restore_kv(handle, id).await {
            Ok(()) => {
                tracing::info!(
                    reused_tokens = m.tokens,
                    prompt_tokens = prompt_tokens.len(),
                    "prefix cache: hit — skipping prefill of cached prefix"
                );
                return Ok(m.tokens);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "prefix cache: restore failed; dropping entry, full prefill"
                );
                let removed = lock_prefix_cache(cache).remove_covering(prompt_tokens, m.tokens);
                if let Some(KvSnapshotRef::Worker(id)) = removed {
                    let _ = worker.drop_kv_snapshot(handle, id).await;
                }
            }
        }
    }
    worker
        .clear_kv_cache(handle)
        .await
        .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e}"))?;
    Ok(0)
}

/// Capture-and-register half of prefix caching: snapshot the live
/// cache state at the prefill boundary (covering exactly
/// `prompt_tokens`) and register it. Eviction decided by the
/// registry; evicted worker snapshots are dropped here. Best-effort —
/// a failed snapshot only costs the next request its prefill saving.
#[cfg(feature = "cuda")]
async fn store_prefix_snapshot_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
    prefix_cache: Option<&ModelPrefixCache>,
    prompt_tokens: Vec<u32>,
) {
    let Some(cache) = prefix_cache else { return };
    match worker.snapshot_kv(handle).await {
        Ok((id, bytes)) => {
            let evicted =
                lock_prefix_cache(cache).insert(prompt_tokens, KvSnapshotRef::Worker(id), bytes);
            for r in evicted {
                if let KvSnapshotRef::Worker(evicted_id) = r {
                    let _ = worker.drop_kv_snapshot(handle, evicted_id).await;
                }
            }
        }
        Err(e) => tracing::debug!(error = %e, "prefix cache: snapshot failed; not cached"),
    }
}

/// CPU-path counterpart of [`restore_or_clear_via_worker`]: same
/// decision, directly against the locally-held [`ModelArch`].
fn restore_or_clear_local(
    arch: &mut ModelArch,
    prefix_cache: Option<&ModelPrefixCache>,
    prompt_tokens: &[u32],
) -> Result<usize> {
    let hit = prefix_cache.and_then(|cache| lock_prefix_cache(cache).longest_match(prompt_tokens));
    if let (Some(cache), Some(m)) = (prefix_cache, hit) {
        let KvSnapshotRef::Local(snap) = m.snapshot else {
            anyhow::bail!(
                "prefix cache: worker snapshot ref on a CPU-loaded model — load-path bug"
            );
        };
        match arch.restore_kv_cache(&snap) {
            Ok(()) => {
                tracing::info!(
                    reused_tokens = m.tokens,
                    prompt_tokens = prompt_tokens.len(),
                    "prefix cache: hit — skipping prefill of cached prefix"
                );
                return Ok(m.tokens);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "prefix cache: restore failed; dropping entry, full prefill"
                );
                lock_prefix_cache(cache).remove_covering(prompt_tokens, m.tokens);
            }
        }
    }
    arch.clear_kv_cache()?;
    Ok(0)
}

/// TP counterpart of [`restore_or_clear_via_worker`]: restore the
/// matched snapshot on **every rank** (pool fan-out + leader), or
/// clear everywhere. A failed restore can leave ranks inconsistent
/// (some restored, some not) — the clear fallback resets every rank,
/// restoring consistency.
#[cfg(feature = "cuda")]
async fn restore_or_clear_tp(
    pool: &mut super::tp::WorkerPool,
    tp: &TpLoadedModel,
    prompt_tokens: &[u32],
) -> Result<usize> {
    let hit = tp
        .prefix_cache
        .as_ref()
        .and_then(|cache| lock_prefix_cache(cache).longest_match(prompt_tokens));
    if let (Some(cache), Some(m)) = (tp.prefix_cache.as_ref(), hit) {
        let KvSnapshotRef::Tp(id) = m.snapshot else {
            anyhow::bail!("prefix cache: non-TP snapshot ref on a TP model — load-path bug");
        };
        match pool
            .restore_kv_cache(&tp.model_id, tp.leader_handle, id)
            .await
        {
            Ok(()) => {
                tracing::info!(
                    reused_tokens = m.tokens,
                    prompt_tokens = prompt_tokens.len(),
                    "prefix cache (TP): hit — skipping prefill of cached prefix"
                );
                return Ok(m.tokens);
            }
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "prefix cache (TP): restore failed; dropping entry, full prefill"
                );
                // Bind the removed ref before awaiting — the registry
                // guard must not live across a suspension point.
                let removed = lock_prefix_cache(cache).remove_covering(prompt_tokens, m.tokens);
                if let Some(KvSnapshotRef::Tp(id)) = removed
                    && let Err(e2) = pool
                        .drop_kv_snapshot(&tp.model_id, tp.leader_handle, id)
                        .await
                {
                    tracing::debug!(
                        error = %format!("{e2:#}"),
                        "prefix cache (TP): cleanup of failed-restore snapshot failed"
                    );
                }
            }
        }
    }
    pool.clear_kv_cache(&tp.model_id, tp.leader_handle)
        .await
        .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e:#}"))?;
    Ok(0)
}

/// TP counterpart of [`store_prefix_snapshot_via_worker`]. Mints a
/// pool-wide snapshot id, snapshots every rank under it, registers it.
/// On any rank failing, drops the id everywhere (idempotent) so no
/// rank leaks a half-stored snapshot.
#[cfg(feature = "cuda")]
async fn store_prefix_snapshot_tp(
    pool: &mut super::tp::WorkerPool,
    tp: &TpLoadedModel,
    prompt_tokens: Vec<u32>,
) {
    let Some(cache) = tp.prefix_cache.as_ref() else {
        return;
    };
    let id = tp.next_snapshot_id.fetch_add(1, Ordering::Relaxed);
    match pool
        .snapshot_kv_cache(&tp.model_id, tp.leader_handle, id)
        .await
    {
        Ok(bytes) => {
            let evicted =
                lock_prefix_cache(cache).insert(prompt_tokens, KvSnapshotRef::Tp(id), bytes);
            for r in evicted {
                if let KvSnapshotRef::Tp(evicted_id) = r
                    && let Err(e) = pool
                        .drop_kv_snapshot(&tp.model_id, tp.leader_handle, evicted_id)
                        .await
                {
                    tracing::debug!(
                        error = %format!("{e:#}"),
                        "prefix cache (TP): drop of evicted snapshot failed"
                    );
                }
            }
        }
        Err(e) => {
            tracing::debug!(
                error = %format!("{e:#}"),
                "prefix cache (TP): snapshot failed; cleaning up partial snapshot"
            );
            if let Err(e2) = pool
                .drop_kv_snapshot(&tp.model_id, tp.leader_handle, id)
                .await
            {
                tracing::debug!(
                    error = %format!("{e2:#}"),
                    "prefix cache (TP): partial-snapshot cleanup failed"
                );
            }
        }
    }
}

/// CPU-path counterpart of [`store_prefix_snapshot_via_worker`].
/// Evicted local snapshots free when their `Arc` drops.
fn store_prefix_snapshot_local(
    arch: &ModelArch,
    prefix_cache: Option<&ModelPrefixCache>,
    prompt_tokens: Vec<u32>,
) {
    let Some(cache) = prefix_cache else { return };
    match arch.snapshot_kv_cache() {
        Ok(snap) => {
            let bytes = snap.size_bytes();
            drop(lock_prefix_cache(cache).insert(
                prompt_tokens,
                KvSnapshotRef::Local(Arc::new(snap)),
                bytes,
            ));
        }
        Err(e) => tracing::debug!(error = %e, "prefix cache: snapshot failed; not cached"),
    }
}

#[cfg(feature = "cuda")]
async fn run_inference_via_worker(
    worker: &super::device_worker::DeviceWorkerHandle,
    handle: super::device_worker::ArchHandle,
    prompt_tokens: &[u32],
    prefix_cache: Option<&ModelPrefixCache>,
    im_start_id: Option<u32>,
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

    let reused = restore_or_clear_via_worker(worker, handle, prefix_cache, prompt_tokens).await?;

    // Prefill the prompt in `prefill_chunk_tokens()`-sized chunks so
    // activation memory is bounded per step rather than scaling with
    // prompt length. The KV cache accumulates across chunks; we keep
    // only the final chunk's logits for sampling the first generated
    // token. A restored prefix snapshot skips the first `reused`
    // tokens entirely.
    //
    // When caching is active, prefill pauses at the retokenization-
    // stable boundary (see `stable_snapshot_cut`) to capture the
    // snapshot the next turn can actually match, then finishes the
    // volatile tail. A `reused >= cut` hit means an entry already
    // covers the boundary — no new snapshot needed.
    let cut = if prefix_cache.is_some() {
        stable_snapshot_cut(prompt_tokens, im_start_id).filter(|&c| c > reused)
    } else {
        None
    };
    let logits_vec = match cut {
        Some(c) => {
            chunked_prefill_via_worker(worker, handle, &prompt_tokens[..c], reused).await?;
            store_prefix_snapshot_via_worker(
                worker,
                handle,
                prefix_cache,
                prompt_tokens[..c].to_vec(),
            )
            .await;
            chunked_prefill_via_worker(worker, handle, prompt_tokens, c).await?
        }
        None => chunked_prefill_via_worker(worker, handle, prompt_tokens, reused).await?,
    };
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

    let mut finish_reason = "length";
    if Some(next_token) == eos_id {
        finish_reason = "stop";
    } else {
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
                finish_reason = "stop";
                break;
            }
            generated.push(next_token);
        }
    }

    Ok((generated, finish_reason.into()))
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
    prefix_cache: Option<&ModelPrefixCache>,
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    eos_id: Option<u32>,
    reasoning_tokens: Option<ReasoningTokenPair>,
    tool_call_tokens: Option<ToolCallTokenPair>,
    tool_schemas: ToolSchemas,
    tx: mpsc::Sender<InferenceEvent>,
) -> Result<String> {
    // Image content isn't part of the token sequence, so token-prefix
    // identity would be unsound for vision requests — they bypass the
    // prefix cache entirely (no restore, no snapshot).
    let prefix_cache = if images.is_some() { None } else { prefix_cache };
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
    // `prompt_opens_reasoning`: Qwen3.6 starts the generation prompt
    // inside a <think> block, so begin in reasoning mode if it does.
    let mut in_reasoning = prompt_opens_reasoning(&prompt_tokens, reasoning_tokens.as_ref());
    let mut in_tool_call = false;
    let mut tool_call_buf = String::new();
    let mut tool_call_idx: usize = 0;
    // See `inference_tp_stream`: promotes finish_reason to ToolCalls.
    let mut emitted_tool_call = false;

    // Prefill. Vision-bearing requests (`images = Some`) clear the
    // cache and do a single-shot prefill that splices the image
    // embeddings; text-only requests consult the prefix cache
    // (restore-or-clear) and chunk-prefill only the uncached suffix
    // (see `chunked_prefill_via_worker`) to bound activation memory.
    // Either way the owning `prompt_tokens: Vec<u32>` outlives this
    // step; we use `prompt_len` (already extracted above) for the
    // decode-step offset arithmetic.
    let logits_vec = match images {
        Some((imgs, image_token_id)) => {
            worker
                .clear_kv_cache(handle)
                .await
                .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e}"))?;
            worker
                .forward_logits_with_images(handle, prompt_tokens.clone(), 0, imgs, image_token_id)
                .await
                .map_err(|e| anyhow::anyhow!("forward_logits_with_images: {e}"))?
        }
        None => {
            let reused =
                restore_or_clear_via_worker(&worker, handle, prefix_cache, &prompt_tokens).await?;
            // Two-stage prefill around the retokenization-stable
            // snapshot boundary — see `run_inference_via_worker`.
            let cut = if prefix_cache.is_some() {
                stable_snapshot_cut(&prompt_tokens, tokenizer.token_to_id("<|im_start|>"))
                    .filter(|&c| c > reused)
            } else {
                None
            };
            match cut {
                Some(c) => {
                    chunked_prefill_via_worker(&*worker, handle, &prompt_tokens[..c], reused)
                        .await?;
                    store_prefix_snapshot_via_worker(
                        &worker,
                        handle,
                        prefix_cache,
                        prompt_tokens[..c].to_vec(),
                    )
                    .await;
                    chunked_prefill_via_worker(&*worker, handle, &prompt_tokens, c).await?
                }
                None => {
                    chunked_prefill_via_worker(&*worker, handle, &prompt_tokens, reused).await?
                }
            }
        }
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
                        match parse_tool_call_body(&buffer, idx, &tool_schemas) {
                            Some((id, name, arguments)) => {
                                emitted_tool_call = true;
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
    if emitted_tool_call && finish_reason == FinishReason::Stop {
        finish_reason = FinishReason::ToolCalls;
    }
    let _ = tx
        .send(InferenceEvent::Finish {
            reason: finish_reason,
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: all_tokens.len() as u32,
        })
        .await;

    Ok(finish_reason.as_openai_str().to_string())
}

#[allow(clippy::too_many_arguments)]
fn run_inference(
    arch: &mut ModelArch,
    device: &Device,
    prompt_tokens: &[u32],
    prefix_cache: Option<&ModelPrefixCache>,
    im_start_id: Option<u32>,
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

    let reused = restore_or_clear_local(arch, prefix_cache, prompt_tokens)?;
    // Two-stage prefill around the retokenization-stable snapshot
    // boundary — see `run_inference_via_worker`.
    let cut = if prefix_cache.is_some() {
        stable_snapshot_cut(prompt_tokens, im_start_id).filter(|&c| c > reused)
    } else {
        None
    };
    let logits = match cut {
        Some(c) => {
            chunked_prefill_local(arch, device, &prompt_tokens[..c], reused)?;
            store_prefix_snapshot_local(arch, prefix_cache, prompt_tokens[..c].to_vec());
            chunked_prefill_local(arch, device, prompt_tokens, c)?
        }
        None => chunked_prefill_local(arch, device, prompt_tokens, reused)?,
    };
    let mut next_token = sample_with_penalty(&logits, &generated, &mut logits_processor)?;

    let mut finish_reason = "length";
    if Some(next_token) == eos_id {
        finish_reason = "stop";
    } else {
        generated.push(next_token);
        for index in 0..max_new.saturating_sub(1) {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = arch.forward(&input, prompt_tokens.len() + index)?;
            next_token = sample_with_penalty(&logits, &generated, &mut logits_processor)?;
            if Some(next_token) == eos_id {
                finish_reason = "stop";
                break;
            }
            generated.push(next_token);
        }
    }

    Ok((generated, finish_reason.into()))
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
    prefix_cache: Option<&ModelPrefixCache>,
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    eos_id: Option<u32>,
    reasoning_tokens: Option<&ReasoningTokenPair>,
    tool_call_tokens: Option<&ToolCallTokenPair>,
    tool_schemas: ToolSchemas,
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
    // they exist purely as state transitions. Seeded from the prompt:
    // Qwen3.6 opens a <think> block in the generation prompt itself.
    let mut in_reasoning = prompt_opens_reasoning(prompt_tokens, reasoning_tokens);
    // Tool-call state. While `in_tool_call`, content tokens get
    // accumulated into `tool_call_buf` instead of emitted; on the
    // close marker we parse the buffer and emit a structured
    // ToolCall event (or fall back to passing the raw text
    // through if the buffer doesn't parse).
    let mut in_tool_call = false;
    let mut tool_call_buf = String::new();
    let mut tool_call_idx: usize = 0;
    // See `inference_tp_stream`: promotes finish_reason to ToolCalls.
    let mut emitted_tool_call = false;

    let reused = restore_or_clear_local(arch, prefix_cache, prompt_tokens)?;
    // Two-stage prefill around the retokenization-stable snapshot
    // boundary — see `run_inference_via_worker`.
    let cut = if prefix_cache.is_some() {
        stable_snapshot_cut(prompt_tokens, tokenizer.token_to_id("<|im_start|>"))
            .filter(|&c| c > reused)
    } else {
        None
    };
    let logits = match cut {
        Some(c) => {
            chunked_prefill_local(arch, device, &prompt_tokens[..c], reused)?;
            store_prefix_snapshot_local(arch, prefix_cache, prompt_tokens[..c].to_vec());
            chunked_prefill_local(arch, device, prompt_tokens, c)?
        }
        None => chunked_prefill_local(arch, device, prompt_tokens, reused)?,
    };
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
                    match parse_tool_call_body(&buffer, idx, &tool_schemas) {
                        Some((id, name, arguments)) => {
                            emitted_tool_call = true;
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

    if emitted_tool_call && finish_reason == FinishReason::Stop {
        finish_reason = FinishReason::ToolCalls;
    }
    let _ = tx.blocking_send(InferenceEvent::Finish {
        reason: finish_reason,
        prompt_tokens: prompt_tokens.len() as u32,
        completion_tokens: all_tokens.len() as u32,
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

    const IM_START: u32 = 999;

    #[test]
    fn stable_snapshot_cut_lands_after_last_im_start() {
        // ChatML shape: [im_start, "system", ..., im_start, "user",
        // ..., im_start, "assistant", "\n"] — the cut must be one past
        // the final im_start, leaving the volatile "assistant\n" tail
        // outside the snapshot.
        let prompt = [IM_START, 1, 2, 3, IM_START, 4, 5, IM_START, 6, 7];
        assert_eq!(stable_snapshot_cut(&prompt, Some(IM_START)), Some(8));
    }

    #[test]
    fn stable_snapshot_cut_rejects_degenerate_boundaries() {
        // Marker id unknown → no snapshot.
        assert_eq!(stable_snapshot_cut(&[1, 2, 3], None), None);
        // No marker in the prompt → no snapshot.
        assert_eq!(stable_snapshot_cut(&[1, 2, 3], Some(IM_START)), None);
        // Marker is the final token → cut == len leaves nothing to
        // prefill after the snapshot → no snapshot.
        assert_eq!(stable_snapshot_cut(&[1, IM_START], Some(IM_START)), None);
        // Empty prompt.
        assert_eq!(stable_snapshot_cut(&[], Some(IM_START)), None);
    }

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

    /// #20: a model mid-auto-recovery — registry slot absent during
    /// the unload→reload window — must stay listed in `list_models`
    /// as `recovering`, carrying its trigger-time snapshot, so cortex
    /// holds the route instead of reporting it evicted/unknown.
    #[tokio::test]
    async fn list_models_includes_recovering_models() {
        use crate::config::CandleHarnessConfig;

        let cfg = CandleHarnessConfig::default();
        let harness = CandleHarness::new("http://localhost:13131".into(), &cfg);
        harness.recovering.write().await.insert(
            "Qwen/Qwen3.6-27B".to_string(),
            RecoveringSnapshot {
                devices: vec![0, 1],
                capabilities: vec!["text".into(), "vision".into()],
            },
        );

        let models = harness.list_models().await.expect("list_models");
        let entry = models
            .iter()
            .find(|m| m.id == "Qwen/Qwen3.6-27B")
            .expect("recovering model must remain listed");
        assert_eq!(entry.status, "recovering");
        assert_eq!(entry.devices, vec![0, 1]);
        assert_eq!(
            entry.capabilities,
            vec!["text".to_string(), "vision".to_string()]
        );
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

    // ── Tool-call body parsing ───────────────────────────────────────

    fn weather_schemas() -> ToolSchemas {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                            "metric": {"type": "boolean"}
                        }
                    }
                }
            }]
        }))
        .unwrap();
        build_tool_schemas(&req)
    }

    #[test]
    fn parses_json_tool_call_body() {
        let body = r#"{"name": "get_weather", "arguments": {"city": "Brno"}}"#;
        let (id, name, args) = parse_tool_call_body(body, 0, &ToolSchemas::new()).expect("parsed");
        assert!(id.starts_with("call_"));
        assert_eq!(name, "get_weather");
        assert_eq!(args, r#"{"city":"Brno"}"#);
    }

    #[test]
    fn parses_qwen_xml_tool_call_with_schema_coercion() {
        // The exact shape Qwen3.6-27B emitted on the live fleet.
        let body = "\n<function=get_weather>\n<parameter=city>\nBrno\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n</function>\n";
        let (_, name, args) = parse_tool_call_body(body, 1, &weather_schemas()).expect("parsed");
        assert_eq!(name, "get_weather");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        // string stays a string; integer + boolean are coerced per schema.
        assert_eq!(v["city"], "Brno");
        assert_eq!(v["days"], 3);
        assert_eq!(v["metric"], true);
        assert!(v["days"].is_number());
        assert!(v["metric"].is_boolean());
    }

    #[test]
    fn qwen_xml_without_schema_sniffs_types() {
        let body = "<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=s>\nhello\n</parameter>\n</function>";
        let (_, name, args) = parse_tool_call_body(body, 0, &ToolSchemas::new()).expect("parsed");
        assert_eq!(name, "f");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["n"], 42); // sniffed to number
        assert_eq!(v["s"], "hello"); // un-JSON-parseable → string
    }

    #[test]
    fn unparseable_tool_call_body_returns_none() {
        // Neither JSON nor a `<function=…>` block — caller re-emits as text.
        assert!(parse_tool_call_body("just some prose", 0, &ToolSchemas::new()).is_none());
    }

    #[test]
    fn coerce_falls_back_to_string_on_type_mismatch() {
        use serde_json::Value;
        // Declared integer but value isn't numeric → keep as string,
        // never drop the argument.
        assert_eq!(
            coerce_param_value("not-a-number", Some("integer")),
            Value::String("not-a-number".into())
        );
        assert_eq!(
            coerce_param_value("Brno", Some("string")),
            Value::String("Brno".into())
        );
        assert_eq!(
            coerce_param_value("true", Some("boolean")),
            Value::Bool(true)
        );
    }

    #[test]
    fn prompt_opens_reasoning_tracks_marker_state() {
        let pair = ReasoningTokenPair {
            open_id: 100,
            close_id: 101,
            open_text: "<think>".into(),
            close_text: "</think>".into(),
        };
        // Generation prompt ends with an unclosed <think> (Qwen3.6).
        assert!(prompt_opens_reasoning(&[1, 2, 100], Some(&pair)));
        // A complete <think>…</think> in history leaves us closed.
        assert!(!prompt_opens_reasoning(&[100, 5, 101, 7], Some(&pair)));
        // Last marker wins: closed then reopened at the prompt tail.
        assert!(prompt_opens_reasoning(&[100, 101, 100], Some(&pair)));
        // No markers → closed.
        assert!(!prompt_opens_reasoning(&[1, 2, 3], Some(&pair)));
        // Non-reasoning model (no pair) → always false.
        assert!(!prompt_opens_reasoning(&[100], None));
    }

    #[test]
    fn render_failure_with_tools_errors_instead_of_silent_fallback() {
        // A template that always raises — stands in for the real
        // incompatibilities (system-message position, tool_call arg
        // shape) that made neuron silently drop tools.
        let bad = "{{ raise_exception('boom') }}";

        // Tools present → must surface as an error, never a tool-less
        // fallback prompt.
        let with_tools: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x"}}]
        }))
        .unwrap();
        let err = build_prompt_for_request(Some(bad), &with_tools).unwrap_err();
        assert!(matches!(err, InferenceError::TemplateRenderFailed { .. }));

        // No tools → falling back is harmless, so it stays Ok.
        let no_tools: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert!(build_prompt_for_request(Some(bad), &no_tools).is_ok());
    }
}
