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
    ChatMessage, ChunkChoice, MessageContent, Usage,
};
use serde_json::json;
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
    hf_cache: Option<PathBuf>,
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
fn check_dense_config_supported(config_json: &str, model_id: &str) -> Result<()> {
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

impl CandleHarness {
    pub fn new(bind_url: String, hf_cache: Option<PathBuf>) -> Self {
        let hf_cache = resolve_hf_cache(hf_cache);
        if let Some(p) = &hf_cache {
            tracing::info!(path = %p.display(), "candle harness using HuggingFace cache");
        }
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            hf_cache,
            bind_url,
            device_workers: Arc::new(RwLock::new(HashMap::new())),
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
        device: &Device,
    ) -> Result<(PathBuf, ModelArch)> {
        let (config_path, tokenizer_path, safetensors_paths) =
            self.resolve_dense_files(spec).await?;
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

        let result = async {
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

            let (vram_free_mb, vram_total_mb) = loaded.query_vram().await;
            tracing::info!(
                prompt_len,
                max_new,
                temperature,
                ?top_p,
                ?eos_id,
                vram_free_mb,
                vram_total_mb,
                "chat_completion: starting"
            );

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
                    match run_inference_via_worker(
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
                    {
                        Ok(v) => v,
                        Err(e) => {
                            loaded.poisoned.store(true, Ordering::Release);
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

                // Any failure inside the spawn_blocking touched CUDA via
                // candle's forward / cache code, so we treat it as a
                // device-poisoning event. The terminal log at the bottom
                // of the wrapper reports the error; this flag stops the
                // NEXT request from going down the same path.
                match inference_result {
                    Ok(Ok(v)) => v,
                    Ok(Err(e)) => {
                        loaded.poisoned.store(true, Ordering::Release);
                        return Err(InferenceError::Other(e));
                    }
                    Err(e) => {
                        loaded.poisoned.store(true, Ordering::Release);
                        return Err(InferenceError::Other(anyhow::anyhow!(
                            "inference task panicked: {e}"
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
                return self.chat_completion_tp_stream(m, request).await;
            }
        };

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
        // Refuse if the model is already poisoned. No point opening
        // an SSE stream just to send the role chunk and then bail.
        if loaded.poisoned.load(Ordering::Acquire) {
            return Err(poisoned_error(&model_id));
        }

        // If sending the role chunk fails the receiver is already gone;
        // bail before kicking off the heavy blocking work.
        tx.send(role_chunk)
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
                "chat_completion (stream): starting"
            );
        }
        // Routing parallel to the non-streaming chat_completion: CUDA
        // goes through the worker (async task), CPU keeps the
        // spawn_blocking + Arc<Mutex<ModelArch>> path.
        if let (Some(worker), Some(handle)) = (loaded.worker.clone(), loaded.arch_handle) {
            #[cfg(feature = "cuda")]
            {
                let prompt_tokens = prompt_tokens.clone();
                tokio::spawn(
                    async move {
                        match stream_inference_via_worker(
                            worker,
                            handle,
                            tokenizer,
                            prompt_tokens,
                            max_new,
                            temperature,
                            top_p,
                            seed,
                            eos_id,
                            id,
                            created,
                            model_id,
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
                                loaded_for_task.poisoned.store(true, Ordering::Release);
                                tracing::error!(
                                    error = %format!("{e:#}"),
                                    prompt_tokens = prompt_len,
                                    total_ms = req_start.elapsed().as_millis(),
                                    "chat_completion (stream): failed, model marked poisoned"
                                );
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
            tokio::task::spawn_blocking(move || {
                let _g = span_for_task.enter();
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
                    &id,
                    created,
                    &model_id,
                    &tx,
                ) {
                    Ok(()) => tracing::info!(
                        prompt_tokens = prompt_len,
                        total_ms = req_start.elapsed().as_millis(),
                        "chat_completion (stream): done"
                    ),
                    Err(e) => {
                        loaded_for_task.poisoned.store(true, Ordering::Release);
                        tracing::error!(
                            error = %format!("{e:#}"),
                            prompt_tokens = prompt_len,
                            total_ms = req_start.elapsed().as_millis(),
                            "chat_completion (stream): failed, model marked poisoned"
                        );
                    }
                }
            });
        } else {
            return Err(InferenceError::Other(anyhow::anyhow!(
                "LoadedModel has neither worker handle nor local arch — load-path bug"
            )));
        }

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

        let tp_size = spec.tensor_parallel.unwrap_or(1);
        if tp_size > 1 {
            #[cfg(feature = "cuda")]
            {
                return self.load_tp(spec, tp_size).await;
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

        // Worker thread for the chosen device. CPU loads (CUDA
        // unavailable / not requested) skip the worker — there's no
        // context to own. For CUDA loads, the arch is transferred
        // into the worker's slab now so the inference path can
        // reference it via the returned `ArchHandle`. The explicit
        // type annotation lets the no-cuda build resolve `None` to
        // the right `Option<Arc<DeviceWorkerHandle>>` type.
        let worker: Option<Arc<super::device_worker::DeviceWorkerHandle>> = match &device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Some(self.ensure_device_worker(devices[0]).await?),
            _ => None,
        };
        let (arch_local, arch_handle) = match &worker {
            Some(w) => {
                let handle = w
                    .transfer_in(Box::new(arch))
                    .await
                    .map_err(|e| anyhow::anyhow!("transfer arch into device worker: {e}"))?;
                (None, Some(handle))
            }
            None => (Some(Arc::new(Mutex::new(arch))), None),
        };

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
    async fn load_tp(&self, spec: &ModelSpec, tp_size: u32) -> Result<()> {
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
            self.resolve_dense_files(spec).await?;
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

        let tp_for_marker = Arc::clone(&tp);
        let handle = tokio::spawn(chat_completion_tp_inner(tp, request).instrument(span.clone()));
        let result = match handle.await {
            Ok(r) => r,
            Err(join_err) => Err(InferenceError::Other(anyhow::anyhow!(
                "TP inference task panicked or was cancelled: {join_err}"
            ))),
        };
        if let Err(ref e) = result {
            // Mark poisoned: a failure inside the spawned task either
            // hit a CUDA/NCCL driver error directly or surfaced as a
            // task panic. Both cases leave the worker subprocesses in
            // an unknown state — refuse subsequent requests until an
            // operator unload+reloads. This is the gate that turned
            // the 2026-05-26 silent-hang into a clean 5xx.
            tp_for_marker.poisoned.store(true, Ordering::Release);
            let _g = span.enter();
            tracing::error!(
                error = %format!("{e:#}"),
                total_ms = req_start.elapsed().as_millis(),
                "TP chat_completion: failed, model marked poisoned"
            );
        }
        result
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
    async fn chat_completion_tp_stream(
        &self,
        tp: Arc<TpLoadedModel>,
        request: ChatCompletionRequest,
    ) -> Result<mpsc::Receiver<ChatCompletionChunk>, InferenceError> {
        if tp.poisoned.load(Ordering::Acquire) {
            return Err(poisoned_error(&request.model));
        }

        let prompt = format_qwen3_prompt(&request.messages);
        let encoding = tp
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p;
        let max_new = request.max_tokens.unwrap_or(512) as usize;
        let seed = unix_subsec_nanos();

        let eos_id = tp
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tp.tokenizer.token_to_id("<|endoftext|>"));

        let model_id = request.model.clone();
        let id = format!("chatcmpl-{:x}", unix_subsec_nanos());
        let created = unix_now_secs();
        let tokenizer = tp.tokenizer.clone();

        // Bounded channel — back-pressures the producer when the SSE
        // writer is slow.
        let (tx, rx) = mpsc::channel::<ChatCompletionChunk>(32);

        // Role chunk first, before kicking off the heavy work — if the
        // receiver is gone by now there's no point starting inference.
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
        tx.send(role_chunk)
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
        let tp_for_task = Arc::clone(&tp);
        tokio::spawn(
            async move {
                let mut failure: Option<String> = None;
                let mut pool = acquire_pool_lock(&tp_for_task.pool, &model_id).await;
                let leader_handle = tp_for_task.leader_handle;

                let mut all_tokens: Vec<u32> = Vec::new();
                let mut decoded_prefix = String::new();
                let mut finish_reason = "length".to_string();

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

                    // Prefill — every rank embeds the prompt, offset = 0.
                    let logits_vec = match pool
                        .generate_step(&model_id, leader_handle, prompt_tokens.clone(), 0)
                        .await
                    {
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
                        finish_reason = "stop".into();
                    } else {
                        all_tokens.push(next_token);
                        if !emit_chunk(
                            &all_tokens,
                            &mut decoded_prefix,
                            &tokenizer,
                            &tx,
                            &id,
                            created,
                            &model_id,
                        )
                        .await
                        {
                            // Client gone — treat as normal stream end,
                            // not a failure. No log spam.
                            break 'work;
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
                                finish_reason = "stop".into();
                                break;
                            }
                            all_tokens.push(next_token);
                            if !emit_chunk(
                                &all_tokens,
                                &mut decoded_prefix,
                                &tokenizer,
                                &tx,
                                &id,
                                created,
                                &model_id,
                            )
                            .await
                            {
                                break 'work;
                            }
                        }
                    }
                }

                // One terminal line per request, success or failure. The
                // success branch was previously implicit (the SSE final
                // chunk went out and the spawned task just ended); now
                // there's always a log line for the operator.
                if let Some(err) = &failure {
                    tp_for_task.poisoned.store(true, Ordering::Release);
                    tracing::error!(
                        error = %err,
                        completion_tokens = all_tokens.len(),
                        total_ms = req_start.elapsed().as_millis(),
                        "TP chat_completion (stream): failed, model marked poisoned"
                    );
                } else {
                    tracing::info!(
                        prompt_tokens = prompt_len,
                        completion_tokens = all_tokens.len(),
                        finish_reason = %finish_reason,
                        total_ms = req_start.elapsed().as_millis(),
                        "TP chat_completion (stream): done"
                    );
                }

                // Final chunk carrying finish_reason — only on the success
                // path. On failure we drop the channel so the client sees
                // the SSE stream end abruptly (matches pre-change behaviour
                // when the failed-path early-returned without final chunk).
                if failure.is_none() {
                    let final_chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: serde_json::Value::Object(Default::default()),
                            finish_reason: Some(finish_reason),
                            extra: serde_json::Value::Object(Default::default()),
                        }],
                        usage: None,
                        extra: serde_json::Value::Object(Default::default()),
                    };
                    let _ = tx.send(final_chunk).await;
                }
            }
            .instrument(span),
        );

        Ok(rx)
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

    let prompt = format_qwen3_prompt(&request.messages);
    let encoding = tp
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| InferenceError::Other(anyhow::anyhow!("tokenize: {e}")))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    let temperature = request.temperature.unwrap_or(0.7);
    let top_p = request.top_p;
    let max_new = request.max_tokens.unwrap_or(512) as usize;
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

    // Prefill: every rank embeds the whole prompt, offset = 0.
    let prefill_start = std::time::Instant::now();
    let logits_vec = pool
        .generate_step(&model_id, leader_handle, prompt_tokens.clone(), 0)
        .await
        .map_err(InferenceError::Other)?;
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

/// Decode the cumulative token list, emit the delta (substring appended
/// since the last chunk) as a `chat.completion.chunk`. Returns `false`
/// if the receiver has hung up — the caller should bail.
#[cfg(feature = "cuda")]
async fn emit_chunk(
    all_tokens: &[u32],
    decoded_prefix: &mut String,
    tokenizer: &Tokenizer,
    tx: &mpsc::Sender<ChatCompletionChunk>,
    id: &str,
    created: u64,
    model_id: &str,
) -> bool {
    let full = match tokenizer.decode(all_tokens, true) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "TP stream: decode failed");
            return false;
        }
    };
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
        if tx.send(chunk).await.is_err() {
            return false;
        }
    }
    true
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

    // Prefill — every rank embeds the prompt with offset 0.
    let logits_vec = worker
        .forward_logits(handle, prompt_tokens.to_vec(), 0)
        .await
        .map_err(|e| anyhow::anyhow!("prefill forward: {e}"))?;
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
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn stream_inference_via_worker(
    worker: Arc<super::device_worker::DeviceWorkerHandle>,
    handle: super::device_worker::ArchHandle,
    tokenizer: Tokenizer,
    prompt_tokens: Vec<u32>,
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    eos_id: Option<u32>,
    id: String,
    created: u64,
    model_id: String,
    tx: mpsc::Sender<ChatCompletionChunk>,
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
    let mut decoded_prefix = String::new();
    let prompt_len = prompt_tokens.len();
    let mut finish_reason = "length".to_string();

    worker
        .clear_kv_cache(handle)
        .await
        .map_err(|e| anyhow::anyhow!("clear_kv_cache: {e}"))?;

    let logits_vec = worker
        .forward_logits(handle, prompt_tokens, 0)
        .await
        .map_err(|e| anyhow::anyhow!("prefill forward: {e}"))?;
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

    if Some(next_token) == eos_id {
        finish_reason = "stop".into();
    } else {
        all_tokens.push(next_token);
        if !emit_chunk(
            &all_tokens,
            &mut decoded_prefix,
            &tokenizer,
            &tx,
            &id,
            created,
            &model_id,
        )
        .await
        {
            return Ok(finish_reason); // Client gone — clean stream end.
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
                finish_reason = "stop".into();
                break;
            }
            all_tokens.push(next_token);
            if !emit_chunk(
                &all_tokens,
                &mut decoded_prefix,
                &tokenizer,
                &tx,
                &id,
                created,
                &model_id,
            )
            .await
            {
                return Ok(finish_reason);
            }
        }
    }

    // Final chunk carrying finish_reason. Matches the run_inference_streaming
    // shape so the SSE consumer sees an identical termination sequence.
    let final_chunk = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: serde_json::Value::Object(Default::default()),
            finish_reason: Some(finish_reason.clone()),
            extra: serde_json::Value::Object(Default::default()),
        }],
        usage: None,
        extra: serde_json::Value::Object(Default::default()),
    };
    let _ = tx.send(final_chunk).await;

    Ok(finish_reason)
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
    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = arch.forward(&input, 0)?;
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

    arch.clear_kv_cache()?;
    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = arch.forward(&input, 0)?;
    let mut next_token = sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?;

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
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = arch.forward(&input, prompt_tokens.len() + index)?;
            next_token = sample_with_penalty(&logits, &all_tokens, &mut logits_processor)?;
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
}
