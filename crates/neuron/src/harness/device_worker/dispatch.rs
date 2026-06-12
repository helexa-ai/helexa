//! Synchronous dispatch loop running on the device worker thread.
//!
//! `run()` is the thread's entry point. It binds the CUDA context for
//! its device on startup, then pulls `Job`s off the channel one at a
//! time and runs the corresponding handler. The handlers are
//! synchronous by design — the only async on this thread is the
//! one-line `oneshot::Sender::send` call to ship the reply back, which
//! is non-blocking.
//!
//! Phase 2 handles QueryVram, TransferIn, DropArch, ClearKv,
//! ForwardLogits, Shutdown. Phase 3 will add the TP variants
//! (NcclInit, NcclSanity, TpLoadShard, TpForward, TpClearKv) and the
//! ARCH model state in this state slab will gain a companion
//! `tp_models: HashMap<TpHandle, Box<TpLeaderModel>>`.

use crate::harness::arch::qwen3_5::snapshot::KvCacheSnapshot;
use crate::harness::candle::ModelArch;
#[cfg(feature = "cuda")]
use crate::harness::device_worker::jobs::TpHandle;
use crate::harness::device_worker::jobs::{ArchHandle, ImageInput, Job, KvSnapshotId};
#[cfg(feature = "cuda")]
use crate::harness::tp::TpLeaderModel;
use crate::harness::tp::nccl_state::NcclState;
use anyhow::Context as _;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;

/// Per-thread state owned by the worker. On CUDA builds the `Arc<CudaContext>`
/// is created and bound at thread startup; on CPU builds the struct
/// is mostly empty.
struct DeviceWorkerState {
    #[allow(dead_code)]
    device_index: u32,
    /// Candle `Device` constructed at startup. Used by handlers (e.g.
    /// `ForwardLogits`) to build input tensors against the right
    /// device. Falls back to `Device::Cpu` if CUDA init fails.
    device: candle_core::Device,
    /// Boxed `ModelArch` slab. Indexed by an opaque `ArchHandle` minted
    /// by `TransferIn`. The Box means the entry's address is stable
    /// across HashMap rehashes (relevant only when we later hand out
    /// `&mut ModelArch` references — for Phase 2 every handler runs
    /// `&mut` via `get_mut`, no long-lived borrows).
    models: HashMap<ArchHandle, Box<ModelArch>>,
    /// Counter for minting fresh `ArchHandle`s. Each `TransferIn`
    /// increments and returns the new value. Wraps at u64::MAX after
    /// ~10^19 model loads — not a practical concern.
    next_handle: u64,
    /// Prefix-cache snapshots (#11), keyed by the owning model's
    /// handle plus a per-worker snapshot counter. Kept beside the
    /// model slab (not inside it) so every existing `get_mut` on
    /// `models` stays untouched; `DropArch` retains this map down so
    /// snapshot tensors drop on this thread alongside the model's.
    kv_snapshots: HashMap<(ArchHandle, u64), KvCacheSnapshot>,
    /// Counter for minting fresh `KvSnapshotId`s.
    next_kv_snapshot_id: u64,
    /// Leader's NCCL state. Populated by `Job::NcclInit`; the
    /// underlying `Comm`'s libnccl handle lives bound to this thread
    /// for its entire lifetime. Subprocess workers maintain their own
    /// `NcclState` in their own processes — that's not visible from
    /// here.
    #[allow(dead_code)] // Read only via methods on NcclState
    nccl: NcclState,
    /// TP leader model slab. Same lifecycle as `models`; separate
    /// namespace so `ArchHandle` and `TpHandle` can't collide.
    #[cfg(feature = "cuda")]
    tp_models: HashMap<TpHandle, Box<TpLeaderModel>>,
    /// Counter for minting fresh `TpHandle`s.
    #[cfg(feature = "cuda")]
    next_tp_handle: u64,
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    /// `None` only if `CudaContext::new()` failed — in that case the
    /// thread still runs so the handle's lifecycle stays uniform, but
    /// every job that touches CUDA falls through to a zero reply with
    /// a log warning.
    ctx: Option<Arc<candle_core::cuda::cudarc::driver::CudaContext>>,
}

/// Worker thread entry point. Runs until `Job::Shutdown` arrives or
/// the channel sender is dropped (which happens when the last
/// `DeviceWorkerHandle` `Arc` is dropped without an explicit
/// `shutdown()`).
pub(crate) fn run(device_index: u32, rx: Receiver<Job>, poisoned: Arc<AtomicBool>) {
    let mut state = init_state(device_index);
    tracing::info!(device_index, "device worker started");

    while let Ok(job) = rx.recv() {
        // Shutdown is processed unconditionally so a poisoned worker
        // still exits when asked. Matching by reference first so we
        // can fall through to the consume-match below.
        if matches!(&job, Job::Shutdown) {
            break;
        }
        if poisoned.load(Ordering::Acquire) {
            // Drain-only mode: reply with a poisoned error without
            // touching CUDA. Phase 1/2 never set the flag from the
            // dispatch loop itself (no driver errors classified yet),
            // but tests use `DeviceWorkerHandle::set_poisoned()` to
            // simulate this state.
            drain_poisoned(job, device_index);
            continue;
        }
        match job {
            Job::QueryVram { reply } => {
                let result = query_vram(&state);
                // If the caller dropped its receiver (request cancelled,
                // gateway timed out) the send fails — fine, we just
                // discard the reply.
                let _ = reply.send(result);
            }
            Job::LoadGguf {
                gguf_path,
                model_id,
                reply,
            } => {
                let result = load_gguf_inner(&state.device, &gguf_path, &model_id)
                    .map(|arch| insert_arch(&mut state, Box::new(arch)));
                let _ = reply.send(result);
            }
            Job::LoadDense {
                config_path,
                safetensors_paths,
                model_id,
                reply,
            } => {
                let result =
                    load_dense_inner(&state.device, &config_path, &safetensors_paths, &model_id)
                        .map(|arch| insert_arch(&mut state, Box::new(arch)));
                let _ = reply.send(result);
            }
            Job::DropArch { handle, reply } => {
                let removed = state.models.remove(&handle);
                let was_present = removed.is_some();
                // Prefix snapshots are scoped to the model: drop them
                // here (on this thread) so a stale async-side id can
                // never resurrect tensors from an unloaded model.
                state.kv_snapshots.retain(|(h, _), _| *h != handle);
                // Explicit drop on this thread — runs the Box<ModelArch>
                // Drop with the CUDA context bound here, which frees
                // all device tensors on the right context. The Drop is
                // implicit on the `removed` value going out of scope at
                // the end of the arm; calling drop() explicitly just
                // makes the intent visible.
                drop(removed);
                tracing::debug!(
                    device_index,
                    handle = handle.0,
                    was_present,
                    slab_size = state.models.len(),
                    "device worker: model dropped"
                );
                let _ = reply.send(());
            }
            Job::ClearKv { handle, reply } => {
                let result = match state.models.get_mut(&handle) {
                    Some(arch) => arch.clear_kv_cache(),
                    None => Err(anyhow::anyhow!("ClearKv: no model for handle {}", handle.0)),
                };
                if result.is_ok() {
                    trim_device_pool(&state);
                }
                let _ = reply.send(result);
            }
            Job::SnapshotKv { handle, reply } => {
                let result = match state.models.get(&handle) {
                    Some(arch) => arch.snapshot_kv_cache().map(|snap| {
                        let id = KvSnapshotId(state.next_kv_snapshot_id);
                        state.next_kv_snapshot_id = state.next_kv_snapshot_id.wrapping_add(1);
                        let bytes = snap.size_bytes();
                        state.kv_snapshots.insert((handle, id.0), snap);
                        tracing::debug!(
                            device_index,
                            handle = handle.0,
                            snapshot = id.0,
                            bytes,
                            stored = state.kv_snapshots.len(),
                            "device worker: kv snapshot captured"
                        );
                        (id, bytes)
                    }),
                    None => Err(anyhow::anyhow!(
                        "SnapshotKv: no model for handle {}",
                        handle.0
                    )),
                };
                let _ = reply.send(result);
            }
            Job::RestoreKv {
                handle,
                snapshot,
                reply,
            } => {
                let result = match (
                    state.models.get_mut(&handle),
                    state.kv_snapshots.get(&(handle, snapshot.0)),
                ) {
                    (Some(arch), Some(snap)) => arch.restore_kv_cache(snap),
                    (None, _) => Err(anyhow::anyhow!(
                        "RestoreKv: no model for handle {}",
                        handle.0
                    )),
                    (_, None) => Err(anyhow::anyhow!(
                        "RestoreKv: no snapshot {} for handle {}",
                        snapshot.0,
                        handle.0
                    )),
                };
                // The replaced live cache state just freed its
                // tensors — same release-to-driver point as ClearKv.
                if result.is_ok() {
                    trim_device_pool(&state);
                }
                let _ = reply.send(result);
            }
            Job::DropKvSnapshot {
                handle,
                snapshot,
                reply,
            } => {
                let was_present = state.kv_snapshots.remove(&(handle, snapshot.0)).is_some();
                if was_present {
                    trim_device_pool(&state);
                }
                tracing::debug!(
                    device_index,
                    handle = handle.0,
                    snapshot = snapshot.0,
                    was_present,
                    stored = state.kv_snapshots.len(),
                    "device worker: kv snapshot dropped"
                );
                let _ = reply.send(());
            }
            Job::ForwardLogits {
                handle,
                tokens,
                offset,
                reply,
            } => {
                let result = forward_logits(&mut state, handle, &tokens, offset);
                let _ = reply.send(result);
            }
            Job::EncodeImage {
                handle,
                pixels,
                c,
                h,
                w,
                reply,
            } => {
                let result = encode_image(&mut state, handle, pixels, c, h, w);
                let _ = reply.send(result);
            }
            Job::ForwardLogitsWithImages {
                handle,
                tokens,
                offset,
                images,
                image_token_id,
                reply,
            } => {
                let result = forward_logits_with_images(
                    &mut state,
                    handle,
                    &tokens,
                    offset,
                    images,
                    image_token_id,
                );
                let _ = reply.send(result);
            }
            Job::NcclInit {
                cfg,
                comm_id_hex,
                reply,
            } => {
                let resp = state.nccl.init(cfg, &comm_id_hex);
                let _ = reply.send(resp);
            }
            Job::NcclSanity { reply } => {
                let resp = state.nccl.sanity_check();
                let _ = reply.send(resp);
            }
            #[cfg(feature = "cuda")]
            Job::GetLeaderComm { reply } => {
                // Clone the leader's Arc<Comm> out for the async-side
                // watchdog. `None` before NcclInit. (#17 Stage 2)
                let comm = state
                    .nccl
                    .comm()
                    .map(crate::harness::tp::nccl_state::SendComm);
                let _ = reply.send(comm);
            }
            #[cfg(feature = "cuda")]
            Job::TpLoadShard {
                model_id,
                config_json,
                safetensors_paths,
                dtype,
                quant,
                world_size,
                reply,
            } => {
                let result = tp_load_shard_inner(
                    &mut state,
                    &model_id,
                    &config_json,
                    &safetensors_paths,
                    dtype,
                    quant.as_deref(),
                    world_size,
                );
                let _ = reply.send(result);
            }
            #[cfg(feature = "cuda")]
            Job::DropTp { handle, reply } => {
                let removed = state.tp_models.remove(&handle);
                let was_present = removed.is_some();
                drop(removed);
                tracing::debug!(
                    device_index,
                    tp_handle = handle.0,
                    was_present,
                    slab_size = state.tp_models.len(),
                    "device worker: TP model dropped"
                );
                let _ = reply.send(());
            }
            #[cfg(feature = "cuda")]
            Job::TpClearKv { handle, reply } => {
                let result = match state.tp_models.get_mut(&handle) {
                    Some(model) => {
                        model.clear_kv_cache();
                        Ok(())
                    }
                    None => Err(anyhow::anyhow!(
                        "TpClearKv: no TP model for handle {}",
                        handle.0
                    )),
                };
                if result.is_ok() {
                    trim_device_pool(&state);
                }
                let _ = reply.send(result);
            }
            #[cfg(feature = "cuda")]
            Job::TpForwardLogits {
                handle,
                tokens,
                offset,
                reply,
            } => {
                let result = tp_forward_logits(&mut state, handle, &tokens, offset);
                let _ = reply.send(result);
            }
            #[cfg(feature = "cuda")]
            Job::TpForwardLogitsWithImages {
                handle,
                tokens,
                offset,
                image_token_id,
                image_data_uris,
                chunk_size,
                reply,
            } => {
                let result = tp_forward_logits_with_images(
                    &mut state,
                    handle,
                    &tokens,
                    offset,
                    image_token_id,
                    &image_data_uris,
                    chunk_size,
                );
                let _ = reply.send(result);
            }
            // Handled by the matches!() check above; reaching here
            // means a Shutdown slipped past which is a bug.
            Job::Shutdown => unreachable!("Shutdown should break above"),
        }
    }

    #[cfg(feature = "cuda")]
    let tp_slab_size = state.tp_models.len();
    #[cfg(not(feature = "cuda"))]
    let tp_slab_size = 0_usize;
    tracing::info!(
        device_index,
        slab_size = state.models.len(),
        tp_slab_size,
        "device worker exiting; dropping remaining models"
    );
    // Drops every model in the slab on this thread before the function
    // returns. Critical for CUDA tensors: dropping on a thread that
    // doesn't have the context bound is UB. Phase 2 still runs Drop
    // via the slab going out of scope, which is correct as long as no
    // pre-poisoned state lurks in here — see the poisoned-mode
    // semantics in mod.rs for the Phase 3+ refinement.
}

fn init_state(device_index: u32) -> DeviceWorkerState {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda::cudarc::driver::CudaContext;
        // Construct a candle Device first — cudarc returns the
        // primary context for this index on subsequent calls, so
        // CudaContext::new and Device::new_cuda end up sharing state.
        let (device, ctx) = match candle_core::Device::new_cuda(device_index as usize) {
            Ok(device) => match CudaContext::new(device_index as usize) {
                Ok(ctx) => {
                    if let Err(e) = ctx.bind_to_thread() {
                        tracing::warn!(
                            device_index,
                            error = ?e,
                            "device worker: bind_to_thread failed; \
                             operations will still rebind per-call"
                        );
                    } else {
                        tracing::info!(device_index, "device worker bound CUDA context");
                    }
                    (device, Some(ctx))
                }
                Err(e) => {
                    tracing::warn!(
                        device_index,
                        error = ?e,
                        "device worker: CudaContext::new failed; \
                         vram queries will return (0, 0), forward will error"
                    );
                    (device, None)
                }
            },
            Err(e) => {
                tracing::warn!(
                    device_index,
                    error = %e,
                    "device worker: Device::new_cuda failed; falling back to CPU device"
                );
                (candle_core::Device::Cpu, None)
            }
        };
        DeviceWorkerState {
            device_index,
            device,
            models: HashMap::new(),
            next_handle: 1,
            kv_snapshots: HashMap::new(),
            next_kv_snapshot_id: 1,
            nccl: NcclState::new(),
            tp_models: HashMap::new(),
            next_tp_handle: 1,
            ctx,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        DeviceWorkerState {
            device_index,
            device: candle_core::Device::Cpu,
            models: HashMap::new(),
            next_handle: 1,
            kv_snapshots: HashMap::new(),
            next_kv_snapshot_id: 1,
            nccl: NcclState::new(),
        }
    }
}

#[cfg(feature = "cuda")]
fn query_vram(state: &DeviceWorkerState) -> anyhow::Result<(u64, u64)> {
    use candle_core::cuda::cudarc::driver::result;
    if state.ctx.is_none() {
        return Ok((0, 0));
    }
    // The context was bound in init_state. cudarc's `mem_get_info`
    // reads from the current context on the calling thread; since we
    // bound on startup and we never spawn child threads from this
    // worker, the binding holds.
    match result::mem_get_info() {
        Ok((free, total)) => Ok((
            (free / (1024 * 1024)) as u64,
            (total / (1024 * 1024)) as u64,
        )),
        Err(e) => Err(anyhow::anyhow!("mem_get_info: {e:?}")),
    }
}

#[cfg(not(feature = "cuda"))]
fn query_vram(_state: &DeviceWorkerState) -> anyhow::Result<(u64, u64)> {
    Ok((0, 0))
}

/// Force cudarc's stream-ordered memory pool to release every block it
/// is holding back to the system. After `ConcatKvCache::reset()` drops
/// its tensors, the underlying `CudaSlice::drop` calls `cuMemFreeAsync`,
/// which returns the blocks to the device's default mempool but not to
/// the OS — `mem_get_info` still reports them as used. The next
/// request's prefill then sees a falsely-small free pool and either
/// OOMs or trips cuBLAS into `CUBLAS_STATUS_INTERNAL_ERROR`.
///
/// Calling `cuMemPoolTrimTo(pool, 0)` after each `clear_kv_cache`
/// returns those blocks. We synchronize first so any pending
/// `cuMemFreeAsync` operations have settled. Failures are non-fatal:
/// the pool may not exist on legacy drivers, or a transient driver
/// error may prevent the trim — neither breaks correctness, the next
/// request just sees a less-recovered free pool.
#[cfg(feature = "cuda")]
fn trim_device_pool(state: &DeviceWorkerState) {
    use candle_core::cuda::cudarc::driver::result::{device, mem_pool};
    let Some(ctx) = state.ctx.as_ref() else {
        return;
    };
    let (before_free, _) = match query_vram(state) {
        Ok(v) => v,
        Err(_) => (0, 0),
    };
    if let Err(e) = ctx.synchronize() {
        tracing::debug!(
            device_index = state.device_index,
            error = ?e,
            "trim_device_pool: synchronize failed; skipping trim"
        );
        return;
    }
    let dev = ctx.cu_device();
    let pool = match unsafe { device::get_default_mem_pool(dev) } {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(
                device_index = state.device_index,
                error = ?e,
                "trim_device_pool: get_default_mem_pool failed"
            );
            return;
        }
    };
    if let Err(e) = unsafe { mem_pool::trim_to(pool, 0) } {
        tracing::debug!(
            device_index = state.device_index,
            error = ?e,
            "trim_device_pool: cuMemPoolTrimTo failed"
        );
        return;
    }
    let (after_free, _) = match query_vram(state) {
        Ok(v) => v,
        Err(_) => (0, 0),
    };
    let freed_mb = after_free.saturating_sub(before_free);
    tracing::debug!(
        device_index = state.device_index,
        before_free_mb = before_free,
        after_free_mb = after_free,
        freed_mb,
        "trim_device_pool: trimmed pool"
    );
}

#[cfg(not(feature = "cuda"))]
fn trim_device_pool(_state: &DeviceWorkerState) {}

/// Insert a freshly-built `ModelArch` into the slab and mint a fresh
/// `ArchHandle`. Used by both `LoadGguf` and `LoadDense` dispatch
/// handlers — they differ only in *how* the arch is built; the
/// post-construction bookkeeping is identical.
fn insert_arch(state: &mut DeviceWorkerState, arch: Box<ModelArch>) -> ArchHandle {
    let handle = ArchHandle(state.next_handle);
    state.next_handle = state.next_handle.wrapping_add(1);
    state.models.insert(handle, arch);
    tracing::debug!(
        device_index = state.device_index,
        handle = handle.0,
        slab_size = state.models.len(),
        "device worker: model inserted"
    );
    handle
}

/// Load a GGUF (pre-quantized) model on the worker thread. Pulled
/// verbatim from the spawn_blocking closure that used to live in
/// `CandleHarness::load_arch_gguf`; the only change is that `device`
/// is now `state.device` (the worker's permanently-bound device).
fn load_gguf_inner(
    device: &candle_core::Device,
    gguf_path: &std::path::Path,
    model_id: &str,
) -> anyhow::Result<ModelArch> {
    use anyhow::Context;
    use candle_core::DType;
    use candle_core::quantized::gguf_file;
    use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaWeights;
    use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Weights;
    use candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE;

    tracing::info!(model = %model_id, path = ?gguf_path, "loading GGUF");
    let mut file = std::fs::File::open(gguf_path).context("open GGUF file")?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("parse GGUF: {e}"))?;

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
            let weights = QuantizedQwen3Weights::from_gguf(content, &mut file, device)
                .map_err(|e| anyhow::anyhow!("from_gguf qwen3: {e}"))?;
            Ok(ModelArch::Qwen3Quantized(weights))
        }
        "qwen3moe" => {
            // GGUFQWenMoE takes an explicit compute dtype alongside
            // the device — F16 matches the GGUF weights' typical
            // accumulation precision and gives the best tokens/sec on
            // consumer cards.
            let weights = GGUFQWenMoE::from_gguf(content, &mut file, device, DType::F16)
                .map_err(|e| anyhow::anyhow!("from_gguf qwen3_moe: {e}"))?;
            Ok(ModelArch::Qwen3MoeQuantized(weights))
        }
        "llama" => {
            let weights = QuantizedLlamaWeights::from_gguf(content, &mut file, device)
                .map_err(|e| anyhow::anyhow!("from_gguf llama: {e}"))?;
            Ok(ModelArch::LlamaQuantized(weights))
        }
        other => anyhow::bail!(
            "unsupported GGUF architecture '{other}'; quantized path supports \
             qwen3, qwen3moe, llama"
        ),
    }
}

/// Load a dense safetensors model on the worker thread.
fn load_dense_inner(
    device: &candle_core::Device,
    config_path: &std::path::Path,
    safetensors_paths: &[std::path::PathBuf],
    model_id: &str,
) -> anyhow::Result<ModelArch> {
    use anyhow::Context;
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use candle_transformers::models::llama as llama_dense;
    use candle_transformers::models::qwen3 as qwen3_dense;
    use candle_transformers::models::qwen3_moe as qwen3_moe_dense;

    let cfg_text = std::fs::read_to_string(config_path).context("read config.json")?;
    crate::harness::candle::check_dense_config_supported(&cfg_text, model_id)?;
    // Peek at model_type to choose the family before the typed
    // deserialize — each family has its own Config.
    let model_type = serde_json::from_str::<serde_json::Value>(&cfg_text)
        .ok()
        .as_ref()
        .and_then(|v| v.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    tracing::info!(
        model = %model_id,
        model_type = %model_type,
        shards = safetensors_paths.len(),
        "loading dense model from safetensors"
    );

    // bf16 is the canonical distribution dtype for Qwen3 / Llama 3 /
    // Qwen3 MoE. CUDA on Ada+ has hardware bf16; Ampere has it too.
    // CPU emulates.
    let dtype = DType::BF16;
    // SAFETY: VarBuilder::from_mmaped_safetensors mmaps the files;
    // mutation by another process while we hold the mapping is UB.
    // We trust the HF cache is immutable-by-design.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(safetensors_paths, dtype, device)
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
            let config = cfg.into_config(false);
            let cache = llama_dense::Cache::new(true, dtype, &config, device)
                .context("build Llama Cache")?;
            let model = llama_dense::Llama::load(vb, &config)
                .map_err(|e| anyhow::anyhow!("build Llama dense model: {e}"))?;
            Ok(ModelArch::LlamaDense(Box::new(
                crate::harness::candle::LlamaDense::from_parts(
                    model,
                    cache,
                    config,
                    dtype,
                    device.clone(),
                ),
            )))
        }
        "qwen3_5" => {
            let cfg: crate::harness::arch::qwen3_5::Config = serde_json::from_str(&cfg_text)
                .context("parse Qwen3-Next (qwen3_5) config.json")?;
            let sharded_vb = unsafe {
                candle_nn::var_builder::ShardedSafeTensors::var_builder(
                    safetensors_paths,
                    dtype,
                    device,
                )
                .context("build ShardedVarBuilder for Qwen3-Next")?
            };
            let model = crate::harness::arch::qwen3_5::Qwen3_5ForCausalLM::new(cfg, sharded_vb)
                .context("build Qwen3-Next dense model")?;
            Ok(ModelArch::Qwen3_5Dense(model))
        }
        other => anyhow::bail!(
            "unrouted supported model_type '{other}' — \
             DENSE_SUPPORTED_MODEL_TYPES and load_dense_inner \
             must stay in sync"
        ),
    }
}

/// Load the leader's TP shard on the worker thread. Reads the Comm
/// directly from `state.nccl`; no cross-thread Arc<Comm> transfer.
#[cfg(feature = "cuda")]
fn tp_load_shard_inner(
    state: &mut DeviceWorkerState,
    model_id: &str,
    config_json: &str,
    safetensors_paths: &[std::path::PathBuf],
    dtype: candle_core::DType,
    quant: Option<&str>,
    world_size: u32,
) -> anyhow::Result<TpHandle> {
    use anyhow::Context;
    use candle_nn::var_builder::ShardedSafeTensors;

    let comm = state.nccl.comm().ok_or_else(|| {
        anyhow::anyhow!("TpLoadShard: NcclState has no Comm; call NcclInit first")
    })?;

    let model_type = serde_json::from_str::<serde_json::Value>(config_json)
        .ok()
        .as_ref()
        .and_then(|v| v.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // SAFETY: same invariant as the single-GPU dense path — the HF
    // cache files are treated as immutable while the mmap is held.
    let vb = unsafe {
        ShardedSafeTensors::var_builder(safetensors_paths, dtype, &state.device)
            .context("build ShardedVarBuilder over safetensors")?
    };
    let mmap = unsafe {
        candle_core::safetensors::MmapedSafetensors::multi(safetensors_paths)
            .context("build MmapedSafetensors for leader load")?
    };

    let loaded = match model_type.as_str() {
        "qwen3" => {
            let cfg: crate::harness::tp::tp_qwen3::Config = serde_json::from_str(config_json)
                .context("parse Qwen3 Config JSON for leader load")?;
            TpLeaderModel::Qwen3(crate::harness::tp::tp_qwen3::TpQwen3ForCausalLM::load(
                &cfg, &vb, 0, world_size, comm,
            )?)
        }
        "qwen3_5" => {
            let cfg: crate::harness::tp::tp_qwen3_5::Config = serde_json::from_str(config_json)
                .context("parse Qwen3-Next Config JSON for leader load")?;
            let quant_dtype = crate::harness::tp::worker::parse_quant_string(quant)?;
            TpLeaderModel::Qwen3_5(crate::harness::tp::tp_qwen3_5::TpQwen3_5ForCausalLM::load(
                cfg,
                &vb,
                &mmap,
                0,
                world_size,
                comm,
                quant_dtype,
            )?)
        }
        other => anyhow::bail!(
            "TP dispatch: unsupported model_type '{other}' on leader (supported: qwen3, qwen3_5)"
        ),
    };

    tracing::info!(
        rank = 0,
        model = %model_id,
        model_type = %model_type,
        "loaded TP shard (leader)"
    );

    let handle = TpHandle(state.next_tp_handle);
    state.next_tp_handle = state.next_tp_handle.wrapping_add(1);
    state.tp_models.insert(handle, Box::new(loaded));
    tracing::debug!(
        device_index = state.device_index,
        tp_handle = handle.0,
        slab_size = state.tp_models.len(),
        "device worker: TP model inserted"
    );
    Ok(handle)
}

/// TP-equivalent of [`forward_logits`]: looks up the leader's
/// [`TpLeaderModel`] in the slab, runs its forward, copies the
/// `[vocab]` logits to a CPU `Vec<f32>`. The leader's `Arc<Comm>`
/// clones embedded in the TP layers' AllReduce ops fire from this
/// thread — same thread that bound the CUDA context and that holds
/// the `Comm` in `state.nccl`.
#[cfg(feature = "cuda")]
fn tp_forward_logits(
    state: &mut DeviceWorkerState,
    handle: TpHandle,
    tokens: &[u32],
    offset: usize,
) -> anyhow::Result<Vec<f32>> {
    use candle_core::{DType, Tensor};

    let input = Tensor::new(tokens, &state.device)?.unsqueeze(0)?;

    let model = state
        .tp_models
        .get_mut(&handle)
        .ok_or_else(|| anyhow::anyhow!("TpForwardLogits: no model for handle {}", handle.0))?;

    let logits = model.forward(&input, offset)?;
    // ForCausalLM forward returns [B, 1, V] after the trailing
    // .i((.., l - 1.., ..))?.apply(lm_head); squeeze both leading
    // singleton dims to a rank-1 [V] tensor for sampling.
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let logits = logits.to_dtype(DType::F32)?.flatten_all()?;
    let values = logits.to_vec1::<f32>()?;
    Ok(values)
}

/// Image-bearing leader forward (rank 0). Preprocesses each source
/// `image_data_uris` entry through the same deterministic
/// `preprocess_data_uri` every rank runs, uploads to the leader's
/// device, encodes + splices + forwards via
/// `TpLeaderModel::forward_with_images`, and copies the `[vocab]`
/// logits to CPU. Mirrors the single-GPU `forward_logits_with_images`
/// but on the TP leader's replicated tower.
#[cfg(feature = "cuda")]
fn tp_forward_logits_with_images(
    state: &mut DeviceWorkerState,
    handle: TpHandle,
    tokens: &[u32],
    offset: usize,
    image_token_id: u32,
    image_data_uris: &[String],
    chunk_size: usize,
) -> anyhow::Result<Vec<f32>> {
    use crate::harness::preprocess::{PreprocessProfile, preprocess_data_uri};
    use candle_core::{DType, Tensor};

    if image_data_uris.is_empty() {
        anyhow::bail!("TpForwardLogitsWithImages dispatched with zero images");
    }

    // Preprocess every image into a device-resident (C, H, W) tensor at
    // its native-aspect resized dims (#14). Same `smart_resize` + decode
    // path the subprocess workers run, so the encoded embeddings — and
    // the per-image grids derived from these dims — match across ranks
    // bit-for-bit.
    let profile = PreprocessProfile::qwen3_6();
    let mut pixels: Vec<Tensor> = Vec::with_capacity(image_data_uris.len());
    for (idx, uri) in image_data_uris.iter().enumerate() {
        let (px, h, w) = preprocess_data_uri(uri, &profile)
            .with_context(|| format!("preprocess image[{idx}] (TP leader)"))?;
        let t = Tensor::from_vec(px, (3, h as usize, w as usize), &state.device)?;
        pixels.push(t);
    }

    let model = state.tp_models.get_mut(&handle).ok_or_else(|| {
        anyhow::anyhow!(
            "TpForwardLogitsWithImages: no model for handle {}",
            handle.0
        )
    })?;

    // Chunked prefill (encode once, splice per chunk) — bounded
    // activation, in lockstep with the subprocess ranks.
    let logits =
        model.prefill_with_images_chunked(tokens, offset, &pixels, image_token_id, chunk_size)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let logits = logits.to_dtype(DType::F32)?.flatten_all()?;
    let values = logits.to_vec1::<f32>()?;
    Ok(values)
}

/// Forward step + copy the `[vocab]` logits to a CPU `Vec<f32>` ready
/// for sampling on the async caller. The model's `device()` (CUDA or
/// CPU) determines where the kernel runs; this fn doesn't care.
///
/// On CUDA, the `to_dtype(F32).flatten_all().to_vec1::<f32>()` chain
/// triggers the device → host copy. The copy runs synchronously on
/// this worker thread; the bound context owns the source allocation
/// so the transfer is straightforward.
fn forward_logits(
    state: &mut DeviceWorkerState,
    handle: ArchHandle,
    tokens: &[u32],
    offset: usize,
) -> anyhow::Result<Vec<f32>> {
    use candle_core::{DType, Tensor};

    // Build the input tensor on the worker's own device. cudarc's
    // primary-context model means `Device::new_cuda(idx)` shares state
    // with the `CudaContext` we bound at startup, so this is the same
    // device the ModelArch was loaded against.
    let input = Tensor::new(tokens, &state.device)?.unsqueeze(0)?;

    let arch = state
        .models
        .get_mut(&handle)
        .ok_or_else(|| anyhow::anyhow!("ForwardLogits: no model for handle {}", handle.0))?;

    let logits = arch.forward(&input, offset)?;
    // Copy to CPU f32. logits is already `[vocab]` (squeeze_to_vocab
    // inside ModelArch::forward). The to_dtype handles bf16/f16 →
    // f32 promotion for the sampler.
    let logits = logits.to_dtype(DType::F32)?.flatten_all()?;
    let values = logits.to_vec1::<f32>()?;
    Ok(values)
}

/// Run the LM forward with vision-tower image splicing. Stage B3.
///
/// Encodes each image through the vision tower (`VisionTower::forward`,
/// dispatched via `ModelArch::encode_image`), concatenates the
/// resulting embeddings into a single `(N_total, hidden)` tensor, and
/// passes it to `ModelArch::forward_with_vision` along with the
/// prompt-expanded `tokens`. Image embeddings never leave the device.
///
/// Returns CPU `[vocab]` logits — same shape contract as
/// `ForwardLogits` so the async sampler doesn't have to branch on the
/// presence of images.
fn forward_logits_with_images(
    state: &mut DeviceWorkerState,
    handle: ArchHandle,
    tokens: &[u32],
    offset: usize,
    images: Vec<ImageInput>,
    image_token_id: u32,
) -> anyhow::Result<Vec<f32>> {
    use candle_core::{DType, Tensor};

    if images.is_empty() {
        anyhow::bail!("ForwardLogitsWithImages dispatched with zero images");
    }

    let arch = state.models.get_mut(&handle).ok_or_else(|| {
        anyhow::anyhow!("ForwardLogitsWithImages: no model for handle {}", handle.0)
    })?;

    // pixel→LM-grid divisor (patch×merge) for this tower; each image's
    // LM grid is (h/factor, w/factor) (#14 dynamic resolution).
    let factor = arch.vision_grid_factor().ok_or_else(|| {
        anyhow::anyhow!("ForwardLogitsWithImages: loaded model has no vision tower")
    })?;

    // Encode every image on the worker's device, collecting per-image
    // post-merger embeddings as device-resident tensors plus their LM
    // grids (for the interleaved-M-RoPE position ids).
    let mut per_image: Vec<Tensor> = Vec::with_capacity(images.len());
    let mut grids: Vec<(usize, usize)> = Vec::with_capacity(images.len());
    for (idx, img) in images.into_iter().enumerate() {
        anyhow::ensure!(
            img.pixels.len() == img.c * img.h * img.w,
            "ForwardLogitsWithImages: image[{idx}] pixels length {} does not match shape ({}, {}, {})",
            img.pixels.len(),
            img.c,
            img.h,
            img.w,
        );
        grids.push((img.h / factor, img.w / factor));
        let image = Tensor::from_vec(img.pixels, (img.c, img.h, img.w), &state.device)?;
        let embed = arch
            .encode_image(&image)
            .with_context(|| format!("encode image[{idx}]"))?;
        per_image.push(embed);
    }
    // Concatenate per-image embeddings along the patch axis →
    // (sum_of_patches, hidden). `Tensor::cat` keeps the result
    // device-resident.
    let image_embeds = Tensor::cat(&per_image.iter().collect::<Vec<_>>(), 0)?;

    let input = Tensor::new(tokens, &state.device)?.unsqueeze(0)?;
    let logits = arch.forward_with_vision(&input, offset, &image_embeds, image_token_id, &grids)?;
    let values = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values)
}

/// Run the vision tower on a single preprocessed image. Stage A5.
///
/// `pixels` is a row-major `(c, h, w)` f32 image that the async-side
/// `harness::preprocess` produced. We reconstruct the tensor on the
/// worker's device (the same device the model was loaded against),
/// call `arch.encode_image`, and copy the resulting
/// `(N_lm_tokens, hidden_size)` embedding back to CPU f32.
///
/// Returns the flattened embedding as a `Vec<f32>` — the caller knows
/// the LM-side token count from `VisionTower::lm_tokens_for(h, w)`
/// and reshapes accordingly. Stage B introduces a device-resident
/// embedding-slab variant that avoids this round-trip when the next
/// forward call needs the result.
fn encode_image(
    state: &mut DeviceWorkerState,
    handle: ArchHandle,
    pixels: Vec<f32>,
    c: usize,
    h: usize,
    w: usize,
) -> anyhow::Result<Vec<f32>> {
    use candle_core::{DType, Tensor};

    anyhow::ensure!(
        pixels.len() == c * h * w,
        "EncodeImage: pixels length {} does not match shape ({c}, {h}, {w})",
        pixels.len()
    );
    let image = Tensor::from_vec(pixels, (c, h, w), &state.device)?;

    let arch = state
        .models
        .get(&handle)
        .ok_or_else(|| anyhow::anyhow!("EncodeImage: no model for handle {}", handle.0))?;

    let embed = arch.encode_image(&image)?;
    let values = embed
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values)
}

/// Reply to a job with the poisoned-worker error. Used when the worker
/// has flipped into drain-only mode after a CUDA driver error.
///
/// `Job::Shutdown` is filtered before reaching this fn so the match
/// only needs the data-carrying variants. As phases 2–4 add more
/// variants the match here grows; every variant must reply with the
/// poisoned error so callers never hang waiting for a worker that's
/// no longer running CUDA.
fn drain_poisoned(job: Job, device_index: u32) {
    let err = || anyhow::anyhow!("device worker for device {device_index} is poisoned");
    match job {
        Job::QueryVram { reply } => {
            let _ = reply.send(Err(err()));
        }
        Job::LoadGguf { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::LoadDense { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::DropArch { reply, .. } => {
            // Drop reply is `()` — no error path. Send the unit so the
            // caller's await resolves; the model handle is leaked in
            // the worker's slab, but the whole slab gets `mem::forget`
            // on shutdown anyway per the poisoned-thread design.
            let _ = reply.send(());
        }
        Job::ClearKv { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::SnapshotKv { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::RestoreKv { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::DropKvSnapshot { reply, .. } => {
            // Same shape as DropArch: unit reply so the caller's await
            // resolves; the snapshot leaks with the rest of the slab
            // per the poisoned-thread design.
            let _ = reply.send(());
        }
        Job::ForwardLogits { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::EncodeImage { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::ForwardLogitsWithImages { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::NcclInit { reply, .. } => {
            let _ = reply.send(crate::harness::tp::rpc::WorkerResponse::Error {
                kind: "device_worker_poisoned".into(),
                message: format!("device worker {device_index} poisoned"),
            });
        }
        #[cfg(feature = "cuda")]
        Job::GetLeaderComm { reply } => {
            let _ = reply.send(None);
        }
        Job::NcclSanity { reply } => {
            let _ = reply.send(crate::harness::tp::rpc::WorkerResponse::Error {
                kind: "device_worker_poisoned".into(),
                message: format!("device worker {device_index} poisoned"),
            });
        }
        #[cfg(feature = "cuda")]
        Job::TpLoadShard { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        #[cfg(feature = "cuda")]
        Job::DropTp { reply, .. } => {
            let _ = reply.send(());
        }
        #[cfg(feature = "cuda")]
        Job::TpClearKv { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        #[cfg(feature = "cuda")]
        Job::TpForwardLogits { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        #[cfg(feature = "cuda")]
        Job::TpForwardLogitsWithImages { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::Shutdown => {
            // Filtered by the matches!() guard in run(); reaching
            // here would be a logic error.
            unreachable!("Shutdown is filtered before drain_poisoned");
        }
    }
}
