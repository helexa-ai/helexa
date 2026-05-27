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

use crate::harness::candle::ModelArch;
use crate::harness::device_worker::jobs::{ArchHandle, Job};
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
            Job::TransferIn { arch, reply } => {
                let handle = ArchHandle(state.next_handle);
                state.next_handle = state.next_handle.wrapping_add(1);
                state.models.insert(handle, arch);
                tracing::debug!(
                    device_index,
                    handle = handle.0,
                    slab_size = state.models.len(),
                    "device worker: model transferred in"
                );
                let _ = reply.send(Ok(handle));
            }
            Job::DropArch { handle, reply } => {
                let removed = state.models.remove(&handle);
                let was_present = removed.is_some();
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
                let _ = reply.send(result);
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
            // Handled by the matches!() check above; reaching here
            // means a Shutdown slipped past which is a bug.
            Job::Shutdown => unreachable!("Shutdown should break above"),
        }
    }

    tracing::info!(
        device_index,
        slab_size = state.models.len(),
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
        Job::TransferIn { reply, .. } => {
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
        Job::ForwardLogits { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        Job::Shutdown => {
            // Filtered by the matches!() guard in run(); reaching
            // here would be a logic error.
            unreachable!("Shutdown is filtered before drain_poisoned");
        }
    }
}
