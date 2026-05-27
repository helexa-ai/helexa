//! Synchronous dispatch loop running on the device worker thread.
//!
//! `run()` is the thread's entry point. It binds the CUDA context for
//! its device on startup, then pulls `Job`s off the channel one at a
//! time and runs the corresponding handler. The handlers are
//! synchronous by design — the only async on this thread is the
//! one-line `oneshot::Sender::send` call to ship the reply back, which
//! is non-blocking.
//!
//! Phase 1 handles only `QueryVram` and `Shutdown`. Later phases add
//! Forward, ClearKv, NCCL, and load handlers as separate match arms.

use crate::harness::device_worker::jobs::Job;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;

/// Per-thread state owned by the worker. On CUDA builds the `Arc<CudaContext>`
/// is created and bound at thread startup; on CPU builds the struct is
/// empty save for the device index (kept for log clarity).
#[cfg(feature = "cuda")]
struct DeviceWorkerState {
    device_index: u32,
    /// `None` only if `CudaContext::new()` failed — in that case the
    /// thread still runs so the handle's lifecycle stays uniform, but
    /// every job that touches CUDA falls through to a zero reply with
    /// a log warning.
    ctx: Option<Arc<candle_core::cuda::cudarc::driver::CudaContext>>,
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
struct DeviceWorkerState {
    device_index: u32,
}

/// Worker thread entry point. Runs until `Job::Shutdown` arrives or
/// the channel sender is dropped (which happens when the last
/// `DeviceWorkerHandle` `Arc` is dropped without an explicit
/// `shutdown()`).
pub(crate) fn run(device_index: u32, rx: Receiver<Job>, poisoned: Arc<AtomicBool>) {
    let state = init_state(device_index);
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
            // touching CUDA. Phase 1 never sets the flag from the
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
            // Handled by the matches!() check above; reaching here
            // means a Shutdown slipped past which is a bug.
            Job::Shutdown => unreachable!("Shutdown should break above"),
        }
    }

    tracing::info!(device_index, "device worker exiting");
}

#[cfg(feature = "cuda")]
fn init_state(device_index: u32) -> DeviceWorkerState {
    use candle_core::cuda::cudarc::driver::CudaContext;
    match CudaContext::new(device_index as usize) {
        Ok(ctx) => {
            // Make sure the context is current on this thread. cudarc
            // is generally fine with lazy binding, but doing it once
            // here gives us a deterministic moment to log "context
            // bound" — and makes `mem_get_info()` work without further
            // bind dances inside the dispatch handlers.
            if let Err(e) = ctx.bind_to_thread() {
                tracing::warn!(
                    device_index,
                    error = ?e,
                    "device worker: bind_to_thread failed; \
                     vram queries will still rebind per-call"
                );
            } else {
                tracing::info!(device_index, "device worker bound CUDA context");
            }
            DeviceWorkerState {
                device_index,
                ctx: Some(ctx),
            }
        }
        Err(e) => {
            tracing::warn!(
                device_index,
                error = ?e,
                "device worker: CudaContext::new failed; \
                 vram queries will return (0, 0)"
            );
            DeviceWorkerState {
                device_index,
                ctx: None,
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn init_state(device_index: u32) -> DeviceWorkerState {
    DeviceWorkerState { device_index }
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

/// Reply to a job with the poisoned-worker error. Used when the worker
/// has flipped into drain-only mode after a CUDA driver error.
///
/// `Job::Shutdown` is filtered before reaching this fn so the match
/// only needs the data-carrying variants. As phases 2–4 add more
/// variants the match here grows; every variant must reply with the
/// poisoned error so callers never hang waiting for a worker that's
/// no longer running CUDA.
fn drain_poisoned(job: Job, device_index: u32) {
    match job {
        Job::QueryVram { reply } => {
            let _ = reply.send(Err(anyhow::anyhow!(
                "device worker for device {device_index} is poisoned"
            )));
        }
        Job::Shutdown => {
            // Filtered by the matches!() guard in run(); reaching
            // here would be a logic error.
            unreachable!("Shutdown is filtered before drain_poisoned");
        }
    }
}
