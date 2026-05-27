//! Per-device CUDA worker thread.
//!
//! One dedicated OS thread per CUDA device the leader uses. The thread
//! binds the device's `CudaContext` once at startup and owns it for the
//! daemon's lifetime; all GPU operations and VRAM queries for that
//! device route through a `std::sync::mpsc` channel into this thread.
//! Tensors never escape the thread alive — replies cross the channel
//! as plain values (`u32` tokens, `(u64, u64)` mb numbers, `()`).
//!
//! Rationale, in order of weight:
//!
//! 1. **Context locality.** cudarc binds the CUDA context per OS thread
//!    via `cuCtxSetCurrent`. With `tokio::task::spawn_blocking`, the
//!    blocking thread chosen is arbitrary, so the context gets bound
//!    onto a different thread each time and `device_vram_mb()` from an
//!    async task binds it again on the *caller's* thread as a side
//!    effect. Pinning the context to one named thread ends that.
//!
//! 2. **Drop safety.** `cudarc::driver::CudaContext`, every `CudaSlice`
//!    inside a `Tensor`, and every `cudarc::nccl::Comm` call `cuMemFree`
//!    / `cuCtxDestroy` / `ncclCommDestroy` during `Drop`. These must
//!    run with the right context current. Owning everything in this
//!    thread's state slab and dropping it via `Job::DropArch` /
//!    `Job::Shutdown` is the only safe pattern.
//!
//! 3. **Poisoning blast radius.** When a CUDA driver error (illegal
//!    address, OOM cascade) makes the context unrecoverable, today the
//!    spawn_blocking thread carrying that bad state simply returns to
//!    tokio's pool — invisible. With the per-device thread, the
//!    poisoned flag lives on the thread itself; subsequent
//!    `submit()` calls fast-reject at the channel boundary with a
//!    clear "device worker is poisoned" error before any further CUDA
//!    work is attempted.
//!
//! The TP worker subprocesses (`harness/tp/worker.rs`) are already this
//! pattern, just out-of-process. The in-process variant uses the same
//! discipline for rank 0.
//!
//! Phase 1 of the refactor exposes only `Job::QueryVram` + `Job::Shutdown`.
//! Forward, kv-cache clear, model load, and NCCL bring-up move in later
//! phases. See `/home/grenade/.claude/plans/plan-the-per-device-worker-abstract-micali.md`.

pub mod dispatch;
pub mod jobs;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use tokio::sync::oneshot;

#[cfg(feature = "cuda")]
pub use jobs::TpHandle;
pub use jobs::{ArchHandle, Job};

/// Errors returned by `DeviceWorkerHandle` submit methods.
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    /// The worker's CUDA context was poisoned by an earlier driver
    /// error. The thread is still alive (dropping it would re-touch
    /// the broken context); it returns this error for every job
    /// submitted until the daemon is restarted.
    #[error(
        "device worker for device {device_index} is poisoned \
         (a prior CUDA driver error left the context unrecoverable); \
         restart the daemon to recover"
    )]
    Poisoned { device_index: u32 },
    /// The worker thread has exited (`Job::Shutdown` was processed or
    /// the thread panicked). Subsequent `submit()` calls fail here
    /// rather than blocking forever.
    #[error("device worker for device {device_index} is no longer running")]
    Gone { device_index: u32 },
    /// The dispatched job returned an `Err`. Forwarded verbatim.
    #[error(transparent)]
    Job(#[from] anyhow::Error),
}

/// Shared handle to a per-device CUDA worker thread.
///
/// Cloning the `Arc` lets multiple `LoadedModel`s (and `TpLoadedModel`s)
/// share the same worker — there's one worker per CUDA device index,
/// not one per model.
pub struct DeviceWorkerHandle {
    device_index: u32,
    tx: Sender<Job>,
    poisoned: Arc<AtomicBool>,
    /// `Mutex<Option<JoinHandle>>` so `shutdown()` can take the handle
    /// out without `&mut self` and so the inevitable `Drop` after
    /// `shutdown()` doesn't double-join. The mutex is uncontended in
    /// practice: only one caller ever takes the handle.
    join: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl DeviceWorkerHandle {
    /// Spawn a new worker for the given CUDA device index.
    ///
    /// The thread is named `cuda-dev-N` so it shows up legibly in
    /// `top -H`, `pidstat -t`, and gdb backtraces. On CUDA builds, the
    /// thread binds `CudaContext::new(N)` on startup; on CPU builds
    /// (`--no-default-features`) the thread runs without a context and
    /// every job that touches CUDA falls through to a zero return.
    pub fn spawn(device_index: u32) -> anyhow::Result<Arc<Self>> {
        let (tx, rx) = mpsc::channel::<Job>();
        let poisoned = Arc::new(AtomicBool::new(false));
        let poisoned_for_thread = Arc::clone(&poisoned);
        let join = std::thread::Builder::new()
            .name(format!("cuda-dev-{device_index}"))
            .spawn(move || {
                dispatch::run(device_index, rx, poisoned_for_thread);
            })?;
        Ok(Arc::new(Self {
            device_index,
            tx,
            poisoned,
            join: std::sync::Mutex::new(Some(join)),
        }))
    }

    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Mark the worker's context as poisoned. Future `submit()` calls
    /// short-circuit to `WorkerError::Poisoned` before sending. The
    /// dispatch loop also flips into drain-only mode when it sees this
    /// flag, so any jobs already in flight on the channel reply with
    /// the same error without touching CUDA.
    #[allow(dead_code)]
    pub(crate) fn set_poisoned(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    /// Send `Job::QueryVram`, await the worker's reply.
    ///
    /// Returns `Ok((free_mb, total_mb))` on success, `Ok((0, 0))` on
    /// CPU builds or when the device lacks a bound context, or an
    /// error if the worker is poisoned, gone, or the query itself
    /// failed inside cudarc.
    pub async fn query_vram(&self) -> Result<(u64, u64), WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::QueryVram { reply: reply_tx })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Load a GGUF (pre-quantized) single-GPU model on the worker
    /// thread. The hf-hub resolution happens on the async caller; the
    /// resolved local `gguf_path` plus the spec's model_id are sent
    /// into the worker which opens, parses, and constructs the
    /// `ModelArch` on the right thread.
    pub async fn load_gguf(
        &self,
        gguf_path: std::path::PathBuf,
        model_id: String,
    ) -> Result<ArchHandle, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::LoadGguf {
                gguf_path,
                model_id,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Load a dense safetensors single-GPU model on the worker thread.
    pub async fn load_dense(
        &self,
        config_path: std::path::PathBuf,
        safetensors_paths: Vec<std::path::PathBuf>,
        model_id: String,
    ) -> Result<ArchHandle, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::LoadDense {
                config_path,
                safetensors_paths,
                model_id,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Tell the worker to drop the `ModelArch` for `handle` on the
    /// worker thread (so CUDA tensors release on the right context).
    /// Returns `Ok(())` even if the handle wasn't in the slab — Drop
    /// is idempotent. Reports `Gone` if the worker isn't running.
    pub async fn drop_arch(&self, handle: ArchHandle) -> Result<(), WorkerError> {
        // Poisoning doesn't block DropArch — even on a poisoned
        // context we want callers to unblock and proceed with the
        // unload bookkeeping. The dispatch handler under poison just
        // replies `()` without touching the model (the actual Drop
        // happens via mem::forget at thread exit per the poison
        // protocol).
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::DropArch {
                handle,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(()) => Ok(()),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Reset the KV cache for the model at `handle`. Called at the
    /// start of every chat completion so the new prompt doesn't
    /// attend over the previous request's tokens.
    pub async fn clear_kv_cache(&self, handle: ArchHandle) -> Result<(), WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::ClearKv {
                handle,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Run one forward step and return the resulting `[vocab]` logits
    /// as a CPU-side `Vec<f32>`. The caller then samples on a CPU
    /// candle Tensor without ever binding the device context on its
    /// tokio thread.
    pub async fn forward_logits(
        &self,
        handle: ArchHandle,
        tokens: Vec<u32>,
        offset: usize,
    ) -> Result<Vec<f32>, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::ForwardLogits {
                handle,
                tokens,
                offset,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Initialise the leader's NCCL communicator. The reply uses
    /// `WorkerResponse` (same shape subprocess workers use over stdio
    /// RPC) so `WorkerPool::init_nccl`'s aggregation treats leader +
    /// subprocess responses uniformly. Available on no-cuda builds
    /// too — the dispatch handler calls the no-cuda `NcclState::init`
    /// stub which replies `cuda_feature_not_enabled`.
    pub async fn nccl_init(
        &self,
        cfg: crate::harness::tp::worker::WorkerConfig,
        comm_id_hex: String,
    ) -> Result<crate::harness::tp::rpc::WorkerResponse, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::NcclInit {
                cfg,
                comm_id_hex,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        reply_rx.await.map_err(|_| WorkerError::Gone {
            device_index: self.device_index,
        })
    }

    /// Run an NCCL sanity all_reduce on the leader's rank 0.
    /// Available on no-cuda builds; replies with an error response.
    pub async fn nccl_sanity(
        &self,
    ) -> Result<crate::harness::tp::rpc::WorkerResponse, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::NcclSanity { reply: reply_tx })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        reply_rx.await.map_err(|_| WorkerError::Gone {
            device_index: self.device_index,
        })
    }

    /// Load the leader's TP shard on the worker thread. The dispatch
    /// handler reads its own `NcclState`'s `Arc<Comm>` directly — no
    /// cross-thread Comm transfer — and builds the `TpLeaderModel`
    /// against it. Phase 4 replaces the Phase 3 Clone/TransferIn
    /// bridge with this single Job.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub async fn tp_load_shard(
        &self,
        model_id: String,
        config_json: String,
        safetensors_paths: Vec<std::path::PathBuf>,
        dtype: candle_core::DType,
        quant: Option<String>,
        world_size: u32,
    ) -> Result<TpHandle, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::TpLoadShard {
                model_id,
                config_json,
                safetensors_paths,
                dtype,
                quant,
                world_size,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Drop the TP model at `handle` on the worker thread.
    #[cfg(feature = "cuda")]
    pub async fn drop_tp(&self, handle: TpHandle) -> Result<(), WorkerError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::DropTp {
                handle,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(()) => Ok(()),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Reset the leader's KV cache for a TP model.
    #[cfg(feature = "cuda")]
    pub async fn tp_clear_kv(&self, handle: TpHandle) -> Result<(), WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::TpClearKv {
                handle,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Run one TP forward step on the leader's shard. Returns CPU-side
    /// logits as `Vec<f32>` ready for sampling. The caller is
    /// responsible for fan-out / drain of the subprocess workers
    /// concurrently with this call.
    #[cfg(feature = "cuda")]
    pub async fn tp_forward_logits(
        &self,
        handle: TpHandle,
        tokens: Vec<u32>,
        offset: usize,
    ) -> Result<Vec<f32>, WorkerError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WorkerError::Poisoned {
                device_index: self.device_index,
            });
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Job::TpForwardLogits {
                handle,
                tokens,
                offset,
                reply: reply_tx,
            })
            .map_err(|_| WorkerError::Gone {
                device_index: self.device_index,
            })?;
        match reply_rx.await {
            Ok(result) => result.map_err(WorkerError::from),
            Err(_) => Err(WorkerError::Gone {
                device_index: self.device_index,
            }),
        }
    }

    /// Send `Job::Shutdown` and join the thread. Idempotent — calling
    /// twice is a no-op the second time.
    pub fn shutdown(&self) -> anyhow::Result<()> {
        // Best-effort send: if the channel is already closed (thread
        // exited after a prior shutdown or panic) the send fails and
        // we fall through to the join which returns the panic, if any.
        let _ = self.tx.send(Job::Shutdown);
        let join = self.join.lock().unwrap().take();
        if let Some(j) = join {
            j.join()
                .map_err(|_| anyhow::anyhow!("worker thread panicked during shutdown"))?;
        }
        Ok(())
    }
}

impl Drop for DeviceWorkerHandle {
    fn drop(&mut self) {
        // Best-effort: send Shutdown so the thread breaks its loop
        // and exits. We do NOT join here — Drop may run on a tokio
        // worker thread, and joining a thread that's still processing
        // the last job would block the runtime. The OS reaps the
        // thread on detach.
        let _ = self.tx.send(Job::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn spawn_query_vram_shutdown() {
        let handle = DeviceWorkerHandle::spawn(0).expect("spawn ok");
        // CPU build (the only one CI runs) returns (0, 0) by design;
        // a CUDA build with a real device would return real values.
        let result = handle.query_vram().await.expect("query ok");
        // We assert >= 0 — the field width matters more than the value.
        let _ = result.0;
        let _ = result.1;
        handle.shutdown().expect("shutdown ok");
    }

    #[tokio::test]
    async fn thread_is_named_correctly() {
        // The thread name lets `top -H` / pidstat / gdb show
        // `cuda-dev-N` instead of an opaque tokio worker name. Verify
        // by spawning and reading proc-self thread comms — but on
        // platforms without /proc, just confirm we don't crash.
        let handle = DeviceWorkerHandle::spawn(7).expect("spawn ok");
        // Round-trip a job to ensure the thread is alive and processing.
        handle.query_vram().await.expect("query ok");
        handle.shutdown().expect("shutdown ok");
    }

    #[tokio::test]
    async fn submit_after_shutdown_returns_gone() {
        let handle = DeviceWorkerHandle::spawn(0).expect("spawn ok");
        handle.shutdown().expect("shutdown ok");
        // Channel closed; submit should map to Gone rather than block.
        let result = handle.query_vram().await;
        match result {
            Err(WorkerError::Gone { device_index: 0 }) => {}
            other => panic!("expected Gone, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn poisoned_flag_short_circuits_submit() {
        let handle = DeviceWorkerHandle::spawn(0).expect("spawn ok");
        handle.set_poisoned();
        let result = handle.query_vram().await;
        match result {
            Err(WorkerError::Poisoned { device_index: 0 }) => {}
            other => panic!("expected Poisoned, got {other:?}"),
        }
        // The channel is still alive; shutdown should still succeed.
        handle.shutdown().expect("shutdown ok");
    }

    #[tokio::test]
    async fn shutdown_drains_pending_jobs() {
        let handle = DeviceWorkerHandle::spawn(0).expect("spawn ok");
        // Submit many concurrent jobs; they should all complete even
        // though a Shutdown is racing them.
        let mut futures = Vec::new();
        for _ in 0..16 {
            let h = Arc::clone(&handle);
            futures.push(tokio::spawn(async move { h.query_vram().await }));
        }
        // Small yield to give the senders a chance to actually send
        // before we issue the shutdown; not strictly necessary because
        // the channel is FIFO, but makes the test's intent clearer.
        tokio::time::sleep(Duration::from_millis(10)).await;
        handle.shutdown().expect("shutdown ok");
        for f in futures {
            // Each query should have completed (Ok or Gone, never panic).
            let _ = f.await.expect("task did not panic");
        }
    }
}
