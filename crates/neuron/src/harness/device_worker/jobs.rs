//! Job variants accepted by the per-device worker thread.
//!
//! Each variant carries the inputs the synchronous dispatch handler
//! needs plus a `tokio::sync::oneshot::Sender` for the reply. The
//! async-side `DeviceWorkerHandle` constructs a job, sends it down the
//! `std::sync::mpsc` channel, and `await`s the oneshot for the reply.

use crate::harness::candle::ModelArch;
use anyhow::Result;
use tokio::sync::oneshot;

/// Opaque handle to a `ModelArch` stored in the worker thread's state
/// slab. Cheap to copy; `Send + Sync` so it crosses task boundaries
/// freely. The actual `Box<ModelArch>` it points to is owned by the
/// worker thread for the duration of the handle's lifetime — the only
/// way to drop the model is to send `Job::DropArch { handle }` so the
/// `Drop` impl runs on the thread with the bound CUDA context (the
/// invariant the whole refactor exists to guarantee).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchHandle(pub u64);

/// Opaque handle to a `TpLeaderModel` stored in the worker thread's
/// state slab. Same shape as [`ArchHandle`] but in a separate
/// namespace so the two slabs can coexist without ambiguity. Phase 3
/// introduces it; Phase 4 may unify the two slabs after the TP forward
/// path proves out.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TpHandle(pub u64);

/// One unit of work for the device worker.
///
/// Phase 1 had only `QueryVram` and `Shutdown`. Phase 2 adds the
/// single-GPU inference primitives: transfer-in a freshly-loaded
/// `ModelArch`, drop it, clear its KV cache, and run one forward step
/// returning CPU-side logits ready for sampling on the async caller.
///
/// Sampling stays on the async side intentionally. The worker copies
/// logits to CPU (`Vec<f32>`) before reply, so the device-resident
/// tensor never escapes the worker thread and the async caller's
/// `LogitsProcessor::sample` runs entirely on the CPU candle backend
/// — no incidental context binding on a tokio worker thread.
pub enum Job {
    /// Query free / total VRAM on the device. Returns
    /// `(free_mb, total_mb)`. CPU builds and contexts that failed to
    /// initialise reply with `(0, 0)` — matches today's
    /// `device_vram_mb` sentinel so the log field values don't change.
    QueryVram {
        reply: oneshot::Sender<Result<(u64, u64)>>,
    },
    /// Move a freshly-loaded `ModelArch` into the worker's state slab.
    /// Returns an `ArchHandle` the caller stores on `LoadedModel` and
    /// passes back in subsequent `ClearKv` / `ForwardLogits` /
    /// `DropArch` jobs.
    TransferIn {
        arch: Box<ModelArch>,
        reply: oneshot::Sender<Result<ArchHandle>>,
    },
    /// Remove the model from the slab and drop it. The `Drop` runs on
    /// the worker thread so CUDA tensors release their memory on the
    /// same context that allocated them.
    DropArch {
        handle: ArchHandle,
        reply: oneshot::Sender<()>,
    },
    /// Reset the KV cache for this model. Called at the start of every
    /// chat completion so a new request doesn't attend over the
    /// previous one's tokens.
    ClearKv {
        handle: ArchHandle,
        reply: oneshot::Sender<Result<()>>,
    },
    /// Run one forward step and copy the resulting `[vocab]` logits to
    /// CPU. The caller takes the returned `Vec<f32>`, wraps it in a
    /// CPU `Tensor`, and runs `apply_repeat_penalty` + sampling
    /// without touching the device context. `offset` is the KV-cache
    /// position before this step (0 for prefill, `prompt_len + i` for
    /// the i-th decode step).
    ForwardLogits {
        handle: ArchHandle,
        tokens: Vec<u32>,
        offset: usize,
        reply: oneshot::Sender<Result<Vec<f32>>>,
    },
    /// Initialize the leader's NCCL communicator. The worker's
    /// `NcclState` mints the `Comm` here so its underlying
    /// `ncclComm_t` and `CudaContext` live on the same thread as
    /// every later `Comm::all_reduce` call. Reply is the worker
    /// response shape used by the subprocess workers (`InitOk` on
    /// success, `Error` on failure) so the calling
    /// `WorkerPool::init_nccl` orchestration stays uniform.
    ///
    /// Available on both cuda and no-cuda builds — the dispatch
    /// handler calls `NcclState::init` which has a no-cuda stub that
    /// replies with `cuda_feature_not_enabled`. Keeping the Job
    /// variant ungated lets `WorkerPool::init_nccl` stay uniform.
    NcclInit {
        cfg: crate::harness::tp::worker::WorkerConfig,
        comm_id_hex: String,
        reply: oneshot::Sender<crate::harness::tp::rpc::WorkerResponse>,
    },
    /// Run NCCL's all_reduce sanity check on the leader's rank 0.
    /// Same response shape as `NcclInit`; also available on both
    /// builds via the no-cuda `NcclState::sanity_check` stub.
    NcclSanity {
        reply: oneshot::Sender<crate::harness::tp::rpc::WorkerResponse>,
    },
    /// Clone the leader's `Arc<Comm>` out of the worker's `NcclState`
    /// so a spawn_blocking-based load (Phase 3 bridge) can hand it to
    /// the row-parallel layers. Wrapped in `SendComm` because
    /// `Arc<Comm>` is `!Send` at the type level (the NCCL contract
    /// requires serialised access, which we provide structurally).
    /// Phase 4 eliminates this when `TpLoadShard` becomes a Job and
    /// the load runs entirely on the worker thread.
    #[cfg(feature = "cuda")]
    CloneLeaderComm {
        reply: oneshot::Sender<Result<crate::harness::tp::nccl_state::SendComm>>,
    },
    /// Move a freshly-built `TpLeaderModel` into the worker's tp slab.
    /// Returns a `TpHandle` the caller stores on `TpLoadedModel`.
    #[cfg(feature = "cuda")]
    TransferInTp {
        model: Box<crate::harness::tp::TpLeaderModel>,
        reply: oneshot::Sender<Result<TpHandle>>,
    },
    /// Drop the TP leader model on the worker thread. CUDA tensors
    /// and `Arc<Comm>` clones held inside the model release on the
    /// thread that allocated them.
    #[cfg(feature = "cuda")]
    DropTp {
        handle: TpHandle,
        reply: oneshot::Sender<()>,
    },
    /// Reset the leader's KV cache for a TP model. Mirrors `ClearKv`
    /// for single-GPU.
    #[cfg(feature = "cuda")]
    TpClearKv {
        handle: TpHandle,
        reply: oneshot::Sender<Result<()>>,
    },
    /// Run one TP forward step on the leader's shard. Returns CPU-
    /// side logits as a `Vec<f32>` so the async caller can sample
    /// without holding a device tensor. The caller is also
    /// responsible for fan-out to subprocess ranks and drain — only
    /// the leader's forward moves into the worker thread.
    #[cfg(feature = "cuda")]
    TpForwardLogits {
        handle: TpHandle,
        tokens: Vec<u32>,
        offset: usize,
        reply: oneshot::Sender<Result<Vec<f32>>>,
    },
    /// Tell the worker to break its dispatch loop and exit. Any jobs
    /// queued after this in the channel reply `Err` to their oneshot
    /// senders (the senders are dropped on the worker's exit, which
    /// the async-side `Receiver::await` maps to `WorkerError::Gone`).
    Shutdown,
}
