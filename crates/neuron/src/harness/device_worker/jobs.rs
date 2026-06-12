//! Job variants accepted by the per-device worker thread.
//!
//! Each variant carries the inputs the synchronous dispatch handler
//! needs plus a `tokio::sync::oneshot::Sender` for the reply. The
//! async-side `DeviceWorkerHandle` constructs a job, sends it down the
//! `std::sync::mpsc` channel, and `await`s the oneshot for the reply.

use anyhow::Result;
use std::path::PathBuf;
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

/// Opaque handle to a prefix-cache snapshot (#11) stored worker-side
/// next to the model slab. Scoped to the `ArchHandle` it was captured
/// from — `Job::DropArch` drops every snapshot under its handle. The
/// snapshot's tensors never leave the worker thread; the async side
/// holds only this id plus the token sequence it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvSnapshotId(pub u64);

/// One image payload for `Job::ForwardLogitsWithImages` /
/// `Job::EncodeImage`. Pixels are row-major `(c, h, w)` f32 — the
/// shape `harness::preprocess::preprocess` produces. Carries the
/// shape inline since `Vec<f32>` is rank-1.
///
/// `Clone` so the vision-aware dispatch in `chat_completion` can
/// match `&vision_route` (carrying borrowed images) and still hand
/// owned `Vec<ImageInput>` to the worker job. The clone cost is one
/// pixel-buffer memcpy per image — now variable with dynamic resolution
/// (#14): `3 × h × w × 4` bytes, up to ~6.3 MiB at the default 1024²
/// `max_pixels` budget.
///
/// `h`/`w` are the **resized** dims (factor-aligned), so the per-image LM
/// grid is `(h/factor, w/factor)` — derived downstream for the splice
/// and the interleaved-M-RoPE position ids.
#[derive(Clone)]
pub struct ImageInput {
    pub pixels: Vec<f32>,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

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
    /// Load a GGUF (pre-quantized) single-GPU model on the worker
    /// thread. The dispatch handler opens the GGUF file, parses
    /// metadata, dispatches on `general.architecture`, and inserts
    /// the resulting `ModelArch` into the slab. Returns the fresh
    /// `ArchHandle`.
    LoadGguf {
        gguf_path: PathBuf,
        model_id: String,
        reply: oneshot::Sender<Result<ArchHandle>>,
    },
    /// Load a dense safetensors single-GPU model on the worker
    /// thread. The dispatch handler reads `config.json`, dispatches on
    /// `model_type`, builds a `VarBuilder` over the mmap'd
    /// safetensors, and inserts the resulting `ModelArch`.
    LoadDense {
        config_path: PathBuf,
        safetensors_paths: Vec<PathBuf>,
        model_id: String,
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
    /// Capture the model's live cache state (attention KV + GDN
    /// recurrent state + position counters) as a prefix snapshot
    /// (#11). The snapshot stays in the worker's state, keyed by the
    /// returned id; the reply carries `(id, bytes)` so the async side
    /// can do budget accounting without touching tensors. Errors on
    /// archs without snapshot support.
    SnapshotKv {
        handle: ArchHandle,
        reply: oneshot::Sender<Result<(KvSnapshotId, u64)>>,
    },
    /// Replace the model's live cache state with a stored snapshot,
    /// instead of `ClearKv`, so prefill can resume at the snapshot's
    /// token boundary. The snapshot remains stored (restorable again).
    RestoreKv {
        handle: ArchHandle,
        snapshot: KvSnapshotId,
        reply: oneshot::Sender<Result<()>>,
    },
    /// Drop one stored snapshot (prefix-cache eviction). Idempotent.
    DropKvSnapshot {
        handle: ArchHandle,
        snapshot: KvSnapshotId,
        reply: oneshot::Sender<()>,
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
    /// Run the LM forward with vision splicing in one round-trip.
    /// Stage B3 of the vision plan.
    ///
    /// Inputs:
    /// - `tokens`: prompt-expanded token ids (the caller has already
    ///   replaced each `<|image_pad|>` with N copies per the
    ///   per-image patch count, so `tokens` already contains exactly
    ///   `sum(n_i)` `image_token_id` entries across all images).
    /// - `offset`: KV-cache position (same contract as `ForwardLogits`).
    /// - `images`: one entry per image — preprocessed pixels plus the
    ///   `(c, h, w)` shape. Images are encoded on the worker via the
    ///   model's vision tower (`VisionTower::forward`), concatenated
    ///   in order, and spliced into the LM input embeddings at
    ///   `image_token_id` positions.
    /// - `image_token_id`: the sentinel token (248056 for Qwen3.6).
    ///
    /// Returns flat CPU `[vocab]` logits, same as `ForwardLogits`.
    /// Image embeddings stay device-resident — they're never copied
    /// to CPU. The "tensors don't escape the worker" invariant holds.
    ForwardLogitsWithImages {
        handle: ArchHandle,
        tokens: Vec<u32>,
        offset: usize,
        images: Vec<ImageInput>,
        image_token_id: u32,
        reply: oneshot::Sender<Result<Vec<f32>>>,
    },
    /// Encode one image through the model's vision tower. Stage A5 of
    /// the vision plan (`doc/vision-qwen3_6-spec.md`).
    ///
    /// `pixels` is the CPU-side preprocessed image tensor in row-major
    /// `(C, H, W)` f32 layout — what `harness::preprocess::preprocess`
    /// produces. `c`, `h`, `w` carry the shape since `Vec<f32>` itself
    /// is rank-1. The handler reconstructs the tensor on the worker's
    /// device, runs `VisionTower::forward`, copies the resulting
    /// `(N_lm_tokens, hidden_size)` embedding back to CPU as a flat
    /// `Vec<f32>` (the caller knows the expected shape from
    /// `VisionTower::lm_tokens_for(h, w) * hidden_size`).
    ///
    /// Mirrors the `ForwardLogits` "tensors don't escape" invariant —
    /// device-side image embeddings are dropped at handler return.
    /// Stage B will introduce a follow-up variant that keeps the
    /// embeddings device-resident and references them from the next
    /// `ForwardLogits` call, avoiding the round-trip copy.
    EncodeImage {
        handle: ArchHandle,
        pixels: Vec<f32>,
        c: usize,
        h: usize,
        w: usize,
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
    /// Hand a clonable handle to the leader's NCCL `Comm` back to the
    /// async side, so the TP step watchdog can call `ncclCommAbort` on
    /// it from a *different* thread to unblock a wedged collective
    /// (#17 Stage 2). Fetched once at init while the worker thread is
    /// still responsive — a thread already wedged in a collective can't
    /// service this job, which is exactly why the handle is cached
    /// up front. Replies `None` before `NcclInit` has run.
    #[cfg(feature = "cuda")]
    GetLeaderComm {
        reply: oneshot::Sender<Option<crate::harness::tp::nccl_state::SendComm>>,
    },
    /// Load the leader's TP shard on the worker thread. The dispatch
    /// handler reads `state.nccl.comm()` directly (no cross-thread
    /// `Arc<Comm>` transfer, no `SendComm` wrapper) and builds the
    /// `TpLeaderModel` against that Comm. The model's embedded
    /// `Arc<Comm>` clones, `CudaContext`, and all per-rank CUDA
    /// tensors live on this thread for the model's lifetime.
    /// Inserts into the TP slab and returns the fresh `TpHandle`.
    #[cfg(feature = "cuda")]
    TpLoadShard {
        model_id: String,
        config_json: String,
        safetensors_paths: Vec<PathBuf>,
        dtype: candle_core::DType,
        quant: Option<String>,
        world_size: u32,
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
    /// Image-bearing leader (rank 0) forward for the single-shot vision
    /// prefill. The handler preprocesses each `image_data_uris` entry
    /// (the same deterministic path every rank runs), encodes through
    /// the leader's replicated tower, splices at `image_token_id`, and
    /// returns CPU-side `[vocab]` logits. Image tensors never escape the
    /// worker thread. Caller fans out `GenerateStepWithImages` to the
    /// subprocess ranks and drains them; only the leader forward moves
    /// here.
    #[cfg(feature = "cuda")]
    TpForwardLogitsWithImages {
        handle: TpHandle,
        tokens: Vec<u32>,
        offset: usize,
        image_token_id: u32,
        image_data_uris: Vec<String>,
        chunk_size: usize,
        reply: oneshot::Sender<Result<Vec<f32>>>,
    },
    /// Tell the worker to break its dispatch loop and exit. Any jobs
    /// queued after this in the channel reply `Err` to their oneshot
    /// senders (the senders are dropped on the worker's exit, which
    /// the async-side `Receiver::await` maps to `WorkerError::Gone`).
    Shutdown,
}
