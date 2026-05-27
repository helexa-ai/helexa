//! Job variants accepted by the per-device worker thread.
//!
//! Each variant carries the inputs the synchronous dispatch handler
//! needs plus a `tokio::sync::oneshot::Sender` for the reply. The
//! async-side `DeviceWorkerHandle` constructs a job, sends it down the
//! `std::sync::mpsc` channel, and `await`s the oneshot for the reply.
//!
//! Phase 1 includes only `QueryVram` and `Shutdown`. Phases 2–4 add
//! forward, kv-cache clear, drop-arch, NCCL init/sanity, and the load
//! variants. Each new variant lands as a separate PR so the worker
//! thread stays small at every checkpoint.

use anyhow::Result;
use tokio::sync::oneshot;

/// One unit of work for the device worker.
pub enum Job {
    /// Query free / total VRAM on the device. Returns
    /// `(free_mb, total_mb)`. CPU builds and contexts that failed to
    /// initialise reply with `(0, 0)` — matches today's
    /// `device_vram_mb` sentinel so the log field values don't change.
    QueryVram {
        reply: oneshot::Sender<Result<(u64, u64)>>,
    },
    /// Tell the worker to break its dispatch loop and exit. The
    /// channel is then drained — any further jobs already queued get
    /// dropped (their oneshot senders are dropped, causing the async
    /// caller's receiver to return `Err` which `DeviceWorkerHandle`
    /// maps to `WorkerError::Gone`).
    Shutdown,
}
