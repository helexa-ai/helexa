//! Tensor-parallel inference plumbing.
//!
//! The leader process (the neuron daemon proper) drives one
//! subprocess per non-zero NCCL rank — `tokio::process::Command` on
//! `/proc/self/exe --worker --rank N --tp-size N --cuda-device N` —
//! and talks to each over a newline-delimited JSON RPC channel on
//! the worker's stdin/stdout (see `rpc.rs`).
//!
//! Sub-staging:
//!
//! - **7a-i (this commit):** process lifecycle. `WorkerPool::spawn`
//!   forks N workers; `ping` round-trips every worker to confirm
//!   they're alive; `shutdown` cleanly drains and reaps. `Init` /
//!   `NcclSanityCheck` are stubbed.
//! - **7a-ii:** real NCCL `Comm` setup via `Init`, sanity check via
//!   `NcclSanityCheck`. CUDA-gated.
//! - **7b:** TP-aware Qwen3 inference dispatched through the pool.
//! - **7c:** crash detection, streaming SSE, graceful unload.

pub mod all_reduce;
pub mod fused_load;
pub mod nccl_state;
pub mod rpc;
pub mod tp_linear;
pub mod tp_qwen3;
pub mod tp_qwen3_5;
pub mod worker;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use rpc::{WorkerRequest, WorkerResponse};

/// Leader-side handle for any TP-loaded model. The pool's
/// `load_dense_shard` dispatches on `config.json#/model_type` to build
/// the right variant; downstream callers (the harness's
/// `chat_completion_tp` path, `generate_step`, `clear_kv_cache`,
/// `unload_model`) all hold this enum and let the variant dispatch
/// determine the concrete forward.
///
/// Variants gated on `cuda` because the underlying TP models hold
/// `Arc<cudarc::nccl::Comm>` references — irrelevant on CPU builds.
#[cfg(feature = "cuda")]
pub enum TpLeaderModel {
    Qwen3(tp_qwen3::TpQwen3ForCausalLM),
    Qwen3_5(tp_qwen3_5::TpQwen3_5ForCausalLM),
}

#[cfg(feature = "cuda")]
impl TpLeaderModel {
    pub fn forward(
        &mut self,
        input: &candle_core::Tensor,
        offset: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            TpLeaderModel::Qwen3(m) => m.forward(input, offset),
            TpLeaderModel::Qwen3_5(m) => m.forward(input, offset),
        }
    }

    /// Chunked image prefill on rank 0. Only the vision-capable
    /// `qwen3_5` arch supports it; the dense `qwen3` arch has no tower.
    pub fn prefill_with_images_chunked(
        &mut self,
        tokens: &[u32],
        base_offset: usize,
        image_pixels: &[candle_core::Tensor],
        image_token_id: u32,
        chunk_size: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            TpLeaderModel::Qwen3_5(m) => m.prefill_with_images_chunked(
                tokens,
                base_offset,
                image_pixels,
                image_token_id,
                chunk_size,
            ),
            TpLeaderModel::Qwen3(_) => {
                candle_core::bail!("prefill_with_images_chunked: qwen3 (dense) has no vision tower")
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            TpLeaderModel::Qwen3(m) => m.clear_kv_cache(),
            TpLeaderModel::Qwen3_5(m) => m.clear_kv_cache(),
        }
    }

    pub fn device(&self) -> &candle_core::Device {
        match self {
            TpLeaderModel::Qwen3(m) => m.device(),
            TpLeaderModel::Qwen3_5(m) => m.device(),
        }
    }
}

/// One worker subprocess plus its bidirectional stdio handles.
struct Worker {
    rank: u32,
    /// Captured so the leader can log "spawned rank N on device M" and
    /// future stages can re-issue Init after a CUDA reset. Unused in
    /// the Stage 7a-i RPC paths themselves.
    #[allow(dead_code)]
    cuda_device: u32,
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl Worker {
    /// Send a request and wait for the response. Used for sequenced
    /// ops like `Ping` / `Shutdown` where the caller doesn't need to
    /// overlap the worker's execution with the leader's.
    async fn request(&mut self, req: &WorkerRequest) -> Result<WorkerResponse> {
        self.send_only(req).await?;
        self.recv_only().await
    }

    /// Write a request without awaiting its response. Pair with
    /// `recv_only` from the caller when leader and worker need to do
    /// work concurrently — e.g. during `Init`, where the leader
    /// itself calls `Comm::from_rank` on rank 0 in parallel with the
    /// workers, then collects `InitOk` after NCCL completes.
    async fn send_only(&mut self, req: &WorkerRequest) -> Result<()> {
        let mut line = serde_json::to_string(req).context("serialise WorkerRequest")?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .await
            .with_context(|| format!("write request to rank {}", self.rank))?;
        self.stdin
            .flush()
            .await
            .with_context(|| format!("flush stdin to rank {}", self.rank))?;
        Ok(())
    }

    async fn recv_only(&mut self) -> Result<WorkerResponse> {
        let reply = self
            .stdout
            .next_line()
            .await
            .with_context(|| format!("read reply from rank {}", self.rank))?
            .ok_or_else(|| anyhow::anyhow!("rank {} stdout closed before reply", self.rank))?;
        serde_json::from_str(&reply)
            .with_context(|| format!("parse reply from rank {}: {reply:?}", self.rank))
    }
}

/// Drain one response from every worker, classifying each via the
/// supplied checker. Always reads from every worker — even if some
/// fail — so the next call's recv doesn't pick up stale responses
/// from this one (pipe-poisoning was the cause of the
/// "ClearKvCache: expected KvCacheCleared, got GenerateStepOk" class
/// of bugs).
///
/// Returns a vector of `rank N: detail` strings for any worker that
/// errored, expected-mismatched, or failed to respond. Caller decides
/// how to combine these with the leader's outcome.
async fn drain_workers(
    workers: &mut [Worker],
    mut check: impl FnMut(WorkerResponse) -> std::result::Result<(), String>,
) -> Vec<String> {
    let mut errs = Vec::new();
    for w in workers {
        match w.recv_only().await {
            Ok(resp) => {
                if let Err(detail) = check(resp) {
                    errs.push(format!("rank {} {detail}", w.rank));
                }
            }
            Err(e) => errs.push(format!("rank {} recv: {e:#}", w.rank)),
        }
    }
    errs
}

/// Combine a leader's `Result<Result<T>>` (the typical
/// `spawn_blocking → JoinHandle<Result<T>>` shape) with the worker
/// drain results into a single `Result<T>`. Leader failures take
/// precedence in the error message but worker errors get appended so
/// the operator sees both halves.
#[cfg(feature = "cuda")]
fn combine_leader_workers<T>(
    leader: Result<Result<T>>,
    worker_errors: Vec<String>,
    op: &str,
) -> Result<T> {
    match leader {
        Ok(Ok(value)) => {
            if worker_errors.is_empty() {
                Ok(value)
            } else {
                anyhow::bail!(
                    "{op}: leader succeeded but workers failed: {}",
                    worker_errors.join("; ")
                )
            }
        }
        Ok(Err(e)) => {
            if worker_errors.is_empty() {
                Err(e.context(format!("{op}: leader forward failed")))
            } else {
                Err(e.context(format!(
                    "{op}: leader forward failed and workers also failed: {}",
                    worker_errors.join("; ")
                )))
            }
        }
        Err(panic_err) => {
            if worker_errors.is_empty() {
                Err(panic_err)
            } else {
                Err(panic_err.context(format!(
                    "{op}: leader task panicked and workers failed: {}",
                    worker_errors.join("; ")
                )))
            }
        }
    }
}

/// A live pool of worker subprocesses. Owns the `Child` handles so
/// dropping the pool kills the children; explicit `shutdown()` is
/// the graceful path.
pub struct WorkerPool {
    world_size: u32,
    workers: Vec<Worker>,
    /// Path to the neuron binary used to launch workers.
    #[allow(dead_code)]
    exe: PathBuf,
    /// The leader's per-device CUDA worker thread. Phase 3 moved the
    /// leader's `NcclState` (rank-0 NCCL Comm) into this thread, so
    /// every NCCL op (init, sanity, all_reduce inside forward) issues
    /// from one OS thread for the daemon's lifetime. The handle is
    /// also used by `load_dense_shard` to clone the leader's
    /// `Arc<Comm>` for the row-parallel layers' AllReduce ops; in
    /// Phase 4 the load itself moves onto the worker and that bridge
    /// goes away.
    pub(crate) leader_worker: std::sync::Arc<super::device_worker::DeviceWorkerHandle>,
    /// Cached handle to the leader's NCCL `Comm`, fetched at `init_nccl`
    /// while the worker thread is responsive. The TP step watchdog uses
    /// it to `ncclCommAbort` a wedged collective from the async thread —
    /// the one NCCL op allowed concurrently with an in-flight collective,
    /// and the only way to unblock the in-process leader thread so
    /// recovery's `unload` doesn't itself hang (#17 Stage 2). `None` if
    /// init couldn't cache it; the watchdog then logs that it can't abort.
    #[cfg(feature = "cuda")]
    leader_comm: Option<nccl_state::SendComm>,
}

/// Per-step deadline for a TP forward (#17 Stage 2). A healthy decode
/// step or chunked prefill completes in well under a second; a wedged
/// NCCL collective never returns. Generous default so no legitimate step
/// trips it; overridable via `NEURON_TP_STEP_TIMEOUT_S` (seconds).
#[cfg(feature = "cuda")]
fn tp_step_timeout() -> std::time::Duration {
    let secs = std::env::var("NEURON_TP_STEP_TIMEOUT_S")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(120);
    std::time::Duration::from_secs(secs)
}

impl WorkerPool {
    /// Abort the leader's NCCL comm to unblock a collective the watchdog
    /// found wedged (#17 Stage 2). Logs the whole sequence loudly so a
    /// real-world hang leaves a greppable forensic trail
    /// (`tp watchdog:` / `ncclCommAbort`). Calling abort from this async
    /// thread while the worker thread is blocked inside the collective is
    /// the one concurrent NCCL op the library sanctions — it is how a
    /// stuck/failed collective is unblocked.
    #[cfg(feature = "cuda")]
    fn watchdog_abort_leader_comm(&self, model_id: &str, secs: u64) {
        tracing::error!(
            model = %model_id,
            timeout_s = secs,
            "tp watchdog: leader forward exceeded deadline — NCCL collective wedged; \
             aborting comm to unblock the leader thread for auto-recovery"
        );
        match &self.leader_comm {
            Some(c) => match c.0.abort() {
                Ok(()) => tracing::error!(
                    model = %model_id,
                    "tp watchdog: ncclCommAbort succeeded — wedged collective unblocked; \
                     failing the step so the model auto-recovers (unload+reload)"
                ),
                Err(e) => tracing::error!(
                    model = %model_id, error = ?e,
                    "tp watchdog: ncclCommAbort failed — recovery may stall until a process restart"
                ),
            },
            None => tracing::error!(
                model = %model_id,
                "tp watchdog: no cached leader comm handle — cannot abort; recovery will rely \
                 on a process restart"
            ),
        }
    }

    /// Spawn `world_size - 1` worker subprocesses. Rank 0 is the
    /// leader (in-process) and is *not* spawned here — the leader
    /// holds rank 0's NCCL Comm and shard in its own address space.
    ///
    /// `binary` is the path to the neuron executable to run for each
    /// worker (production passes `/proc/self/exe`; tests pass the
    /// sibling-binary path from `env!("CARGO_BIN_EXE_neuron")`).
    /// `cuda_devices` is one entry per rank including rank 0. Worker
    /// `i` (rank `i`) gets `cuda_devices[i]` as its `--cuda-device`.
    pub async fn spawn(
        binary: &Path,
        world_size: u32,
        cuda_devices: &[u32],
        leader_worker: std::sync::Arc<super::device_worker::DeviceWorkerHandle>,
    ) -> Result<Self> {
        if world_size < 2 {
            anyhow::bail!(
                "WorkerPool::spawn called with world_size={world_size}; \
                 use the single-process path for world_size < 2"
            );
        }
        if cuda_devices.len() as u32 != world_size {
            anyhow::bail!(
                "expected {world_size} cuda_devices entries, got {}",
                cuda_devices.len()
            );
        }
        let exe = binary.to_path_buf();

        let mut workers = Vec::with_capacity(world_size as usize - 1);
        // Rank 0 stays in-process. Spawn ranks 1..world_size.
        for rank in 1..world_size {
            let cuda_device = cuda_devices[rank as usize];
            let mut cmd = Command::new(&exe);
            cmd.arg("--worker")
                .arg("--rank")
                .arg(rank.to_string())
                .arg("--tp-size")
                .arg(world_size.to_string())
                .arg("--cuda-device")
                .arg(cuda_device.to_string())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                // Inherit stderr so worker tracing surfaces alongside
                // the leader's journalctl stream.
                .stderr(Stdio::inherit())
                .kill_on_drop(true);

            let mut child = cmd
                .spawn()
                .with_context(|| format!("spawn worker rank {rank}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("rank {rank}: no stdin handle"))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| anyhow::anyhow!("rank {rank}: no stdout handle"))?;
            let stdout = BufReader::new(stdout).lines();

            workers.push(Worker {
                rank,
                cuda_device,
                child,
                stdin,
                stdout,
            });
            tracing::info!(rank, cuda_device, "spawned tp worker");
        }

        Ok(Self {
            world_size,
            workers,
            exe,
            leader_worker,
            #[cfg(feature = "cuda")]
            leader_comm: None,
        })
    }

    /// Establish the NCCL communicator across the leader (rank 0) and
    /// every worker subprocess. Rendezvous is via a freshly-generated
    /// `Id` broadcast over the RPC stream; the actual handshake blocks
    /// inside `Comm::from_rank` until all `world_size` ranks check in.
    ///
    /// `leader_cuda_device` is the CUDA device the leader binds rank 0
    /// to — typically the first entry of the `cuda_devices` slice
    /// originally passed to `spawn()`.
    ///
    /// On the non-cuda build this immediately fails because the leader
    /// can't generate an `Id` without libnccl. The same call works in
    /// the worker path (returning a no-cuda error response) so the
    /// failure surface is uniform.
    pub async fn init_nccl(&mut self, leader_cuda_device: u32) -> Result<()> {
        let comm_id = nccl_state::generate_comm_id_hex()
            .map_err(|m| anyhow::anyhow!("generate NCCL id: {m}"))?;

        // 1. Write Init to every worker's stdin without awaiting the
        //    response. Workers will parse and call Comm::from_rank
        //    concurrently with the leader below.
        for w in &mut self.workers {
            let req = WorkerRequest::Init {
                comm_id: comm_id.clone(),
            };
            w.send_only(&req).await?;
        }

        // 2. Leader rank 0 calls Comm::from_rank on its own device.
        //    Phase 3 moved this from spawn_blocking onto the leader's
        //    device worker thread (`Job::NcclInit`); the underlying
        //    `Comm` now lives on the same OS thread for its entire
        //    lifetime, including every later `Comm::all_reduce` issued
        //    by the row-parallel layers during forward.
        //
        //    NCCL's init blocks until every rank has called in — the
        //    subprocess workers above and the leader's device worker
        //    here. The Job's reply unblocks when the leader's
        //    Comm::from_rank returns.
        let leader_cfg = worker::WorkerConfig {
            rank: 0,
            world_size: self.world_size,
            cuda_device: leader_cuda_device,
        };
        let leader_resp = self
            .leader_worker
            .nccl_init(leader_cfg, comm_id.clone())
            .await
            .map_err(|e| anyhow::anyhow!("leader NCCL init via device worker: {e}"))?;
        match leader_resp {
            rpc::WorkerResponse::InitOk => {}
            rpc::WorkerResponse::Error { kind, message } => {
                anyhow::bail!("leader rank 0 init failed [{kind}]: {message}");
            }
            other => anyhow::bail!("leader rank 0 init: unexpected {other:?}"),
        }

        // 3. Read InitOk from each worker. By now every worker has
        //    completed its Comm::from_rank call (NCCL released them
        //    when the leader joined the handshake) and is writing its
        //    response.
        for w in &mut self.workers {
            let resp = w.recv_only().await?;
            match &resp {
                rpc::WorkerResponse::InitOk => {}
                rpc::WorkerResponse::Error { kind, message } => {
                    anyhow::bail!("worker rank {} init failed [{kind}]: {message}", w.rank);
                }
                other => anyhow::bail!(
                    "worker rank {} init: expected InitOk, got {other:?}",
                    w.rank
                ),
            }
        }
        tracing::info!(
            world_size = self.world_size,
            "NCCL communicator established across all ranks"
        );

        // Cache the leader's Comm handle now, while the worker thread is
        // responsive, so the TP step watchdog can abort a wedged
        // collective later (it can't fetch it then — the thread is stuck).
        // (#17 Stage 2.)
        #[cfg(feature = "cuda")]
        {
            self.leader_comm = self.leader_worker.get_leader_comm().await;
            if self.leader_comm.is_some() {
                tracing::debug!("cached leader NCCL comm handle for the TP step watchdog");
            } else {
                tracing::warn!(
                    "could not cache leader NCCL comm handle; the TP step watchdog will be \
                     unable to abort a wedged collective (a hang would need a process restart)"
                );
            }
        }
        Ok(())
    }

    /// Validate the NCCL communicator: every rank `all_reduce`s a
    /// sentinel `1u32` with `ReduceOp::Sum`; the expected total is
    /// `world_size`. Confirms the handshake is live, not just
    /// configured.
    ///
    /// Must be called after `init_nccl()`; before that the leader has
    /// no Comm and the workers reply with `nccl_not_initialised`.
    pub async fn nccl_sanity_check(&mut self) -> Result<()> {
        // 1. Trigger the all_reduce on every worker (write-only).
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::NcclSanityCheck).await?;
        }

        // 2. Leader's own all_reduce, on its device worker thread.
        //    NCCL operations block until every rank participates;
        //    Job::NcclSanity returns once the leader's side completes
        //    (which happens when every subprocess worker reaches its
        //    all_reduce call too).
        let leader_resp = self
            .leader_worker
            .nccl_sanity()
            .await
            .map_err(|e| anyhow::anyhow!("leader NCCL sanity via device worker: {e}"))?;

        let expected = self.world_size;
        let leader_sum = match leader_resp {
            rpc::WorkerResponse::NcclSanityResult { observed_sum } => observed_sum,
            rpc::WorkerResponse::Error { kind, message } => {
                anyhow::bail!("leader rank 0 sanity failed [{kind}]: {message}");
            }
            other => anyhow::bail!("leader rank 0 sanity: unexpected {other:?}"),
        };
        if leader_sum != expected {
            anyhow::bail!("leader observed_sum={leader_sum}, expected {expected}");
        }

        // 3. Read sanity result from each worker. All must match
        //    world_size — anything else means the collective didn't
        //    complete consistently across ranks.
        for w in &mut self.workers {
            let resp = w.recv_only().await?;
            match resp {
                rpc::WorkerResponse::NcclSanityResult { observed_sum }
                    if observed_sum == expected => {}
                rpc::WorkerResponse::NcclSanityResult { observed_sum } => {
                    anyhow::bail!(
                        "worker rank {} observed_sum={observed_sum}, expected {expected}",
                        w.rank
                    );
                }
                rpc::WorkerResponse::Error { kind, message } => {
                    anyhow::bail!("worker rank {} sanity failed [{kind}]: {message}", w.rank);
                }
                other => anyhow::bail!("worker rank {} sanity: unexpected {other:?}", w.rank),
            }
        }
        tracing::info!(
            world_size = expected,
            "NCCL sanity check OK across all ranks"
        );
        Ok(())
    }

    /// Ping every worker and return their Pong payloads in rank order.
    /// Useful right after `spawn` to confirm the lifecycle plumbing is
    /// intact before kicking off any heavier work.
    pub async fn ping_all(&mut self) -> Result<Vec<WorkerResponse>> {
        let mut out = Vec::with_capacity(self.workers.len());
        for w in &mut self.workers {
            let resp = w.request(&WorkerRequest::Ping).await?;
            match &resp {
                WorkerResponse::Pong { rank, .. } if *rank == w.rank => {}
                WorkerResponse::Pong { rank, .. } => {
                    anyhow::bail!("rank mismatch: expected {}, got {rank}", w.rank);
                }
                other => anyhow::bail!("expected Pong from rank {}, got {other:?}", w.rank),
            }
            out.push(resp);
        }
        Ok(out)
    }

    /// Load this rank's shard of a dense Qwen3 model on every rank.
    ///
    /// The leader builds rank 0's `TpQwen3ForCausalLM` directly into
    /// the returned `Arc<Mutex<_>>` — workers build their rank-local
    /// shards in their own address spaces and confirm via
    /// `LoadDenseShardOk`. All ranks see the same `safetensors_paths`;
    /// `ShardedVarBuilder` slices each tensor by rank at materialisation
    /// time, so the per-rank VRAM footprint is roughly `1/world_size`
    /// of the full model (plus the replicated embedding/norm/lm_head).
    ///
    /// `leader_device` is the candle `Device` the leader's shard lives
    /// on — typically `Device::new_cuda(leader_cuda_device)` matching
    /// the same index passed to `init_nccl`. `dtype` is the on-device
    /// element type; bf16 is the canonical Qwen3 distribution dtype.
    ///
    /// `init_nccl` must have completed first. Bails if the leader's
    /// NCCL comm isn't set up yet.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub async fn load_dense_shard(
        &mut self,
        model_id: &str,
        config_json: &str,
        safetensors_paths: &[std::path::PathBuf],
        _leader_device: &candle_core::Device,
        dtype: candle_core::DType,
        quant: Option<String>,
    ) -> Result<super::device_worker::TpHandle> {
        let world_size = self.world_size;
        let safetensors_str: Vec<String> = safetensors_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        // 1. Fan out the LoadDenseShard request to every subprocess
        //    worker without awaiting their replies — they'll build
        //    their shards in parallel with the leader below.
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::LoadDenseShard {
                model_id: model_id.to_string(),
                config_json: config_json.to_string(),
                safetensors_paths: safetensors_str.clone(),
                quant: quant.clone(),
            })
            .await?;
        }

        // 2. Build rank 0's shard on the leader's device worker
        //    thread. Phase 4 moved the load itself onto the worker —
        //    the dispatch handler reads `state.nccl.comm()` directly
        //    so the leader's `Arc<Comm>` clones embedded in the
        //    row-parallel layers are constructed and used on the same
        //    OS thread for the model's entire lifetime. No
        //    spawn_blocking, no SendComm bridge.
        let handle = self
            .leader_worker
            .tp_load_shard(
                model_id.to_string(),
                config_json.to_string(),
                safetensors_paths.to_vec(),
                dtype,
                quant.clone(),
                world_size,
            )
            .await
            .map_err(|e| anyhow::anyhow!("leader TP shard load via device worker: {e}"))?;

        // 3. Collect worker confirmations. Anything other than
        //    LoadDenseShardOk aborts the whole load — the leader's
        //    already-inserted shard would leak in the worker slab
        //    until the daemon restarts; an explicit DropTp would be
        //    cleaner but the failure here is rare and the operator's
        //    next step is to restart anyway.
        for w in &mut self.workers {
            let resp = w.recv_only().await?;
            match resp {
                WorkerResponse::LoadDenseShardOk => {}
                WorkerResponse::Error { kind, message } => {
                    anyhow::bail!("worker rank {} LoadDenseShard [{kind}]: {message}", w.rank)
                }
                other => anyhow::bail!(
                    "worker rank {} LoadDenseShard: expected LoadDenseShardOk, got {other:?}",
                    w.rank
                ),
            }
        }

        Ok(handle)
    }

    /// Run one forward step across every rank. The leader's forward
    /// runs on the device worker thread via `Job::TpForwardLogits` and
    /// returns CPU-side `[vocab]` logits as `Vec<f32>`; the async
    /// caller wraps them in a CPU tensor for `apply_repeat_penalty` +
    /// sampling without holding a device-resident tensor on a tokio
    /// thread.
    ///
    /// Subprocess workers run their own forwards in parallel (the
    /// AllReduce CustomOps inside row-parallel layers are what let
    /// the leader's collective complete) and reply with
    /// `GenerateStepOk` over the RPC stream — they do not ship logits.
    ///
    /// `tokens` is the input for this step (prompt for prefill, the
    /// previously-sampled token for decode). `offset` is the KV-cache
    /// position before this step.
    #[cfg(feature = "cuda")]
    pub async fn generate_step(
        &mut self,
        model_id: &str,
        leader_handle: super::device_worker::TpHandle,
        tokens: Vec<u32>,
        offset: usize,
    ) -> Result<Vec<f32>> {
        let step_start = std::time::Instant::now();
        let tokens_len = tokens.len();
        tracing::debug!(
            model = %model_id,
            tokens = tokens_len,
            offset,
            "WorkerPool::generate_step: fan-out"
        );
        // 1. Fan-out to subprocess workers.
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::GenerateStep {
                model_id: model_id.to_string(),
                tokens: tokens.clone(),
                offset,
            })
            .await?;
        }

        // 2. Leader's forward on its device worker thread. The
        //    AllReduce CustomOps inside the row-parallel layers block
        //    until every subprocess worker's forward issues the
        //    matching collective. Returning CPU-side `Vec<f32>` keeps
        //    the device tensor from escaping the worker thread —
        //    that's the invariant the whole refactor exists to
        //    preserve.
        let leader_start = std::time::Instant::now();
        let timeout = tp_step_timeout();
        let leader_fut = self
            .leader_worker
            .tp_forward_logits(leader_handle, tokens, offset);
        let leader_result = match tokio::time::timeout(timeout, leader_fut).await {
            Ok(r) => r,
            Err(_elapsed) => {
                // Watchdog (#17 Stage 2): the NCCL collective is wedged.
                // Abort the leader comm to unblock its thread, then fail
                // the step WITHOUT draining (the subprocess workers are
                // wedged too; recovery's unload kills them). The error
                // poisons the model → auto-recovery, which no longer hangs
                // because the leader thread is now responsive.
                self.watchdog_abort_leader_comm(model_id, timeout.as_secs());
                anyhow::bail!(
                    "tp watchdog: leader forward exceeded {}s deadline; aborted wedged NCCL \
                     comm — model will auto-recover",
                    timeout.as_secs()
                );
            }
        };
        let leader_ok = leader_result.is_ok();
        let leader_ms = leader_start.elapsed().as_millis();
        // Surface the leader's own error at WARN before draining
        // workers so the operator can correlate it with whatever the
        // subprocess workers logged. Previously this was silently
        // coerced to a bool.
        if !leader_ok {
            let detail = leader_result
                .as_ref()
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default();
            tracing::warn!(
                model = %model_id,
                tokens = tokens_len,
                offset,
                leader_ms,
                error = %detail,
                "WorkerPool::generate_step: leader forward failed"
            );
        }
        tracing::debug!(
            model = %model_id,
            tokens = tokens_len,
            leader_ms,
            leader_ok,
            "WorkerPool::generate_step: leader forward returned"
        );

        // 3. ALWAYS drain worker responses, regardless of whether the
        //    leader succeeded. Skipping this on the leader's error
        //    path leaves stale GenerateStepOk replies in the worker
        //    pipes that poison the NEXT request's recv (was seeing
        //    "ClearKvCache: expected KvCacheCleared, got
        //    GenerateStepOk" the call after any forward-time failure).
        let drain_start = std::time::Instant::now();
        let worker_errors = drain_workers(&mut self.workers, |r| match r {
            WorkerResponse::GenerateStepOk => Ok(()),
            WorkerResponse::Error { kind, message } => Err(format!("[{kind}]: {message}")),
            other => Err(format!("expected GenerateStepOk, got {other:?}")),
        })
        .await;
        tracing::debug!(
            model = %model_id,
            drain_ms = drain_start.elapsed().as_millis(),
            errors = worker_errors.len(),
            total_ms = step_start.elapsed().as_millis(),
            "WorkerPool::generate_step: workers drained"
        );

        // Combine the leader's Result + the workers' string-error
        // list. Phase 3 inlines this because the upstream
        // `combine_leader_workers` expects the spawn_blocking-shaped
        // `Result<Result<T>>`; the new device-worker path produces a
        // single `Result<T, WorkerError>` instead.
        match leader_result {
            Ok(values) => {
                if worker_errors.is_empty() {
                    Ok(values)
                } else {
                    anyhow::bail!(
                        "GenerateStep: leader succeeded but workers failed: {}",
                        worker_errors.join("; ")
                    )
                }
            }
            Err(e) => {
                if worker_errors.is_empty() {
                    Err(anyhow::Error::new(e).context("GenerateStep: leader forward failed"))
                } else {
                    Err(anyhow::Error::new(e).context(format!(
                        "GenerateStep: leader forward failed and workers also failed: {}",
                        worker_errors.join("; ")
                    )))
                }
            }
        }
    }

    /// Image-bearing variant of [`Self::generate_step`] for the
    /// single-shot vision prefill. Identical fan-out / leader-forward /
    /// drain shape, but every rank runs the encode + splice path:
    ///
    /// - subprocess workers get `GenerateStepWithImages` (carrying the
    ///   source `image_data_uris`); each preprocesses + encodes through
    ///   its replicated tower and splices locally;
    /// - the leader runs the same encode + splice + forward on its
    ///   device worker thread via `tp_forward_logits_with_images`.
    ///
    /// The row-parallel `AllReduce`s synchronise the ranks exactly as in
    /// the text path. Because the tower is replicated and the preprocess
    /// is deterministic, every rank's spliced hidden state matches — no
    /// embedding broadcast. Only used for prefill; decode reuses
    /// `generate_step`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_step_with_images(
        &mut self,
        model_id: &str,
        leader_handle: super::device_worker::TpHandle,
        tokens: Vec<u32>,
        offset: usize,
        image_token_id: u32,
        image_data_uris: Vec<String>,
        chunk_size: usize,
    ) -> Result<Vec<f32>> {
        let step_start = std::time::Instant::now();
        let tokens_len = tokens.len();
        tracing::debug!(
            model = %model_id,
            tokens = tokens_len,
            offset,
            images = image_data_uris.len(),
            chunk_size,
            "WorkerPool::generate_step_with_images: fan-out"
        );

        // 1. Fan-out the image-bearing prefill to subprocess workers.
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::GenerateStepWithImages {
                model_id: model_id.to_string(),
                tokens: tokens.clone(),
                offset,
                image_token_id,
                image_data_uris: image_data_uris.clone(),
                chunk_size,
            })
            .await?;
        }

        // 2. Leader's image forward on its device worker thread. The
        //    AllReduce CustomOps block until every worker issues the
        //    matching collective; CPU-side logits keep the device tensor
        //    from escaping the worker thread.
        let leader_start = std::time::Instant::now();
        let timeout = tp_step_timeout();
        let leader_fut = self.leader_worker.tp_forward_logits_with_images(
            leader_handle,
            tokens,
            offset,
            image_token_id,
            image_data_uris,
            chunk_size,
        );
        let leader_result = match tokio::time::timeout(timeout, leader_fut).await {
            Ok(r) => r,
            Err(_elapsed) => {
                // Watchdog (#17 Stage 2) — see generate_step. Vision
                // prefill is still well under the deadline on healthy
                // hardware; a timeout means a wedged collective.
                self.watchdog_abort_leader_comm(model_id, timeout.as_secs());
                anyhow::bail!(
                    "tp watchdog: leader image forward exceeded {}s deadline; aborted wedged \
                     NCCL comm — model will auto-recover",
                    timeout.as_secs()
                );
            }
        };
        let leader_ok = leader_result.is_ok();
        let leader_ms = leader_start.elapsed().as_millis();
        if !leader_ok {
            let detail = leader_result
                .as_ref()
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default();
            tracing::warn!(
                model = %model_id,
                tokens = tokens_len,
                offset,
                leader_ms,
                error = %detail,
                "WorkerPool::generate_step_with_images: leader forward failed"
            );
        }

        // 3. ALWAYS drain worker responses, regardless of the leader's
        //    outcome, so stale GenerateStepOk replies don't poison the
        //    next request's recv (same invariant as generate_step).
        let worker_errors = drain_workers(&mut self.workers, |r| match r {
            WorkerResponse::GenerateStepOk => Ok(()),
            WorkerResponse::Error { kind, message } => Err(format!("[{kind}]: {message}")),
            other => Err(format!("expected GenerateStepOk, got {other:?}")),
        })
        .await;
        tracing::debug!(
            model = %model_id,
            leader_ms,
            leader_ok,
            errors = worker_errors.len(),
            total_ms = step_start.elapsed().as_millis(),
            "WorkerPool::generate_step_with_images: workers drained"
        );

        match leader_result {
            Ok(values) => {
                if worker_errors.is_empty() {
                    Ok(values)
                } else {
                    anyhow::bail!(
                        "GenerateStepWithImages: leader succeeded but workers failed: {}",
                        worker_errors.join("; ")
                    )
                }
            }
            Err(e) => {
                if worker_errors.is_empty() {
                    Err(anyhow::Error::new(e)
                        .context("GenerateStepWithImages: leader forward failed"))
                } else {
                    Err(anyhow::Error::new(e).context(format!(
                        "GenerateStepWithImages: leader forward failed and workers also failed: {}",
                        worker_errors.join("; ")
                    )))
                }
            }
        }
    }

    /// Reset the KV cache for `model_id` on every rank. Called at the
    /// start of every inference so a fresh request doesn't attend over
    /// the previous one's tokens.
    pub async fn clear_kv_cache(
        &mut self,
        model_id: &str,
        #[cfg(feature = "cuda")] leader_handle: super::device_worker::TpHandle,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        tracing::debug!(model = %model_id, "WorkerPool::clear_kv_cache: fan-out");
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::ClearKvCache {
                model_id: model_id.to_string(),
            })
            .await?;
        }
        #[cfg(feature = "cuda")]
        {
            // Leader-side clear on the device worker thread —
            // `TpLeaderModel::clear_kv_cache` is infallible but still
            // routes through Job::TpClearKv so the cache reset runs
            // on the same thread that owns the model's CUDA tensors.
            if let Err(e) = self.leader_worker.tp_clear_kv(leader_handle).await {
                anyhow::bail!("leader TP clear_kv_cache via device worker: {e}");
            }
        }
        // Drain workers — same rationale as `generate_step`. The
        // leader's clear_kv_cache is now async-via-channel but still
        // returns before the drain so the workers' KvCacheCleared
        // replies are processed in order.
        let worker_errors = drain_workers(&mut self.workers, |r| match r {
            WorkerResponse::KvCacheCleared => Ok(()),
            WorkerResponse::Error { kind, message } => Err(format!("[{kind}]: {message}")),
            other => Err(format!("expected KvCacheCleared, got {other:?}")),
        })
        .await;
        tracing::debug!(
            model = %model_id,
            elapsed_ms = start.elapsed().as_millis(),
            errors = worker_errors.len(),
            "WorkerPool::clear_kv_cache: workers drained"
        );
        if !worker_errors.is_empty() {
            anyhow::bail!("ClearKvCache: {}", worker_errors.join("; "));
        }
        Ok(())
    }

    /// Drop this model's shards on every rank. The leader's shard is
    /// expected to have been dropped by the caller (its `Arc` was held
    /// in the TpLoadedModel and goes away when that's removed).
    pub async fn unload_model(&mut self, model_id: &str) -> Result<()> {
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::UnloadModel {
                model_id: model_id.to_string(),
            })
            .await?;
        }
        for w in &mut self.workers {
            let resp = w.recv_only().await?;
            match resp {
                WorkerResponse::Unloaded => {}
                WorkerResponse::Error { kind, message } => {
                    anyhow::bail!("worker rank {} UnloadModel [{kind}]: {message}", w.rank)
                }
                other => anyhow::bail!(
                    "worker rank {} UnloadModel: expected Unloaded, got {other:?}",
                    w.rank
                ),
            }
        }
        Ok(())
    }

    /// Send `Shutdown` to every worker, await each `Bye`, and reap the
    /// children. Best-effort — individual worker failures are logged
    /// but don't abort the rest of the sweep.
    pub async fn shutdown(mut self) -> Result<()> {
        for w in &mut self.workers {
            match w.request(&WorkerRequest::Shutdown).await {
                Ok(WorkerResponse::Bye) => {}
                Ok(other) => tracing::warn!(
                    rank = w.rank,
                    response = ?other,
                    "expected Bye on shutdown"
                ),
                Err(e) => tracing::warn!(rank = w.rank, error = %e, "shutdown request failed"),
            }
        }
        for w in &mut self.workers {
            match w.child.wait().await {
                Ok(status) => tracing::info!(rank = w.rank, %status, "worker exited"),
                Err(e) => tracing::warn!(rank = w.rank, error = %e, "wait on worker failed"),
            }
        }
        Ok(())
    }

    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    pub fn binary_path(&self) -> &PathBuf {
        &self.exe
    }
}
