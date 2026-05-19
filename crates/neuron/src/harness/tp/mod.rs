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
pub mod nccl_state;
pub mod rpc;
pub mod tp_linear;
pub mod worker;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use rpc::{WorkerRequest, WorkerResponse};

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

/// A live pool of worker subprocesses. Owns the `Child` handles so
/// dropping the pool kills the children; explicit `shutdown()` is
/// the graceful path.
pub struct WorkerPool {
    world_size: u32,
    workers: Vec<Worker>,
    /// Path to the neuron binary used to launch workers.
    #[allow(dead_code)]
    exe: PathBuf,
    /// Leader's own NCCL rank-0 state. Defaults to empty; populated by
    /// `init_nccl()`. Held here so the leader can participate in
    /// collectives (rank 0) without spawning a fourth subprocess.
    leader_nccl: nccl_state::NcclState,
}

impl WorkerPool {
    /// Spawn `world_size - 1` worker subprocesses. Rank 0 is the
    /// leader (in-process) and is *not* spawned here — the leader
    /// holds rank 0's NCCL Comm and shard in its own address space.
    ///
    /// `binary` is the path to the neuron executable to run for each
    /// worker (production passes `/proc/self/exe`; tests pass the
    /// sibling-binary path from `env!("CARGO_BIN_EXE_neuron")`).
    /// `cuda_devices` is one entry per rank including rank 0. Worker
    /// `i` (rank `i`) gets `cuda_devices[i]` as its `--cuda-device`.
    pub async fn spawn(binary: &Path, world_size: u32, cuda_devices: &[u32]) -> Result<Self> {
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
            leader_nccl: nccl_state::NcclState::new(),
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
        //    Runs on spawn_blocking because NCCL's init blocks until
        //    every rank has called in — that's exactly the workers
        //    above. The leader's NcclState is moved through the
        //    blocking task and returned to the pool.
        let leader_cfg = worker::WorkerConfig {
            rank: 0,
            world_size: self.world_size,
            cuda_device: leader_cuda_device,
        };
        let comm_id_for_leader = comm_id.clone();
        // Swap out the leader's NcclState into a fresh empty one so we
        // can move it into spawn_blocking; restore after the task
        // returns. (NcclState isn't Clone — it owns a real NCCL Comm.)
        let mut leader_state = std::mem::take(&mut self.leader_nccl);
        let (returned_state, leader_resp) = tokio::task::spawn_blocking(move || {
            let resp = leader_state.init(leader_cfg, &comm_id_for_leader);
            (leader_state, resp)
        })
        .await
        .context("leader NCCL init task panicked")?;
        self.leader_nccl = returned_state;
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

        // 2. Leader's own all_reduce, in spawn_blocking. NCCL operations
        //    block until every rank participates.
        let mut leader_state = std::mem::take(&mut self.leader_nccl);
        let (returned_state, leader_resp) = tokio::task::spawn_blocking(move || {
            let resp = leader_state.sanity_check();
            (leader_state, resp)
        })
        .await
        .context("leader NCCL sanity task panicked")?;
        self.leader_nccl = returned_state;

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
