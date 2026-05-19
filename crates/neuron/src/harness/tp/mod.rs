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

pub mod rpc;
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
    async fn request(&mut self, req: &WorkerRequest) -> Result<WorkerResponse> {
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
    /// Path to the neuron binary used to launch workers — captured at
    /// `spawn()` time via `/proc/self/exe` so the workers run the same
    /// binary the leader is running.
    exe: PathBuf,
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
        })
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
