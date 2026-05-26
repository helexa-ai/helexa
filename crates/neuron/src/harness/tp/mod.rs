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
        leader_device: &candle_core::Device,
        dtype: candle_core::DType,
        quant: Option<String>,
    ) -> Result<std::sync::Arc<tokio::sync::Mutex<TpLeaderModel>>> {
        use candle_nn::var_builder::ShardedSafeTensors;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        // Wrap the comm in SendComm immediately so it stays Send across
        // the await points in this method — bare Arc<Comm> would
        // poison the async fn's Send bound (Comm's raw NCCL pointer is
        // !Send). The wrapper's safety contract is satisfied by the
        // pool's outer Mutex serialising callers + the spawn_blocking
        // thread being the only place ops are issued.
        let leader_comm =
            nccl_state::SendComm(self.leader_nccl.comm().ok_or_else(|| {
                anyhow::anyhow!("leader NCCL not initialised; call init_nccl first")
            })?);
        let world_size = self.world_size;
        let safetensors_str: Vec<String> = safetensors_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        // 1. Fan out the LoadDenseShard request to every worker without
        //    awaiting their replies — they'll build their shards in
        //    parallel with the leader below.
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::LoadDenseShard {
                model_id: model_id.to_string(),
                config_json: config_json.to_string(),
                safetensors_paths: safetensors_str.clone(),
                quant: quant.clone(),
            })
            .await?;
        }

        // 2. Build rank 0's shard on the leader. Dispatch on model_type
        //    — for `qwen3` we build a `TpQwen3ForCausalLM`, for
        //    `qwen3_5` (Qwen3-Next, Qwen3.6's architecture) we build
        //    `TpQwen3_5ForCausalLM`. Both end up wrapped in the
        //    `TpLeaderModel` enum so downstream callers don't care.
        let model_type = serde_json::from_str::<serde_json::Value>(config_json)
            .ok()
            .as_ref()
            .and_then(|v| v.get("model_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let paths_for_leader: Vec<std::path::PathBuf> = safetensors_paths.to_vec();
        let device_for_leader = leader_device.clone();
        let comm_for_leader = leader_comm;
        let model_id_for_log = model_id.to_string();
        let config_json_for_leader = config_json.to_string();
        let quant_for_leader = quant.clone();

        let leader_model = tokio::task::spawn_blocking(move || -> Result<TpLeaderModel> {
            // SAFETY: same invariant as the single-GPU dense path —
            // the HF cache files are treated as immutable while the
            // mmap is held.
            let vb = unsafe {
                ShardedSafeTensors::var_builder(&paths_for_leader, dtype, &device_for_leader)
                    .context("build ShardedVarBuilder over safetensors")?
            };
            // SAFETY: as above — the HF cache files are immutable.
            let mmap = unsafe {
                candle_core::safetensors::MmapedSafetensors::multi(&paths_for_leader)
                    .context("build MmapedSafetensors for leader load")?
            };
            let comm = comm_for_leader.into_inner();
            let loaded = match model_type.as_str() {
                "qwen3" => {
                    let cfg: super::tp::tp_qwen3::Config = serde_json::from_str(&config_json_for_leader)
                        .context("parse Qwen3 Config JSON for leader load")?;
                    TpLeaderModel::Qwen3(super::tp::tp_qwen3::TpQwen3ForCausalLM::load(
                        &cfg, &vb, 0, world_size, comm,
                    )?)
                }
                "qwen3_5" => {
                    let cfg: super::tp::tp_qwen3_5::Config =
                        serde_json::from_str(&config_json_for_leader)
                            .context("parse Qwen3-Next Config JSON for leader load")?;
                    let quant_dtype =
                        super::tp::worker::parse_quant_string(quant_for_leader.as_deref())?;
                    TpLeaderModel::Qwen3_5(super::tp::tp_qwen3_5::TpQwen3_5ForCausalLM::load(
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
            tracing::info!(rank = 0, model = %model_id_for_log, model_type = %model_type, "loaded TP shard (leader)");
            Ok(loaded)
        })
        .await
        .context("leader load task panicked")??;

        // 3. Collect worker confirmations. Anything other than
        //    LoadDenseShardOk aborts the whole load — the leader's
        //    already-loaded shard drops when this fn returns Err.
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

        Ok(Arc::new(Mutex::new(leader_model)))
    }

    /// Run one forward step across every rank. The leader's forward
    /// returns the last-position logits as a candle Tensor on the
    /// leader's device; the caller does sampling out-of-band. Workers
    /// run their own forwards (the AllReduce inside row-parallel layers
    /// is what lets the leader's collective complete) and reply with
    /// `GenerateStepOk` — they do not ship logits over the wire.
    ///
    /// `tokens` is the input for this step (prompt for prefill, the
    /// previously-sampled token for decode). `offset` is the KV-cache
    /// position before this step.
    #[cfg(feature = "cuda")]
    pub async fn generate_step(
        &mut self,
        model_id: &str,
        leader_model: std::sync::Arc<tokio::sync::Mutex<TpLeaderModel>>,
        tokens: Vec<u32>,
        offset: usize,
    ) -> Result<candle_core::Tensor> {
        let step_start = std::time::Instant::now();
        let tokens_len = tokens.len();
        tracing::debug!(
            model = %model_id,
            tokens = tokens_len,
            offset,
            "WorkerPool::generate_step: fan-out"
        );
        // 1. Fan-out to workers.
        for w in &mut self.workers {
            w.send_only(&WorkerRequest::GenerateStep {
                model_id: model_id.to_string(),
                tokens: tokens.clone(),
                offset,
            })
            .await?;
        }

        // 2. Leader's forward in spawn_blocking. The AllReduce CustomOps
        //    inside the row-parallel layers block until every worker's
        //    forward issues the matching collective.
        let leader_start = std::time::Instant::now();
        let leader_result = tokio::task::spawn_blocking(move || -> Result<candle_core::Tensor> {
            let mut model = leader_model.blocking_lock();
            let device = model.device().clone();
            let input = candle_core::Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            // ForCausalLM::forward returns [B, 1, V] — squeeze both
            // leading dims to the rank-1 vocab logits the sampler wants.
            let logits = model.forward(&input, offset)?.squeeze(0)?.squeeze(0)?;
            Ok(logits)
        })
        .await
        .context("leader forward task panicked");
        let leader_ok = matches!(leader_result, Ok(Ok(_)));
        let leader_ms = leader_start.elapsed().as_millis();
        // Surface the leader's own error at WARN. Previously this was
        // silently coerced to `leader_ok=false` while only worker
        // ranks' errors got logged — when both the leader and a worker
        // fail together (the typical "CUDA context is now poisoned"
        // pattern after an OOM), the operator could see only the
        // worker side and had to guess what hit rank 0.
        if !leader_ok {
            let detail = match &leader_result {
                Ok(Err(e)) => format!("{e:#}"),
                Err(e) => format!("task: {e:#}"),
                Ok(Ok(_)) => unreachable!("leader_ok=false implies an error path"),
            };
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

        combine_leader_workers(leader_result, worker_errors, "GenerateStep")
    }

    /// Reset the KV cache for `model_id` on every rank. Called at the
    /// start of every inference so a fresh request doesn't attend over
    /// the previous one's tokens.
    pub async fn clear_kv_cache(
        &mut self,
        model_id: &str,
        #[cfg(feature = "cuda")] leader_model: std::sync::Arc<tokio::sync::Mutex<TpLeaderModel>>,
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
            let mut m = leader_model.lock().await;
            m.clear_kv_cache();
        }
        // Drain workers — same rationale as `generate_step`. The
        // leader's clear_kv_cache is in-process and infallible, but we
        // still always drain so an error on one worker doesn't leave
        // pending responses for the others.
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
