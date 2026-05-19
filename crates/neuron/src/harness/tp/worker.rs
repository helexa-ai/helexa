//! Entry point for `neuron --worker`.
//!
//! Stage 7a-i: bare RPC loop — `Ping` and `Shutdown` work, `Init` and
//! `NcclSanityCheck` return `Error{kind = "not_implemented_7a_i"}`.
//! Stage 7a-ii will replace the latter with real `cudarc::nccl` calls
//! behind the `cuda` feature.
//!
//! The worker reads one newline-delimited JSON `WorkerRequest` from
//! stdin per loop iteration, dispatches synchronously, and writes
//! exactly one `WorkerResponse` JSON line to stdout. tracing goes to
//! stderr so it doesn't collide with the RPC stream.

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::rpc::{WorkerRequest, WorkerResponse};

#[derive(Debug, Clone, Copy)]
pub struct WorkerConfig {
    pub rank: u32,
    pub world_size: u32,
    pub cuda_device: u32,
}

/// Drive the worker RPC loop until `Shutdown` or EOF on stdin.
pub async fn run(config: WorkerConfig) -> Result<()> {
    tracing::info!(
        rank = config.rank,
        world_size = config.world_size,
        cuda_device = config.cuda_device,
        "tp worker starting"
    );

    let mut state = WorkerState::new(config);
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let req: WorkerRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = WorkerResponse::Error {
                    kind: "bad_request".into(),
                    message: format!("parse {line:?}: {e}"),
                };
                write_response(&mut stdout, &resp).await?;
                continue;
            }
        };

        let resp = state.handle(req).await;
        let is_bye = matches!(resp, WorkerResponse::Bye);
        write_response(&mut stdout, &resp).await?;
        if is_bye {
            break;
        }
    }

    tracing::info!(rank = config.rank, "tp worker exiting");
    Ok(())
}

async fn write_response(stdout: &mut tokio::io::Stdout, resp: &WorkerResponse) -> Result<()> {
    let mut line = serde_json::to_string(resp)?;
    line.push('\n');
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

/// Per-worker state. In Stage 7a-i this only carries the static
/// config; 7a-ii adds an `Option<cudarc::nccl::safe::Comm>` populated
/// by `Init`.
struct WorkerState {
    config: WorkerConfig,
}

impl WorkerState {
    fn new(config: WorkerConfig) -> Self {
        Self { config }
    }

    async fn handle(&mut self, req: WorkerRequest) -> WorkerResponse {
        match req {
            WorkerRequest::Ping => WorkerResponse::Pong {
                rank: self.config.rank,
                world_size: self.config.world_size,
                cuda_device: self.config.cuda_device,
            },
            WorkerRequest::Init { comm_id: _ } => WorkerResponse::Error {
                kind: "not_implemented_7a_i".into(),
                message: "NCCL init lands in Stage 7a-ii (CUDA-gated)".into(),
            },
            WorkerRequest::NcclSanityCheck => WorkerResponse::Error {
                kind: "not_implemented_7a_i".into(),
                message: "NCCL sanity check lands in Stage 7a-ii (CUDA-gated)".into(),
            },
            WorkerRequest::Shutdown => WorkerResponse::Bye,
        }
    }
}
