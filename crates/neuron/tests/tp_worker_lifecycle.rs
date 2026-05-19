//! Stage 7a-i: confirm the TP worker subprocess lifecycle round-trips.
//!
//! Spawns two worker subprocesses via the leader→worker stdio RPC,
//! pings each, and cleanly shuts them down. No CUDA required —
//! `Init` and `NcclSanityCheck` are stubbed in 7a-i, so this test
//! runs on any host the workspace builds on.

use neuron::harness::tp::{WorkerPool, rpc::WorkerResponse};

/// Path to the neuron binary built by cargo for this test process.
/// cargo populates `CARGO_BIN_EXE_neuron` at compile time for sibling-
/// binary tests; production paths in main.rs use `/proc/self/exe`.
const NEURON_BIN: &str = env!("CARGO_BIN_EXE_neuron");

/// Two workers (so we spawn one subprocess: rank 0 is in-process,
/// rank 1 is the child). Verify the spawned worker responds to Ping
/// with its own identity, then shut it down cleanly.
#[tokio::test]
async fn test_spawn_ping_shutdown() {
    // cuda_devices: rank 0 → device 0 (leader, unused here),
    //               rank 1 → device 1 (worker; not actually opened in 7a-i).
    let mut pool = WorkerPool::spawn(NEURON_BIN.as_ref(), 2, &[0, 1])
        .await
        .expect("spawn worker pool");

    let pongs = pool.ping_all().await.expect("ping all workers");
    assert_eq!(pongs.len(), 1, "expected one Pong (rank 1 only)");
    match &pongs[0] {
        WorkerResponse::Pong {
            rank,
            world_size,
            cuda_device,
        } => {
            assert_eq!(*rank, 1);
            assert_eq!(*world_size, 2);
            assert_eq!(*cuda_device, 1);
        }
        other => panic!("expected Pong, got {other:?}"),
    }

    pool.shutdown().await.expect("clean shutdown");
}

/// Three workers — exercise the loop in `ping_all` / `shutdown`.
#[tokio::test]
async fn test_spawn_three_workers() {
    let mut pool = WorkerPool::spawn(NEURON_BIN.as_ref(), 3, &[0, 1, 2])
        .await
        .expect("spawn worker pool");

    let pongs = pool.ping_all().await.expect("ping all workers");
    assert_eq!(pongs.len(), 2, "expected two Pongs (ranks 1 and 2)");
    for (i, resp) in pongs.iter().enumerate() {
        match resp {
            WorkerResponse::Pong {
                rank,
                world_size,
                cuda_device,
            } => {
                let expected_rank = (i + 1) as u32;
                assert_eq!(*rank, expected_rank);
                assert_eq!(*world_size, 3);
                assert_eq!(*cuda_device, expected_rank);
            }
            other => panic!("expected Pong, got {other:?}"),
        }
    }

    pool.shutdown().await.expect("clean shutdown");
}

/// 7a-i's Init/NcclSanityCheck handlers return an error rather than
/// silently no-op, so the leader can tell the difference between
/// "haven't implemented yet" and "succeeded vacuously". Confirm the
/// shape so 7a-ii's replacement is a drop-in (same wire op names).
#[tokio::test]
async fn test_init_returns_not_implemented_in_7a_i() {
    use neuron::harness::tp::rpc::WorkerRequest;
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

    // Spawn a single worker by hand to send Init directly (the pool's
    // public API doesn't expose Init yet — that lands in 7a-ii).
    let mut child = Command::new(NEURON_BIN)
        .arg("--worker")
        .arg("--rank")
        .arg("1")
        .arg("--tp-size")
        .arg("2")
        .arg("--cuda-device")
        .arg("1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn worker");

    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut lines = BufReader::new(stdout).lines();

    let req = WorkerRequest::Init {
        comm_id: "ff".repeat(128),
    };
    let mut payload = serde_json::to_string(&req).unwrap();
    payload.push('\n');
    stdin.write_all(payload.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();

    let reply = lines
        .next_line()
        .await
        .expect("read line")
        .expect("got line");
    let resp: WorkerResponse = serde_json::from_str(&reply).expect("parse reply");
    match resp {
        WorkerResponse::Error { kind, .. } => {
            assert_eq!(kind, "not_implemented_7a_i");
        }
        other => panic!("expected Error{{kind=not_implemented_7a_i}}, got {other:?}"),
    }

    // Clean shutdown.
    stdin.write_all(b"{\"op\":\"shutdown\"}\n").await.unwrap();
    stdin.flush().await.unwrap();
    let _ = lines.next_line().await; // Bye
    let _ = child.wait().await;
}
