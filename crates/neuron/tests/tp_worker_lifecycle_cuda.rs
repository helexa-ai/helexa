//! Stage 7a-ii: real NCCL handshake across the worker pool.
//!
//! Gated behind the `cuda-integration` feature because it requires
//! libnccl AND multiple CUDA devices on the running host. Run on
//! beast (2× RTX 5090) via:
//!
//!   cargo test -p neuron --features cuda-integration \
//!         --test tp_worker_lifecycle_cuda
//!
//! Steps: spawn N-1 workers, call `init_nccl`, run `nccl_sanity_check`
//! (every rank `all_reduce`s `1u32` with Sum; expected total =
//! world_size), shut down cleanly.

#![cfg(feature = "cuda-integration")]

use neuron::harness::tp::WorkerPool;

const NEURON_BIN: &str = env!("CARGO_BIN_EXE_neuron");

#[tokio::test]
async fn test_init_and_sanity_check_two_ranks() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("info,neuron=debug")
        .try_init();

    // 2 ranks: leader = rank 0 on device 0, worker = rank 1 on device 1.
    let leader_worker = neuron::harness::device_worker::DeviceWorkerHandle::spawn(0)
        .expect("spawn leader device worker");
    let mut pool = WorkerPool::spawn(NEURON_BIN.as_ref(), 2, &[0, 1], leader_worker)
        .await
        .expect("spawn worker pool");

    pool.ping_all().await.expect("pong all workers");

    pool.init_nccl(0)
        .await
        .expect("init_nccl: NCCL handshake across all ranks");

    pool.nccl_sanity_check()
        .await
        .expect("nccl_sanity_check: observed_sum == world_size on all ranks");

    pool.shutdown().await.expect("clean shutdown");
}
