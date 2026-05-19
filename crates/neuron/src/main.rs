use anyhow::{Context, Result};
use clap::Parser;
use neuron::{
    api,
    config::NeuronConfig,
    discovery,
    harness::{HarnessRegistry, tp},
    health, startup,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

/// Top-level CLI. The same binary runs as either the public neuron
/// daemon (default), a tensor-parallel worker subprocess (when
/// `--worker` is set, spawned by the leader on the same host), or a
/// one-shot TP NCCL handshake check (when `--tp-smoke` is set).
#[derive(Parser)]
#[command(name = "neuron")]
#[command(about = "Per-node daemon for cortex inference clusters")]
#[command(version)]
struct Args {
    /// Run in tensor-parallel worker mode. The leader process spawns
    /// one of these per non-zero NCCL rank and drives it over
    /// newline-delimited JSON on stdin/stdout. Worker mode skips
    /// discovery, the HTTP listener, and the health poller — it's a
    /// pure RPC loop.
    #[arg(long, default_value_t = false)]
    worker: bool,

    /// Run a one-shot TP smoke test: spawn `--tp-size - 1` worker
    /// subprocesses on `--cuda-devices`, build the NCCL communicator,
    /// run an `AllReduce` sanity check across every rank, and exit.
    /// Used to validate the TP plumbing in isolation from model load
    /// and inference. Diagnostic-only — not exposed through the daemon
    /// HTTP API.
    #[arg(long, default_value_t = false)]
    tp_smoke: bool,

    /// NCCL rank for worker mode. Ignored when `--worker` is not set.
    #[arg(long, default_value_t = 0)]
    rank: u32,

    /// Total NCCL world size for worker mode or TP smoke mode.
    #[arg(long, default_value_t = 1)]
    tp_size: u32,

    /// CUDA device index for worker mode. Ignored when `--worker` is
    /// not set.
    #[arg(long, default_value_t = 0)]
    cuda_device: u32,

    /// Comma-separated CUDA device indices for TP smoke mode (one per
    /// rank, starting with rank 0). Must have `tp_size` entries.
    #[arg(long, value_delimiter = ',')]
    cuda_devices: Vec<u32>,

    /// Port to listen on (overrides config file). Daemon mode only.
    #[arg(short, long)]
    port: Option<u16>,

    /// Path to the neuron config file. Daemon mode only.
    #[arg(short, long, default_value = "neuron.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,neuron=debug")),
        )
        .init();

    let args = Args::parse();

    if args.worker {
        return tp::worker::run(tp::worker::WorkerConfig {
            rank: args.rank,
            world_size: args.tp_size,
            cuda_device: args.cuda_device,
        })
        .await;
    }

    if args.tp_smoke {
        return tp_smoke(args.tp_size, args.cuda_devices).await;
    }

    daemon(args).await
}

/// One-shot tensor-parallel handshake. Spawns N-1 worker subprocesses
/// (rank 0 stays in this process), builds the NCCL communicator across
/// the full world, runs an AllReduce sanity check, and shuts everyone
/// down. Output is plain log lines on stderr + a final summary on
/// stdout in `key=value` form so an outer script can parse it.
async fn tp_smoke(tp_size: u32, cuda_devices: Vec<u32>) -> Result<()> {
    if tp_size < 2 {
        anyhow::bail!("--tp-size must be at least 2 (got {tp_size})");
    }
    if cuda_devices.len() as u32 != tp_size {
        anyhow::bail!(
            "--cuda-devices must list exactly {tp_size} entries (got {})",
            cuda_devices.len()
        );
    }

    let exe = std::env::current_exe().context("resolve current_exe for worker spawn")?;
    let leader_device = cuda_devices[0];

    tracing::info!(
        tp_size,
        ?cuda_devices,
        binary = %exe.display(),
        "tp-smoke: spawning worker pool"
    );
    let mut pool = tp::WorkerPool::spawn(&exe, tp_size, &cuda_devices).await?;

    tracing::info!("tp-smoke: pinging every worker");
    let pongs = pool.ping_all().await?;
    for p in &pongs {
        tracing::info!(?p, "tp-smoke: pong");
    }

    tracing::info!(leader_device, "tp-smoke: initialising NCCL");
    pool.init_nccl(leader_device).await?;

    tracing::info!("tp-smoke: running AllReduce sanity check");
    pool.nccl_sanity_check().await?;

    tracing::info!("tp-smoke: shutting down pool");
    pool.shutdown().await?;

    println!("status=ok");
    println!("tp_size={tp_size}");
    println!(
        "cuda_devices={}",
        cuda_devices
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}

async fn daemon(args: Args) -> Result<()> {
    let cfg = NeuronConfig::load(&args.config).unwrap_or_else(|e| {
        tracing::warn!(path = %args.config, error = %e, "config not found, using defaults");
        NeuronConfig::default()
    });

    let port = args.port.unwrap_or(cfg.port);
    let bind_url = format!("http://localhost:{port}");
    let start_time = Instant::now();

    tracing::info!("running hardware discovery");
    let mut discovery_result = discovery::discover_system().await?;
    tracing::info!(
        hostname = %discovery_result.hostname,
        devices = discovery_result.devices.len(),
        "discovery complete"
    );

    // Build harness registry from config. In-process harnesses (candle)
    // need to know neuron's own bind URL so they can return it from
    // inference_endpoint.
    let registry = HarnessRegistry::from_configs(&cfg.harnesses, &bind_url, &cfg.harness);
    discovery_result.harnesses = registry.names();
    let candle = registry.candle();

    // Activation: load default models before binding the listener.
    // Each load may take tens of seconds to several minutes depending
    // on model size and HF cache state — keep TimeoutStartSec in the
    // systemd unit generous enough to cover the slowest entry.
    startup::load_default_models(&registry, &cfg.default_models).await;

    let health_cache = Arc::new(health::HealthCache::new());
    health_cache
        .set_has_gpus(!discovery_result.devices.is_empty())
        .await;

    let poller_cache = Arc::clone(&health_cache);
    tokio::spawn(async move {
        poller_cache.poll_loop(start_time).await;
    });

    let state = Arc::new(api::NeuronState {
        discovery: discovery_result,
        health_cache,
        registry: RwLock::new(registry),
        candle,
    });

    let app = api::neuron_routes().with_state(Arc::clone(&state));
    let addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse()?;
    tracing::info!("neuron listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(startup::shutdown_signal())
        .await?;

    // Deactivation: serve has returned (graceful shutdown signal
    // received and connections drained). Release CUDA contexts / VRAM
    // by unloading every model before exiting; systemd's TimeoutStopSec
    // bounds how long this phase may take.
    let registry = state.registry.read().await;
    startup::unload_all_models(&registry).await;
    tracing::info!("shutdown complete");

    Ok(())
}
