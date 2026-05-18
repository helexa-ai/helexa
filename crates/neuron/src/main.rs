use anyhow::Result;
use clap::Parser;
use neuron::{api, config::NeuronConfig, discovery, harness::HarnessRegistry, health};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "neuron")]
#[command(about = "Per-node daemon for cortex inference clusters")]
#[command(version)]
struct Args {
    /// Port to listen on (overrides config file).
    #[arg(short, long)]
    port: Option<u16>,

    /// Path to the neuron config file.
    #[arg(short, long, default_value = "neuron.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,neuron=debug")),
        )
        .init();

    let args = Args::parse();

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

    let app = api::neuron_routes().with_state(state);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse()?;
    tracing::info!("neuron listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
