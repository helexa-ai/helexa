use anyhow::Result;
use clap::Parser;
use cortex_neuron::{api, discovery, health};
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "cortex-neuron")]
#[command(about = "Per-node daemon for cortex inference clusters")]
#[command(version)]
struct Args {
    /// Port to listen on.
    #[arg(short, long, default_value = "9090")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,cortex_neuron=debug")),
        )
        .init();

    let args = Args::parse();
    let start_time = Instant::now();

    tracing::info!("running hardware discovery");
    let discovery_result = discovery::discover_system().await?;
    tracing::info!(
        hostname = %discovery_result.hostname,
        devices = discovery_result.devices.len(),
        "discovery complete"
    );

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
    });

    let app = api::neuron_routes().with_state(state);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", args.port).parse()?;
    tracing::info!("cortex-neuron listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
