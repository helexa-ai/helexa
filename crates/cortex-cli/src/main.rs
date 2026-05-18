use anyhow::Result;
use clap::{Parser, Subcommand};
use cortex_core::config::GatewayConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "cortex")]
#[command(about = "Unified inference gateway for multi-node GPU clusters")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the gateway server.
    Serve {
        /// Path to the gateway config file.
        #[arg(short, long, default_value = "cortex.toml")]
        config: String,
    },
    /// Print the fleet status (models, nodes, health).
    Status {
        /// Gateway API endpoint to query.
        #[arg(short, long, default_value = "http://localhost:31313")]
        endpoint: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing with env filter (e.g. RUST_LOG=cortex_gateway=debug).
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,cortex_gateway=debug")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => {
            let cfg = GatewayConfig::load(&config)
                .map_err(|e| anyhow::anyhow!("failed to load config from '{config}': {e}"))?;

            tracing::info!(
                neurons = cfg.neurons.len(),
                listen = %cfg.gateway.listen,
                "starting cortex"
            );

            // Install Prometheus metrics exporter on a separate port.
            cortex_gateway::metrics::install(&cfg.gateway.metrics_listen)?;

            cortex_gateway::run(cfg).await?;
        }
        Commands::Status { endpoint } => {
            print_status(&endpoint).await?;
        }
    }

    Ok(())
}

async fn print_status(endpoint: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // Fetch health.
    let health: serde_json::Value = client
        .get(format!("{endpoint}/health"))
        .send()
        .await?
        .json()
        .await?;

    println!("Fleet health: {}", serde_json::to_string_pretty(&health)?);

    // Fetch models.
    let models: serde_json::Value = client
        .get(format!("{endpoint}/v1/models"))
        .send()
        .await?
        .json()
        .await?;

    println!("\nModels:");
    if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
        for model in data {
            let id = model.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let locations = model
                .get("locations")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|l| {
                            let node = l.get("node")?.as_str()?;
                            let status = l.get("status")?.as_str()?;
                            Some(format!("{node}({status})"))
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            println!("  {id:40} {locations}");
        }
    }

    Ok(())
}
