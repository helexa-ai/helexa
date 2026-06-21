use anyhow::Result;
use clap::{Parser, Subcommand};
use helexa_router::config::RouterConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "helexa-router")]
#[command(about = "Public multi-operator ingress proxy for helexa")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the router server.
    Serve {
        /// Path to the router config file.
        #[arg(short, long, default_value = "helexa-router.toml")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,helexa_router=debug")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => {
            let cfg = RouterConfig::load(&config)
                .map_err(|e| anyhow::anyhow!("failed to load config from '{config}': {e}"))?;

            tracing::info!(
                cortexes = cfg.cortexes.len(),
                listen = %cfg.router.listen,
                "starting helexa-router"
            );

            helexa_router::run(cfg).await?;
        }
    }

    Ok(())
}
