use anyhow::Result;
use clap::{Parser, Subcommand};
use helexa_upstream::config::UpstreamConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "helexa-upstream")]
#[command(about = "Mesh-level account & authorization authority for helexa")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the upstream server.
    Serve {
        /// Path to the config file.
        #[arg(short, long, default_value = "helexa-upstream.toml")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,helexa_upstream=debug")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => {
            let cfg = UpstreamConfig::load(&config)
                .map_err(|e| anyhow::anyhow!("failed to load config from '{config}': {e}"))?;
            tracing::info!(listen = %cfg.server.listen, "starting helexa-upstream");
            helexa_upstream::run(cfg).await?;
        }
    }

    Ok(())
}
