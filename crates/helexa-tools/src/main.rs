use anyhow::Result;
use clap::{Parser, Subcommand};
use helexa_tools::config::ToolsConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "helexa-tools")]
#[command(about = "Tool-execution service for helexa grounding (fetch/readability)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the tools server.
    Serve {
        /// Path to the config file.
        #[arg(short, long, default_value = "helexa-tools.toml")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,helexa_tools=debug")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Serve { config } => {
            // Config file is optional: the defaults are complete, and
            // the RPM ships one only for operator overrides.
            let cfg = ToolsConfig::load(&config).unwrap_or_else(|e| {
                tracing::warn!(config, error = %e, "config not loaded; using defaults");
                ToolsConfig::default()
            });
            let listen = cfg.listen.clone();
            let listener = tokio::net::TcpListener::bind(&listen).await?;
            tracing::info!(listen, "helexa-tools serving");
            axum::serve(listener, helexa_tools::app(cfg)).await?;
        }
    }
    Ok(())
}
