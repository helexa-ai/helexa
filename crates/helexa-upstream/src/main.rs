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
    /// Mint single-use top-up codes and print them (one per line). The raw
    /// codes are shown only here — only their hash is stored. (The future
    /// faucet bot calls the same path.)
    Mint {
        #[arg(short, long, default_value = "helexa-upstream.toml")]
        config: String,
        /// Tokens each code grants.
        #[arg(long)]
        value: i64,
        /// How many codes to mint.
        #[arg(long, default_value_t = 1)]
        count: u32,
        /// Optional human label (e.g. "small", "beta-launch").
        #[arg(long)]
        denomination: Option<String>,
    },
    /// Roll up not-yet-reconciled served usage per operator/period (#58),
    /// stamp it reconciled, and print the totals. Payout is out of scope.
    Reconcile {
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
        Commands::Mint {
            config,
            value,
            count,
            denomination,
        } => {
            let cfg = UpstreamConfig::load(&config)
                .map_err(|e| anyhow::anyhow!("failed to load config from '{config}': {e}"))?;
            let pool =
                helexa_upstream::db::connect_and_migrate(&cfg.db.url, cfg.db.max_connections)
                    .await?;
            let codes =
                helexa_upstream::topup::mint(&pool, value, count, denomination.as_deref()).await?;
            // Raw codes to stdout (one per line) for the operator to distribute;
            // logs/diagnostics go to stderr via tracing.
            for code in codes {
                println!("{code}");
            }
        }
        Commands::Reconcile { config } => {
            let cfg = UpstreamConfig::load(&config)
                .map_err(|e| anyhow::anyhow!("failed to load config from '{config}': {e}"))?;
            let pool =
                helexa_upstream::db::connect_and_migrate(&cfg.db.url, cfg.db.max_connections)
                    .await?;
            let rollup = helexa_upstream::reconcile::reconcile(&pool).await?;
            for r in &rollup {
                println!("{}\t{}\t{}", r.operator_id, r.period, r.total_served_tokens);
            }
            tracing::info!(operators_periods = rollup.len(), "reconciliation complete");
        }
    }

    Ok(())
}
