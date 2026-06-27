//! helexa-bench CLI.
//!
//! - `run`    — continuous daemon (systemd default): sweep, sleep, repeat.
//! - `once`   — a single sweep, then exit (manual / CI).
//! - `report` — render the SQLite store as a results table.
//!
//! Runs on a single-threaded runtime: the workload is batch-1 sequential
//! (one request at a time, the regime we measure), and it lets the
//! SQLite connection live across awaits without `Sync` gymnastics.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use helexa_bench::api;
use helexa_bench::config::BenchConfig;
use helexa_bench::report;
use helexa_bench::store::Store;
use helexa_bench::sweep::Sweeper;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "helexa-bench")]
#[command(about = "Continuous version-aware benchmark harness for the neuron fleet")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run sweeps continuously, pausing `sweep_interval_secs` between them.
    Run {
        #[arg(short, long, default_value = "helexa-bench.toml")]
        config: String,
    },
    /// Run a single sweep over all targets, then exit.
    Once {
        #[arg(short, long, default_value = "helexa-bench.toml")]
        config: String,
    },
    /// Serve the read-only JSON API only (no sweeping).
    Serve {
        #[arg(short, long, default_value = "helexa-bench.toml")]
        config: String,
    },
    /// Render recorded results. Uses `--db` if given, else the db_path
    /// from `--config`.
    Report {
        #[arg(short, long, default_value = "helexa-bench.toml")]
        config: String,
        /// Override the SQLite path (skips reading the config file).
        #[arg(long)]
        db: Option<String>,
        /// Output format.
        #[arg(long, default_value = "md")]
        format: Format,
        /// Render the context-length scaling view (prefill & decode tok/s
        /// vs context per model, with decode-flatness) instead of the flat
        /// results table (#88).
        #[arg(long)]
        scaling: bool,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Format {
    Md,
    Json,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    rt.block_on(run(cli))
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Run { config } => {
            let cfg = load_config(&config)?;
            require_targets(&cfg)?;
            // Bind the read API alongside the sweep loop (one bob service
            // does both). Its own store connection; WAL keeps the sweep
            // writer and the API readers from blocking each other.
            if cfg.api.enabled {
                let state = api::open_state(&cfg.bench.db_path)?;
                let listen = cfg.api.listen.clone();
                tokio::spawn(async move {
                    if let Err(e) = api::serve(&listen, state).await {
                        tracing::error!(error = %format!("{e:#}"), "bench API server exited");
                    }
                });
            }
            let sweeper = Sweeper::new(cfg)?;
            tracing::info!("helexa-bench started; entering continuous sweep loop");
            sweeper.run_forever().await
        }
        Command::Serve { config } => {
            let cfg = load_config(&config)?;
            if !cfg.api.enabled {
                anyhow::bail!("[api] enabled = false — nothing to serve");
            }
            let state = api::open_state(&cfg.bench.db_path)?;
            tracing::info!("helexa-bench serving API only");
            api::serve(&cfg.api.listen, state).await
        }
        Command::Once { config } => {
            let cfg = load_config(&config)?;
            require_targets(&cfg)?;
            let sweeper = Sweeper::new(cfg)?;
            let summary = sweeper.run_once().await?;
            tracing::info!(
                measured = summary.measured,
                skipped = summary.skipped,
                failed = summary.failed,
                unreachable = summary.targets_unreachable,
                "single sweep complete"
            );
            Ok(())
        }
        Command::Report {
            config,
            db,
            format,
            scaling,
        } => {
            let db_path = match db {
                Some(p) => p,
                None => load_config(&config)?.bench.db_path,
            };
            let store = Store::open(&db_path)?;
            let rendered = if scaling {
                let curves = store.scaling()?;
                match format {
                    Format::Md => report::render_scaling_markdown(&curves),
                    Format::Json => report::render_scaling_json(&curves)?,
                }
            } else {
                let rows = store.report_rows()?;
                match format {
                    Format::Md => report::render_markdown(&rows),
                    Format::Json => report::render_json(&rows)?,
                }
            };
            println!("{rendered}");
            Ok(())
        }
    }
}

fn load_config(path: &str) -> Result<BenchConfig> {
    BenchConfig::load(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("loading config {path}"))
}

fn require_targets(cfg: &BenchConfig) -> Result<()> {
    if cfg.targets.is_empty() {
        anyhow::bail!("no targets configured — add at least one [[targets]] entry");
    }
    Ok(())
}
