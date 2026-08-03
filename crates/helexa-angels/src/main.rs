use anyhow::Result;
use clap::{Parser, Subcommand};
use helexa_angels::config::AngelsConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "helexa-angels")]
#[command(about = "Confidential investor portal for helexa (angels.helexa.ai)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the portal.
    Serve {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
    },
    /// Mint an invitation code and print it. The plaintext is shown only
    /// here — only its hash is stored, so it cannot be recovered later.
    Invite {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
        /// Round the code grants access to.
        #[arg(long)]
        round: String,
        /// Human label, so codes are tellable apart in a listing.
        #[arg(long)]
        label: String,
        /// Maximum redemptions. Omit for unlimited — reusability is the
        /// point; this is a blast-radius control for a code that travels
        /// further than intended.
        #[arg(long)]
        max_uses: Option<i32>,
        /// Expire the code after this many days.
        #[arg(long)]
        expires_days: Option<i64>,
    },
    /// List invitation codes and their state. Never prints a code.
    Invites {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
    },
    /// Who holds access to a round.
    Access {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
        #[arg(long)]
        round: String,
    },
    /// Who has read what, most recent first.
    Reads {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
        #[arg(long)]
        round: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: i64,
    },
    /// Withdraw one person's access to a round.
    Revoke {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
        #[arg(long)]
        round: String,
        /// Email address of the person to cut off.
        #[arg(long)]
        email: String,
    },
    /// Approve a pending grant (rounds running without auto-grant).
    Approve {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
        #[arg(long)]
        round: String,
        #[arg(long)]
        email: String,
    },
    /// Stop an invitation code issuing further grants. Existing grants are
    /// untouched — revoking a code and revoking a person are different acts.
    RevokeInvite {
        #[arg(short, long, default_value = "/etc/helexa-angels/helexa-angels.toml")]
        config: String,
        #[arg(long)]
        label: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,helexa_angels=debug")),
        )
        .init();

    match Cli::parse().command {
        Commands::Serve { config } => {
            helexa_angels::run(AngelsConfig::load(&config)?).await?;
        }
        Commands::Invite {
            config,
            round,
            label,
            max_uses,
            expires_days,
        } => {
            let (pool, cfg) = open(&config).await?;
            let code = helexa_angels::invites::mint(&pool, &round, &label, max_uses, expires_days)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("{}/i/{}", cfg.site.base_url.trim_end_matches('/'), code);
            eprintln!(
                "\nThis link is shown once. Only its hash is stored, so it cannot be \
                 recovered — send it now or mint another."
            );
        }
        Commands::Invites { config } => {
            let (pool, _) = open(&config).await?;
            let rows = helexa_angels::invites::list(&pool)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("{:<28} {:<18} {:<10} USES", "LABEL", "ROUND", "STATUS");
            for (label, round, status, uses) in rows {
                println!("{label:<28} {round:<18} {status:<10} {uses}");
            }
        }
        Commands::Access { config, round } => {
            let (pool, _) = open(&config).await?;
            let rows = helexa_angels::grants::holders(&pool, &round)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("{:<40} {:<10} GRANTED", "EMAIL", "STATE");
            for (email, state, granted) in rows {
                println!("{email:<40} {state:<10} {granted}");
            }
        }
        Commands::Reads {
            config,
            round,
            limit,
        } => {
            let (pool, _) = open(&config).await?;
            let rows = helexa_angels::audit::recent(&pool, round.as_deref(), limit).await?;
            println!("{:<18} {:<36} {:<18} DOCUMENT", "WHEN", "WHO", "ROUND");
            for (at, who, round, doc) in rows {
                println!("{at:<18} {who:<36} {round:<18} {doc}");
            }
        }
        Commands::Revoke {
            config,
            round,
            email,
        } => {
            let (pool, _) = open(&config).await?;
            let n = helexa_angels::grants::revoke(&pool, &email, &round)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("revoked {n} grant(s) for {email} on {round}");
        }
        Commands::Approve {
            config,
            round,
            email,
        } => {
            let (pool, _) = open(&config).await?;
            let n = helexa_angels::grants::approve(&pool, &email, &round, "cli")
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("approved {n} grant(s) for {email} on {round}");
        }
        Commands::RevokeInvite { config, label } => {
            let (pool, _) = open(&config).await?;
            let n = helexa_angels::invites::revoke(&pool, &label)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("revoked {n} invite code(s) labelled {label}");
        }
    }
    Ok(())
}

/// Shared setup for the CLI subcommands.
async fn open(path: &str) -> Result<(sqlx::postgres::PgPool, AngelsConfig)> {
    let cfg = AngelsConfig::load(path)?;
    let pool = helexa_angels::db::connect_and_migrate(&cfg.db.url, cfg.db.max_connections).await?;
    // So `invite --round <slug>` works the moment a manifest exists on
    // disk, without needing the service restarted first.
    let content = helexa_angels::content::Content::new(cfg.content.dir.clone());
    if let Err(e) = helexa_angels::content::sync_rounds(&pool, &content).await {
        eprintln!("warning: content sync failed ({e}); round list may be stale");
    }
    Ok((pool, cfg))
}
