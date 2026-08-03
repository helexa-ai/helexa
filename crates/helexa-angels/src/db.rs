//! PostgreSQL pool + embedded migrations, in a dedicated schema.
//!
//! angels shares helexa-upstream's database so that credential auth is
//! genuinely shared (D2) — a cross-database join against `users` is not
//! possible, so "same database" is a requirement, not a convenience.
//!
//! Sharing it safely needs one precaution. Both services call
//! `sqlx::migrate!`, and sqlx records applied migrations in an
//! **unqualified** `_sqlx_migrations` table. Two migrators in one schema
//! would therefore write to the same bookkeeping table, each seeing the
//! other's versions as unknown and its checksums as corrupt. Giving angels
//! its own schema gives it its own `_sqlx_migrations`.
//!
//! Every pooled connection runs with `search_path = angels, public`, so
//! angels' own unqualified names resolve to its schema while
//! `public.users` stays reachable for the credential join.

use anyhow::{Context, Result};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{Executor, PgConnection};

/// The schema angels owns. Nothing outside it is ever written by this
/// service — upstream-owned tables are read-only from here.
pub const SCHEMA: &str = "angels";

/// Connect, ensure the schema exists, pin `search_path`, and migrate.
pub async fn connect_and_migrate(url: &str, max_connections: u32) -> Result<PgPool> {
    // Bootstrap on a throwaway connection: the schema must exist before a
    // pooled connection can set `search_path` to it.
    {
        use sqlx::Connection;
        let mut conn = PgConnection::connect(url)
            .await
            .with_context(|| "connecting to PostgreSQL (schema bootstrap)")?;
        conn.execute(format!("CREATE SCHEMA IF NOT EXISTS {SCHEMA}").as_str())
            .await
            .with_context(|| format!("creating schema {SCHEMA}"))?;
        let _ = conn.close().await;
    }

    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .after_connect(|conn, _meta| {
            Box::pin(async move {
                conn.execute(format!("SET search_path = {SCHEMA}, public").as_str())
                    .await?;
                Ok(())
            })
        })
        .connect(url)
        .await
        .with_context(|| "connecting to PostgreSQL")?;

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .with_context(|| "running angels migrations")?;

    Ok(pool)
}
