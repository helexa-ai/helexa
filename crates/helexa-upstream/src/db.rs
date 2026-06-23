//! PostgreSQL pool + embedded migrations.

use anyhow::{Context, Result};
use sqlx::postgres::{PgPool, PgPoolOptions};

/// Connect to Postgres and run embedded migrations (`./migrations`).
pub async fn connect_and_migrate(url: &str, max_connections: u32) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(url)
        .await
        .with_context(|| "connecting to PostgreSQL")?;

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .with_context(|| "running migrations")?;

    Ok(pool)
}
