//! helexa-upstream — the mesh-level account/authorization authority (#59).
//!
//! The clearing house above cortex: it issues accounts and API keys, holds
//! the real token-allocation ledger, authorizes inference in real time
//! (reserve → settle, fail-closed), and tracks served usage for operator
//! reconciliation. cortex's `UpstreamEntitlementProvider` (#57) is a client
//! of the `/authz/v1` surface; the helexa.ai frontend is a client of the
//! `/web/v1` surface.
//!
//! Landed so far: B1 — schema + reserve→settle [`ledger`] (no-overshoot) +
//! `/health`. B2 — the `/authz/v1` [`authz`] surface (resolve/reserve/
//! settle/release/snapshot) with shared-bearer client auth and a
//! stale-reservation sweeper.

pub mod authz;
pub mod config;
pub mod crypto;
pub mod db;
pub mod error;
pub mod handlers;
pub mod ledger;
pub mod state;

use anyhow::Result;
use config::UpstreamConfig;
use state::AppState;
use std::time::Duration;
use tower_http::trace::TraceLayer;

/// Build the axum application.
pub fn build_app(state: AppState) -> axum::Router {
    axum::Router::new()
        .merge(handlers::routes())
        .merge(authz::router(&state))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the service: connect Postgres, run migrations, spawn the
/// reservation sweeper, bind the listener.
pub async fn run(config: UpstreamConfig) -> Result<()> {
    let pool = db::connect_and_migrate(&config.db.url, config.db.max_connections).await?;
    let listen = config.server.listen.clone();
    let state = AppState::new(pool, config);

    if state.config.client_auth.tokens.is_empty() {
        tracing::warn!(
            "no [client_auth] tokens configured — the /authz/v1 surface is OPEN (dev only)"
        );
    }

    // Stale-reservation sweeper: releases open reservations whose
    // settle/release from cortex was lost, self-healing allocation_reserved.
    spawn_sweeper(&state);

    let addr = listen.parse::<std::net::SocketAddr>()?;
    tracing::info!("helexa-upstream listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}

fn spawn_sweeper(state: &AppState) {
    let pool = state.pool.clone();
    let ttl = state.config.authz.reservation_ttl_secs as i64;
    let interval = Duration::from_secs(state.config.authz.sweep_interval_secs);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            match ledger::sweep_stale(&pool, ttl).await {
                Ok(n) if n > 0 => tracing::info!(swept = n, "released stale reservations"),
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "reservation sweep failed"),
            }
        }
    });
}
