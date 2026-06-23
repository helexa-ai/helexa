//! helexa-upstream — the mesh-level account/authorization authority (#59).
//!
//! The clearing house above cortex: it issues accounts and API keys, holds
//! the real token-allocation ledger, authorizes inference in real time
//! (reserve → settle, fail-closed), and tracks served usage for operator
//! reconciliation. cortex's `UpstreamEntitlementProvider` (#57) is a client
//! of the `/authz/v1` surface; the helexa.ai frontend is a client of the
//! `/web/v1` surface.
//!
//! B1 (this milestone) lands the crate skeleton, the full Postgres schema
//! (`migrations/`), the reserve→settle ledger ([`ledger`]) with its
//! no-overshoot guarantee, and `/health`.

pub mod config;
pub mod db;
pub mod handlers;
pub mod ledger;
pub mod state;

use anyhow::Result;
use config::UpstreamConfig;
use state::AppState;
use tower_http::trace::TraceLayer;

/// Build the axum application.
pub fn build_app(state: AppState) -> axum::Router {
    axum::Router::new()
        .merge(handlers::routes())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the service: connect Postgres, run migrations, bind the listener.
pub async fn run(config: UpstreamConfig) -> Result<()> {
    let pool = db::connect_and_migrate(&config.db.url, config.db.max_connections).await?;
    let listen = config.server.listen.clone();
    let state = AppState::new(pool, config);

    let addr = listen.parse::<std::net::SocketAddr>()?;
    tracing::info!("helexa-upstream listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}
