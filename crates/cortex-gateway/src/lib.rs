pub mod evictor;
pub mod handlers;
pub mod metrics;
pub mod poller;
pub mod proxy;
pub mod router;
pub mod state;

use anyhow::Result;
use axum::Router;
use cortex_core::config::GatewayConfig;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Build the Axum application router with all routes wired up.
pub fn build_app(fleet: Arc<state::CortexState>) -> Router {
    Router::new()
        .merge(handlers::api_routes())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(fleet)
}

/// Start the gateway: build state from config, spawn background tasks,
/// bind the HTTP server.
pub async fn run(config: GatewayConfig) -> Result<()> {
    let fleet = Arc::new(state::CortexState::from_config(&config));

    // Spawn the background poller that refreshes node/model status.
    let poller_fleet = Arc::clone(&fleet);
    tokio::spawn(async move {
        poller::poll_loop(poller_fleet).await;
    });

    // Spawn the evictor (reacts to VRAM pressure events from the router).
    let evictor_fleet = Arc::clone(&fleet);
    tokio::spawn(async move {
        evictor::eviction_loop(evictor_fleet).await;
    });

    let app = build_app(Arc::clone(&fleet));

    let listen_addr = config.gateway.listen.parse::<std::net::SocketAddr>()?;
    tracing::info!("cortex listening on {listen_addr}");

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
