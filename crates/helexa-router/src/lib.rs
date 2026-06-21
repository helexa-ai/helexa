//! helexa-router — public multi-operator ingress proxy (router.helexa.ai).
//!
//! The router is the data-plane *ingress* tier: a geo-distributed,
//! capacity-aware, OpenAI/Anthropic-compatible reverse proxy in front of
//! many operator-run cortexes ("cortex-of-cortexes"). End users configure
//! one `baseURL` and the router forwards their request to a cortex with
//! capacity, proxying #63-shaped rejections back verbatim.
//!
//! It holds **zero entitlement logic** — auth/budget stays at cortex
//! (epic #47); the router forwards the client bearer unchanged and routes
//! on capacity (epic #69). This crate is the binary skeleton (#70):
//! a plaintext axum server reusing `cortex-core` types, serving `/health`
//! and a stub `/v1/models`.

pub mod config;
pub mod handlers;
pub mod state;

use anyhow::Result;
use config::RouterConfig;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Build the axum application: handlers + CORS + tracing. No auth layer —
/// the router asserts no identity of its own and forwards the client bearer
/// to the downstream cortex, which authenticates it (#69).
pub fn build_app(state: Arc<state::RouterState>) -> axum::Router {
    axum::Router::new()
        .merge(handlers::api_routes())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the router: build state from config and bind the plaintext HTTP
/// listener. TLS is terminated by edge nginx ahead of this process.
pub async fn run(config: RouterConfig) -> Result<()> {
    let state = Arc::new(state::RouterState::from_config(&config));
    let app = build_app(state);

    let listen_addr = config.router.listen.parse::<std::net::SocketAddr>()?;
    tracing::info!("helexa-router listening on {listen_addr}");

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
