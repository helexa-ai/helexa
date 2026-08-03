//! helexa-angels — the investor portal served at `angels.helexa.ai`.
//!
//! # Why this is a separate service
//!
//! `helexa.ai` is a static SPA: `vite build` → `dist` → nginx. Every
//! string in it, including all 42 locale bundles, is compiled into
//! `/assets/index-*.js` and served to anyone who asks. A React route guard
//! therefore gates *navigation*, not *access* — confidential material
//! placed in that bundle is one `curl` away from disclosure. That is a
//! property of static hosting, not a routing bug to be fixed with a better
//! guard.
//!
//! This service inverts it. Pages are assembled on the server, so the only
//! thing an unauthenticated request can obtain is a sign-in form. Three
//! further properties follow from that choice rather than being bolted on:
//!
//! - **The audit is honest.** One document render is one server request,
//!   so [`audit`] records reading rather than fetching. An SPA against a
//!   JSON API could pull every document once and re-read them offline for
//!   a week while the log showed a single view.
//! - **Watermarking is possible.** The viewer's identity is in scope at
//!   render time, so every page carries it.
//! - **No second asset pipeline**, and no i18n machinery — the portal is
//!   English-only by decision, and all its wording is operator-reviewed.
//!
//! # What is shared with helexa.ai, and what is not
//!
//! Credentials are shared: one helexa account signs in on both properties,
//! so an investor can evaluate the platform they are being asked to back.
//! Sessions are **not** — see [`auth`] for why that separation is
//! load-bearing rather than fussy.

pub mod audit;
pub mod auth;
pub mod config;
pub mod crypto;
pub mod db;
pub mod error;
pub mod grants;
pub mod invites;
pub mod state;
pub mod templates;
pub mod upstream;
pub mod web;

use anyhow::Result;
use config::AngelsConfig;
use state::AppState;
use tower_http::trace::TraceLayer;

/// The entity contractually responsible for the current round.
///
/// Bears Lairs EOOD is helexa's operator zero; Helexa AI (a Bulgarian VCC)
/// is not yet registered. Named in the footer of every page and in the
/// privacy note, because a confidential document should always say who is
/// holding the material.
pub const ENTITY_NAME: &str = "Bears Lairs EOOD";

/// How long access records are kept. Stated in the privacy note, enforced
/// by the retention sweep — the two must not drift apart.
pub const RETENTION_MONTHS: i64 = 24;

/// Build the axum application.
pub fn build_app(state: AppState) -> axum::Router {
    axum::Router::new()
        .merge(web::router())
        .route("/i/{code}", axum::routing::get(invites::enter))
        .fallback(web::fallback)
        // No CORS layer, deliberately: nothing here is meant to be
        // fetched by another origin's JavaScript. The absence is the
        // policy.
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the service.
pub async fn run(config: AngelsConfig) -> Result<()> {
    let pool = db::connect_and_migrate(&config.db.url, config.db.max_connections).await?;
    let listen = config.server.listen.clone();
    let state = AppState::new(pool, config);

    spawn_session_reaper(&state);
    spawn_retention_sweep(&state);

    let addr = listen.parse::<std::net::SocketAddr>()?;
    tracing::info!("helexa-angels listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}

/// Delete dead sessions. Expiry is enforced on every lookup, so this is
/// hygiene rather than a control — it stops the table growing without
/// bound.
fn spawn_session_reaper(state: &AppState) {
    let pool = state.pool.clone();
    let idle = state.config.session.idle_secs as f64;
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            let res = sqlx::query(
                "DELETE FROM sessions \
                 WHERE expires_at < now() \
                    OR last_seen_at < now() - make_interval(secs => $1)",
            )
            .bind(idle)
            .execute(&pool)
            .await;
            match res {
                Ok(r) if r.rows_affected() > 0 => {
                    tracing::debug!(n = r.rows_affected(), "reaped dead sessions")
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "session reap failed"),
            }
        }
    });
}

/// Enforce the retention window on the access log. These rows identify
/// people; keeping them indefinitely is neither necessary nor defensible.
fn spawn_retention_sweep(state: &AppState) {
    let pool = state.pool.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(24 * 3600)).await;
            match audit::prune(&pool, RETENTION_MONTHS).await {
                Ok(n) if n > 0 => tracing::info!(pruned = n, "access records past retention"),
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "retention sweep failed"),
            }
        }
    });
}
