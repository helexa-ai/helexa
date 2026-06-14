//! Read-only JSON API over the bench SQLite store.
//!
//! Consumed by the `bench/` visualisation app and for programmatic
//! access. Served by the `run` daemon (alongside the sweep loop) and by
//! the standalone `serve` subcommand. CORS is permissive because the UI
//! is hosted separately (different origin); the API is internal-only
//! (WireGuard + firewalld) and read-only, so this predates the auth epic.

use crate::store::{RunFilter, Store};
use anyhow::Result;
use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::get;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

/// Shared API state: a dedicated read connection to the store, guarded
/// (rusqlite `Connection` isn't `Sync`). Separate from the sweep's
/// writer connection — WAL lets them run concurrently.
pub type ApiState = Arc<Mutex<Store>>;

/// Open an API state over the store at `db_path`.
pub fn open_state(db_path: &str) -> Result<ApiState> {
    Ok(Arc::new(Mutex::new(Store::open(db_path)?)))
}

/// Build the API router.
pub fn api_routes(state: ApiState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/dimensions", get(dimensions))
        .route("/api/summary", get(summary))
        .route("/api/series", get(series))
        .route("/api/runs", get(runs))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Bind `listen` and serve the API until the process exits.
pub async fn serve(listen: &str, state: ApiState) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(%listen, "bench API listening");
    axum::serve(listener, api_routes(state)).await?;
    Ok(())
}

type ApiError = (StatusCode, String);

fn err500(e: anyhow::Error) -> ApiError {
    (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
}

async fn health(State(s): State<ApiState>) -> Result<Json<serde_json::Value>, ApiError> {
    let store = s.lock().await;
    let count = store.run_count().map_err(err500)?;
    Ok(Json(json!({ "status": "ok", "run_count": count })))
}

async fn dimensions(State(s): State<ApiState>) -> Result<Json<crate::store::Dimensions>, ApiError> {
    let store = s.lock().await;
    store.dimensions().map(Json).map_err(err500)
}

async fn summary(
    State(s): State<ApiState>,
) -> Result<Json<Vec<crate::store::ReportRow>>, ApiError> {
    let store = s.lock().await;
    store.summary().map(Json).map_err(err500)
}

#[derive(Debug, Deserialize)]
struct SeriesQuery {
    /// Optional — when omitted the store resolves the host serving this model.
    host: Option<String>,
    model: String,
    scenario: String,
}

async fn series(
    State(s): State<ApiState>,
    Query(q): Query<SeriesQuery>,
) -> Result<Json<Vec<crate::store::SeriesPoint>>, ApiError> {
    let store = s.lock().await;
    store
        .series(q.host.as_deref(), &q.model, &q.scenario)
        .map(Json)
        .map_err(err500)
}

#[derive(Debug, Deserialize)]
struct RunsQuery {
    host: Option<String>,
    model: Option<String>,
    scenario: Option<String>,
    sha: Option<String>,
    ok: Option<bool>,
    limit: Option<u32>,
}

async fn runs(
    State(s): State<ApiState>,
    Query(q): Query<RunsQuery>,
) -> Result<Json<Vec<crate::store::RunRow>>, ApiError> {
    let filter = RunFilter {
        host: q.host,
        model: q.model,
        scenario: q.scenario,
        sha: q.sha,
        ok: q.ok,
        limit: q.limit,
    };
    let store = s.lock().await;
    store.runs(&filter).map(Json).map_err(err500)
}
