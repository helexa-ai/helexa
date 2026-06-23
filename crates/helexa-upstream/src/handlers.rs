//! HTTP handlers. B1 ships `/health`; the authz (`/authz/v1`) and web
//! (`/web/v1`) surfaces land in later phases.

use crate::state::AppState;
use axum::{Json, Router, extract::State, routing::get};
use serde_json::{Value, json};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/", get(health))
}

/// `GET /health` — liveness + a database round-trip (`SELECT 1`).
async fn health(State(state): State<AppState>) -> Json<Value> {
    let db_ok = sqlx::query("SELECT 1").execute(&state.pool).await.is_ok();
    Json(json!({
        "status": if db_ok { "ok" } else { "degraded" },
        "db": if db_ok { "ok" } else { "unreachable" },
    }))
}
