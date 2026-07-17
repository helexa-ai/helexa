//! helexa-tools — the tool-execution service backing the chat app's
//! grounding tools (#177). Today: `GET /fetch?url=…`, an SSRF-guarded
//! page fetch + readability extraction so the model can read a search
//! result instead of guessing from its snippet. Exposed to the world
//! only via the edge proxies' rate-limited `/tools/*` locations.

pub mod config;
pub mod fetch;
pub mod ssrf;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Deserialize;
use serde_json::json;

use config::ToolsConfig;

pub fn app(cfg: ToolsConfig) -> Router {
    let cfg = Arc::new(cfg);
    Router::new()
        .route("/health", get(health))
        .route("/fetch", get(fetch_handler))
        .with_state(cfg)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok", "service": "helexa-tools" }))
}

#[derive(Deserialize)]
struct FetchParams {
    url: String,
}

async fn fetch_handler(
    State(cfg): State<Arc<ToolsConfig>>,
    Query(params): Query<FetchParams>,
) -> Response {
    match fetch::fetch_page(&params.url, &cfg).await {
        Ok(page) => {
            tracing::info!(url = %page.url, title = %page.title, chars = page.text.len(), truncated = page.truncated, "fetch: ok");
            Json(page).into_response()
        }
        Err(e) => {
            let status = e.status();
            // Denied/invalid URLs are the caller's fault and common
            // (models invent URLs); log at debug. Upstream failures at
            // info — they're the model's cue to fall back to snippets.
            if status == 400 {
                tracing::debug!(url = %params.url, error = %e, "fetch: denied");
            } else {
                tracing::info!(url = %params.url, error = %e, "fetch: failed");
            }
            (
                StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_ok() {
        let app = app(ToolsConfig::default());
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn fetch_denies_internal_url_with_400() {
        let app = app(ToolsConfig::default());
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/fetch?url=http%3A%2F%2F10.3.0.1%2F")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
