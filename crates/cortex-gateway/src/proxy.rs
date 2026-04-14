//! Streaming HTTP reverse proxy to mistral.rs backends.
//!
//! For streaming requests, SSE chunks are forwarded as they arrive.
//! The proxy captures timing information for metrics but does not
//! buffer the full response.

use crate::router::RouteDecision;
use anyhow::Result;
use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use reqwest::Client;

/// Proxy a request body to the resolved backend node and stream the response.
pub async fn forward_request(
    client: &Client,
    route: &RouteDecision,
    path: &str,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Result<Response, ProxyError> {
    let url = format!("{}{}", route.endpoint, path);
    tracing::info!(
        node = %route.node_name,
        url = %url,
        cold_start = route.cold_start,
        "proxying request"
    );

    let mut req_builder = client.post(&url).body(body);

    // Forward relevant headers.
    for (key, value) in headers.iter() {
        if key == "host" || key == "content-length" {
            continue; // reqwest sets these
        }
        req_builder = req_builder.header(key, value);
    }

    let upstream_resp = req_builder.send().await.map_err(ProxyError::Upstream)?;

    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);

    let resp_headers = upstream_resp.headers().clone();
    let stream = upstream_resp.bytes_stream();

    let body = Body::from_stream(stream);

    let mut response = Response::builder().status(status);
    for (key, value) in resp_headers.iter() {
        response = response.header(key, value);
    }

    response
        .body(body)
        .map_err(|e| ProxyError::ResponseBuild(e.to_string()))
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("upstream request failed: {0}")]
    Upstream(reqwest::Error),
    #[error("failed to build response: {0}")]
    ResponseBuild(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = match &self {
            ProxyError::Upstream(_) => StatusCode::BAD_GATEWAY,
            ProxyError::ResponseBuild(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": "proxy_error",
            }
        });
        (status, axum::Json(body)).into_response()
    }
}
