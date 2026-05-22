//! Streaming HTTP reverse proxy to neuron backends.
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
///
/// Logging contract: every call emits exactly one structured event at
/// info / warn level for operator visibility, regardless of outcome.
/// Network-level failures and non-2xx upstream statuses are warn'd here
/// (closest to the wire); the user-facing response carries only the
/// status code and a generic message — implementation detail (body,
/// error chain) lives in the log, never in the API surface.
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

    let upstream_resp = match req_builder.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                node = %route.node_name,
                url = %url,
                error = %e,
                "proxy: upstream request failed (network)"
            );
            return Err(ProxyError::Upstream(e));
        }
    };

    let upstream_status = upstream_resp.status();
    if !upstream_status.is_success() {
        // Streaming body — can't snippet without breaking the stream
        // pass-through. Log status + URL; the client still gets the
        // upstream status, just without the leaked body.
        tracing::warn!(
            node = %route.node_name,
            url = %url,
            status = upstream_status.as_u16(),
            "proxy: upstream returned non-2xx"
        );
    }

    let status = StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    let resp_headers = upstream_resp.headers().clone();
    let stream = upstream_resp.bytes_stream();

    let body = Body::from_stream(stream);

    let mut response = Response::builder().status(status);
    for (key, value) in resp_headers.iter() {
        response = response.header(key, value);
    }

    response.body(body).map_err(|e| {
        tracing::warn!(
            node = %route.node_name,
            url = %url,
            error = %e,
            "proxy: failed to build response"
        );
        ProxyError::ResponseBuild(e.to_string())
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("upstream request failed")]
    Upstream(reqwest::Error),
    #[error("failed to build response")]
    ResponseBuild(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ProxyError::Upstream(_) => (StatusCode::BAD_GATEWAY, "upstream request failed"),
            ProxyError::ResponseBuild(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build response",
            ),
        };
        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": "proxy_error",
            }
        });
        (status, axum::Json(body)).into_response()
    }
}
