//! Router adapter from the shared, axum-agnostic
//! [`cortex_core::error_envelope::OpenAiError`] (#60/#63) to an axum
//! [`Response`], setting `Retry-After` when the envelope carries one.
//!
//! cortex-core owns the envelope shape; this is the only place the router
//! crosses from that data into axum. Mirrors cortex-gateway's adapter so
//! the router's own rejections (no feasible operator, all unreachable) are
//! the same #63-shaped envelopes clients already understand — distinct from
//! cortex's rejections, which the router proxies through verbatim.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use cortex_core::error_envelope::OpenAiError;

/// Render an [`OpenAiError`] as an axum response (status + JSON envelope +
/// optional `Retry-After`).
pub fn envelope_response(err: OpenAiError) -> Response {
    let status = StatusCode::from_u16(err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let retry_after = err.retry_after_secs;
    let mut response = (status, Json(err.body())).into_response();
    if let Some(secs) = retry_after
        && let Ok(value) = HeaderValue::from_str(&secs.to_string())
    {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}
