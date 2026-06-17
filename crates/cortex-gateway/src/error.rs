//! Gateway adapter that turns the shared, axum-agnostic
//! [`cortex_core::error_envelope::OpenAiError`] into an axum [`Response`],
//! setting the `Retry-After` header when the envelope carries one.
//!
//! cortex-core owns the envelope shape and the rejection contract (#60/#63);
//! this is the only place the gateway crosses from that data into axum.

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
