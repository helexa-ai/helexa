//! Adapter from the shared, axum-agnostic
//! [`cortex_core::error_envelope::OpenAiError`] (#60/#63) to an axum
//! response, with `Retry-After`. The `/authz/v1` surface speaks the #63
//! envelope so cortex (an OpenAI-compatible proxy) can forward rejections
//! verbatim. (The future `/web/v1` surface uses a plain JSON error shape.)

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use cortex_core::error_envelope::OpenAiError;

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
