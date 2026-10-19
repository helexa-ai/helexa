//! Async audio generation jobs through the router (#382).
//!
//! Submitting a job is ordinary capacity dispatch: the router picks a cortex
//! that serves the model and forwards the body, and that cortex owns the job
//! from then on. Every later call names only a job id, which carries no hint
//! of its cortex, so those calls are asked of each cortex in turn and the
//! first answer that is not `404` wins. Listing merges every cortex's list.
//!
//! Ownership is enforced by the cortex holding the job; the router forwards
//! the caller's bearer unchanged, exactly as for inference.

use crate::dispatch;
use crate::error::envelope_response;
use crate::state::RouterState;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use cortex_core::audio::{AudioJob, AudioJobList};
use cortex_core::error_envelope::OpenAiError;
use std::sync::Arc;

const PATH: &str = "/v1/audio/generations";

pub fn routes() -> Router<Arc<RouterState>> {
    Router::new()
        .route(PATH, get(list).post(submit))
        .route("/v1/audio/generations/{id}", get(get_job).delete(cancel))
        .route("/v1/audio/generations/{id}/content", get(content))
}

async fn submit(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    dispatch::dispatch(&state, PATH, headers, body).await
}

async fn get_job(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    find(&state, Method::GET, &format!("{PATH}/{id}"), &headers, &id).await
}

async fn cancel(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    find(
        &state,
        Method::DELETE,
        &format!("{PATH}/{id}"),
        &headers,
        &id,
    )
    .await
}

async fn content(
    State(state): State<Arc<RouterState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    RawQuery(query): RawQuery,
) -> Response {
    // The query carries the download format (`?format=mp3`).
    let path = match query {
        Some(q) => format!("{PATH}/{id}/content?{q}"),
        None => format!("{PATH}/{id}/content"),
    };
    find(&state, Method::GET, &path, &headers, &id).await
}

/// Ask each cortex for the job until one answers with something other than
/// `404`, and stream that answer back verbatim.
async fn find(
    state: &RouterState,
    method: Method,
    path: &str,
    headers: &HeaderMap,
    id: &str,
) -> Response {
    for ep in &state.cortexes {
        let Some(client) = state.client_for(&ep.name) else {
            continue;
        };
        let url = format!("{}{path}", ep.endpoint);
        let resp = match client
            .request(method.clone(), &url)
            .headers(forward_headers(headers))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(cortex = %ep.name, url = %url, error = %e, "cortex unreachable");
                continue;
            }
        };
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let mut out = Response::builder().status(status);
        for (k, v) in resp.headers() {
            if k == "transfer-encoding" || k == "connection" {
                continue;
            }
            out = out.header(k, v);
        }
        return out
            .body(Body::from_stream(resp.bytes_stream()))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response());
    }
    envelope_response(OpenAiError::new(
        404,
        "invalid_request_error",
        "job_not_found",
        format!("no audio job '{id}'"),
    ))
}

/// Merge every reachable cortex's job list, newest first.
async fn list(State(state): State<Arc<RouterState>>, headers: HeaderMap) -> Response {
    let mut data: Vec<AudioJob> = Vec::new();
    for ep in &state.cortexes {
        let Some(client) = state.client_for(&ep.name) else {
            continue;
        };
        let url = format!("{}{PATH}", ep.endpoint);
        let page = client
            .get(&url)
            .headers(forward_headers(&headers))
            .send()
            .await
            .ok()
            .filter(|r| r.status().is_success());
        let Some(page) = page else {
            tracing::warn!(cortex = %ep.name, "audio job list unavailable from cortex");
            continue;
        };
        if let Ok(list) = page.json::<AudioJobList>().await {
            data.extend(list.data);
        }
    }
    data.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
    Json(AudioJobList {
        object: "list".into(),
        data,
    })
    .into_response()
}

fn forward_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in headers {
        if matches!(
            k.as_str(),
            "host" | "content-length" | "connection" | "transfer-encoding"
        ) {
            continue;
        }
        out.append(k.clone(), v.clone());
    }
    out
}
