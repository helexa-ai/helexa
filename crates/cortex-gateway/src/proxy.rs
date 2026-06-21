//! Streaming HTTP reverse proxy to neuron backends.
//!
//! The streaming *mechanism* — forward an SSE body chunk-for-chunk without
//! buffering, observing the bytes for metrics — lives in the shared
//! [`helexa_stream`] crate (#71), so cortex and helexa-router use one
//! implementation. This module supplies cortex's *policy*: the
//! [`CortexMetrics`] observer (per-request token metrics + per-principal
//! reservation settle), cortex's logging contract, and the cortex error
//! envelope. The usage-extraction helper is re-exported from the shared
//! crate so existing call sites keep working.

use crate::router::RouteDecision;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use helexa_stream::{BodyTail, ChunkObserver, StreamError};
use reqwest::Client;
use std::time::Instant;

/// Re-export the shared usage-extraction helper. Several cortex modules
/// (`handlers`, `anthropic_sse`) pull token counts out of a buffered body
/// tail via this function; it lives in `helexa-stream` now.
pub use helexa_stream::last_count_for;

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
    model_id: &str,
    usage_sink: Option<crate::metering::UsageSink>,
) -> Result<Response, ProxyError> {
    let request_start = Instant::now();
    let url = format!("{}{}", route.endpoint, path);
    tracing::info!(
        node = %route.node_name,
        url = %url,
        cold_start = route.cold_start,
        "proxying request"
    );

    let observer = CortexMetrics::new(model_id, &route.node_name, request_start, usage_sink);

    let response = helexa_stream::forward_streaming(client, &url, headers, body, observer)
        .await
        .map_err(|e| {
            match &e {
                StreamError::Upstream(err) => tracing::warn!(
                    node = %route.node_name,
                    url = %url,
                    error = %err,
                    "proxy: upstream request failed (network)"
                ),
                StreamError::ResponseBuild(err) => tracing::warn!(
                    node = %route.node_name,
                    url = %url,
                    error = %err,
                    "proxy: failed to build response"
                ),
            }
            ProxyError::from(e)
        })?;

    if !response.status().is_success() {
        // Streaming body — can't snippet without breaking the stream
        // pass-through. Log status + URL; the client still gets the
        // upstream status, just without the leaked body.
        tracing::warn!(
            node = %route.node_name,
            url = %url,
            status = response.status().as_u16(),
            "proxy: upstream returned non-2xx"
        );
    }

    Ok(response)
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("upstream request failed")]
    Upstream(reqwest::Error),
    #[error("failed to build response")]
    ResponseBuild(String),
}

impl From<StreamError> for ProxyError {
    fn from(e: StreamError) -> Self {
        match e {
            StreamError::Upstream(err) => ProxyError::Upstream(err),
            StreamError::ResponseBuild(msg) => ProxyError::ResponseBuild(msg),
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ProxyError::Upstream(_) => (
                StatusCode::BAD_GATEWAY,
                "upstream_connection_error",
                "upstream request failed",
            ),
            ProxyError::ResponseBuild(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_server_error",
                "failed to build response",
            ),
        };
        crate::error::envelope_response(cortex_core::error_envelope::OpenAiError::new(
            status.as_u16(),
            "api_error",
            code,
            message,
        ))
    }
}

// ── Per-request token metrics (#21) ─────────────────────────────────
//
// The proxy never buffers or re-serialises the upstream body — chunks
// are forwarded verbatim. For metrics it observes each chunk's arrival
// time and keeps a bounded tail of the body text (via the shared
// `helexa_stream::BodyTail`), from which the final OpenAI `usage` object
// (present on the last SSE chunk and on non-streaming JSON bodies alike)
// yields engine-truth token counts.
//
// Emitted per request, labelled {model, node}:
//   cortex_time_to_first_token_seconds  (histogram) — first body chunk
//   cortex_tokens_per_second            (histogram) — completion tokens
//       over the decode window (first→last chunk); falls back to the
//       full request duration for single-chunk (non-streaming) bodies
//   cortex_prompt_tokens_total / cortex_completion_tokens_total (counters)

/// Cap on the retained body tail. The usage object rides on the final
/// chunk, so a generous tail is plenty; the cap bounds memory on huge
/// non-streaming bodies.
const TAIL_CAP_BYTES: usize = 64 * 1024;

/// cortex's [`ChunkObserver`]: per-request token metrics plus the
/// per-principal reservation settle. Drives cortex policy over the shared
/// streaming mechanism.
struct CortexMetrics {
    labels: [(&'static str, String); 2],
    request_start: Instant,
    first_chunk: Option<Instant>,
    last_chunk: Option<Instant>,
    tail: BodyTail,
    finished: bool,
    /// Per-principal metering hook (#51). Invoked exactly once in `finish`
    /// with the observed `(prompt, completion)` so the reservation can be
    /// settled and spend recorded. `None` for anonymous requests.
    usage_sink: Option<crate::metering::UsageSink>,
}

impl CortexMetrics {
    fn new(
        model_id: &str,
        node_name: &str,
        request_start: Instant,
        usage_sink: Option<crate::metering::UsageSink>,
    ) -> Self {
        Self {
            labels: [
                ("model", model_id.to_string()),
                ("node", node_name.to_string()),
            ],
            request_start,
            first_chunk: None,
            last_chunk: None,
            tail: BodyTail::new(TAIL_CAP_BYTES),
            finished: false,
            usage_sink,
        }
    }
}

impl ChunkObserver for CortexMetrics {
    fn observe(&mut self, chunk: &[u8]) {
        let now = Instant::now();
        self.first_chunk.get_or_insert(now);
        self.last_chunk = Some(now);
        self.tail.push(chunk);
    }

    /// Emit the metrics exactly once — called on clean stream end and
    /// from Drop (client disconnect mid-stream still records what we
    /// saw).
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;

        let prompt = last_count_for(self.tail.as_str(), "prompt_tokens");
        let completion = last_count_for(self.tail.as_str(), "completion_tokens");

        // Per-model metrics — only when body chunks actually arrived.
        if let Some(first) = self.first_chunk {
            let ttft = first.duration_since(self.request_start).as_secs_f64();
            metrics::histogram!("cortex_time_to_first_token_seconds", &self.labels).record(ttft);

            if let Some(prompt) = prompt {
                metrics::counter!("cortex_prompt_tokens_total", &self.labels).increment(prompt);
            }
            if let Some(completion) = completion.filter(|c| *c > 0) {
                metrics::counter!("cortex_completion_tokens_total", &self.labels)
                    .increment(completion);

                let last = self.last_chunk.unwrap_or(first);
                let decode_window = last.duration_since(first).as_secs_f64();
                // Streaming: rate over the decode window (first→last chunk).
                // Non-streaming bodies arrive as ~one chunk (window ≈ 0),
                // where the only honest denominator is the full request
                // duration.
                let secs = if decode_window >= 0.1 {
                    decode_window
                } else {
                    last.duration_since(self.request_start).as_secs_f64()
                };
                if secs > 0.0 {
                    metrics::histogram!("cortex_tokens_per_second", &self.labels)
                        .record(completion as f64 / secs);
                }
            }
        }

        // Per-principal metering + reservation settle (#51). Always runs so
        // the reservation is resolved even when no usage/body was observed
        // (sink with (0, 0) → settle 0 → release).
        if let Some(sink) = self.usage_sink.take() {
            sink(prompt.unwrap_or(0), completion.unwrap_or(0));
        }
    }
}
