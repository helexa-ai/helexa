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
use futures::Stream;
use futures::stream::BoxStream;
use reqwest::Client;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

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
    let stream = TokenMetricsStream::new(
        Box::pin(upstream_resp.bytes_stream()),
        TokenMetrics::new(model_id, &route.node_name, request_start, usage_sink),
    );

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
// time and keeps a bounded tail of the body text, from which the final
// OpenAI `usage` object (present on the last SSE chunk and on
// non-streaming JSON bodies alike) yields engine-truth token counts.
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

/// Find the value of the LAST `"key": <integer>` occurrence in `tail`.
/// Pure and chunk-boundary-safe (the tail is contiguous appended text).
/// The quoted-needle form means `completion_tokens` never matches
/// `completion_tokens_details`.
pub(crate) fn last_count_for(tail: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let mut result = None;
    for (idx, _) in tail.match_indices(&needle) {
        let rest = tail[idx + needle.len()..].trim_start();
        let Some(rest) = rest.strip_prefix(':') else {
            continue;
        };
        let rest = rest.trim_start();
        let digits: &str = &rest[..rest
            .char_indices()
            .find(|(_, c)| !c.is_ascii_digit())
            .map(|(i, _)| i)
            .unwrap_or(rest.len())];
        if let Ok(v) = digits.parse::<u64>() {
            result = Some(v);
        }
    }
    result
}

struct TokenMetrics {
    labels: [(&'static str, String); 2],
    request_start: Instant,
    first_chunk: Option<Instant>,
    last_chunk: Option<Instant>,
    tail: String,
    finished: bool,
    /// Per-principal metering hook (#51). Invoked exactly once in `finish`
    /// with the observed `(prompt, completion)` so the reservation can be
    /// settled and spend recorded. `None` for anonymous requests.
    usage_sink: Option<crate::metering::UsageSink>,
}

impl TokenMetrics {
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
            tail: String::new(),
            finished: false,
            usage_sink,
        }
    }

    fn observe(&mut self, chunk: &[u8]) {
        let now = Instant::now();
        self.first_chunk.get_or_insert(now);
        self.last_chunk = Some(now);
        self.tail.push_str(&String::from_utf8_lossy(chunk));
        if self.tail.len() > TAIL_CAP_BYTES {
            // Keep the newest half; the usage object is always at the
            // very end of the body. Split at a char boundary.
            let mut cut = self.tail.len() - TAIL_CAP_BYTES / 2;
            while !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }
    }

    /// Emit the metrics exactly once — called on clean stream end and
    /// from Drop (client disconnect mid-stream still records what we
    /// saw).
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;

        let prompt = last_count_for(&self.tail, "prompt_tokens");
        let completion = last_count_for(&self.tail, "completion_tokens");

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

/// Pass-through stream wrapper that feeds [`TokenMetrics`]. Emits on
/// clean end-of-stream; the Drop impl covers client disconnects.
struct TokenMetricsStream {
    inner: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    metrics: TokenMetrics,
}

impl TokenMetricsStream {
    fn new(
        inner: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
        metrics: TokenMetrics,
    ) -> Self {
        Self { inner, metrics }
    }
}

impl Stream for TokenMetricsStream {
    type Item = Result<bytes::Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.metrics.observe(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                this.metrics.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for TokenMetricsStream {
    fn drop(&mut self) {
        self.metrics.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::last_count_for;

    #[test]
    fn extracts_counts_from_final_sse_usage_chunk() {
        let tail = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":225,",
            "\"completion_tokens\":42,\"total_tokens\":267}}\n\n",
            "data: [DONE]\n\n"
        );
        assert_eq!(last_count_for(tail, "prompt_tokens"), Some(225));
        assert_eq!(last_count_for(tail, "completion_tokens"), Some(42));
    }

    #[test]
    fn extracts_counts_from_non_streaming_body() {
        let tail = "{\"choices\":[{\"message\":{\"content\":\"hi\"}}],\
                    \"usage\":{\"prompt_tokens\": 12, \"completion_tokens\": 7}}";
        assert_eq!(last_count_for(tail, "prompt_tokens"), Some(12));
        assert_eq!(last_count_for(tail, "completion_tokens"), Some(7));
    }

    #[test]
    fn ignores_details_variants_and_takes_last_occurrence() {
        // completion_tokens_details must not shadow completion_tokens,
        // and the LAST usage object wins (matters when content echoes
        // a usage-shaped string earlier in the stream).
        let tail = concat!(
            "data: {\"usage\":{\"completion_tokens\":1}}\n\n",
            "data: {\"usage\":{\"completion_tokens\":99,",
            "\"completion_tokens_details\":{\"reasoning_tokens\":3}}}\n\n"
        );
        assert_eq!(last_count_for(tail, "completion_tokens"), Some(99));
    }

    #[test]
    fn absent_keys_yield_none() {
        assert_eq!(
            last_count_for("data: [DONE]\n\n", "completion_tokens"),
            None
        );
        assert_eq!(last_count_for("", "prompt_tokens"), None);
        // key present but non-numeric value
        assert_eq!(
            last_count_for("\"completion_tokens\": null", "completion_tokens"),
            None
        );
    }
}
