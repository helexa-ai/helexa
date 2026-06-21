//! Shared streaming reverse-proxy mechanism (#71).
//!
//! cortex and helexa-router both need to proxy an OpenAI/Anthropic SSE
//! response from a downstream backend **verbatim** — chunks forwarded as
//! they arrive, never buffering the full body — while observing the bytes
//! for metrics/metering. This crate owns that mechanism so there is one
//! implementation, not one per tier.
//!
//! The split is mechanism vs policy:
//!
//! - **Mechanism (here):** [`forward_streaming`] POSTs to a backend and
//!   streams the response body back through an [`ObservedStream`], which
//!   feeds every chunk to a caller-supplied [`ChunkObserver`] and calls
//!   [`ChunkObserver::finish`] exactly once on clean end-of-stream or on
//!   drop (client disconnect mid-stream). [`BodyTail`] and
//!   [`last_count_for`] are the reusable pieces an observer uses to pull
//!   the trailing OpenAI `usage` object out of the streamed bytes.
//! - **Policy (caller):** what to *do* with the observed bytes — which
//!   metric names to emit, which labels, whether to settle a per-principal
//!   reservation — lives in the consumer's `ChunkObserver` impl, not here.
//!
//! The proxy is status-agnostic: a non-2xx upstream response (e.g. a
//! cortex `429 rate_limit_exceeded`) is streamed back with its status and
//! headers intact, so honest backpressure reaches the client unchanged.
//! Only a network failure or a malformed response build is an error.

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures::Stream;
use futures::stream::BoxStream;
use reqwest::Client;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Observes the bytes of a streamed proxy response without altering them.
///
/// `observe` is called for each forwarded chunk; `finish` is called
/// exactly once — on clean end-of-stream or on drop — and implementations
/// must be idempotent (the [`ObservedStream`] guards against a double call,
/// but a `finish` that runs side effects should still self-guard).
pub trait ChunkObserver: Send + Unpin + 'static {
    /// A body chunk has been forwarded downstream. The slice is the exact
    /// bytes the client receives.
    fn observe(&mut self, chunk: &[u8]);

    /// The stream has ended (cleanly or via client disconnect). Called once.
    fn finish(&mut self);
}

/// A bounded accumulator for the tail of a streamed body.
///
/// The OpenAI `usage` object rides on the final SSE chunk (and sits at the
/// end of a non-streaming JSON body), so retaining a generous tail is
/// enough to recover token counts via [`last_count_for`]; the cap bounds
/// memory on huge bodies. Appends are char-boundary-safe.
#[derive(Debug)]
pub struct BodyTail {
    tail: String,
    cap: usize,
}

impl BodyTail {
    /// Create a tail retaining at most `cap` bytes.
    pub fn new(cap: usize) -> Self {
        Self {
            tail: String::new(),
            cap,
        }
    }

    /// Append a chunk, trimming from the front past the cap. When trimming,
    /// the newest half is kept (the usage object is always at the very end).
    pub fn push(&mut self, chunk: &[u8]) {
        self.tail.push_str(&String::from_utf8_lossy(chunk));
        if self.tail.len() > self.cap {
            let mut cut = self.tail.len() - self.cap / 2;
            while !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }
    }

    /// The retained tail text.
    pub fn as_str(&self) -> &str {
        &self.tail
    }
}

/// Find the value of the LAST `"key": <integer>` occurrence in `tail`.
///
/// Pure and chunk-boundary-safe (the tail is contiguous appended text).
/// The quoted-needle form means `completion_tokens` never matches
/// `completion_tokens_details`, and taking the last occurrence means the
/// final `usage` object wins even if content earlier in the stream echoed
/// a usage-shaped string.
pub fn last_count_for(tail: &str, key: &str) -> Option<u64> {
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

/// Error from [`forward_streaming`]. Distinguishes a network/transport
/// failure reaching the backend from a failure assembling the downstream
/// response. A non-2xx upstream *status* is not an error — it is streamed
/// through verbatim.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("upstream request failed")]
    Upstream(reqwest::Error),
    #[error("failed to build response")]
    ResponseBuild(String),
}

/// POST `body` to `url` and stream the response back verbatim through
/// `observer`.
///
/// Request headers are forwarded except `host` / `content-length` (reqwest
/// sets these). The returned [`Response`] carries the upstream status and
/// headers unchanged — including non-2xx — with a body that streams the
/// upstream bytes chunk-for-chunk, feeding each chunk to `observer`.
pub async fn forward_streaming<O: ChunkObserver>(
    client: &Client,
    url: &str,
    headers: HeaderMap,
    body: Bytes,
    observer: O,
) -> Result<Response, StreamError> {
    let mut req_builder = client.post(url).body(body);
    for (key, value) in headers.iter() {
        if key == "host" || key == "content-length" {
            continue; // reqwest sets these
        }
        req_builder = req_builder.header(key, value);
    }

    let upstream = req_builder.send().await.map_err(StreamError::Upstream)?;

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let resp_headers = upstream.headers().clone();

    let stream = ObservedStream::new(Box::pin(upstream.bytes_stream()), observer);
    let body = Body::from_stream(stream);

    let mut response = Response::builder().status(status);
    for (key, value) in resp_headers.iter() {
        response = response.header(key, value);
    }
    response
        .body(body)
        .map_err(|e| StreamError::ResponseBuild(e.to_string()))
}

/// Pass-through stream wrapper that feeds a [`ChunkObserver`]. Forwards
/// each chunk verbatim, calls `observe` per chunk, and `finish` once on
/// clean end-of-stream; the `Drop` impl covers client disconnects.
pub struct ObservedStream<O: ChunkObserver> {
    inner: BoxStream<'static, Result<Bytes, reqwest::Error>>,
    observer: O,
    finished: bool,
}

impl<O: ChunkObserver> ObservedStream<O> {
    /// Wrap a byte stream with an observer.
    pub fn new(inner: BoxStream<'static, Result<Bytes, reqwest::Error>>, observer: O) -> Self {
        Self {
            inner,
            observer,
            finished: false,
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.observer.finish();
    }
}

impl<O: ChunkObserver> Stream for ObservedStream<O> {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.observer.observe(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                this.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<O: ChunkObserver> Drop for ObservedStream<O> {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn body_tail_retains_usage_after_cap_trim() {
        // Cap small enough that the filler forces several front-trims, but
        // (as in production, where cap ≫ the usage object) large enough that
        // the trailing usage object survives the newest-half retention.
        let mut tail = BodyTail::new(512);
        for _ in 0..100 {
            tail.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
        }
        assert!(tail.as_str().len() <= 512, "cap must bound the tail");
        tail.push(b"data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n");
        assert_eq!(last_count_for(tail.as_str(), "prompt_tokens"), Some(5));
        assert_eq!(last_count_for(tail.as_str(), "completion_tokens"), Some(9));
    }
}
