//! Integration tests for the shared streaming proxy (#71): proves a backend
//! SSE response is forwarded chunk-for-chunk (no buffering), the observer
//! sees every byte and finishes once, and non-2xx is streamed through with
//! its status intact — the behaviours both cortex and helexa-router rely on.

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::post;
use helexa_stream::{BodyTail, ChunkObserver, forward_streaming, last_count_for};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

/// Observer that records what it saw, for assertions.
#[derive(Clone, Default)]
struct RecordingObserver {
    inner: Arc<Mutex<Recorded>>,
}

#[derive(Default)]
struct Recorded {
    chunks: usize,
    finished: usize,
    tail: String,
}

impl ChunkObserver for RecordingObserver {
    fn observe(&mut self, chunk: &[u8]) {
        let mut r = self.inner.lock().unwrap();
        r.chunks += 1;
        r.tail.push_str(&String::from_utf8_lossy(chunk));
    }
    fn finish(&mut self) {
        self.inner.lock().unwrap().finished += 1;
    }
}

/// Mock backend that streams 5 SSE chunks with 30ms gaps, then a usage
/// chunk and `[DONE]`.
async fn sse_handler() -> Response {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"d\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"e\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5}}\n\n",
        "data: [DONE]\n\n",
    ];
    let stream = async_stream::stream! {
        for c in chunks {
            tokio::time::sleep(Duration::from_millis(30)).await;
            yield Ok::<_, std::io::Error>(axum::body::Bytes::from_static(c.as_bytes()));
        }
    };
    Response::new(Body::from_stream(stream))
}

async fn rate_limited_handler() -> Response {
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .body(Body::from("{\"error\":{\"type\":\"rate_limit_exceeded\"}}"))
        .unwrap()
}

async fn spawn_backend(router: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn streams_chunks_incrementally_and_observes_usage() {
    let base = spawn_backend(Router::new().route("/v1/chat/completions", post(sse_handler))).await;
    let observer = RecordingObserver::default();
    let probe = observer.clone();

    let client = reqwest::Client::new();
    let resp = forward_streaming(
        &client,
        &format!("{base}/v1/chat/completions"),
        HeaderMap::new(),
        axum::body::Bytes::from_static(b"{\"model\":\"x\",\"stream\":true}"),
        observer,
    )
    .await
    .expect("forward ok");

    assert_eq!(resp.status(), StatusCode::OK);

    // Read the proxied body as a stream, timestamping arrivals.
    let mut body = resp.into_body().into_data_stream();
    let mut arrivals: Vec<Instant> = Vec::new();
    let mut collected = String::new();
    use futures::StreamExt;
    while let Some(item) = body.next().await {
        let bytes = item.unwrap();
        arrivals.push(Instant::now());
        collected.push_str(&String::from_utf8_lossy(&bytes));
    }

    // Incremental delivery: first and last chunk are meaningfully apart
    // (5×30ms gaps), proving no full-response buffering.
    let spread = *arrivals.last().unwrap() - arrivals[0];
    assert!(
        spread >= Duration::from_millis(100),
        "expected incremental delivery, spread was {spread:?}"
    );

    // The client received the terminator and the usage object verbatim.
    assert!(collected.contains("data: [DONE]"));

    // The observer saw the bytes and finished exactly once.
    let r = probe.inner.lock().unwrap();
    assert!(r.chunks >= 5, "observer saw {} chunks", r.chunks);
    assert_eq!(r.finished, 1, "finish must run exactly once");
    assert_eq!(last_count_for(&r.tail, "prompt_tokens"), Some(11));
    assert_eq!(last_count_for(&r.tail, "completion_tokens"), Some(5));
}

#[tokio::test]
async fn non_2xx_is_streamed_through_verbatim() {
    let base =
        spawn_backend(Router::new().route("/v1/chat/completions", post(rate_limited_handler)))
            .await;
    let observer = RecordingObserver::default();
    let probe = observer.clone();

    let client = reqwest::Client::new();
    let resp = forward_streaming(
        &client,
        &format!("{base}/v1/chat/completions"),
        HeaderMap::new(),
        axum::body::Bytes::new(),
        observer,
    )
    .await
    .expect("forward ok");

    // Backpressure status reaches the client unchanged.
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&body).contains("rate_limit_exceeded"));

    // finish still runs once even with a tiny non-streaming body.
    assert_eq!(probe.inner.lock().unwrap().finished, 1);
}

#[test]
fn body_tail_smoke() {
    let mut tail = BodyTail::new(128);
    tail.push(b"hello ");
    tail.push(b"world");
    assert_eq!(tail.as_str(), "hello world");
}
