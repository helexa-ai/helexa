mod common;

use futures::StreamExt;
use serde_json::json;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_streaming_sse_passthrough() {
    let chunk_count = 5;
    let chunk_delay = Duration::from_millis(50);
    let mock_url = common::spawn_streaming_mock_backend(chunk_count, chunk_delay).await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "text/event-stream"
    );

    // Collect SSE chunks as they arrive, recording arrival times.
    let start = Instant::now();
    let mut chunk_times = Vec::new();
    let mut chunks = Vec::new();
    let mut stream = resp.bytes_stream();

    while let Some(result) = stream.next().await {
        let bytes = result.expect("chunk should be valid");
        let text = String::from_utf8_lossy(&bytes);
        for line in text.split("data: ").filter(|s| !s.is_empty()) {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                chunk_times.push(start.elapsed());
                chunks.push(trimmed.to_string());
            }
        }
    }

    // Verify we got all content chunks plus [DONE].
    assert!(
        chunks.len() >= chunk_count + 1,
        "expected at least {} chunks (got {}): {:?}",
        chunk_count + 1,
        chunks.len(),
        chunks,
    );

    // The last chunk should be [DONE].
    assert_eq!(chunks.last().unwrap(), "[DONE]");

    // Verify the content chunks contain expected tokens.
    for i in 0..chunk_count {
        let chunk_json: serde_json::Value =
            serde_json::from_str(&chunks[i]).expect("chunk should be valid JSON");
        assert_eq!(
            chunk_json["choices"][0]["delta"]["content"],
            format!("token{i}")
        );
    }

    // Verify streaming behavior: total time should reflect incremental delivery,
    // not a single batch. With 5 chunks at 50ms each + [DONE], we expect ~300ms total.
    // If buffered, all chunks would arrive at once after ~300ms with no spread.
    // We verify that the last chunk arrived noticeably after the first.
    let first = chunk_times.first().unwrap();
    let last = chunk_times.last().unwrap();
    let spread = *last - *first;
    assert!(
        spread >= Duration::from_millis(100),
        "chunks should arrive incrementally (spread: {spread:?})",
    );
}

#[tokio::test]
async fn test_streaming_done_terminator() {
    let mock_url = common::spawn_streaming_mock_backend(2, Duration::from_millis(10)).await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("request should succeed");

    let body = resp.text().await.unwrap();
    assert!(
        body.contains("data: [DONE]"),
        "response must contain [DONE] terminator"
    );
    assert!(body.contains("token0"), "response must contain first token");
    assert!(
        body.contains("token1"),
        "response must contain second token"
    );
}
