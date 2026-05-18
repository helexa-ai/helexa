mod common;

use futures::StreamExt;
use serde_json::json;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_streaming_sse_passthrough() {
    let chunk_count = 5;
    let chunk_delay = Duration::from_millis(50);
    let mock_url = common::spawn_streaming_mock_neuron(chunk_count, chunk_delay).await;
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

    assert!(
        chunks.len() > chunk_count,
        "expected more than {} chunks (got {}): {:?}",
        chunk_count,
        chunks.len(),
        chunks,
    );

    assert_eq!(chunks.last().unwrap(), "[DONE]");

    for (i, chunk) in chunks.iter().enumerate().take(chunk_count) {
        let chunk_json: serde_json::Value =
            serde_json::from_str(chunk).expect("chunk should be valid JSON");
        assert_eq!(
            chunk_json["choices"][0]["delta"]["content"],
            format!("token{i}")
        );
    }

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
    let mock_url = common::spawn_streaming_mock_neuron(2, Duration::from_millis(10)).await;
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
