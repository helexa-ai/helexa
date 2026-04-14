mod common;

use serde_json::json;

#[tokio::test]
async fn test_anthropic_to_openai_round_trip() {
    let mock_url = common::spawn_mock_backend().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");

    // Response should be in Anthropic format.
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "test-model");

    // Content should be an array of content blocks.
    let content = body["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Hello from mock backend");

    // Stop reason should be translated from "stop" to "end_turn".
    assert_eq!(body["stop_reason"], "end_turn");

    // Usage should have Anthropic field names.
    assert_eq!(body["usage"]["input_tokens"], 10);
    assert_eq!(body["usage"]["output_tokens"], 5);
}

#[tokio::test]
async fn test_anthropic_with_system_prompt() {
    let mock_url = common::spawn_mock_backend().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["text"], "Hello from mock backend");
}

#[tokio::test]
async fn test_anthropic_with_content_blocks() {
    let mock_url = common::spawn_mock_backend().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"}
                    ]
                }
            ]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["text"], "Hello from mock backend");
}

#[tokio::test]
async fn test_anthropic_model_not_found() {
    let mock_url = common::spawn_mock_backend().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "nonexistent",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_anthropic_invalid_request() {
    let mock_url = common::spawn_mock_backend().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "not_a_valid": "request"
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid Anthropic request")
    );
}
