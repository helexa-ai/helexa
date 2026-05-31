//! Integration tests for the `/v1/responses` proxy route.
//!
//! The gateway forwards the request body to whichever neuron has the
//! model loaded. These tests exercise the routing decision (200 on a
//! known model, 404 on an unknown model, 400 on a missing model
//! field) and confirm the response body round-trips verbatim.

mod common;

use serde_json::json;

/// Happy path: gateway routes a `/v1/responses` request to the neuron
/// that has the model loaded, and the neuron's response body
/// arrives at the client unchanged.
#[tokio::test]
async fn test_responses_proxy() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/responses"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "input": "Hi"
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON response");
    assert_eq!(body["id"], "resp-test-001");
    assert_eq!(body["object"], "response");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["status"], "completed");
    assert_eq!(
        body["output"][0]["content"][0]["text"],
        "Hello from mock backend"
    );
    // Usage shape is the Responses-specific (input/output_tokens),
    // not the chat-completions one (prompt/completion_tokens). Asserts
    // the proxy didn't accidentally route through the wrong handler.
    assert_eq!(body["usage"]["total_tokens"], 10);
    assert!(body["usage"].get("input_tokens").is_some());
}

/// A request that targets a model not present in the catalogue gets
/// 404 from the router. This matches the chat-completions handler's
/// behaviour — same error path, same status code, so a client can
/// share retry logic across the two routes.
#[tokio::test]
async fn test_responses_model_not_found() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/responses"))
        .json(&json!({
            "model": "not-in-catalogue",
            "input": "Hi"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// A request body without a `model` field can't be routed; the
/// gateway returns 400 before reaching a backend. Same as the
/// chat-completions handler — extracted via the same `extract_model`
/// helper.
#[tokio::test]
async fn test_responses_missing_model_field() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/responses"))
        .json(&json!({
            "input": "Hi"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}
