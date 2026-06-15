mod common;

use serde_json::json;

#[tokio::test]
async fn test_anthropic_to_openai_round_trip() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "test-model");

    let content = body["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Hello from mock backend");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["input_tokens"], 10);
    assert_eq!(body["usage"]["output_tokens"], 5);
}

#[tokio::test]
async fn test_anthropic_with_system_prompt() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn test_anthropic_with_content_blocks() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "What is this?"}]
            }]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("valid JSON");
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn test_anthropic_model_not_found() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "nonexistent",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_anthropic_invalid_request() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({"not_a_valid": "request"}))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);
}

/// Tool round-trip: an Anthropic `/v1/messages` request carrying tools
/// (the Claude Code shape: `{name, description, input_schema}`) must
/// reach the upstream neuron reshaped into OpenAI function-tool form,
/// and tool history (`tool_use` / `tool_result` blocks) must become
/// `tool_calls` / `role:"tool"` messages. This is the fix for the
/// failure where the model received malformed tool defs and improvised
/// an unparseable `<tool_use_name>` format.
#[tokio::test]
async fn test_anthropic_tools_reshaped_for_upstream() {
    let (mock_url, captured) = common::spawn_capturing_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "tools": [{
                "name": "Read",
                "description": "Read a file from disk",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }],
            "tool_choice": {"type": "auto"},
            "messages": [
                {"role": "user", "content": "read /etc/hosts"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Reading it."},
                    {"type": "tool_use", "id": "toolu_42", "name": "Read",
                     "input": {"path": "/etc/hosts"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_42",
                     "content": "127.0.0.1 localhost"}
                ]}
            ]
        }))
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);

    let forwarded = {
        let guard = captured.lock().unwrap();
        guard.last().cloned().expect("upstream received a request")
    };

    // Tool definitions reshaped to OpenAI function form.
    let tools = forwarded["tools"].as_array().expect("tools array");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "Read");
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["path"]["type"],
        "string"
    );
    assert!(tools[0]["function"].get("input_schema").is_none());

    // tool_choice mapped.
    assert_eq!(forwarded["tool_choice"], "auto");

    // Message history: user, assistant(+tool_calls), tool, user.
    let msgs = forwarded["messages"].as_array().expect("messages array");
    let assistant = msgs
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("assistant turn");
    assert_eq!(assistant["tool_calls"][0]["id"], "toolu_42");
    assert_eq!(assistant["tool_calls"][0]["function"]["name"], "Read");
    assert_eq!(
        assistant["tool_calls"][0]["function"]["arguments"],
        "{\"path\":\"/etc/hosts\"}"
    );

    let tool_msg = msgs
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("tool result turn");
    assert_eq!(tool_msg["tool_call_id"], "toolu_42");
    assert_eq!(tool_msg["content"], "127.0.0.1 localhost");
}

/// #24: a streaming Anthropic request gets a translated Anthropic SSE
/// stream — not raw OpenAI frames. Verifies the full event sequence,
/// text reassembly, and the content type.
#[tokio::test]
async fn test_anthropic_streaming_sse_translation() {
    let mock_url =
        common::spawn_streaming_mock_neuron(4, std::time::Duration::from_millis(20)).await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .starts_with("text/event-stream"),
        "anthropic stream must be SSE"
    );

    let body = resp.text().await.expect("stream should complete");
    assert!(
        !body.contains("chat.completion.chunk"),
        "raw OpenAI frames must not leak through:\n{body}"
    );

    let event_names: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();
    assert_eq!(
        event_names,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
        "unexpected event sequence:\n{body}"
    );

    // Reassemble the text deltas: the mock emits token0..token3.
    let text: String = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .filter(|v| v["type"] == "content_block_delta")
        .filter_map(|v| v["delta"]["text"].as_str().map(String::from))
        .collect();
    assert_eq!(text, "token0token1token2token3");

    // The mock sends no finish_reason — stop_reason defaults to
    // end_turn, and output_tokens falls back to the delta count.
    let message_delta = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .find(|v| v["type"] == "message_delta")
        .expect("message_delta event present");
    assert_eq!(message_delta["delta"]["stop_reason"], "end_turn");
    assert_eq!(message_delta["usage"]["output_tokens"], 4);
}

/// #24: an upstream usage frame (stream_options include_usage shape)
/// rides into message_delta as input/output token counts.
#[tokio::test]
async fn test_anthropic_streaming_usage_propagation() {
    let mock_url = common::spawn_streaming_mock_neuron_with_usage(
        3,
        std::time::Duration::from_millis(10),
        225,
        42,
    )
    .await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let body = client
        .post(format!("{gw_url}/v1/messages"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed")
        .text()
        .await
        .expect("stream should complete");

    let message_delta = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .find(|v| v["type"] == "message_delta")
        .expect("message_delta event present");
    assert_eq!(message_delta["usage"]["output_tokens"], 42);
    assert_eq!(message_delta["usage"]["input_tokens"], 225);
}
