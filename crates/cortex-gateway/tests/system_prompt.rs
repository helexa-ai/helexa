//! Application-owned system prompts (#179).
//!
//! The platform guarantee: a caller's system prompt reaches the model
//! **verbatim** on every API surface — helexa never injects, rewrites or
//! defaults it. Applications tailor model behaviour through the standard
//! API fields; operators do not curate prompts.
//!
//! These pin the gateway half of that guarantee by asserting what cortex
//! *actually forwarded upstream*, not merely that a response came back —
//! a gateway that silently dropped or replaced the system prompt would
//! still return a perfectly valid-looking answer.

mod common;

use axum::Router;
use axum::extract::Path;
use axum::response::Json;
use axum::routing::{get, post};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

/// Mock neuron capturing every inference body it receives, across the
/// chat-completions **and** responses surfaces (the shared helper only
/// covers the former).
async fn spawn_capturing_neuron() -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();
    let captured: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let chat_sink = captured.clone();
    let resp_sink = captured.clone();

    let app = Router::new()
        .route(
            "/models",
            get(|| async {
                Json(json!([{
                    "id": "test-model", "harness": "candle", "status": "loaded",
                    "devices": [0], "vram_used_mb": 8000,
                    "capabilities": ["text"], "tool_call": false, "reasoning": false
                }]))
            }),
        )
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let sink = chat_sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    Json(json!({
                        "id": "chatcmpl-sys", "object": "chat.completion",
                        "created": 1700000000_u64, "model": "test-model",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    }))
                }
            }),
        )
        .route(
            "/v1/responses",
            post(move |Json(body): Json<Value>| {
                let sink = resp_sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    Json(json!({
                        "id": "resp-sys", "object": "response", "created_at": 1700000000_u64,
                        "model": "test-model", "status": "completed",
                        "output": [{"type": "message", "id": "msg-1", "status": "completed",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "ok"}]}],
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                    }))
                }
            }),
        );

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (base_url, captured)
}

const SYS: &str = "You must reply with exactly the word PONG and nothing else.";

/// Collect the `system`-role message contents from a forwarded
/// chat-completions body.
fn system_messages(body: &Value) -> Vec<String> {
    body["messages"]
        .as_array()
        .map(|msgs| {
            msgs.iter()
                .filter(|m| m["role"] == "system")
                .map(|m| m["content"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn chat_completions_system_message_is_forwarded_verbatim() {
    let (mock, captured) = spawn_capturing_neuron().await;
    let gw = common::spawn_gateway(&mock).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "hello"}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap();
    let body = seen.first().expect("neuron received a request");
    assert_eq!(
        system_messages(body),
        vec![SYS.to_string()],
        "system prompt must reach the neuron unaltered: {body}"
    );
    // Order matters: a system turn after the conversation reads as a late
    // instruction rather than a role.
    assert_eq!(body["messages"][0]["role"], "system");
}

#[tokio::test]
async fn responses_instructions_are_forwarded() {
    let (mock, captured) = spawn_capturing_neuron().await;
    let gw = common::spawn_gateway(&mock).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw}/v1/responses"))
        .json(&json!({
            "model": "test-model",
            "instructions": SYS,
            "input": "hello"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap();
    let body = seen.first().expect("neuron received a request");
    // cortex proxies Responses verbatim — neuron owns the mapping of
    // `instructions` onto the system slot.
    assert_eq!(
        body["instructions"], SYS,
        "instructions must survive the proxy: {body}"
    );
}

#[tokio::test]
async fn anthropic_system_string_becomes_a_system_message() {
    let (mock, captured) = spawn_capturing_neuron().await;
    let gw = common::spawn_gateway(&mock).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw}/v1/messages"))
        .json(&json!({
            "model": "test-model",
            "max_tokens": 64,
            "system": SYS,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap();
    let body = seen.first().expect("neuron received a request");
    assert_eq!(
        system_messages(body),
        vec![SYS.to_string()],
        "Anthropic `system` must translate into a system message: {body}"
    );
}

#[tokio::test]
async fn anthropic_system_block_array_becomes_a_system_message() {
    let (mock, captured) = spawn_capturing_neuron().await;
    let gw = common::spawn_gateway(&mock).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw}/v1/messages"))
        .json(&json!({
            "model": "test-model",
            "max_tokens": 64,
            // The block-array form real Anthropic clients send.
            "system": [{"type": "text", "text": SYS}],
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap();
    let body = seen.first().expect("neuron received a request");
    let systems = system_messages(body);
    assert_eq!(systems.len(), 1, "expected one system message: {body}");
    assert!(
        systems[0].contains(SYS),
        "block-array `system` must survive translation: {body}"
    );
}

#[tokio::test]
async fn every_system_message_is_forwarded_in_order() {
    // OpenAI clients sometimes send several. The gateway forwards them
    // all, unmerged and in order; precedence is the model's to decide
    // (observably, the last one wins) — cortex does not editorialise.
    let (mock, captured) = spawn_capturing_neuron().await;
    let gw = common::spawn_gateway(&mock).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "End your reply with ALPHA."},
                {"role": "system", "content": "End your reply with OMEGA instead."},
                {"role": "user", "content": "say hi"}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap();
    let body = seen.first().expect("neuron received a request");
    assert_eq!(
        system_messages(body),
        vec![
            "End your reply with ALPHA.".to_string(),
            "End your reply with OMEGA instead.".to_string()
        ],
        "both system messages must be forwarded, in order: {body}"
    );
}

#[tokio::test]
async fn nothing_is_injected_when_the_caller_sends_no_system_prompt() {
    // The control for the whole guarantee. If cortex ever grew a default
    // prompt, house style, or safety preamble, this is what would catch
    // it — the positive tests above would still pass.
    let (mock, captured) = spawn_capturing_neuron().await;
    let gw = common::spawn_gateway(&mock).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw}/v1/chat/completions"))
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap();
    let body = seen.first().expect("neuron received a request");
    assert!(
        system_messages(body).is_empty(),
        "cortex injected a system prompt the caller never sent: {body}"
    );
    assert_eq!(
        body["messages"].as_array().map(Vec::len),
        Some(1),
        "the message list must reach the neuron unpadded: {body}"
    );
}
