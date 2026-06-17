//! Integration tests for API-key auth + principal resolution (#49).
//!
//! Verifies the #63 rejection contract (401 invalid_api_key via the #60
//! envelope) and that an authenticated request reaches neuron carrying the
//! internal principal headers — while a client-supplied principal header is
//! stripped (anti-spoofing).

use axum::Json;
use axum::extract::Path;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use cortex_core::config::{
    ApiKeyConfig, EntitlementsConfig, EvictionSettings, EvictionStrategy, GatewayConfig,
    GatewaySettings, NeuronEndpoint,
};
use cortex_core::entitlements::{CapWindow, HEADER_ACCOUNT_ID, HEADER_KEY_ID};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

/// What the mock neuron observed on the inbound `/v1/chat/completions`
/// request: the principal headers cortex stamped (or didn't).
#[derive(Default)]
struct Seen {
    account_id: Option<String>,
    key_id: Option<String>,
}

/// Spawn a mock neuron that records the principal headers it receives and
/// returns a trivial chat completion. Returns (base_url, observed).
async fn spawn_capturing_neuron() -> (String, Arc<Mutex<Seen>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();
    let seen: Arc<Mutex<Seen>> = Arc::new(Mutex::new(Seen::default()));
    let sink = Arc::clone(&seen);

    let app = axum::Router::new()
        .route(
            "/models/{model_id}/endpoint",
            get(move |Path(_): Path<String>| {
                let url = inference_url.clone();
                async move { Json(json!({ "url": url })) }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let sink = Arc::clone(&sink);
                async move {
                    {
                        let mut s = sink.lock().unwrap();
                        s.account_id = headers
                            .get(HEADER_ACCOUNT_ID)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_string);
                        s.key_id = headers
                            .get(HEADER_KEY_ID)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_string);
                    }
                    let model = body.get("model").and_then(Value::as_str).unwrap_or("m");
                    Json(json!({
                        "id": "chatcmpl-auth-001",
                        "object": "chat.completion",
                        "created": 1700000000_u64,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
                    }))
                }
            }),
        )
        .with_state(());

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (base_url, seen)
}

/// Spawn a gateway with the given entitlements config, a single neuron, and
/// `test-model` seeded as loaded (build_app spawns no poller).
async fn spawn_gateway(neuron_url: &str, entitlements: EntitlementsConfig) -> String {
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![NeuronEndpoint {
            name: "mock-node".into(),
            endpoint: neuron_url.to_string(),
        }],
        models_config: "/dev/null".into(),
        entitlements,
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").unwrap();
        node.healthy = true;
        node.models.insert(
            "test-model".into(),
            ModelEntry {
                id: "test-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
            },
        );
    }

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn one_key_config(require_auth: bool) -> EntitlementsConfig {
    EntitlementsConfig {
        require_auth,
        keys: vec![ApiKeyConfig {
            key: "sk-good".into(),
            account_id: "acct-1".into(),
            key_id: Some("key-1".into()),
            hard_cap: None,
            window: CapWindow::Balance,
        }],
    }
}

fn chat_body() -> Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    })
}

#[tokio::test]
async fn missing_key_when_required_is_401_invalid_api_key() {
    let (neuron, _seen) = spawn_capturing_neuron().await;
    let gateway = spawn_gateway(&neuron, one_key_config(true)).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_api_key");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn invalid_key_is_401_even_when_auth_not_required() {
    let (neuron, seen) = spawn_capturing_neuron().await;
    // A present-but-wrong credential is always an error.
    let gateway = spawn_gateway(&neuron, one_key_config(false)).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .bearer_auth("sk-wrong")
        .json(&chat_body())
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_api_key");
    // Rejected before dispatch — neuron never saw the request.
    assert!(seen.lock().unwrap().account_id.is_none());
}

#[tokio::test]
async fn valid_key_reaches_neuron_with_principal_headers() {
    let (neuron, seen) = spawn_capturing_neuron().await;
    let gateway = spawn_gateway(&neuron, one_key_config(true)).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .bearer_auth("sk-good")
        // A spoofed principal header must be stripped, not forwarded.
        .header(HEADER_ACCOUNT_ID, "attacker")
        .json(&chat_body())
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let s = seen.lock().unwrap();
    assert_eq!(s.account_id.as_deref(), Some("acct-1"));
    assert_eq!(s.key_id.as_deref(), Some("key-1"));
}

#[tokio::test]
async fn anonymous_allowed_when_auth_not_required() {
    let (neuron, seen) = spawn_capturing_neuron().await;
    let gateway = spawn_gateway(&neuron, EntitlementsConfig::default()).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    // No principal resolved → no principal headers stamped.
    let s = seen.lock().unwrap();
    assert!(s.account_id.is_none());
    assert!(s.key_id.is_none());
}

#[tokio::test]
async fn health_is_public_even_when_auth_required() {
    let (neuron, _seen) = spawn_capturing_neuron().await;
    let gateway = spawn_gateway(&neuron, one_key_config(true)).await;

    let resp = reqwest::Client::new()
        .get(format!("{gateway}/health"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
}
