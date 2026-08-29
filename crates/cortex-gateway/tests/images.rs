//! `/v1/images/generations` proxy tests (#201).
//!
//! The mock neuron answers the images endpoint with a tiny valid
//! envelope; the gateway must route by model, forward verbatim, and
//! pass the response (including `usage.helexa_image_units`) through
//! untouched.

mod common;

use axum::Router;
use axum::extract::Path;
use axum::response::Json;
use axum::routing::{get, post};
use common::spawn_gateway;
use serde_json::{Value, json};
use tokio::net::TcpListener;

/// Mock neuron that serves the images endpoint for "test-model".
async fn spawn_images_mock_neuron() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();

    let app = Router::new()
        .route(
            "/models",
            get(|| async {
                Json(json!([
                    {"id": "test-model", "harness": "candle", "status": "loaded",
                     "devices": [0], "vram_used_mb": 14000,
                     "capabilities": ["image"], "tool_call": false, "reasoning": false}
                ]))
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
            "/v1/images/generations",
            post(|Json(body): Json<Value>| async move {
                // Echo enough of the request to prove verbatim forwarding.
                let seed = body.get("seed").cloned().unwrap_or(Value::Null);
                Json(json!({
                    "created": 1700000000_u64,
                    "data": [{"b64_json": "aGVsZXhh"}],
                    "usage": {
                        "helexa_image_units": 9.437184,
                        "helexa_timing": {
                            "encode_ms": 1673, "denoise_ms": 7894,
                            "decode_ms": 1835, "steps": 9, "cfg": false
                        }
                    },
                    "echo_seed": seed
                }))
            }),
        );

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    base_url
}

#[tokio::test]
async fn test_images_proxy_round_trip() {
    let mock_url = spawn_images_mock_neuron().await;
    let gateway_url = spawn_gateway(&mock_url).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/generations"))
        .json(&json!({
            "model": "test-model",
            "prompt": "a neon sign reading HELEXA",
            "size": "1024x1024",
            "seed": 42
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["data"][0]["b64_json"], "aGVsZXhh");
    // Metering unit passthrough (#202 reads this at the gateway).
    assert!(
        (body["usage"]["helexa_image_units"].as_f64().unwrap() - 9.437184).abs() < 1e-9,
        "helexa_image_units must pass through untouched"
    );
    assert_eq!(body["usage"]["helexa_timing"]["steps"], 9);
    // The request body reached the neuron verbatim.
    assert_eq!(body["echo_seed"], 42);
}

#[tokio::test]
async fn test_images_missing_model_field() {
    let mock_url = spawn_images_mock_neuron().await;
    let gateway_url = spawn_gateway(&mock_url).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/generations"))
        .json(&json!({"prompt": "no model here"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "missing_model_field");
}

#[tokio::test]
async fn test_images_unknown_model_404() {
    let mock_url = spawn_images_mock_neuron().await;
    let gateway_url = spawn_gateway(&mock_url).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/generations"))
        .json(&json!({"model": "no-such-model", "prompt": "x"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 404);
}

// ── image budget enforcement (#202) ──────────────────────────────

use cortex_core::config::{
    ApiKeyConfig, EntitlementsConfig, EvictionSettings, EvictionStrategy, GatewayConfig,
    GatewaySettings, NeuronEndpoint,
};
use cortex_core::entitlements::CapWindow;
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use std::sync::Arc;

async fn spawn_keyed_gateway(neuron_url: &str, hard_cap: u64) -> String {
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
        entitlements: EntitlementsConfig {
            require_auth: true,
            keys: vec![ApiKeyConfig {
                key: "sk-img".into(),
                account_id: "acct-img".into(),
                key_id: Some("key-img".into()),
                hard_cap: Some(hard_cap),
                window: CapWindow::Balance,
            }],
        },
        upstream: Default::default(),
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
                vram_estimate_mb: Some(14000),
                capabilities: vec!["image".into()],
                tool_call: false,
                reasoning: false,
                limit: None,
                servable: None,
                reasoning_budget: Vec::new(),
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

#[tokio::test]
async fn test_images_budget_rejected_before_dispatch() {
    let mock_url = spawn_images_mock_neuron().await;
    // A 1024²/9-step image reserves ~9.44 units ≈ 9438 tokens; a cap of
    // 100 tokens must fail-close before the neuron is touched.
    let gateway_url = spawn_keyed_gateway(&mock_url, 100).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/generations"))
        .bearer_auth("sk-img")
        .json(&json!({"model": "test-model", "prompt": "a cat"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "insufficient_quota");
}

#[tokio::test]
async fn test_images_within_budget_succeeds_and_settles() {
    let mock_url = spawn_images_mock_neuron().await;
    // Plenty of budget: reservation ~9438 tokens, settle at actual
    // 9.437184 units ≈ 9438 tokens.
    let gateway_url = spawn_keyed_gateway(&mock_url, 1_000_000).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/generations"))
        .bearer_auth("sk-img")
        .json(&json!({"model": "test-model", "prompt": "a cat", "seed": 42}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["data"][0]["b64_json"], "aGVsZXhh");
}

#[tokio::test]
async fn test_images_cfg_doubles_reservation() {
    let mock_url = spawn_images_mock_neuron().await;
    // Cap sits between the plain (9.44u ≈ 9438t) and CFG (18.88u ≈
    // 18875t) reservations: plain passes, CFG fail-closes.
    let gateway_url = spawn_keyed_gateway(&mock_url, 12_000).await;
    let client = reqwest::Client::new();

    let cfg_resp = client
        .post(format!("{gateway_url}/v1/images/generations"))
        .bearer_auth("sk-img")
        .json(&json!({
            "model": "test-model", "prompt": "a cat",
            "negative_prompt": "blurry"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(cfg_resp.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);

    let plain_resp = client
        .post(format!("{gateway_url}/v1/images/generations"))
        .bearer_auth("sk-img")
        .json(&json!({"model": "test-model", "prompt": "a cat"}))
        .send()
        .await
        .unwrap();
    assert_eq!(plain_resp.status(), 200);
}
