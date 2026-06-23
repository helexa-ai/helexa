//! Integration tests for budget enforcement (#52) — the A0 seatbelt.
//!
//! A reservation over the key's hard cap is refused *before* neuron is hit,
//! with the #63 code matching the cap-window semantics (rate_limit_exceeded
//! + Retry-After for a resetting window, insufficient_quota for a hard
//! balance). Spend never exceeds the cap. No 402, ever.

use axum::Json;
use axum::extract::Path;
use axum::routing::{get, post};
use cortex_core::config::{
    ApiKeyConfig, EntitlementsConfig, EvictionSettings, EvictionStrategy, GatewayConfig,
    GatewaySettings, NeuronEndpoint,
};
use cortex_core::entitlements::{CapWindow, Principal};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::TcpListener;

/// Mock neuron with a hit counter on the inference path, so a test can prove
/// a request was (or wasn't) dispatched.
async fn spawn_counting_neuron() -> (String, Arc<AtomicU64>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();
    let hits = Arc::new(AtomicU64::new(0));
    let sink = Arc::clone(&hits);

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
            post(move |Json(body): Json<Value>| {
                let sink = Arc::clone(&sink);
                async move {
                    sink.fetch_add(1, Ordering::SeqCst);
                    let model = body.get("model").and_then(Value::as_str).unwrap_or("m");
                    Json(json!({
                        "id": "chatcmpl-budget",
                        "object": "chat.completion",
                        "created": 1700000000_u64,
                        "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                    }))
                }
            }),
        );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (base_url, hits)
}

async fn spawn_gateway(neuron_url: &str, key: ApiKeyConfig) -> (Arc<CortexState>, String) {
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
            keys: vec![key],
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
    (fleet, format!("http://{addr}"))
}

fn key(window: CapWindow, hard_cap: u64) -> ApiKeyConfig {
    ApiKeyConfig {
        key: "sk-cap".into(),
        account_id: "acct-cap".into(),
        key_id: Some("key-cap".into()),
        hard_cap: Some(hard_cap),
        window,
    }
}

fn chat(max_tokens: u64) -> Value {
    json!({
        "model": "test-model",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}]
    })
}

#[tokio::test]
async fn balance_over_cap_is_429_insufficient_quota_before_dispatch() {
    let (neuron, hits) = spawn_counting_neuron().await;
    // Cap far below a single request's reservation (max_tokens 1000).
    let (_fleet, gateway) = spawn_gateway(&neuron, key(CapWindow::Balance, 10)).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .bearer_auth("sk-cap")
        .json(&chat(1000))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    // Hard balance → no Retry-After.
    assert!(resp.headers().get(reqwest::header::RETRY_AFTER).is_none());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "insufficient_quota");
    // Refused before dispatch — neuron never saw it.
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rolling_over_cap_is_429_rate_limited_with_retry_after() {
    let (neuron, hits) = spawn_counting_neuron().await;
    let (_fleet, gateway) =
        spawn_gateway(&neuron, key(CapWindow::Rolling { seconds: 3600 }, 10)).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .bearer_auth("sk-cap")
        .json(&chat(1000))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    let retry = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .expect("rolling-window rejection must carry Retry-After");
    assert!(retry.to_str().unwrap().parse::<u64>().unwrap() >= 1);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn within_cap_is_served() {
    let (neuron, hits) = spawn_counting_neuron().await;
    let (_fleet, gateway) = spawn_gateway(&neuron, key(CapWindow::Balance, 1_000_000)).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .bearer_auth("sk-cap")
        .json(&chat(50))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let _ = resp.bytes().await.unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a0_seatbelt_caps_a_runaway_fan_out() {
    // An Agent-Zero-style key with a modest cap: a burst of requests drains
    // it, then further requests are refused — the account stops draining and
    // spend never exceeds the cap.
    let (neuron, hits) = spawn_counting_neuron().await;
    let (fleet, gateway) = spawn_gateway(&neuron, key(CapWindow::Balance, 100)).await;
    let client = reqwest::Client::new();

    let mut ok = 0;
    let mut refused = 0;
    for _ in 0..20 {
        let resp = client
            .post(format!("{gateway}/v1/chat/completions"))
            .bearer_auth("sk-cap")
            .json(&chat(20))
            .send()
            .await
            .unwrap();
        match resp.status() {
            reqwest::StatusCode::OK => {
                ok += 1;
                let _ = resp.bytes().await.unwrap();
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                refused += 1;
                let body: Value = resp.json().await.unwrap();
                assert_eq!(body["error"]["code"], "insufficient_quota");
            }
            other => panic!("unexpected status {other}"),
        }
    }

    assert!(ok >= 1, "some requests should be served");
    assert!(refused >= 1, "the cap must eventually refuse the fan-out");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        ok,
        "refused requests never dispatched"
    );

    // Spend never exceeded the hard cap (reservation prevents overshoot).
    // Poll briefly for in-flight settles to land.
    let principal = Principal {
        account_id: "acct-cap".into(),
        key_id: "key-cap".into(),
    };
    for _ in 0..50 {
        let snap = fleet.entitlements.snapshot(&principal).await.unwrap();
        if snap.reserved == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let snap = fleet.entitlements.snapshot(&principal).await.unwrap();
    assert!(snap.spent <= 100, "spent {} exceeded cap", snap.spent);
}
