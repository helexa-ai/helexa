//! Fail-fast prompt pre-validation + advisory client hints (#56).
//!
//! cortex refuses a prompt that already exceeds the model's advertised
//! context window before dispatching to neuron — the same #60
//! `context_length_exceeded` envelope neuron would emit, just earlier — and
//! attaches an advisory `X-Helexa-Advice` header for fingerprinted clients.

use axum::Json;
use axum::extract::Path;
use axum::routing::{get, post};
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::harness::ModelLimit;
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::TcpListener;

/// Mock neuron with a hit counter, so a test can prove a request was (or
/// wasn't) dispatched past the gateway's pre-validation.
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
            post(move || {
                let sink = Arc::clone(&sink);
                async move {
                    sink.fetch_add(1, Ordering::SeqCst);
                    Json(json!({
                        "id": "c", "object": "chat.completion", "created": 1_700_000_000_u64,
                        "model": "test-model",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
                    }))
                }
            }),
        );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (base_url, hits)
}

/// Gateway over one neuron with `test-model` loaded and a tiny advertised
/// context window (so a modest prompt overflows it).
async fn spawn_gateway(neuron: &str, context: usize) -> String {
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
            endpoint: neuron.to_string(),
        }],
        models_config: "/dev/null".into(),
        entitlements: Default::default(),
    };
    let fleet = Arc::new(CortexState::from_config(&config));
    {
        let mut nodes = fleet.nodes.write().await;
        let n = nodes.get_mut("mock-node").unwrap();
        n.healthy = true;
        n.models.insert(
            "test-model".into(),
            ModelEntry {
                id: "test-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: Some(ModelLimit {
                    context,
                    input: None,
                    output: 16,
                }),
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
async fn over_long_prompt_is_rejected_before_dispatch() {
    let (neuron, hits) = spawn_counting_neuron().await;
    let gateway = spawn_gateway(&neuron, 50).await; // tiny 50-token window

    // ~1200 chars → ~300 est tokens, well over 50.
    let big = "word ".repeat(240);
    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .header("user-agent", "litellm/1.0")
        .json(&json!({"model": "test-model", "messages": [{"role": "user", "content": big}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    // Advisory hint for the fingerprinted client (header only, never body).
    assert!(
        resp.headers().get("x-helexa-advice").is_some(),
        "litellm should get advice"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    assert_eq!(body["error"]["max"], 50);
    // Refused at the edge — neuron never saw it.
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn within_context_passes_through() {
    let (neuron, hits) = spawn_counting_neuron().await;
    let gateway = spawn_gateway(&neuron, 4096).await;

    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        .json(&json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let _ = resp.bytes().await.unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1, "served by neuron");
}

#[tokio::test]
async fn unknown_client_gets_no_advice_header() {
    let (neuron, _hits) = spawn_counting_neuron().await;
    let gateway = spawn_gateway(&neuron, 50).await;

    let big = "word ".repeat(240);
    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/chat/completions"))
        // no/unknown User-Agent → no advice, but still a clean 400
        .json(&json!({"model": "test-model", "messages": [{"role": "user", "content": big}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(resp.headers().get("x-helexa-advice").is_none());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "context_length_exceeded");
}
