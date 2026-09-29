//! `/v1/systemone` decision proxy tests (#338).
//!
//! The mock neuron echoes what it received — the `model` field and the
//! routed-model header — so each test can assert both halves of the
//! contract: *where* cortex sent the request, and that it did not
//! rewrite the body on the way.

use axum::Router;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use cortex_core::catalogue::{DECISION_DEFAULT_ALIAS, DECISION_MODEL_HEADER};
use cortex_core::config::{
    ApiKeyConfig, EntitlementsConfig, EvictionSettings, EvictionStrategy, GatewayConfig,
    GatewaySettings, NeuronEndpoint,
};
use cortex_core::entitlements::CapWindow;
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::net::TcpListener;

const DECISION_MODEL: &str = "convaiinnovations/laya";
const TEXT_MODEL: &str = "Qwen/Qwen3-1.7B";

/// Mock neuron: `/v1/systemone` answers with a Jev-shaped body that
/// echoes the forwarded `model` and routed-model header, and reports
/// `usage.input_tokens` = 41 per question.
async fn spawn_decision_mock_neuron() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let inference_url = base_url.clone();
    let app = Router::new()
        .route(
            "/models/{*rest}",
            get(move || {
                let url = inference_url.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .route(
            "/v1/systemone",
            post(|headers: HeaderMap, Json(body): Json<Value>| async move {
                let routed = headers
                    .get(DECISION_MODEL_HEADER)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let questions = body
                    .get("questions")
                    .and_then(Value::as_object)
                    .map(|m| m.len() as u64)
                    .unwrap_or(0);
                (
                    [("server-timing", "inference;dur=12.50")],
                    Json(json!({
                        "model": "laya-rl-agent",
                        "answers": {},
                        "usage": {"input_tokens": 41 * questions, "output_tokens": 0},
                        "echo_model": body.get("model").cloned().unwrap_or(Value::Null),
                        "echo_routed": routed,
                    })),
                )
                    .into_response()
            }),
        );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    base_url
}

fn entry(id: &str, capabilities: &[&str]) -> ModelEntry {
    ModelEntry {
        id: id.into(),
        status: ModelStatus::Loaded,
        last_accessed: None,
        vram_estimate_mb: Some(1000),
        capabilities: capabilities.iter().map(|c| c.to_string()).collect(),
        tool_call: false,
        reasoning: false,
        limit: None,
        servable: None,
        reasoning_budget: Vec::new(),
    }
}

/// A gateway fronting the mock with one decision and one text model
/// loaded. `default_alias` controls whether `helexa/one` is configured;
/// `hard_cap` switches on keyed auth with that balance.
async fn spawn_gateway(neuron_url: &str, default_alias: bool, hard_cap: Option<u64>) -> String {
    let entitlements = match hard_cap {
        Some(cap) => EntitlementsConfig {
            require_auth: true,
            keys: vec![ApiKeyConfig {
                key: "sk-dec".into(),
                account_id: "acct-dec".into(),
                key_id: Some("key-dec".into()),
                hard_cap: Some(cap),
                window: CapWindow::Balance,
            }],
        },
        None => EntitlementsConfig::default(),
    };
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
        upstream: Default::default(),
    };
    let mut state = CortexState::from_config(&config);
    if default_alias {
        state
            .catalogue
            .aliases
            .insert(DECISION_DEFAULT_ALIAS.into(), DECISION_MODEL.into());
    }
    let fleet = Arc::new(state);
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").unwrap();
        node.healthy = true;
        node.models
            .insert(DECISION_MODEL.into(), entry(DECISION_MODEL, &["decision"]));
        node.models
            .insert(TEXT_MODEL.into(), entry(TEXT_MODEL, &[]));
    }
    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn send(gateway: &str, body: Value) -> (u16, HeaderMap, Value) {
    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/systemone"))
        .json(&body)
        .send()
        .await
        .expect("request should reach the gateway");
    let status = resp.status().as_u16();
    let mut headers = HeaderMap::new();
    for (k, v) in resp.headers() {
        headers.insert(k.clone(), v.clone());
    }
    (status, headers, resp.json().await.unwrap())
}

fn request(model: Option<&str>) -> Value {
    let mut body = json!({
        "state": "I was charged twice this month",
        "questions": {"refund": {"type": "noul", "instructions": "Refund request?"}},
    });
    if let Some(m) = model {
        body["model"] = json!(m);
    }
    body
}

#[tokio::test]
async fn test_explicit_decision_model_routes_and_body_is_untouched() {
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, None).await;
    let (status, headers, body) = send(&gateway, request(Some(DECISION_MODEL))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["echo_routed"], DECISION_MODEL);
    assert_eq!(body["echo_model"], DECISION_MODEL);
    assert_eq!(body["usage"]["input_tokens"], 41);
    // Upstream timing headers survive the hop.
    assert_eq!(
        headers.get("server-timing").and_then(|v| v.to_str().ok()),
        Some("inference;dur=12.50")
    );
}

#[tokio::test]
async fn test_foreign_model_name_falls_back_to_default_alias() {
    // A Jev client names its own service. The request must still be
    // served — routed via helexa/one — and the client's value must reach
    // the node unchanged (it may be a checkpoint pin the node honours).
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, None).await;
    let (status, _, body) = send(&gateway, request(Some("jev-1"))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["echo_routed"], DECISION_MODEL);
    assert_eq!(body["echo_model"], "jev-1");
}

#[tokio::test]
async fn test_missing_model_falls_back_to_default_alias() {
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, None).await;
    let (status, _, body) = send(&gateway, request(None)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["echo_routed"], DECISION_MODEL);
    assert_eq!(body["echo_model"], Value::Null);
}

#[tokio::test]
async fn test_default_alias_itself_resolves() {
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, None).await;
    let (status, _, body) = send(&gateway, request(Some(DECISION_DEFAULT_ALIAS))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["echo_routed"], DECISION_MODEL);
}

#[tokio::test]
async fn test_known_non_decision_model_is_not_substituted() {
    // A model the fleet does serve routes to itself; the node owns the
    // wrong_modality answer. Silently swapping in the decision model
    // would hide the client's mistake.
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, None).await;
    let (status, _, body) = send(&gateway, request(Some(TEXT_MODEL))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["echo_routed"], TEXT_MODEL);
}

#[tokio::test]
async fn test_unknown_model_without_default_alias_is_404() {
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, false, None).await;
    let (status, _, body) = send(&gateway, request(Some("jev-1"))).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn test_budget_below_reservation_rejected_before_dispatch() {
    // One question reserves one full sequence (1024 tokens by default);
    // a 100-token balance cannot cover it.
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, Some(100)).await;
    let resp = reqwest::Client::new()
        .post(format!("{gateway}/v1/systemone"))
        .bearer_auth("sk-dec")
        .json(&request(Some(DECISION_MODEL)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "insufficient_quota");
}

#[tokio::test]
async fn test_spend_settles_to_reported_input_tokens() {
    // Balance for exactly two one-question requests at the reserved
    // bound (2 × 1024). Settling to the reported 41 tokens leaves room
    // for many more; settling at the bound would exhaust it on the third.
    let gateway = spawn_gateway(&spawn_decision_mock_neuron().await, true, Some(2048)).await;
    let client = reqwest::Client::new();
    for i in 0..5 {
        let resp = client
            .post(format!("{gateway}/v1/systemone"))
            .bearer_auth("sk-dec")
            .json(&request(Some(DECISION_MODEL)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "request {i} should be affordable");
        // Settlement is spawned; let it land before the next reservation.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}
