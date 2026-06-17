mod common;

use serde_json::json;

#[tokio::test]
async fn error_response_model_not_found() {
    let neuron_url = common::spawn_mock_neuron().await;
    let gateway_url = common::spawn_gateway(&neuron_url).await;

    let client = reqwest::Client::new();

    // Request a model that isn't loaded on the mock neuron.
    let resp = client
        .post(format!("{gateway_url}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);

    let body: serde_json::Value = resp.json().await.expect("valid json");
    let err = body.get("error").expect("response has error object");

    // Broad type categorization
    assert_eq!(err.get("type").unwrap(), "invalid_request_error");
    // Specific machine-readable code
    assert_eq!(
        err.get("code").unwrap().as_str().unwrap(),
        "model_not_found"
    );
    // param is always null
    assert!(err.get("param").unwrap().is_null());
}

#[tokio::test]
async fn error_response_missing_model_field() {
    let neuron_url = common::spawn_mock_neuron().await;
    let gateway_url = common::spawn_gateway(&neuron_url).await;

    let client = reqwest::Client::new();

    // Request without the required `model` field.
    let resp = client
        .post(format!("{gateway_url}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&json!({
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);

    let body: serde_json::Value = resp.json().await.expect("valid json");
    let err = body.get("error").expect("response has error object");

    assert_eq!(err.get("type").unwrap(), "invalid_request_error");
    assert_eq!(
        err.get("code").unwrap().as_str().unwrap(),
        "missing_model_field"
    );
    assert!(err.get("param").unwrap().is_null());
}

#[tokio::test]
async fn error_response_no_healthy_nodes() {
    use cortex_core::config::{EvictionSettings, GatewayConfig, GatewaySettings, NeuronEndpoint};
    use std::sync::Arc;

    // Create a gateway config with a neuron pointing at an unreachable port so no node is ever healthy.
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: cortex_core::config::EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![NeuronEndpoint {
            name: "dead-node".into(),
            endpoint: "http://127.0.0.1:1".into(),
        }],
        models_config: "/dev/null".into(),
        entitlements: Default::default(),
    };

    let fleet = Arc::new(cortex_gateway::state::CortexState::from_config(&config));

    let app = cortex_gateway::build_app(fleet);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Allow the poller a moment to mark the node unhealthy.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": "any-model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);

    // Transient 503 — the gateway advertises Retry-After so OpenAI-compatible
    // clients back off and retry rather than surfacing an opaque error (#63).
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .expect("transient 503 must carry Retry-After")
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(retry_after, "5");

    let body: serde_json::Value = resp.json().await.expect("valid json");
    let err = body.get("error").expect("response has error object");

    assert_eq!(err.get("type").unwrap(), "api_error");
    assert_eq!(
        err.get("code").unwrap().as_str().unwrap(),
        "service_unavailable"
    );
    assert!(err.get("param").unwrap().is_null());
}
