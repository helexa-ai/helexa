mod common;

use serde_json::json;

#[tokio::test]
async fn test_chat_completion_proxy() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("valid JSON response");
    assert_eq!(body["id"], "chatcmpl-test-001");
    assert_eq!(body["model"], "test-model");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from mock backend"
    );
    assert_eq!(body["usage"]["total_tokens"], 15);
}

#[tokio::test]
async fn test_health_endpoint() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{gw_url}/health"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["nodes"]["healthy"], 1);
    assert_eq!(body["nodes"]["total"], 1);
}

#[tokio::test]
async fn test_list_models() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{gw_url}/v1/models"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");

    let data = body["data"].as_array().expect("data should be an array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "test-model");
}

#[tokio::test]
async fn test_model_not_found() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 404);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not found")
    );
}

#[tokio::test]
async fn test_no_healthy_nodes() {
    let config = cortex_core::config::GatewayConfig {
        gateway: cortex_core::config::GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: cortex_core::config::EvictionSettings {
            strategy: cortex_core::config::EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![cortex_core::config::NeuronEndpoint {
            name: "dead-node".into(),
            endpoint: "http://127.0.0.1:1".into(),
        }],
        models_config: "/dev/null".into(),
    };
    let fleet = std::sync::Arc::new(cortex_gateway::state::CortexState::from_config(&config));

    let app = cortex_gateway::build_app(fleet);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "any-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 503);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no healthy nodes")
    );
}

#[tokio::test]
async fn test_missing_model_field() {
    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("model"));
}

#[tokio::test]
async fn test_recovering_model_returns_503_and_stays_listed() {
    // #20: while a model auto-recovers on a neuron, the gateway must
    // hold the route — transient 503 ("retry shortly"), not the 404
    // "not found on any node" that makes a recovering model look
    // evicted — and keep listing it on /v1/models.
    let mock_url = common::spawn_mock_neuron().await;
    let (fleet, gw_url) = common::spawn_gateway_with_state(&mock_url).await;

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").expect("node must exist");
        node.models.insert(
            "recovering-model".into(),
            cortex_core::node::ModelEntry {
                id: "recovering-model".into(),
                status: cortex_core::node::ModelStatus::Recovering,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
                capabilities: Vec::new(),
            },
        );
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "recovering-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("recovering") && message.contains("retry"),
        "503 body must say recovering/retry, got: {message}"
    );

    // The model must still be visible on the unified models endpoint.
    let models: serde_json::Value = client
        .get(format!("{gw_url}/v1/models"))
        .send()
        .await
        .expect("models request should succeed")
        .json()
        .await
        .unwrap();
    let listed = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["id"] == "recovering-model");
    assert!(listed, "recovering model must stay listed on /v1/models");
}
