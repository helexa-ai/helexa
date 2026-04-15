use cortex_core::discovery::{DeviceInfo, DiscoveryResponse};
use cortex_neuron::api::{self, NeuronState};
use cortex_neuron::harness::HarnessRegistry;
use cortex_neuron::health::HealthCache;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

async fn spawn_neuron(discovery: DiscoveryResponse) -> String {
    let health_cache = Arc::new(HealthCache::new());
    let registry = HarnessRegistry::new();

    let state = Arc::new(NeuronState {
        discovery,
        health_cache,
        registry: RwLock::new(registry),
    });

    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn fake_discovery() -> DiscoveryResponse {
    DiscoveryResponse {
        hostname: "test-node".into(),
        os: "Linux".into(),
        kernel: "6.19.0".into(),
        cuda_version: Some("12.8".into()),
        driver_version: Some("570.86.16".into()),
        devices: vec![
            DeviceInfo {
                index: 0,
                name: "NVIDIA GeForce RTX 5090".into(),
                vram_total_mb: 32614,
                compute_capability: "12.0".into(),
            },
            DeviceInfo {
                index: 1,
                name: "NVIDIA GeForce RTX 5090".into(),
                vram_total_mb: 32614,
                compute_capability: "12.0".into(),
            },
        ],
        harnesses: vec![],
    }
}

#[tokio::test]
async fn test_discovery_endpoint() {
    let url = spawn_neuron(fake_discovery()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{url}/discovery"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["hostname"], "test-node");
    assert_eq!(body["cuda_version"], "12.8");

    let devices = body["devices"].as_array().unwrap();
    assert_eq!(devices.len(), 2);
    assert_eq!(devices[0]["name"], "NVIDIA GeForce RTX 5090");
    assert_eq!(devices[0]["vram_total_mb"], 32614);
}

#[tokio::test]
async fn test_health_endpoint() {
    let url = spawn_neuron(fake_discovery()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{url}/health"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["uptime_secs"], 0);
}

#[tokio::test]
async fn test_discovery_no_gpus() {
    let disc = DiscoveryResponse {
        hostname: "cpu-only".into(),
        os: "Linux".into(),
        kernel: "6.19.0".into(),
        cuda_version: None,
        driver_version: None,
        devices: vec![],
        harnesses: vec![],
    };
    let url = spawn_neuron(disc).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{url}/discovery"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["hostname"], "cpu-only");
    assert!(body["cuda_version"].is_null());
    assert!(body["devices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_models_empty_registry() {
    let url = spawn_neuron(fake_discovery()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{url}/models"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body.as_array().unwrap().is_empty());
}

/// Spawn a mock mistral.rs backend and a neuron with the mistralrs harness
/// pointing at it, then test the full model lifecycle through neuron's API.
#[tokio::test]
async fn test_models_via_mistralrs_harness() {
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use cortex_core::harness::HarnessConfig;
    use serde_json::Value;

    // Mock mistral.rs backend.
    let mock_app = Router::new()
        .route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "data": [
                        {"id": "test-model", "status": "loaded"},
                        {"id": "other-model", "status": "unloaded"}
                    ]
                }))
            }),
        )
        .route(
            "/v1/models/unload",
            post(|Json(_body): Json<Value>| async { Json(json!({"status": "ok"})) }),
        )
        .route(
            "/v1/models/reload",
            post(|Json(_body): Json<Value>| async { Json(json!({"status": "ok"})) }),
        );

    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(mock_listener, mock_app).await.unwrap();
    });
    let mock_url = format!("http://{mock_addr}");

    // Build neuron with mistralrs harness pointing at mock.
    let registry = HarnessRegistry::from_configs(&[HarnessConfig {
        name: "mistralrs".into(),
        endpoint: Some(mock_url.clone()),
        systemd_unit: None,
    }]);

    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
    });

    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let neuron_addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let neuron_url = format!("http://{neuron_addr}");

    let client = reqwest::Client::new();

    // GET /models — should return models from mock mistralrs.
    let resp = client
        .get(format!("{neuron_url}/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let models: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0]["id"], "test-model");
    assert_eq!(models[0]["harness"], "mistralrs");
    assert_eq!(models[0]["status"], "loaded");
    assert_eq!(models[1]["id"], "other-model");
    assert_eq!(models[1]["status"], "unloaded");

    // GET /models/test-model/endpoint — should return mock URL.
    let resp = client
        .get(format!("{neuron_url}/models/test-model/endpoint"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["url"], mock_url);

    // POST /models/unload — should succeed.
    let resp = client
        .post(format!("{neuron_url}/models/unload"))
        .json(&json!({"model_id": "test-model"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "unloaded");

    // POST /models/load — should succeed.
    let resp = client
        .post(format!("{neuron_url}/models/load"))
        .json(&json!({
            "model_id": "test-model",
            "harness": "mistralrs"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "loaded");
}
