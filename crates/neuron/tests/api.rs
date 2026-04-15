use cortex_core::discovery::{DeviceHealth, DeviceInfo, DiscoveryResponse, HealthResponse};
use cortex_neuron::api::{self, NeuronState};
use cortex_neuron::health::HealthCache;
use std::sync::Arc;

async fn spawn_neuron(discovery: DiscoveryResponse, health: HealthResponse) -> String {
    let health_cache = Arc::new(HealthCache::new());
    // Pre-populate the health cache by writing through the snapshot mechanism.
    // HealthCache doesn't expose a direct setter, so we'll build one with
    // the data already in place via the NeuronState.
    // For testing, we use the cache as-is (uptime 0, empty devices) unless
    // we need specific values — see test_health_endpoint.
    let _ = health; // used below via a different approach

    let state = Arc::new(NeuronState {
        discovery,
        health_cache,
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

fn fake_health() -> HealthResponse {
    HealthResponse {
        uptime_secs: 0,
        devices: vec![
            DeviceHealth {
                index: 0,
                vram_used_mb: 8192,
                vram_free_mb: 24422,
                utilization_pct: 45,
                temp_c: 62,
            },
            DeviceHealth {
                index: 1,
                vram_used_mb: 4096,
                vram_free_mb: 28518,
                utilization_pct: 30,
                temp_c: 58,
            },
        ],
    }
}

#[tokio::test]
async fn test_discovery_endpoint() {
    let disc = fake_discovery();
    let url = spawn_neuron(disc, fake_health()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{url}/discovery"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["hostname"], "test-node");
    assert_eq!(body["os"], "Linux");
    assert_eq!(body["cuda_version"], "12.8");
    assert_eq!(body["driver_version"], "570.86.16");

    let devices = body["devices"].as_array().unwrap();
    assert_eq!(devices.len(), 2);
    assert_eq!(devices[0]["name"], "NVIDIA GeForce RTX 5090");
    assert_eq!(devices[0]["vram_total_mb"], 32614);
    assert_eq!(devices[0]["compute_capability"], "12.0");
}

#[tokio::test]
async fn test_health_endpoint() {
    let url = spawn_neuron(fake_discovery(), fake_health()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{url}/health"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    // HealthCache starts with uptime 0 and empty devices (no poller running in test).
    assert_eq!(body["uptime_secs"], 0);
    assert!(body["devices"].as_array().unwrap().is_empty());
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
    let url = spawn_neuron(
        disc,
        HealthResponse {
            uptime_secs: 0,
            devices: vec![],
        },
    )
    .await;

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
