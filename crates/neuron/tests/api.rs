use cortex_core::discovery::{DeviceInfo, DiscoveryResponse};
use neuron::activation::ActivationTracker;
use neuron::api::{self, NeuronState};
use neuron::harness::HarnessRegistry;
use neuron::health::HealthCache;
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
        candle: None,
        activation: Arc::new(ActivationTracker::new(&[])),
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
        cuda_unavailable_reason: None,
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
        cuda_unavailable_reason: None,
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

/// Verify the candle harness registers, list is empty by default, and a
/// load attempt for an obviously-bogus model id returns a 4xx error
/// without crashing the daemon. Real load/unload exercising actual GGUF
/// download is covered by `tests/candle_lifecycle.rs` (cuda-integration).
#[tokio::test]
async fn test_candle_harness_registers_and_rejects_bogus_model() {
    use cortex_core::harness::HarnessConfig;
    use neuron::config::HarnessSettings;

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:13131",
        &HarnessSettings::default(),
    );

    let candle = registry.candle();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle,
        activation: Arc::new(ActivationTracker::new(&[])),
    });

    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let neuron_addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let neuron_url = format!("http://{neuron_addr}");

    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{neuron_url}/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let models: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(models.is_empty());

    // Sending a wrong-harness spec should be rejected synchronously
    // without touching the network or the model registry.
    let resp = client
        .post(format!("{neuron_url}/models/load"))
        .json(&json!({"model_id": "definitely/not-real", "harness": "not-candle"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // Registry still empty.
    let resp = client
        .get(format!("{neuron_url}/models"))
        .send()
        .await
        .unwrap();
    let models: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(models.is_empty());
}

/// `/v1/chat/completions` returns 503 when no candle harness is registered.
#[tokio::test]
async fn test_chat_completions_no_candle_harness() {
    let registry = HarnessRegistry::new();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle: None,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

/// `/v1/chat/completions` returns 404 when the requested model isn't loaded.
#[tokio::test]
async fn test_chat_completions_model_not_loaded() {
    use cortex_core::harness::HarnessConfig;
    use neuron::config::HarnessSettings;

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );
    let candle = registry.candle();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "definitely/not-loaded",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// `/v1/chat/completions` with `stream: true` returns 404 when the
/// model isn't loaded — same surface as the non-streaming path. The
/// streaming code only kicks in once the model lookup succeeds.
#[tokio::test]
async fn test_chat_completions_streaming_model_not_loaded() {
    use cortex_core::harness::HarnessConfig;
    use neuron::config::HarnessSettings;

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );
    let candle = registry.candle();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "definitely/not-loaded",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── /v1/responses ────────────────────────────────────────────────────

/// `/v1/responses` returns 503 when no candle harness is registered —
/// matches the chat-completions error shape so a client can swap
/// endpoints without re-handling 503s.
#[tokio::test]
async fn test_responses_no_candle_harness() {
    let registry = HarnessRegistry::new();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle: None,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/responses"))
        .json(&json!({"model": "anything", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

/// `previous_response_id` is rejected at translate time with 400 —
/// we don't store responses server-side yet, so chained
/// conversations can't be honoured.
#[tokio::test]
async fn test_responses_rejects_previous_response_id() {
    use cortex_core::harness::HarnessConfig;
    use neuron::config::HarnessSettings;

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );
    let candle = registry.candle();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/responses"))
        .json(&json!({
            "model": "anything",
            "input": "hi",
            "previous_response_id": "resp_prev_42"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "chained_conversation_not_supported");
}

/// `/v1/responses` returns 404 when the model isn't loaded — same
/// surface as chat completions.
#[tokio::test]
async fn test_responses_model_not_loaded() {
    use cortex_core::harness::HarnessConfig;
    use neuron::config::HarnessSettings;

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );
    let candle = registry.candle();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/responses"))
        .json(&json!({"model": "not-loaded", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// Same model-not-loaded surface on the streaming path. The
/// stream is opened only after model lookup succeeds, so a
/// missing model fails fast with a non-SSE 404 response.
#[tokio::test]
async fn test_responses_streaming_model_not_loaded() {
    use cortex_core::harness::HarnessConfig;
    use neuron::config::HarnessSettings;

    let registry = HarnessRegistry::from_configs(
        &[HarnessConfig {
            name: "candle".into(),
        }],
        "http://localhost:0",
        &HarnessSettings::default(),
    );
    let candle = registry.candle();
    let health_cache = Arc::new(HealthCache::new());
    let state = Arc::new(NeuronState {
        discovery: fake_discovery(),
        health_cache,
        registry: RwLock::new(registry),
        candle,
        activation: Arc::new(ActivationTracker::new(&[])),
    });
    let app = api::neuron_routes().with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/responses"))
        .json(&json!({
            "model": "not-loaded",
            "input": "hi",
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_driver_mismatch_rejects_load_and_rides_discovery() {
    // #19: a host with the driver/library mismatch advertises the
    // reason on /discovery (so cortex routes around it) and fast-
    // rejects /models/load with 503 + the actionable message instead
    // of dying minutes later inside cuInit/NCCL.
    let reason = "host NVIDIA driver/library mismatch (userspace NVML 580.159 vs loaded \
                  kernel module 580.159.03) — reboot the host to reload the kernel module; \
                  all CUDA inference is unavailable until then";
    let disc = DiscoveryResponse {
        hostname: "mismatched".into(),
        os: "Linux".into(),
        kernel: "6.19.0".into(),
        cuda_version: Some("13.0".into()),
        driver_version: None,
        devices: vec![],
        harnesses: vec!["candle".into()],
        cuda_unavailable_reason: Some(reason.into()),
    };
    let url = spawn_neuron(disc).await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{url}/discovery"))
        .send()
        .await
        .expect("discovery request")
        .json()
        .await
        .unwrap();
    assert_eq!(body["cuda_unavailable_reason"], reason);

    let resp = client
        .post(format!("{url}/models/load"))
        .json(&serde_json::json!({
            "model_id": "Qwen/Qwen3.6-27B",
            "harness": "candle",
            "quant": "q6k",
            "tensor_parallel": 2
        }))
        .send()
        .await
        .expect("load request");
    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "cuda_unavailable");
    assert!(
        body["error"].as_str().unwrap().contains("reboot the host"),
        "error must be operator-actionable: {body}"
    );
}

#[tokio::test]
async fn test_healthy_discovery_omits_cuda_unavailable_reason() {
    // No false positives: the field must be absent (not null) from the
    // wire format on healthy hosts.
    let url = spawn_neuron(fake_discovery()).await;
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("{url}/discovery"))
        .send()
        .await
        .expect("discovery request")
        .json()
        .await
        .unwrap();
    assert!(
        body.get("cuda_unavailable_reason").is_none(),
        "healthy host must omit the field entirely: {body}"
    );
}
