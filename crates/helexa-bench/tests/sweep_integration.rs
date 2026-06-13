//! End-to-end sweep against a mock neuron: a sweep records samples, a
//! second sweep skips the satisfied cell, and bumping the reported build
//! SHA resumes fresh sampling.

use axum::Router;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use helexa_bench::config::{BenchConfig, BenchSettings, ScenarioConfig, TargetConfig, TargetKind};
use helexa_bench::sweep::Sweeper;
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct MockState {
    sha: Arc<Mutex<String>>,
}

async fn version(State(s): State<MockState>) -> Json<serde_json::Value> {
    let sha = s.sha.lock().unwrap().clone();
    Json(json!({
        "package_version": "0.1.16",
        "git_sha": sha,
        "git_dirty": false,
        "features": ["cuda", "cudnn"],
        "candle_version": "0.10.2",
    }))
}

async fn discovery() -> Json<serde_json::Value> {
    Json(json!({
        "hostname": "mock-beast",
        "os": "Linux",
        "kernel": "6.19.0",
        "cuda_version": "13.0",
        "driver_version": "580.159",
        "devices": [{"index": 0, "name": "RTX 5090", "vram_total_mb": 32614, "compute_capability": "12.0"}],
        "harnesses": ["candle"],
    }))
}

async fn models() -> Json<serde_json::Value> {
    Json(json!([
        {"id": "Qwen/Qwen3.6-27B", "harness": "candle", "status": "loaded", "devices": [0], "capabilities": ["text"]},
        // A non-warm model the bench must ignore.
        {"id": "Qwen/cold", "harness": "candle", "status": "recovering", "devices": [0]},
    ]))
}

async fn chat() -> impl IntoResponse {
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":130,\"completion_tokens\":2,\"total_tokens\":132}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

async fn spawn_mock(sha: &str) -> (String, Arc<Mutex<String>>) {
    let shared = Arc::new(Mutex::new(sha.to_string()));
    let state = MockState {
        sha: shared.clone(),
    };
    let app = Router::new()
        .route("/version", get(version))
        .route("/discovery", get(discovery))
        .route("/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), shared)
}

fn config_for(endpoint: String, db_path: String) -> BenchConfig {
    BenchConfig {
        bench: BenchSettings {
            sweep_interval_secs: 1,
            samples_per_version: 2,
            iteration_pause_secs: 0,
            request_timeout_secs: 30,
            db_path,
        },
        scenarios: ScenarioConfig {
            prompt_sizes: vec![128], // single scenario keeps assertions simple
            max_tokens: 16,
        },
        targets: vec![TargetConfig {
            name: "mock".into(),
            kind: TargetKind::Neuron,
            endpoint,
            label: None,
        }],
    }
}

#[tokio::test]
async fn sweep_records_skips_and_resumes_on_new_sha() {
    let (endpoint, sha_handle) = spawn_mock("aaaaaaa").await;

    // Unique db path per run (bound port is unique).
    let port = endpoint.rsplit(':').next().unwrap();
    let db_path = std::env::temp_dir().join(format!("helexa-bench-it-{port}.sqlite"));
    let _ = std::fs::remove_file(&db_path);
    let db_str = db_path.to_string_lossy().to_string();

    let sweeper = Sweeper::new(config_for(endpoint, db_str)).unwrap();

    // First sweep: one warm model × one scenario × 2 samples.
    let s1 = sweeper.run_once().await.unwrap();
    assert_eq!(s1.measured, 2, "should record samples_per_version samples");
    assert_eq!(s1.skipped, 0);
    assert_eq!(s1.failed, 0);

    // Second sweep at same SHA: cell satisfied, nothing measured.
    let s2 = sweeper.run_once().await.unwrap();
    assert_eq!(s2.measured, 0, "satisfied cell must be skipped");
    assert_eq!(s2.skipped, 1);

    // Bump the reported build SHA: a new cell → fresh sampling resumes.
    *sha_handle.lock().unwrap() = "bbbbbbb".to_string();
    let s3 = sweeper.run_once().await.unwrap();
    assert_eq!(s3.measured, 2, "new SHA must resume sampling");
    assert_eq!(s3.skipped, 0);

    let _ = std::fs::remove_file(&db_path);
}
