//! End-to-end preflight tests against a mock HF-compatible server.
//!
//! Unit tests in `harness/preflight.rs` exercise the classifier and
//! feasibility table against synthetic file lists. These tests close
//! the loop: spawn an axum server that returns a `RepoInfo`-shaped
//! JSON payload at `/api/models/{org}/{name}`, point `hf_hub::Api` at
//! it, and assert `preflight()` returns the expected outcome.

use axum::Router;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::get;
use cortex_core::harness::ModelSpec;
use neuron::harness::preflight::{PreflightError, SourceFormat, preflight};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::Mutex;

/// Per-test mock state: a map from `{org}/{name}` to the JSON body the
/// mock server returns at the corresponding `/api/models/{org}/{name}`
/// endpoint. `None` means "respond 404".
type MockBodies = Arc<Mutex<std::collections::HashMap<String, Option<Value>>>>;

async fn spawn_mock(bodies: MockBodies) -> String {
    // hf-hub 0.4 calls /api/models/{org}/{name}/revision/main for
    // `repo.info()`. We route both shapes so the test stays robust
    // to a future hf-hub upgrade that drops the `/revision/main`
    // suffix.
    let app = Router::new()
        .route("/api/models/{org}/{name}", get(model_info))
        .route(
            "/api/models/{org}/{name}/revision/{rev}",
            get(model_info_rev),
        )
        .with_state(bodies);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn model_info(
    Path((org, name)): Path<(String, String)>,
    axum::extract::State(bodies): axum::extract::State<MockBodies>,
) -> impl IntoResponse {
    respond(&format!("{org}/{name}"), &bodies)
}

async fn model_info_rev(
    Path((org, name, _rev)): Path<(String, String, String)>,
    axum::extract::State(bodies): axum::extract::State<MockBodies>,
) -> impl IntoResponse {
    respond(&format!("{org}/{name}"), &bodies)
}

fn respond(key: &str, bodies: &MockBodies) -> axum::response::Response {
    let entry = bodies.lock().unwrap().get(key).cloned();
    match entry {
        Some(Some(body)) => Json(body).into_response(),
        Some(None) | None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn build_api(endpoint: &str, cache_dir: &std::path::Path) -> hf_hub::api::tokio::Api {
    hf_hub::api::tokio::ApiBuilder::new()
        .with_endpoint(endpoint.to_string())
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .expect("build hf-hub Api")
}

fn siblings(filenames: &[&str]) -> Value {
    json!({
        "sha": "0000000000000000000000000000000000000000",
        "siblings": filenames.iter().map(|f| json!({ "rfilename": f })).collect::<Vec<_>>(),
    })
}

fn spec(model_id: &str, tp: Option<u32>, quant: Option<&str>) -> ModelSpec {
    ModelSpec {
        model_id: model_id.into(),
        harness: "candle".into(),
        quant: quant.map(String::from),
        tensor_parallel: tp,
        devices: None,
    }
}

#[tokio::test]
async fn preflight_gguf_tp_rejected_over_http() {
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    bodies.lock().unwrap().insert(
        "HauhauCS/Qwen3.6".to_string(),
        Some(siblings(&[
            "README.md",
            ".gitattributes",
            "Qwen3.6-Q4_K_P.gguf",
            "Qwen3.6-Q6_K_P.gguf",
            "Qwen3.6-Q8_K_P.gguf",
        ])),
    );
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    let s = spec("HauhauCS/Qwen3.6", Some(2), Some("q6k"));
    let err = preflight(&api, &s).await.unwrap_err();
    match err {
        PreflightError::TpRequiresSafetensors {
            model_id,
            tp_size,
            gguf_quants,
            ..
        } => {
            assert_eq!(model_id, "HauhauCS/Qwen3.6");
            assert_eq!(tp_size, 2);
            assert_eq!(gguf_quants.len(), 3);
        }
        other => panic!("expected TpRequiresSafetensors, got {other:?}"),
    }
}

#[tokio::test]
async fn preflight_gguf_quant_suggestion_over_http() {
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    bodies.lock().unwrap().insert(
        "HauhauCS/Qwen3.6".to_string(),
        Some(siblings(&[
            "Qwen3.6-Q4_K_P.gguf",
            "Qwen3.6-Q5_K_P.gguf",
            "Qwen3.6-Q6_K_P.gguf",
            "Qwen3.6-Q8_K_P.gguf",
        ])),
    );
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    let s = spec("HauhauCS/Qwen3.6", Some(1), Some("q6k"));
    let err = preflight(&api, &s).await.unwrap_err();
    match err {
        PreflightError::QuantNotFound {
            requested,
            nearest,
            available,
            ..
        } => {
            assert_eq!(requested, "q6k");
            assert_eq!(nearest.as_deref(), Some("q6_k_p"));
            assert_eq!(available.len(), 4);
        }
        other => panic!("expected QuantNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn preflight_dense_safetensors_tp_ok() {
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    bodies.lock().unwrap().insert(
        "Qwen/Q3-30B".to_string(),
        Some(siblings(&[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors",
        ])),
    );
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    let s = spec("Qwen/Q3-30B", Some(2), Some("q5k"));
    let plan = preflight(&api, &s).await.expect("dense+tp should succeed");
    assert_eq!(plan.tp_size, 2);
    assert!(plan.picked_quant_file.is_none());
    assert!(matches!(
        plan.format,
        SourceFormat::DenseSafetensors { sharded: true }
    ));
}

#[tokio::test]
async fn preflight_gguf_single_gpu_good_quant() {
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    bodies.lock().unwrap().insert(
        "HauhauCS/Qwen3.6".to_string(),
        Some(siblings(&["Qwen3.6-Q4_K_P.gguf", "Qwen3.6-Q6_K_P.gguf"])),
    );
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    let s = spec("HauhauCS/Qwen3.6", Some(1), Some("q6_k_p"));
    let plan = preflight(&api, &s)
        .await
        .expect("good quant should succeed");
    assert_eq!(plan.tp_size, 1);
    assert_eq!(
        plan.picked_quant_file.as_deref(),
        Some("Qwen3.6-Q6_K_P.gguf")
    );
}

#[tokio::test]
async fn preflight_repo_fetch_failed_on_404() {
    // Mock server has no entry for this id → 404, exercising the
    // RepoFetchFailed path (the same shape today's HauhauCS scenario
    // would have produced if we'd added preflight before the cache
    // download was attempted).
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    let s = spec("DoesNot/Exist", Some(1), None);
    let err = preflight(&api, &s).await.unwrap_err();
    assert!(
        matches!(err, PreflightError::RepoFetchFailed { .. }),
        "expected RepoFetchFailed, got {err:?}"
    );
}

#[tokio::test]
async fn preflight_empty_repo_rejected() {
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    bodies.lock().unwrap().insert(
        "Empty/Repo".to_string(),
        Some(siblings(&["README.md", "tokenizer.json"])),
    );
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    let s = spec("Empty/Repo", Some(1), None);
    let err = preflight(&api, &s).await.unwrap_err();
    assert!(
        matches!(err, PreflightError::EmptyRepo { .. }),
        "expected EmptyRepo, got {err:?}"
    );
}

#[tokio::test]
async fn preflight_mixed_repo_prefers_safetensors() {
    let cache = tempfile::tempdir().expect("tempdir");
    let bodies: MockBodies = Arc::new(Mutex::new(Default::default()));
    bodies.lock().unwrap().insert(
        "Mixed/Repo".to_string(),
        Some(siblings(&[
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "model-Q4_K_M.gguf",
        ])),
    );
    let endpoint = spawn_mock(bodies).await;

    let api = build_api(&endpoint, cache.path());
    // TP=2 + quant should succeed via the dense path even though a
    // GGUF is present — the dense path handles ISQ.
    let s = spec("Mixed/Repo", Some(2), Some("q5k"));
    let plan = preflight(&api, &s).await.expect("mixed should succeed");
    assert!(matches!(plan.format, SourceFormat::Mixed { .. }));
}
