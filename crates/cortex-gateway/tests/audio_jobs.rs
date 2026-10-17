//! Async audio generation jobs (#382): submit → poll → content, cancel,
//! ownership, retries, busy back-off, and restart recovery, against a
//! scriptable mock neuron.

use axum::Router;
use axum::extract::Path;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, body::Bytes};
use cortex_core::audio::{AudioUsage, NeuronAudioResult, RESULT_HEADER};
use cortex_core::config::{
    ApiKeyConfig, AudioJobsConfig, EntitlementsConfig, EvictionSettings, EvictionStrategy,
    GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::net::TcpListener;

const MODEL: &str = "music-model";
const WAV: &[u8] = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio-bytes";

/// How the mock neuron answers `POST /v1/audio/generations`.
#[derive(Clone, Copy)]
enum Mode {
    /// Render immediately.
    Ok,
    /// Answer 503 + Retry-After for the first N calls, then render.
    BusyThenOk(usize),
    /// Answer 500 for the first N calls, then render.
    FailThenOk(usize),
    /// Always 500.
    AlwaysFail,
    /// Always 422 wrong_modality.
    Reject,
    /// Sleep this long, then render.
    Slow(Duration),
    /// Always 503.
    AlwaysBusy,
}

struct Mock {
    url: String,
    calls: Arc<AtomicUsize>,
    /// Account header seen on each render call.
    accounts: Arc<std::sync::Mutex<Vec<Option<String>>>>,
}

async fn spawn_mock(mode: Mode) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let accounts = Arc::new(std::sync::Mutex::new(Vec::new()));
    let endpoint = url.clone();
    let (c, a) = (Arc::clone(&calls), Arc::clone(&accounts));
    let app = Router::new()
        .route(
            "/models/{id}/endpoint",
            get(move |Path(_id): Path<String>| {
                let url = endpoint.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .route(
            "/v1/audio/generations",
            post(move |headers: HeaderMap, body: Bytes| {
                let (c, a) = (Arc::clone(&c), Arc::clone(&a));
                async move { render(mode, &c, &a, &headers, &body).await }
            }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Mock {
        url,
        calls,
        accounts,
    }
}

async fn render(
    mode: Mode,
    calls: &AtomicUsize,
    accounts: &std::sync::Mutex<Vec<Option<String>>>,
    headers: &HeaderMap,
    body: &[u8],
) -> Response {
    let n = calls.fetch_add(1, Ordering::SeqCst);
    accounts.lock().unwrap().push(
        headers
            .get(cortex_core::entitlements::HEADER_ACCOUNT_ID)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string),
    );
    let busy = || {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", "1")],
            Json(json!({"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "busy"}})),
        )
            .into_response()
    };
    let fail = || {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"type": "api_error", "code": "inference_failed", "message": "boom"}})),
        )
            .into_response()
    };
    match mode {
        Mode::BusyThenOk(k) if n < k => return busy(),
        Mode::AlwaysBusy => return busy(),
        Mode::FailThenOk(k) if n < k => return fail(),
        Mode::AlwaysFail => return fail(),
        Mode::Reject => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": {"type": "invalid_request_error", "code": "wrong_modality", "message": "not an audio model"}})),
            )
                .into_response();
        }
        Mode::Slow(d) => tokio::time::sleep(d).await,
        _ => {}
    }
    let req: Value = serde_json::from_slice(body).unwrap();
    let duration = req["duration"].as_f64().unwrap_or(30.0);
    let result = NeuronAudioResult {
        duration,
        sample_rate: 48_000,
        channels: 2,
        seed: req["seed"].as_u64().unwrap_or(7),
        usage: AudioUsage {
            helexa_audio_units: duration,
            helexa_timing: None,
        },
    };
    (
        StatusCode::OK,
        [
            ("content-type", "audio/wav".to_string()),
            (RESULT_HEADER, serde_json::to_string(&result).unwrap()),
        ],
        WAV,
    )
        .into_response()
}

fn config(neuron: &str, dir: &std::path::Path) -> GatewayConfig {
    GatewayConfig {
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
        entitlements: EntitlementsConfig {
            require_auth: false,
            keys: ["alice", "bob"]
                .iter()
                .map(|who| ApiKeyConfig {
                    key: format!("key-{who}"),
                    account_id: format!("acct-{who}"),
                    key_id: None,
                    hard_cap: None,
                    window: Default::default(),
                })
                .collect(),
        },
        upstream: Default::default(),
        audio_jobs: AudioJobsConfig {
            dir: dir.to_string_lossy().into_owned(),
            queue_timeout_secs: 20,
            ..Default::default()
        },
    }
}

async fn gateway(neuron: &str, dir: &std::path::Path) -> (Arc<CortexState>, String) {
    let fleet = Arc::new(CortexState::from_config(&config(neuron, dir)));
    seed_model(&fleet).await;
    let url = serve(&fleet).await;
    (fleet, url)
}

async fn seed_model(fleet: &CortexState) {
    let mut nodes = fleet.nodes.write().await;
    let node = nodes.get_mut("mock-node").unwrap();
    node.healthy = true;
    node.models.insert(
        MODEL.into(),
        ModelEntry {
            id: MODEL.into(),
            status: ModelStatus::Loaded,
            last_accessed: None,
            vram_estimate_mb: Some(8000),
            capabilities: vec!["audio".into()],
            tool_call: false,
            reasoning: false,
            limit: None,
            servable: None,
            reasoning_budget: Vec::new(),
        },
    );
}

async fn serve(fleet: &Arc<CortexState>) -> String {
    let app = cortex_gateway::build_app(Arc::clone(fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

async fn submit(gw: &str, key: Option<&str>, body: Value) -> reqwest::Response {
    let mut req = reqwest::Client::new()
        .post(format!("{gw}/v1/audio/generations"))
        .json(&body);
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    req.send().await.unwrap()
}

async fn get_job(gw: &str, key: Option<&str>, id: &str) -> reqwest::Response {
    let mut req = reqwest::Client::new().get(format!("{gw}/v1/audio/generations/{id}"));
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    req.send().await.unwrap()
}

/// Poll until the job reaches a terminal status (or panic after `secs`).
async fn wait_terminal(gw: &str, key: Option<&str>, id: &str, secs: u64) -> Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(secs);
    loop {
        let job: Value = get_job(gw, key, id).await.json().await.unwrap();
        if matches!(
            job["status"].as_str(),
            Some("succeeded" | "failed" | "cancelled")
        ) {
            return job;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "job {id} never finished: {job}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn song() -> Value {
    json!({"model": MODEL, "prompt": "warm lofi", "duration": 12.5, "seed": 42})
}

#[tokio::test]
async fn submit_poll_fetch_content() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;

    let resp = submit(&gw, Some("key-alice"), song()).await;
    assert_eq!(resp.status(), 202);
    let job: Value = resp.json().await.unwrap();
    assert_eq!(job["object"], "audio.generation");
    assert_eq!(job["status"], "queued");
    assert_eq!(job["model"], MODEL);
    let id = job["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("audiogen_"));

    let done = wait_terminal(&gw, Some("key-alice"), &id, 10).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["result"]["duration"], 12.5);
    assert_eq!(done["result"]["sample_rate"], 48_000);
    assert_eq!(done["result"]["seed"], 42);
    assert_eq!(done["result"]["bytes"], WAV.len());
    assert_eq!(done["usage"]["helexa_audio_units"], 12.5);

    let content = reqwest::Client::new()
        .get(format!("{gw}/v1/audio/generations/{id}/content"))
        .bearer_auth("key-alice")
        .send()
        .await
        .unwrap();
    assert_eq!(content.status(), 200);
    assert_eq!(content.headers()["content-type"], "audio/wav");
    assert_eq!(content.bytes().await.unwrap().as_ref(), WAV);

    // The neuron saw the submitting account, for fair-share admission.
    assert_eq!(
        mock.accounts.lock().unwrap().as_slice(),
        &[Some("acct-alice".to_string())]
    );
    // Record and audio are on disk.
    assert!(dir.path().join(format!("{id}.json")).exists());
    assert!(dir.path().join(format!("{id}.wav")).exists());
}

#[tokio::test]
async fn content_before_completion_is_409() {
    let mock = spawn_mock(Mode::Slow(Duration::from_secs(3))).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap();
    let resp = reqwest::get(format!("{gw}/v1/audio/generations/{id}/content"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "job_not_complete");
}

#[tokio::test]
async fn jobs_are_private_to_their_account() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;

    let job: Value = submit(&gw, Some("key-alice"), song())
        .await
        .json()
        .await
        .unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    wait_terminal(&gw, Some("key-alice"), &id, 10).await;

    // Another account, and an anonymous caller, cannot see it.
    assert_eq!(get_job(&gw, Some("key-bob"), &id).await.status(), 404);
    assert_eq!(get_job(&gw, None, &id).await.status(), 404);
    let content = reqwest::Client::new()
        .get(format!("{gw}/v1/audio/generations/{id}/content"))
        .bearer_auth("key-bob")
        .send()
        .await
        .unwrap();
    assert_eq!(content.status(), 404);
    let cancel = reqwest::Client::new()
        .delete(format!("{gw}/v1/audio/generations/{id}"))
        .bearer_auth("key-bob")
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 404);

    // Listings are per account.
    let list = |key: &'static str| {
        let gw = gw.clone();
        async move {
            reqwest::Client::new()
                .get(format!("{gw}/v1/audio/generations"))
                .bearer_auth(key)
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        }
    };
    let alice = list("key-alice").await;
    assert_eq!(alice["object"], "list");
    assert_eq!(alice["data"].as_array().unwrap().len(), 1);
    assert_eq!(alice["data"][0]["id"], id.as_str());
    assert_eq!(list("key-bob").await["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn anonymous_jobs_are_reachable_by_id_but_never_listed() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    let done = wait_terminal(&gw, None, &id, 10).await;
    assert_eq!(done["status"], "succeeded");
    let list: Value = reqwest::get(format!("{gw}/v1/audio/generations"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn cancel_running_job() {
    let mock = spawn_mock(Mode::Slow(Duration::from_secs(5))).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, Some("key-alice"), song())
        .await
        .json()
        .await
        .unwrap();
    let id = job["id"].as_str().unwrap().to_string();

    // Wait until it is running on the neuron.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while mock.calls.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let resp = reqwest::Client::new()
        .delete(format!("{gw}/v1/audio/generations/{id}"))
        .bearer_auth("key-alice")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let cancelled: Value = resp.json().await.unwrap();
    assert_eq!(cancelled["status"], "cancelled");

    // It stays cancelled: the slow render finishing must not resurrect it.
    tokio::time::sleep(Duration::from_secs(6)).await;
    let job: Value = get_job(&gw, Some("key-alice"), &id)
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(job["status"], "cancelled");
    assert!(job.get("result").is_none());
}

#[tokio::test]
async fn busy_neuron_is_waited_on_without_spending_attempts() {
    // Three busy answers exceed max_attempts (2), so this only succeeds if
    // busy answers don't count as attempts.
    let mock = spawn_mock(Mode::BusyThenOk(3)).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    let done = wait_terminal(&gw, None, &id, 15).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(mock.calls.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn failing_node_is_retried_then_succeeds() {
    let mock = spawn_mock(Mode::FailThenOk(1)).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let done = wait_terminal(&gw, None, job["id"].as_str().unwrap(), 15).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(mock.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn failing_node_exhausts_attempts() {
    let mock = spawn_mock(Mode::AlwaysFail).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let done = wait_terminal(&gw, None, job["id"].as_str().unwrap(), 15).await;
    assert_eq!(done["status"], "failed");
    assert_eq!(done["error"]["code"], "inference_failed");
    assert_eq!(mock.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn neuron_rejection_fails_without_retry() {
    let mock = spawn_mock(Mode::Reject).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let done = wait_terminal(&gw, None, job["id"].as_str().unwrap(), 10).await;
    assert_eq!(done["status"], "failed");
    assert_eq!(done["error"]["code"], "wrong_modality");
    assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn submit_validation() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;

    let resp = submit(&gw, None, json!({"model": MODEL, "prompt": ""})).await;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_audio_params");

    let resp = submit(
        &gw,
        None,
        json!({"model": MODEL, "prompt": "x", "response_format": "ogg"}),
    )
    .await;
    assert_eq!(resp.status(), 400);

    let resp = submit(&gw, None, json!({"model": "nope", "prompt": "x"})).await;
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "model_not_found");

    let resp = submit(&gw, None, json!({"prompt": "x"})).await;
    assert_eq!(resp.status(), 400);

    // Nothing reached the neuron.
    assert_eq!(mock.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unknown_job_is_404() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let (_fleet, gw) = gateway(&mock.url, dir.path()).await;
    let resp = get_job(&gw, None, "audiogen_doesnotexist").await;
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "job_not_found");
}

#[tokio::test]
async fn jobs_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();

    // First cortex: one job finishes, one is stuck behind a busy model.
    let ok = spawn_mock(Mode::Ok).await;
    let (_a, gw_a) = gateway(&ok.url, dir.path()).await;
    let done: Value = submit(&gw_a, Some("key-alice"), song())
        .await
        .json()
        .await
        .unwrap();
    let done_id = done["id"].as_str().unwrap().to_string();
    wait_terminal(&gw_a, Some("key-alice"), &done_id, 10).await;

    let busy = spawn_mock(Mode::AlwaysBusy).await;
    let (_b, gw_b) = gateway(&busy.url, dir.path()).await;
    let stuck: Value = submit(&gw_b, Some("key-alice"), song())
        .await
        .json()
        .await
        .unwrap();
    let stuck_id = stuck["id"].as_str().unwrap().to_string();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while busy.calls.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // A fresh cortex over the same directory, with a healthy neuron.
    let healthy = spawn_mock(Mode::Ok).await;
    let fleet = Arc::new(CortexState::from_config(&config(&healthy.url, dir.path())));
    seed_model(&fleet).await;
    cortex_gateway::audio_jobs::start(Arc::clone(&fleet)).await;
    let gw = serve(&fleet).await;

    // The finished job and its audio are still there.
    let job: Value = get_job(&gw, Some("key-alice"), &done_id)
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(job["status"], "succeeded");
    let content = reqwest::Client::new()
        .get(format!("{gw}/v1/audio/generations/{done_id}/content"))
        .bearer_auth("key-alice")
        .send()
        .await
        .unwrap();
    assert_eq!(content.bytes().await.unwrap().as_ref(), WAV);

    // The unfinished one was re-queued and now completes.
    let resumed = wait_terminal(&gw, Some("key-alice"), &stuck_id, 10).await;
    assert_eq!(resumed["status"], "succeeded", "{resumed}");
}

#[tokio::test]
async fn expired_jobs_are_swept() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(&mock.url, dir.path());
    cfg.audio_jobs.retention_hours = 0;
    let fleet = Arc::new(CortexState::from_config(&cfg));
    seed_model(&fleet).await;
    let gw = serve(&fleet).await;

    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    wait_terminal(&gw, None, &id, 10).await;
    // completed_at has one-second resolution; let it fall behind the cutoff.
    tokio::time::sleep(Duration::from_millis(1100)).await;

    cortex_gateway::audio_jobs::sweep_expired(&fleet).await;
    assert_eq!(get_job(&gw, None, &id).await.status(), 404);
    assert!(!dir.path().join(format!("{id}.json")).exists());
    assert!(!dir.path().join(format!("{id}.wav")).exists());
}
