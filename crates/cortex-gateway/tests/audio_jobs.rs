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
/// What a neuron on an older build rendered: WAV.
const WAV: &[u8] = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio-bytes";

/// What a neuron renders: half a second of a real 16-bit stereo FLAC, so
/// that cortex can transcode it.
static FLAC: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    let samples: Vec<i32> = (0..48_000)
        .map(|i| ((i as f32 * 0.02).sin() * 8_000.0) as i32)
        .collect();
    let config = flacenc::config::Encoder::default().into_verified().unwrap();
    let source = flacenc::source::MemSource::from_samples(&samples, 2, 16, 48_000);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size).unwrap();
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).unwrap();
    sink.into_inner()
});

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
    /// Render WAV, as neurons did before audio moved to FLAC.
    LegacyWav,
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
    let (content_type, body) = match mode {
        Mode::LegacyWav => ("audio/wav", WAV.to_vec()),
        _ => ("audio/flac", FLAC.clone()),
    };
    (
        StatusCode::OK,
        [
            ("content-type", content_type.to_string()),
            (RESULT_HEADER, serde_json::to_string(&result).unwrap()),
        ],
        body,
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
    assert_eq!(job["prompt"], "warm lofi");
    assert!(job.get("lyrics").is_none());
    let id = job["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("audiogen_"));

    let done = wait_terminal(&gw, Some("key-alice"), &id, 10).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["result"]["duration"], 12.5);
    assert_eq!(done["result"]["sample_rate"], 48_000);
    assert_eq!(done["result"]["seed"], 42);
    assert_eq!(done["result"]["bytes"], FLAC.len());
    assert_eq!(done["result"]["format"], "flac");
    assert_eq!(done["usage"]["helexa_audio_units"], 12.5);

    let content = reqwest::Client::new()
        .get(format!("{gw}/v1/audio/generations/{id}/content"))
        .bearer_auth("key-alice")
        .send()
        .await
        .unwrap();
    assert_eq!(content.status(), 200);
    assert_eq!(content.headers()["content-type"], "audio/flac");
    assert_eq!(
        content.headers()["content-disposition"],
        format!("attachment; filename=\"{id}.flac\"").as_str()
    );
    assert_eq!(content.bytes().await.unwrap().as_ref(), FLAC.as_slice());

    // The neuron saw the submitting account, for fair-share admission.
    assert_eq!(
        mock.accounts.lock().unwrap().as_slice(),
        &[Some("acct-alice".to_string())]
    );
    // Record and audio are on disk.
    assert!(dir.path().join(format!("{id}.json")).exists());
    assert!(dir.path().join(format!("{id}.flac")).exists());
}

async fn fetch(gw: &str, key: Option<&str>, path: &str) -> reqwest::Response {
    let mut req = reqwest::Client::new().get(format!("{gw}{path}"));
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    req.send().await.unwrap()
}

fn is_mp3(bytes: &[u8]) -> bool {
    bytes.starts_with(b"ID3") || (bytes.len() > 1 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
}

#[tokio::test]
async fn content_transcodes_to_mp3_and_caches_it() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let fleet = Arc::new(CortexState::from_config(&config(&mock.url, dir.path())));
    seed_model(&fleet).await;
    let gw = serve(&fleet).await;

    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    wait_terminal(&gw, None, &id, 10).await;

    let path = format!("/v1/audio/generations/{id}/content?format=mp3");
    let resp = fetch(&gw, None, &path).await;
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["content-type"], "audio/mpeg");
    assert_eq!(
        resp.headers()["content-disposition"],
        format!("attachment; filename=\"{id}.mp3\"").as_str()
    );
    let mp3 = resp.bytes().await.unwrap();
    assert!(is_mp3(&mp3), "not an mp3: {:02x?}", &mp3[..8]);
    let cached = dir.path().join(format!("{id}.mp3"));
    assert_eq!(std::fs::read(&cached).unwrap(), mp3.as_ref());

    // The second fetch is served from the cache.
    std::fs::write(&cached, b"cached").unwrap();
    let again = fetch(&gw, None, &path).await.bytes().await.unwrap();
    assert_eq!(again.as_ref(), b"cached");

    // The master is untouched and still the default.
    let flac = fetch(&gw, None, &format!("/v1/audio/generations/{id}/content")).await;
    assert_eq!(flac.bytes().await.unwrap().as_ref(), FLAC.as_slice());
}

#[tokio::test]
async fn response_format_sets_the_default_download() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let fleet = Arc::new(CortexState::from_config(&config(&mock.url, dir.path())));
    seed_model(&fleet).await;
    let gw = serve(&fleet).await;

    let mut body = song();
    body["response_format"] = json!("mp3");
    let job: Value = submit(&gw, None, body).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    let done = wait_terminal(&gw, None, &id, 10).await;
    // Stored as FLAC whatever the download format.
    assert_eq!(done["result"]["format"], "flac");
    assert!(dir.path().join(format!("{id}.flac")).exists());

    let base = format!("/v1/audio/generations/{id}/content");
    let resp = fetch(&gw, None, &base).await;
    assert_eq!(resp.headers()["content-type"], "audio/mpeg");
    // An explicit format wins over the job's default.
    let resp = fetch(&gw, None, &format!("{base}?format=flac")).await;
    assert_eq!(resp.headers()["content-type"], "audio/flac");
    // Unknown formats are refused.
    for f in ["wav", "ogg"] {
        let resp = fetch(&gw, None, &format!("{base}?format={f}")).await;
        assert_eq!(resp.status(), 400, "format {f}");
        let err: Value = resp.json().await.unwrap();
        assert_eq!(err["error"]["code"], "invalid_audio_params");
    }
}

#[tokio::test]
async fn legacy_wav_results_are_served_as_wav() {
    let mock = spawn_mock(Mode::LegacyWav).await;
    let dir = tempfile::tempdir().unwrap();
    let fleet = Arc::new(CortexState::from_config(&config(&mock.url, dir.path())));
    seed_model(&fleet).await;
    let gw = serve(&fleet).await;

    let job: Value = submit(&gw, None, song()).await.json().await.unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    let done = wait_terminal(&gw, None, &id, 10).await;
    assert_eq!(done["result"]["format"], "wav");
    assert!(dir.path().join(format!("{id}.wav")).exists());

    let base = format!("/v1/audio/generations/{id}/content");
    let resp = fetch(&gw, None, &base).await;
    assert_eq!(resp.headers()["content-type"], "audio/wav");
    assert_eq!(resp.bytes().await.unwrap().as_ref(), WAV);
    let resp = fetch(&gw, None, &format!("{base}?format=mp3")).await;
    assert_eq!(resp.status(), 409);
    let err: Value = resp.json().await.unwrap();
    assert_eq!(err["error"]["code"], "format_unavailable");
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
    assert_eq!(content.bytes().await.unwrap().as_ref(), FLAC.as_slice());

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
    let mp3 = fetch(
        &gw,
        None,
        &format!("/v1/audio/generations/{id}/content?format=mp3"),
    )
    .await;
    assert_eq!(mp3.status(), 200);
    assert!(dir.path().join(format!("{id}.mp3")).exists());
    // completed_at has one-second resolution; let it fall behind the cutoff.
    tokio::time::sleep(Duration::from_millis(1100)).await;

    cortex_gateway::audio_jobs::sweep_expired(&fleet).await;
    assert_eq!(get_job(&gw, None, &id).await.status(), 404);
    assert!(!dir.path().join(format!("{id}.json")).exists());
    assert!(!dir.path().join(format!("{id}.flac")).exists());
    assert!(!dir.path().join(format!("{id}.mp3")).exists());
}

// ── Per-model rates (#386) ───────────────────────────────────────────

/// A gateway whose keys are capped at `cap` tokens and whose audio rate
/// table is `rates` (model id → tokens per second; "*" is the default).
async fn priced_gateway(
    neuron: &str,
    dir: &std::path::Path,
    cap: Option<u64>,
    rates: &[(&str, u64)],
) -> (Arc<CortexState>, String) {
    let mut cfg = config(neuron, dir);
    for k in &mut cfg.entitlements.keys {
        k.hard_cap = cap;
    }
    let fleet = Arc::new(CortexState::from_config(&cfg));
    let mut table = cortex_gateway::rates::RateTable::default();
    for (model, rate) in rates {
        table.set(cortex_gateway::rates::AUDIO_SECOND, model, *rate);
    }
    fleet.rates.replace(table).await;
    seed_model(&fleet).await;
    let url = serve(&fleet).await;
    (fleet, url)
}

async fn spent(fleet: &CortexState, who: &str) -> u64 {
    let principal = cortex_core::entitlements::Principal {
        account_id: format!("acct-{who}"),
        key_id: format!("acct-{who}"),
    };
    fleet
        .entitlements
        .snapshot(&principal)
        .await
        .map(|s| s.spent)
        .unwrap_or(0)
}

#[tokio::test]
async fn reservation_uses_the_models_rate() {
    // A 12.5 s song against a 10,000-token cap: refused at 1000 tokens/s
    // (12,500), accepted at 500 tokens/s (6,250).
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let (_f, gw) = priced_gateway(&mock.url, dir.path(), Some(10_000), &[("*", 1000)]).await;
    let resp = submit(&gw, Some("key-alice"), song()).await;
    assert_eq!(resp.status(), 429, "{:?}", resp.text().await);

    let dir = tempfile::tempdir().unwrap();
    let (_f, gw) = priced_gateway(
        &mock.url,
        dir.path(),
        Some(10_000),
        &[("*", 1000), (MODEL, 500)],
    )
    .await;
    assert_eq!(submit(&gw, Some("key-alice"), song()).await.status(), 202);
}

#[tokio::test]
async fn unpriced_model_is_refused_for_metered_callers_only() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    // An empty table: what a cortex with upstream sees before its first fetch.
    let (_f, gw) = priced_gateway(&mock.url, dir.path(), None, &[]).await;
    let resp = submit(&gw, Some("key-alice"), song()).await;
    assert_eq!(resp.status(), 503);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no audio rate"),
        "{body}"
    );
    // Anonymous work is not metered, so it does not need a price.
    assert_eq!(submit(&gw, None, song()).await.status(), 202);
}

#[tokio::test]
async fn settles_at_the_rate_reserved_even_if_it_changes_mid_job() {
    let mock = spawn_mock(Mode::Slow(Duration::from_millis(800))).await;
    let dir = tempfile::tempdir().unwrap();
    let (fleet, gw) = priced_gateway(&mock.url, dir.path(), None, &[(MODEL, 300)]).await;
    let job: Value = submit(&gw, Some("key-alice"), song())
        .await
        .json()
        .await
        .unwrap();
    // Reprice while the render is in flight.
    let mut table = cortex_gateway::rates::RateTable::default();
    table.set(cortex_gateway::rates::AUDIO_SECOND, MODEL, 1);
    fleet.rates.replace(table).await;

    let done = wait_terminal(&gw, Some("key-alice"), job["id"].as_str().unwrap(), 10).await;
    assert_eq!(done["status"], "succeeded");
    // 12.5 s × 300 tokens/s, settled asynchronously.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let s = spent(&fleet, "alice").await;
        if s == 3750 {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "spent {s}, want 3750");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn rates_are_fetched_from_upstream_with_the_client_bearer() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/authz/v1/rates",
        get(|headers: HeaderMap| async move {
            if headers.get("authorization").and_then(|v| v.to_str().ok())
                != Some("Bearer op-secret")
            {
                return StatusCode::UNAUTHORIZED.into_response();
            }
            Json(json!({"data": [
                {"model_id": "*", "unit": "audio_second", "tokens_per_unit": 1000},
                {"model_id": MODEL, "unit": "audio_second", "tokens_per_unit": 40},
                {"model_id": "bad", "unit": "audio_second", "tokens_per_unit": 0}
            ]}))
            .into_response()
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = reqwest::Client::new();
    let table = cortex_gateway::rates::fetch(&client, &url, "op-secret")
        .await
        .unwrap();
    assert_eq!(table.audio_tokens_per_second(MODEL), Some(40));
    assert_eq!(table.audio_tokens_per_second("other"), Some(1000));
    // A non-positive rate is skipped, falling back to the default.
    assert_eq!(table.audio_tokens_per_second("bad"), Some(1000));
    assert!(
        cortex_gateway::rates::fetch(&client, &url, "wrong")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn models_endpoint_advertises_audio_limits_and_rate() {
    let mock = spawn_mock(Mode::Ok).await;
    let dir = tempfile::tempdir().unwrap();
    let catalogue = dir.path().join("models.toml");
    std::fs::write(
        &catalogue,
        format!(
            r#"
[[models]]
id = "{MODEL}"
harness = "candle"
quant = "q8_0"

[models.audio]
min_duration = 10.0
max_duration = 600.0
default_duration = 30.0
vocals = true
sample_rate = 48000
channels = 2
max_steps = 20

[aliases]
"helexa/music" = "{MODEL}"
"#
        ),
    )
    .unwrap();
    let mut cfg = config(&mock.url, dir.path());
    cfg.models_config = catalogue.to_string_lossy().into_owned();
    let fleet = Arc::new(CortexState::from_config(&cfg));
    let mut table = cortex_gateway::rates::RateTable::default();
    table.set(cortex_gateway::rates::AUDIO_SECOND, MODEL, 77);
    fleet.rates.replace(table).await;
    seed_model(&fleet).await;
    let gw = serve(&fleet).await;

    let models: Value = reqwest::get(format!("{gw}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let find = |id: &str| {
        models["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .cloned()
            .unwrap_or_else(|| panic!("{id} missing from {models}"))
    };
    for id in [MODEL, "helexa/music"] {
        let m = find(id);
        assert_eq!(m["audio"]["max_duration"], 600.0, "{m}");
        assert_eq!(m["audio"]["vocals"], true);
        assert_eq!(m["audio_tokens_per_second"], 77, "{m}");
    }

    // The catalogue's limits are enforced at submit.
    let resp = submit(
        &gw,
        None,
        json!({"model": MODEL, "prompt": "x", "duration": 5.0}),
    )
    .await;
    assert_eq!(resp.status(), 400);
    // An alias is accepted and resolved to the model.
    let resp = submit(&gw, None, json!({"model": "helexa/music", "prompt": "x"})).await;
    assert_eq!(resp.status(), 202);
    let job: Value = resp.json().await.unwrap();
    assert_eq!(job["model"], MODEL);
}
