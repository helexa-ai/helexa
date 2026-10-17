//! Async music / audio generation jobs (#382).
//!
//! A render takes seconds to minutes, so the public surface is a job API
//! (see [`cortex_core::audio`] for the routes). cortex owns the job: it
//! records it, queues it, dispatches it to a neuron serving the model, keeps
//! the rendered audio, and meters it. A neuron renders one generation per
//! dispatch, synchronously, and knows nothing about jobs.
//!
//! Records and audio are written under `[audio_jobs] dir`, so the job list
//! and finished results survive a cortex restart. Jobs that were queued or
//! running when cortex stopped are re-queued at startup. A job whose neuron
//! fails is retried on whatever node the router picks next, up to
//! `max_attempts`; a neuron that is merely busy (429/503) is waited on, up
//! to `queue_timeout_secs`, without spending an attempt.
//!
//! Ownership: a job belongs to the account that submitted it, and only that
//! account can see, fetch or cancel it. Anonymous jobs (no principal) are
//! reachable by id alone — the id is an unguessable capability — and never
//! appear in a listing.

use crate::metering::ReservationGuard;
use crate::state::CortexState;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use cortex_core::audio::{
    AudioGenerationRequest, AudioJob, AudioJobList, AudioResult, JOB_OBJECT, JobStatus,
    NeuronAudioResult, RESULT_HEADER,
};
use cortex_core::config::AudioJobsConfig;
use cortex_core::entitlements::{HEADER_ACCOUNT_ID, HEADER_KEY_ID, Principal};
use cortex_core::error_envelope::OpenAiError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;

/// Path of the job collection; `{id}` and `{id}/content` hang off it.
pub const JOBS_PATH: &str = "/v1/audio/generations";

/// Wait between dispatches to a busy model when the neuron sends no
/// `Retry-After`.
const DEFAULT_BUSY_WAIT: Duration = Duration::from_secs(5);

/// How often expired jobs are swept.
const SWEEP_INTERVAL: Duration = Duration::from_secs(600);

/// A job as cortex stores it: the public object plus what is needed to run
/// (or re-run) it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobRecord {
    job: AudioJob,
    request: AudioGenerationRequest,
    /// The submitting principal; `None` for an anonymous submission.
    owner: Option<Principal>,
    /// Ledger tokens per audio-second this job was reserved at (#386), so
    /// it settles at the price it was accepted at even if the rate changes
    /// while it runs. `None` for anonymous (unmetered) jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tokens_per_second: Option<u64>,
}

/// The job store and dispatcher, held in [`CortexState`].
pub struct AudioJobs {
    config: AudioJobsConfig,
    /// See [`AudioJobs::dir`].
    dir: std::sync::OnceLock<Option<PathBuf>>,
    jobs: RwLock<HashMap<String, JobRecord>>,
    /// Worker tasks for jobs that are queued or running, for cancellation.
    workers: Mutex<HashMap<String, JoinHandle<()>>>,
    /// Bounds concurrent renders across the fleet.
    slots: Arc<Semaphore>,
    /// Rendered audio, when there is no job directory to write it to.
    memory_audio: Mutex<HashMap<String, Bytes>>,
    /// Client for neuron dispatches. A neuron sends nothing until the render
    /// is finished, so the read timeout has to cover a whole render.
    http: reqwest::Client,
}

impl AudioJobs {
    pub fn new(config: &AudioJobsConfig) -> Self {
        let http = reqwest::Client::builder()
            .read_timeout(Duration::from_secs(config.queue_timeout_secs.max(1800)))
            .build()
            .expect("failed to build audio HTTP client");
        Self {
            config: config.clone(),
            dir: std::sync::OnceLock::new(),
            jobs: RwLock::new(HashMap::new()),
            workers: Mutex::new(HashMap::new()),
            slots: Arc::new(Semaphore::new(config.max_concurrent.max(1))),
            memory_audio: Mutex::new(HashMap::new()),
            http,
        }
    }

    /// The job directory, created on first use. `None` (memory only) when it
    /// cannot be created. Lazy so that building a gateway does not touch the
    /// filesystem until audio jobs are actually used.
    fn dir(&self) -> Option<&PathBuf> {
        self.dir
            .get_or_init(|| {
                let dir = PathBuf::from(&self.config.dir);
                match std::fs::create_dir_all(&dir) {
                    Ok(()) => Some(dir),
                    Err(e) => {
                        tracing::warn!(
                            dir = %self.config.dir,
                            error = %e,
                            "audio job directory unavailable; jobs will not survive a restart"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    fn record_path(&self, id: &str) -> Option<PathBuf> {
        self.dir().map(|d| d.join(format!("{id}.json")))
    }

    fn audio_path(&self, id: &str) -> Option<PathBuf> {
        self.dir().map(|d| d.join(format!("{id}.wav")))
    }

    async fn persist(&self, record: &JobRecord) {
        let Some(path) = self.record_path(&record.job.id) else {
            return;
        };
        let bytes = match serde_json::to_vec_pretty(record) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(job = %record.job.id, error = %e, "audio job serialise failed");
                return;
            }
        };
        // Write-then-rename so a crash never leaves a torn record.
        let tmp = path.with_extension("json.tmp");
        if let Err(e) = tokio::fs::write(&tmp, &bytes).await {
            tracing::warn!(job = %record.job.id, error = %e, "audio job write failed");
            return;
        }
        if let Err(e) = tokio::fs::rename(&tmp, &path).await {
            tracing::warn!(job = %record.job.id, error = %e, "audio job rename failed");
        }
    }

    async fn update<F: FnOnce(&mut JobRecord)>(&self, id: &str, f: F) -> Option<JobRecord> {
        let snapshot = {
            let mut jobs = self.jobs.write().await;
            let record = jobs.get_mut(id)?;
            f(record);
            record.clone()
        };
        self.persist(&snapshot).await;
        Some(snapshot)
    }

    async fn get(&self, id: &str) -> Option<JobRecord> {
        self.jobs.read().await.get(id).cloned()
    }

    fn forget_worker(&self, id: &str) {
        if let Ok(mut w) = self.workers.lock() {
            w.remove(id);
        }
    }
}

/// Load persisted jobs, re-queue unfinished ones, and start the retention
/// sweep. Called once at startup.
pub async fn start(fleet: Arc<CortexState>) {
    let Some(dir) = fleet.audio_jobs.dir().cloned() else {
        spawn_sweeper(fleet);
        return;
    };
    let mut resumed = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let record: JobRecord = match tokio::fs::read(&path)
                .await
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
            {
                Some(r) => r,
                None => {
                    tracing::warn!(path = %path.display(), "skipping unreadable audio job record");
                    continue;
                }
            };
            if !record.job.status.is_terminal() {
                resumed.push(record.job.id.clone());
            }
            fleet
                .audio_jobs
                .jobs
                .write()
                .await
                .insert(record.job.id.clone(), record);
        }
    }
    for id in resumed {
        let Some(record) = fleet
            .audio_jobs
            .update(&id, |r| {
                r.job.status = JobStatus::Queued;
                r.job.started_at = None;
            })
            .await
        else {
            continue;
        };
        // The reservation died with the old process: take a fresh one, and
        // fail the job if the budget no longer allows it.
        let guard = match reserve(&fleet, record.owner.as_ref(), &record.request).await {
            Ok((g, rate)) => {
                fleet
                    .audio_jobs
                    .update(&id, |r| r.tokens_per_second = rate)
                    .await;
                g
            }
            Err(env) => {
                fail(&fleet, &id, envelope_error(&env)).await;
                continue;
            }
        };
        tracing::info!(job = %id, "re-queued audio job after restart");
        spawn_worker(Arc::clone(&fleet), id, guard);
    }
    spawn_sweeper(fleet);
}

fn spawn_sweeper(fleet: Arc<CortexState>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(SWEEP_INTERVAL).await;
            sweep_expired(&fleet).await;
        }
    });
}

/// Delete finished jobs older than the retention window, with their audio.
pub async fn sweep_expired(fleet: &CortexState) {
    let jobs = &fleet.audio_jobs;
    let cutoff = now_secs().saturating_sub(jobs.config.retention_hours * 3600);
    let expired: Vec<String> = jobs
        .jobs
        .read()
        .await
        .values()
        .filter(|r| r.job.status.is_terminal())
        .filter(|r| r.job.completed_at.unwrap_or(r.job.created_at) < cutoff)
        .map(|r| r.job.id.clone())
        .collect();
    for id in expired {
        jobs.jobs.write().await.remove(&id);
        if let Ok(mut m) = jobs.memory_audio.lock() {
            m.remove(&id);
        }
        for path in [jobs.record_path(&id), jobs.audio_path(&id)]
            .into_iter()
            .flatten()
        {
            let _ = tokio::fs::remove_file(path).await;
        }
        tracing::debug!(job = %id, "expired audio job removed");
    }
}

/// Routes for the job API.
pub fn routes() -> Router<Arc<CortexState>> {
    Router::new()
        .route(JOBS_PATH, get(list_jobs).post(submit))
        .route(
            "/v1/audio/generations/{id}",
            get(get_job).delete(cancel_job),
        )
        .route("/v1/audio/generations/{id}/content", get(get_content))
}

/// `POST /v1/audio/generations` — validate, reserve budget, queue.
async fn submit(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: AudioGenerationRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return invalid_params(format!("request body is not a valid audio request: {e}"));
        }
    };
    if let Err(msg) = request.validate_shape() {
        return invalid_params(msg);
    }

    // The model must be one this fleet knows. Resolving it fully would
    // cold-load it at submit time, so only check that it exists here; the
    // worker resolves (and loads) when the job actually runs.
    let model = fleet.catalogue.resolve_alias(&request.model).to_string();
    let known = fleet.catalogue.get(&model).is_some()
        || fleet
            .nodes
            .read()
            .await
            .values()
            .any(|n| n.models.contains_key(&model));
    if !known {
        return crate::error::envelope_response(OpenAiError::new(
            404,
            "invalid_request_error",
            "model_not_found",
            format!(
                "model '{}' not found on any node and not in catalogue",
                request.model
            ),
        ));
    }
    if let Some(profile) = fleet.catalogue.get(&model)
        && let Some(limits) = &profile.audio
        && let Err(msg) = request.validate_limits(limits)
    {
        return invalid_params(msg);
    }

    let owner = crate::metering::principal_from_headers(&headers);
    let mut request = request;
    request.model = model.clone();
    let (guard, tokens_per_second) = match reserve(&fleet, owner.as_ref(), &request).await {
        Ok(r) => r,
        Err(env) => return crate::error::envelope_response(env),
    };

    let id = format!("audiogen_{}", uuid::Uuid::new_v4().simple());
    let job = AudioJob {
        id: id.clone(),
        object: JOB_OBJECT.into(),
        status: JobStatus::Queued,
        model,
        created_at: now_secs(),
        started_at: None,
        completed_at: None,
        duration: request.duration,
        result: None,
        usage: None,
        error: None,
    };
    let record = JobRecord {
        job: job.clone(),
        request,
        owner,
        tokens_per_second,
    };
    fleet
        .audio_jobs
        .jobs
        .write()
        .await
        .insert(id.clone(), record.clone());
    fleet.audio_jobs.persist(&record).await;
    metrics::counter!("cortex_audio_jobs_submitted_total", "model" => job.model.clone())
        .increment(1);
    tracing::info!(job = %id, model = %job.model, "audio job queued");

    spawn_worker(Arc::clone(&fleet), id, guard);
    (StatusCode::ACCEPTED, Json(job)).into_response()
}

/// `GET /v1/audio/generations` — the caller's jobs, newest first.
async fn list_jobs(State(fleet): State<Arc<CortexState>>, headers: HeaderMap) -> Response {
    let caller = crate::metering::principal_from_headers(&headers);
    let mut data: Vec<AudioJob> = match &caller {
        None => Vec::new(),
        Some(p) => fleet
            .audio_jobs
            .jobs
            .read()
            .await
            .values()
            .filter(|r| {
                r.owner
                    .as_ref()
                    .is_some_and(|o| o.account_id == p.account_id)
            })
            .map(|r| r.job.clone())
            .collect(),
    };
    data.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
    Json(AudioJobList {
        object: "list".into(),
        data,
    })
    .into_response()
}

/// `GET /v1/audio/generations/{id}`.
async fn get_job(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    match visible_record(&fleet, &headers, &id).await {
        Some(r) => Json(r.job).into_response(),
        None => job_not_found(&id),
    }
}

/// `GET /v1/audio/generations/{id}/content` — the rendered audio.
async fn get_content(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let Some(record) = visible_record(&fleet, &headers, &id).await else {
        return job_not_found(&id);
    };
    if record.job.status != JobStatus::Succeeded {
        return crate::error::envelope_response(OpenAiError::new(
            409,
            "invalid_request_error",
            "job_not_complete",
            format!(
                "audio job '{id}' is {}; content is available once it has succeeded",
                status_word(record.job.status)
            ),
        ));
    }
    let bytes = match fleet.audio_jobs.audio_path(&id) {
        Some(path) => tokio::fs::read(&path).await.ok(),
        None => fleet.audio_jobs.memory_audio(&id),
    };
    let Some(bytes) = bytes else {
        return crate::error::envelope_response(OpenAiError::new(
            410,
            "invalid_request_error",
            "content_expired",
            format!("the audio for job '{id}' is no longer stored"),
        ));
    };
    let mut resp = Response::new(Body::from(bytes));
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{id}.wav\"")) {
        resp.headers_mut().insert(header::CONTENT_DISPOSITION, v);
    }
    resp
}

/// `DELETE /v1/audio/generations/{id}` — cancel a queued or running job.
/// Cancelling a finished job is a no-op that returns it unchanged.
async fn cancel_job(
    State(fleet): State<Arc<CortexState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let Some(record) = visible_record(&fleet, &headers, &id).await else {
        return job_not_found(&id);
    };
    if record.job.status.is_terminal() {
        return Json(record.job).into_response();
    }
    // Aborting the worker drops its reservation guard, which releases the
    // budget. A render already running on a neuron finishes there unseen.
    let handle = fleet
        .audio_jobs
        .workers
        .lock()
        .ok()
        .and_then(|mut w| w.remove(&id));
    if let Some(h) = handle {
        h.abort();
    }
    let updated = fleet
        .audio_jobs
        .update(&id, |r| {
            r.job.status = JobStatus::Cancelled;
            r.job.completed_at = Some(now_secs());
        })
        .await;
    metrics::counter!("cortex_audio_jobs_cancelled_total", "model" => record.job.model.clone())
        .increment(1);
    tracing::info!(job = %id, "audio job cancelled");
    match updated {
        Some(r) => Json(r.job).into_response(),
        None => job_not_found(&id),
    }
}

/// A job the caller may see: owned by the caller's account, or anonymous.
async fn visible_record(fleet: &CortexState, headers: &HeaderMap, id: &str) -> Option<JobRecord> {
    let record = fleet.audio_jobs.get(id).await?;
    let caller = crate::metering::principal_from_headers(headers);
    match (&record.owner, &caller) {
        (None, _) => Some(record),
        (Some(owner), Some(caller)) if owner.account_id == caller.account_id => Some(record),
        _ => None,
    }
}

fn spawn_worker(fleet: Arc<CortexState>, id: String, guard: ReservationGuard) {
    let jobs_fleet = Arc::clone(&fleet);
    let task_id = id.clone();
    let handle = tokio::spawn(async move {
        run_job(&jobs_fleet, &task_id, guard).await;
        jobs_fleet.audio_jobs.forget_worker(&task_id);
    });
    if let Ok(mut w) = fleet.audio_jobs.workers.lock() {
        // The task may already have finished (and tried to forget itself)
        // before this insert; only track it while it is still running.
        if !handle.is_finished() {
            w.insert(id, handle);
        }
    }
}

/// Outcome of one dispatch to a neuron.
enum Dispatch {
    Done(Bytes, NeuronAudioResult),
    /// The model is busy; try again after the given wait.
    Busy(Duration),
    /// The request itself was refused; retrying will not help.
    Rejected(Value),
    /// The node failed or was unreachable; another attempt may succeed.
    Failed(Value),
}

async fn run_job(fleet: &Arc<CortexState>, id: &str, guard: ReservationGuard) {
    let jobs = &fleet.audio_jobs;
    let Ok(_slot) = Arc::clone(&jobs.slots).acquire_owned().await else {
        return;
    };
    let Some(record) = jobs
        .update(id, |r| {
            if !r.job.status.is_terminal() {
                r.job.status = JobStatus::Running;
                r.job.started_at = Some(now_secs());
            }
        })
        .await
    else {
        return;
    };
    // Cancelled while it waited for a slot.
    if record.job.status != JobStatus::Running {
        return;
    }
    let model = record.job.model.clone();
    let deadline = Instant::now() + Duration::from_secs(jobs.config.queue_timeout_secs);
    let mut attempts = 0u32;
    let mut last_error = error_value(
        "api_error",
        "audio_generation_failed",
        "the audio job could not be dispatched",
    );

    while attempts < jobs.config.max_attempts.max(1) {
        let outcome = dispatch_once(fleet, &record).await;
        match outcome {
            Dispatch::Done(bytes, result) => {
                complete(fleet, id, &record, bytes, result, guard).await;
                return;
            }
            Dispatch::Busy(wait) => {
                if Instant::now() + wait > deadline {
                    last_error = error_value(
                        "rate_limit_error",
                        "queue_timeout",
                        "the model stayed busy for longer than the job queue timeout",
                    );
                    break;
                }
                tokio::time::sleep(wait).await;
            }
            Dispatch::Rejected(err) => {
                last_error = err;
                break;
            }
            Dispatch::Failed(err) => {
                attempts += 1;
                tracing::warn!(job = %id, model = %model, attempt = attempts, error = %err, "audio dispatch failed");
                last_error = err;
                if attempts < jobs.config.max_attempts {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }
    drop(guard); // releases the reservation: nothing was rendered
    fail(fleet, id, last_error).await;
}

async fn dispatch_once(fleet: &Arc<CortexState>, record: &JobRecord) -> Dispatch {
    let route = match crate::router::resolve(fleet, &record.job.model).await {
        Ok(r) => r,
        Err(e) => {
            let err = error_value(e.broad_type(), e.code(), &e.to_string());
            return match e.retry_after_secs() {
                Some(secs) => Dispatch::Busy(Duration::from_secs(secs.max(1))),
                None if e.http_status() < 500 => Dispatch::Rejected(err),
                None => Dispatch::Failed(err),
            };
        }
    };
    crate::handlers::touch_model(fleet, &route.node_name, &route.resolved_model_id).await;

    let mut body = record.request.clone();
    body.model = route.resolved_model_id.clone();
    let url = format!("{}{JOBS_PATH}", route.endpoint);
    let mut req = fleet.audio_jobs.http.post(&url).json(&body);
    if let Some(owner) = &record.owner {
        req = req
            .header(HEADER_ACCOUNT_ID, &owner.account_id)
            .header(HEADER_KEY_ID, &owner.key_id);
    }
    let labels = [
        ("model", route.resolved_model_id.clone()),
        ("node", route.node_name.clone()),
    ];
    metrics::counter!("cortex_requests_total", &labels).increment(1);
    if route.cold_start {
        metrics::counter!("cortex_cold_starts_total", &labels).increment(1);
    }
    let start = Instant::now();
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            metrics::counter!("cortex_request_errors_total", &labels).increment(1);
            return Dispatch::Failed(error_value(
                "api_error",
                "upstream_unreachable",
                &format!("failed to reach the serving node: {e}"),
            ));
        }
    };
    let status = resp.status();
    let retry_after = resp
        .headers()
        .get(header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|s| Duration::from_secs(s.max(1)));
    let result_header = resp
        .headers()
        .get(RESULT_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            metrics::counter!("cortex_request_errors_total", &labels).increment(1);
            return Dispatch::Failed(error_value(
                "api_error",
                "upstream_read_failed",
                &format!("failed to read the serving node's response: {e}"),
            ));
        }
    };
    metrics::histogram!("cortex_audio_render_seconds", &labels)
        .record(start.elapsed().as_secs_f64());

    if status.is_success() {
        let parsed = result_header
            .as_deref()
            .and_then(|h| serde_json::from_str::<NeuronAudioResult>(h).ok());
        return match parsed {
            Some(result) => Dispatch::Done(bytes, result),
            None => Dispatch::Failed(error_value(
                "api_error",
                "upstream_bad_response",
                "the serving node returned audio without its result metadata",
            )),
        };
    }
    metrics::counter!("cortex_request_errors_total", &labels).increment(1);
    if status.as_u16() == 429 || status.as_u16() == 503 {
        return Dispatch::Busy(retry_after.unwrap_or(DEFAULT_BUSY_WAIT));
    }
    let err = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| v.get("error").cloned())
        .unwrap_or_else(|| {
            error_value(
                "api_error",
                "upstream_error",
                &format!("the serving node answered {status}"),
            )
        });
    if status.is_client_error() {
        Dispatch::Rejected(err)
    } else {
        Dispatch::Failed(err)
    }
}

async fn complete(
    fleet: &Arc<CortexState>,
    id: &str,
    record: &JobRecord,
    bytes: Bytes,
    result: NeuronAudioResult,
    guard: ReservationGuard,
) {
    let jobs = &fleet.audio_jobs;
    // Cancelled while the render was in flight: drop the result, release
    // the budget, and leave the job cancelled.
    if jobs.get(id).await.map(|r| r.job.status) == Some(JobStatus::Cancelled) {
        drop(guard);
        return;
    }
    let size = bytes.len() as u64;
    match jobs.audio_path(id) {
        Some(path) => {
            if let Err(e) = tokio::fs::write(&path, &bytes).await {
                drop(guard);
                fail(
                    fleet,
                    id,
                    error_value(
                        "api_error",
                        "storage_failed",
                        &format!("could not store the rendered audio: {e}"),
                    ),
                )
                .await;
                return;
            }
        }
        None => jobs.keep_in_memory(id, bytes),
    }

    let units = result.usage.helexa_audio_units;
    match &record.owner {
        Some(owner) => {
            // An owned job always carries the rate it was reserved at.
            let rate = record
                .tokens_per_second
                .unwrap_or(crate::rates::LOCAL_DEFAULT_TOKENS_PER_AUDIO_SECOND);
            crate::metering::settle_audio_usage(owner, guard, &fleet.served_usage, units, rate)
        }
        None => drop(guard),
    }
    jobs.update(id, |r| {
        r.job.status = JobStatus::Succeeded;
        r.job.completed_at = Some(now_secs());
        r.job.result = Some(AudioResult {
            duration: result.duration,
            sample_rate: result.sample_rate,
            channels: result.channels,
            seed: result.seed,
            format: "wav".into(),
            bytes: size,
        });
        r.job.usage = Some(result.usage.clone());
    })
    .await;
    metrics::counter!("cortex_audio_jobs_succeeded_total", "model" => record.job.model.clone())
        .increment(1);
    tracing::info!(job = %id, model = %record.job.model, seconds = result.duration, "audio job succeeded");
}

async fn fail(fleet: &CortexState, id: &str, error: Value) {
    let updated = fleet
        .audio_jobs
        .update(id, |r| {
            r.job.status = JobStatus::Failed;
            r.job.completed_at = Some(now_secs());
            r.job.error = Some(error.clone());
        })
        .await;
    if let Some(r) = updated {
        metrics::counter!("cortex_audio_jobs_failed_total", "model" => r.job.model.clone())
            .increment(1);
        tracing::warn!(job = %id, model = %r.job.model, error = %error, "audio job failed");
    }
}

/// Reserve budget for a job at its model's current rate (#386). Returns the
/// guard and the rate it reserved at; anonymous jobs are unmetered.
#[allow(clippy::result_large_err)] // the Err is the #63 envelope; see metering::reserve_or_reject
async fn reserve(
    fleet: &Arc<CortexState>,
    owner: Option<&Principal>,
    request: &AudioGenerationRequest,
) -> Result<(ReservationGuard, Option<u64>), OpenAiError> {
    let Some(p) = owner else {
        return Ok((
            ReservationGuard::anonymous(Arc::clone(&fleet.entitlements)),
            None,
        ));
    };
    let Some(rate) = fleet.rates.audio_tokens_per_second(&request.model).await else {
        return Err(OpenAiError::service_unavailable(
            format!("no audio rate is configured for model '{}'", request.model),
            Some(5),
        ));
    };
    let default = fleet
        .catalogue
        .get(&request.model)
        .and_then(|m| m.audio.as_ref())
        .map(|l| l.default_duration);
    let seconds = crate::metering::audio_reservation_seconds(request.duration, default);
    let guard = crate::metering::reserve_or_reject(
        Arc::clone(&fleet.entitlements),
        p,
        crate::metering::audio_seconds_to_tokens(seconds, rate),
    )
    .await?;
    Ok((guard, Some(rate)))
}

impl AudioJobs {
    fn keep_in_memory(&self, id: &str, bytes: Bytes) {
        if let Ok(mut m) = self.memory_audio.lock() {
            m.insert(id.to_string(), bytes);
        }
    }

    fn memory_audio(&self, id: &str) -> Option<Vec<u8>> {
        self.memory_audio
            .lock()
            .ok()
            .and_then(|m| m.get(id).map(|b| b.to_vec()))
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn status_word(s: JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Running => "running",
        JobStatus::Succeeded => "succeeded",
        JobStatus::Failed => "failed",
        JobStatus::Cancelled => "cancelled",
    }
}

fn error_value(typ: &str, code: &str, message: &str) -> Value {
    serde_json::json!({"type": typ, "code": code, "message": message})
}

fn envelope_error(env: &OpenAiError) -> Value {
    env.body()
        .get("error")
        .cloned()
        .unwrap_or_else(|| error_value("api_error", "budget_rejected", "budget reservation failed"))
}

fn invalid_params(message: impl Into<String>) -> Response {
    crate::error::envelope_response(OpenAiError::new(
        400,
        "invalid_request_error",
        "invalid_audio_params",
        format!("invalid audio generation parameters: {}", message.into()),
    ))
}

fn job_not_found(id: &str) -> Response {
    crate::error::envelope_response(OpenAiError::new(
        404,
        "invalid_request_error",
        "job_not_found",
        format!("no audio job '{id}'"),
    ))
}
