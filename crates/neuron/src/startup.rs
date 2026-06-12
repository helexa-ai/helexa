//! Activation- and deactivation-time orchestration.
//!
//! Wired from `main.rs` around the HTTP listener — activation runs
//! before bind, deactivation runs after axum returns from its
//! graceful-shutdown future. Kept in its own module so the logic is
//! unit-testable without spinning up a full neuron process.

use crate::activation::ActivationTracker;
use crate::harness::HarnessRegistry;
use crate::harness::preflight::PreflightError;
use cortex_core::harness::ModelSpec;
use std::time::{Duration, Instant};
use tokio::signal;

/// Maximum time we wait on a single `unload_model` call during
/// shutdown. The TP unload path tries `Arc::try_unwrap`, which fails
/// fast when an inference is in flight, so a healthy unload returns
/// in milliseconds. The timeout exists to bound a *future* unload
/// path that might genuinely block on a stuck worker, so a single
/// wedged model can't burn the whole systemd TimeoutStopSec window.
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(20);

/// Load each spec sequentially against the registry, treating
/// individual failures as warnings rather than fatal errors.
///
/// VRAM contention makes parallel loads risky; the sequential path is
/// boring but correct. The function logs elapsed time per load and
/// updates `activation` so the `/health` endpoint can tell callers
/// which models are still pre-warming. Caller is expected to run this
/// in a background `tokio::spawn` task — the HTTP listener binds
/// independently so the host is reachable during the pre-warm window.
pub async fn load_default_models(
    registry: &HarnessRegistry,
    specs: &[ModelSpec],
    activation: &ActivationTracker,
    cuda_unavailable_reason: Option<&str>,
) {
    if specs.is_empty() {
        activation.mark_ready().await;
        return;
    }
    // Driver/library mismatch preflight (#19): every CUDA load on this
    // host is guaranteed to fail (cuInit → CUDA_ERROR_SYSTEM_DRIVER
    // MISMATCH, surfacing as a cryptic NCCL/driver error). Don't
    // attempt them — mark each default model failed with the
    // operator-actionable reason so `/health` activation shows the
    // real cause, and let the host run API-only until it's rebooted.
    if let Some(reason) = cuda_unavailable_reason {
        tracing::error!(
            count = specs.len(),
            reason = %reason,
            "skipping default model loads: CUDA unavailable"
        );
        for spec in specs {
            activation.start_loading(&spec.model_id).await;
            activation.fail_loading(&spec.model_id, reason).await;
        }
        activation.mark_ready().await;
        return;
    }
    tracing::info!(count = specs.len(), "loading default models");
    for spec in specs {
        let start = Instant::now();
        activation.start_loading(&spec.model_id).await;
        match registry.load_model(spec).await {
            Ok(()) => {
                activation.complete_loading(&spec.model_id).await;
                tracing::info!(
                    model = %spec.model_id,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "loaded default model"
                );
            }
            Err(e) => {
                let rendered = format!("{e:#}");
                activation.fail_loading(&spec.model_id, &rendered).await;
                // When the underlying failure is a preflight rejection,
                // pull the structured fields out so journalctl shows
                // `reason=tp_requires_safetensors detail="..."` instead
                // of an opaque "fetch config.json … 404". The operator
                // can act on the structured form directly.
                if let Some(pf) = e.downcast_ref::<PreflightError>() {
                    tracing::warn!(
                        model = %spec.model_id,
                        reason = preflight_kind(pf),
                        detail = %pf,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        "failed to load default model, continuing"
                    );
                } else {
                    tracing::warn!(
                        model = %spec.model_id,
                        error = %rendered,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        "failed to load default model, continuing"
                    );
                }
            }
        }
    }
    activation.mark_ready().await;
}

/// Short kebab-case tag for a preflight failure. Used as a structured
/// log field so journalctl filtering can match on the failure class
/// (`reason=tp_requires_safetensors`, `reason=quant_not_found`, etc.).
fn preflight_kind(err: &PreflightError) -> &'static str {
    match err {
        PreflightError::RepoFetchFailed { .. } => "repo_fetch_failed",
        PreflightError::EmptyRepo { .. } => "empty_repo",
        PreflightError::TpRequiresSafetensors { .. } => "tp_requires_safetensors",
        PreflightError::QuantNotFound { .. } => "quant_not_found",
    }
}

/// Future that resolves on SIGINT (Ctrl-C) or SIGTERM (systemd stop).
///
/// Wired into `axum::serve(...).with_graceful_shutdown(shutdown_signal())`
/// so the HTTP listener stops accepting new connections, lets in-flight
/// requests drain, and then yields control back to main for cleanup.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.ok();
    };
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

/// Unload every model currently registered. Called from `main.rs` after
/// axum's graceful shutdown future resolves, so CUDA contexts and VRAM
/// are released before the process exits rather than left to the OS to
/// reclaim. Per-model failures are logged and skipped — keep cleanup
/// going even when one harness is unhealthy.
pub async fn unload_all_models(registry: &HarnessRegistry) {
    let listed = match registry.list_all_models().await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "failed to list models during shutdown");
            return;
        }
    };

    if listed.is_empty() {
        return;
    }

    tracing::info!(count = listed.len(), "unloading models for shutdown");
    let mut stuck = 0;
    for model in listed {
        let start = Instant::now();
        match tokio::time::timeout(UNLOAD_TIMEOUT, registry.unload_model(&model.id)).await {
            Ok(Ok(())) => tracing::info!(
                model = %model.id,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "unloaded"
            ),
            // Most common shape today: TP unload bails because an
            // inference is still mid-flight (the spawned task holds
            // an `Arc<TpLoadedModel>` clone). Promoted from warn to
            // error and tagged with the request-state so the operator
            // can correlate with the chat_completion logs above.
            Ok(Err(e)) => {
                stuck += 1;
                tracing::error!(
                    model = %model.id,
                    error = %e,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "unload failed during shutdown"
                );
            }
            Err(_) => {
                stuck += 1;
                tracing::error!(
                    model = %model.id,
                    timeout_secs = UNLOAD_TIMEOUT.as_secs(),
                    "unload timed out during shutdown, continuing"
                );
            }
        }
    }
    if stuck > 0 {
        tracing::error!(
            stuck,
            "shutdown leaving {stuck} model(s) loaded; VRAM will be \
             reclaimed by the OS on process exit"
        );
    }
}
