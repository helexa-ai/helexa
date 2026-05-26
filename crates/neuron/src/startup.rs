//! Activation- and deactivation-time orchestration.
//!
//! Wired from `main.rs` around the HTTP listener — activation runs
//! before bind, deactivation runs after axum returns from its
//! graceful-shutdown future. Kept in its own module so the logic is
//! unit-testable without spinning up a full neuron process.

use crate::harness::HarnessRegistry;
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
/// boring but correct. The function logs elapsed time per load so an
/// operator can see which model is hogging activation.
pub async fn load_default_models(registry: &HarnessRegistry, specs: &[ModelSpec]) {
    if specs.is_empty() {
        return;
    }
    tracing::info!(count = specs.len(), "loading default models");
    for spec in specs {
        let start = Instant::now();
        match registry.load_model(spec).await {
            Ok(()) => tracing::info!(
                model = %spec.model_id,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "loaded default model"
            ),
            Err(e) => tracing::warn!(
                model = %spec.model_id,
                error = %e,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "failed to load default model, continuing"
            ),
        }
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
