//! Activation-time pre-warm progress tracking.
//!
//! Wraps the [`ActivationStatus`] snapshot in an async RwLock so the
//! background pre-warm task can update it per-model while the
//! `/health` handler reads coherent snapshots. The tracker exists
//! because `default_models` loading moved from synchronous-before-bind
//! to background-after-bind on 2026-05-26: the listener is up
//! immediately, but `/health` now needs to tell callers which of the
//! configured defaults are still warming.

use cortex_core::discovery::{ActivationState, ActivationStatus, PreWarmFailure};
use cortex_core::harness::ModelSpec;
use tokio::sync::RwLock;

/// Shared, async-safe handle to the daemon's activation progress.
///
/// Construct once in `main` with the configured `default_models` so
/// the initial `pending` list matches the spec; clone the `Arc` into
/// the `NeuronState` for HTTP handlers and into the spawned pre-warm
/// task for updates.
pub struct ActivationTracker {
    inner: RwLock<ActivationStatus>,
}

impl ActivationTracker {
    /// Build a tracker primed with one entry per spec. An empty spec
    /// list yields a `Ready` tracker — no point reporting PreWarming
    /// when there's nothing queued.
    pub fn new(default_models: &[ModelSpec]) -> Self {
        let pending: Vec<String> = default_models.iter().map(|s| s.model_id.clone()).collect();
        let state = if pending.is_empty() {
            ActivationState::Ready
        } else {
            ActivationState::PreWarming
        };
        Self {
            inner: RwLock::new(ActivationStatus {
                state,
                pending,
                in_progress: None,
                completed: vec![],
                failed: vec![],
            }),
        }
    }

    /// Mark a model as in-progress: remove it from `pending`, set as
    /// `in_progress`. Called immediately before `registry.load_model`.
    pub async fn start_loading(&self, model_id: &str) {
        let mut s = self.inner.write().await;
        s.pending.retain(|m| m != model_id);
        s.in_progress = Some(model_id.to_string());
    }

    /// Mark a model as completed: clear `in_progress` (if it matches),
    /// append to `completed`.
    pub async fn complete_loading(&self, model_id: &str) {
        let mut s = self.inner.write().await;
        if s.in_progress.as_deref() == Some(model_id) {
            s.in_progress = None;
        }
        s.completed.push(model_id.to_string());
    }

    /// Mark a model as failed: clear `in_progress` (if it matches),
    /// append a `PreWarmFailure` carrying the rendered error chain.
    pub async fn fail_loading(&self, model_id: &str, error: &str) {
        let mut s = self.inner.write().await;
        if s.in_progress.as_deref() == Some(model_id) {
            s.in_progress = None;
        }
        s.failed.push(PreWarmFailure {
            model_id: model_id.to_string(),
            error: error.to_string(),
        });
    }

    /// Flip the high-level `state` to `Ready` once the pre-warm task
    /// is done iterating. Pending should be empty by this point; if a
    /// caller bails early it's a stuck activation and the operator
    /// will see entries in `pending` even with `state=ready` — that's
    /// a useful diagnostic, not an inconsistency to scrub.
    pub async fn mark_ready(&self) {
        let mut s = self.inner.write().await;
        s.state = ActivationState::Ready;
        s.in_progress = None;
    }

    /// Cheap clone of the current state for the `/health` handler.
    pub async fn snapshot(&self) -> ActivationStatus {
        self.inner.read().await.clone()
    }
}
