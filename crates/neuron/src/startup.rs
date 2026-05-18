//! Activation-time orchestration.
//!
//! Wired from `main.rs` after the harness registry is built and before
//! the HTTP listener binds. Kept in its own module so the logic is
//! unit-testable without spinning up a full neuron process.

use crate::harness::HarnessRegistry;
use cortex_core::harness::ModelSpec;
use std::time::Instant;

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
