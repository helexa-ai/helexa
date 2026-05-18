//! Harness trait and supporting types for inference engine management.
//!
//! Defined in cortex-core so both cortex (control plane) and neuron
//! (node plane) share the type definitions. neuron provides the
//! runtime implementations.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Configuration for a harness instance on a neuron.
///
/// All current harnesses are in-process (candle); per-harness tuning
/// (cache paths, device policies, etc.) lives in dedicated config
/// blocks rather than on this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub name: String,
}

/// Health status of a harness process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessHealth {
    pub name: String,
    pub running: bool,
    pub uptime_secs: Option<u64>,
}

/// Specification for loading a model through a harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub harness: String,
    pub quant: Option<String>,
    pub tensor_parallel: Option<u32>,
    pub devices: Option<Vec<u32>>,
}

/// A model as reported by a harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub harness: String,
    pub status: String,
    pub devices: Vec<u32>,
    pub vram_used_mb: Option<u64>,
}

/// What an inference harness must do, from neuron's perspective.
///
/// All current harnesses are in-process — they share neuron's address
/// space and lifecycle. `start`/`stop` therefore default to no-ops; a
/// future process-supervising harness would override them.
#[async_trait]
pub trait Harness: Send + Sync {
    /// Human-readable name (e.g. "candle").
    fn name(&self) -> &str;

    /// Start the harness. Default no-op for in-process harnesses.
    async fn start(&self, _config: &HarnessConfig) -> Result<()> {
        Ok(())
    }

    /// Stop the harness. Default no-op for in-process harnesses.
    async fn stop(&self) -> Result<()> {
        Ok(())
    }

    /// Health check. Returns the harness process status.
    async fn health(&self) -> HarnessHealth;

    /// List models the harness knows about (loaded + unloaded).
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Load a model with the given spec (quant, TP, device assignment).
    async fn load_model(&self, spec: &ModelSpec) -> Result<()>;

    /// Unload a model, freeing device memory.
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// Return the URL where inference requests for this model should
    /// be sent. None if the model is not loaded.
    async fn inference_endpoint(&self, model_id: &str) -> Option<String>;
}
