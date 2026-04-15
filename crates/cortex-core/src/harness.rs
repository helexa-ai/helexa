//! Harness trait and supporting types for inference engine management.
//!
//! Defined in cortex-core so both cortex (control plane) and neuron
//! (node plane) share the type definitions. neuron provides the
//! runtime implementations.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Configuration for a harness instance on a neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub name: String,
    /// Base URL of the harness (e.g. "http://localhost:8080" for mistral.rs).
    pub endpoint: Option<String>,
    /// Systemd unit name, if the harness is managed via systemd.
    pub systemd_unit: Option<String>,
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
#[async_trait]
pub trait Harness: Send + Sync {
    /// Human-readable name (e.g. "mistralrs", "llamacpp", "comfyui").
    fn name(&self) -> &str;

    /// Start the harness process if it is not already running.
    async fn start(&self, config: &HarnessConfig) -> Result<()>;

    /// Stop the harness process gracefully.
    async fn stop(&self) -> Result<()>;

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
