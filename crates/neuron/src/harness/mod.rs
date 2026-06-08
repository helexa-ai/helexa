//! Harness registry — maps harness names to trait implementations.

pub mod arch;
pub mod candle;
pub mod chat_template;
pub mod device_worker;
pub mod preflight;
pub mod preprocess;
pub mod tp;

use anyhow::Result;
use cortex_core::harness::{Harness, HarnessConfig, ModelInfo, ModelSpec};
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of available harness implementations.
///
/// Holds an `Arc<dyn Harness>` per harness for generic lifecycle dispatch
/// (load/unload/list_models). When a candle harness is registered, a typed
/// `Arc<CandleHarness>` is also cached so inference routes can bypass the
/// dyn-Trait dispatch and reach harness-specific methods (chat completion,
/// streaming, etc.).
pub struct HarnessRegistry {
    harnesses: HashMap<String, Arc<dyn Harness>>,
    candle: Option<Arc<candle::CandleHarness>>,
}

impl Default for HarnessRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HarnessRegistry {
    pub fn new() -> Self {
        Self {
            harnesses: HashMap::new(),
            candle: None,
        }
    }

    pub fn register(&mut self, harness: Arc<dyn Harness>) {
        self.harnesses.insert(harness.name().to_string(), harness);
    }

    /// List all registered harness names.
    pub fn names(&self) -> Vec<String> {
        self.harnesses.keys().cloned().collect()
    }

    /// Typed handle to the candle harness, if registered. Used by inference
    /// routes that need methods beyond the `Harness` trait surface.
    pub fn candle(&self) -> Option<Arc<candle::CandleHarness>> {
        self.candle.clone()
    }

    /// List models from all registered harnesses.
    pub async fn list_all_models(&self) -> Result<Vec<ModelInfo>> {
        let mut all = Vec::new();
        for harness in self.harnesses.values() {
            match harness.list_models().await {
                Ok(models) => all.extend(models),
                Err(e) => {
                    tracing::warn!(harness = harness.name(), error = %e, "failed to list models");
                }
            }
        }
        Ok(all)
    }

    /// Load a model on the specified harness.
    pub async fn load_model(&self, spec: &ModelSpec) -> Result<()> {
        let harness = self
            .harnesses
            .get(&spec.harness)
            .ok_or_else(|| anyhow::anyhow!("unknown harness: {}", spec.harness))?;
        harness.load_model(spec).await
    }

    /// Unload a model. Tries each harness until one claims it.
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        for harness in self.harnesses.values() {
            match harness.list_models().await {
                Ok(models) if models.iter().any(|m| m.id == model_id) => {
                    return harness.unload_model(model_id).await;
                }
                _ => continue,
            }
        }
        anyhow::bail!("model '{model_id}' not found on any harness")
    }

    /// Get the inference endpoint for a model.
    pub async fn inference_endpoint(&self, model_id: &str) -> Option<String> {
        for harness in self.harnesses.values() {
            if let Some(url) = harness.inference_endpoint(model_id).await {
                return Some(url);
            }
        }
        None
    }

    /// Build a registry from harness configs.
    ///
    /// `bind_url` is the URL where this neuron serves inference (its own
    /// listen address). In-process harnesses (currently the only kind)
    /// return this URL from `inference_endpoint`.
    pub fn from_configs(
        configs: &[HarnessConfig],
        bind_url: &str,
        settings: &crate::config::HarnessSettings,
    ) -> Self {
        let mut registry = Self::new();
        for config in configs {
            match config.name.as_str() {
                "candle" => {
                    let harness =
                        candle::CandleHarness::new(bind_url.to_string(), &settings.candle);
                    registry.candle = Some(Arc::clone(&harness));
                    registry.harnesses.insert("candle".into(), harness);
                }
                other => {
                    tracing::warn!(harness = other, "unknown harness type, skipping");
                }
            }
        }
        registry
    }
}
