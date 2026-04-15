//! Harness registry — maps harness names to trait implementations.

pub mod llamacpp;
pub mod mistralrs;

use anyhow::Result;
use cortex_core::harness::{Harness, HarnessConfig, ModelInfo, ModelSpec};
use std::collections::HashMap;

/// Registry of available harness implementations.
pub struct HarnessRegistry {
    harnesses: HashMap<String, Box<dyn Harness>>,
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
        }
    }

    pub fn register(&mut self, harness: Box<dyn Harness>) {
        self.harnesses.insert(harness.name().to_string(), harness);
    }

    /// List all registered harness names.
    pub fn names(&self) -> Vec<String> {
        self.harnesses.keys().cloned().collect()
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
    pub fn from_configs(configs: &[HarnessConfig]) -> Self {
        let mut registry = Self::new();
        for config in configs {
            match config.name.as_str() {
                "mistralrs" => {
                    if let Some(endpoint) = &config.endpoint {
                        registry.register(Box::new(mistralrs::MistralRsHarness::new(
                            endpoint.clone(),
                            config.systemd_unit.clone(),
                        )));
                    } else {
                        tracing::warn!("mistralrs harness missing endpoint, skipping");
                    }
                }
                other => {
                    tracing::warn!(harness = other, "unknown harness type, skipping");
                }
            }
        }
        registry
    }
}
