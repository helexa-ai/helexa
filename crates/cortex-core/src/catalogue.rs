//! Model catalogue — profiles describing how to serve each model.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A model serving profile loaded from models.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    pub harness: String,
    #[serde(default)]
    pub quant: Option<String>,
    /// Estimated VRAM usage in MB when loaded.
    #[serde(default)]
    pub vram_mb: Option<u64>,
    /// Minimum number of GPU devices required.
    #[serde(default = "default_min_devices")]
    pub min_devices: u32,
    /// Minimum VRAM per device in MB.
    #[serde(default)]
    pub min_device_vram_mb: Option<u64>,
    /// Neurons where this model should never be evicted.
    #[serde(default)]
    pub pinned_on: Vec<String>,
}

fn default_min_devices() -> u32 {
    1
}

/// The full model catalogue.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCatalogue {
    #[serde(default)]
    pub models: Vec<ModelProfile>,
}

impl ModelCatalogue {
    /// Load the catalogue from a TOML file. Returns empty catalogue if file doesn't exist.
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        if !path.exists() {
            tracing::info!(path = %path.display(), "no model catalogue found, using empty");
            return Self::default();
        }
        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(cat) => cat,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to parse model catalogue");
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to read model catalogue");
                Self::default()
            }
        }
    }

    /// Check if a model is pinned on a given neuron.
    pub fn is_pinned(&self, model_id: &str, neuron_name: &str) -> bool {
        self.models
            .iter()
            .any(|p| p.id == model_id && p.pinned_on.contains(&neuron_name.to_string()))
    }
}
