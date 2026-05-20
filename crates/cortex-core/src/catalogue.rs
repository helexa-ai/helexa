//! Model catalogue — profiles describing how to serve each model.

use crate::discovery::DeviceInfo;
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

    /// Find a profile by model id.
    pub fn get(&self, model_id: &str) -> Option<&ModelProfile> {
        self.models.iter().find(|p| p.id == model_id)
    }
}

impl ModelProfile {
    /// True iff this profile's placement constraints can be satisfied
    /// by the named neuron with the given device topology.
    ///
    /// Constraints checked:
    /// - `pinned_on`: non-empty → neuron must be on the list.
    /// - `min_devices`: neuron must have at least this many devices.
    /// - `min_device_vram_mb`: at least `min_devices` of the neuron's
    ///   devices must each meet this VRAM floor.
    pub fn is_feasible_on(&self, neuron_name: &str, devices: &[DeviceInfo]) -> bool {
        if !self.pinned_on.is_empty() && !self.pinned_on.iter().any(|n| n == neuron_name) {
            return false;
        }
        if (devices.len() as u32) < self.min_devices {
            return false;
        }
        if let Some(min_vram) = self.min_device_vram_mb {
            let big_enough = devices
                .iter()
                .filter(|d| d.vram_total_mb >= min_vram)
                .count() as u32;
            if big_enough < self.min_devices {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::DeviceInfo;

    fn device(idx: u32, vram_mb: u64) -> DeviceInfo {
        DeviceInfo {
            index: idx,
            name: format!("DEV-{idx}"),
            vram_total_mb: vram_mb,
            compute_capability: "8.6".into(),
        }
    }

    fn profile() -> ModelProfile {
        ModelProfile {
            id: "Qwen/Qwen3.6-27B".into(),
            harness: "candle".into(),
            quant: None,
            vram_mb: Some(45_000),
            min_devices: 2,
            min_device_vram_mb: Some(24_000),
            pinned_on: vec![],
        }
    }

    #[test]
    fn feasible_when_two_devices_meet_vram_floor() {
        let p = profile();
        let devices = [device(0, 32_000), device(1, 32_000)];
        assert!(p.is_feasible_on("beast", &devices));
    }

    #[test]
    fn infeasible_when_only_one_device() {
        let p = profile();
        let devices = [device(0, 64_000)];
        assert!(!p.is_feasible_on("benjy", &devices));
    }

    #[test]
    fn infeasible_when_one_device_underspec() {
        let p = profile();
        let devices = [device(0, 32_000), device(1, 12_000)];
        assert!(!p.is_feasible_on("mixed", &devices));
    }

    #[test]
    fn pinned_on_excludes_other_neurons() {
        let mut p = profile();
        p.pinned_on = vec!["beast".into()];
        let devices = [device(0, 32_000), device(1, 32_000)];
        assert!(p.is_feasible_on("beast", &devices));
        assert!(!p.is_feasible_on("benjy", &devices));
    }

    #[test]
    fn no_vram_floor_just_needs_min_devices() {
        let mut p = profile();
        p.min_device_vram_mb = None;
        let devices = [device(0, 1_000), device(1, 1_000)];
        assert!(p.is_feasible_on("anywhere", &devices));
    }
}
