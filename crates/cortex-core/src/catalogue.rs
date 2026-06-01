//! Model catalogue — profiles describing how to serve each model.

use crate::discovery::DeviceInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    /// Source scheme this profile's weights come from. When set, the
    /// router prefixes `id` with `scheme:` before forwarding the load
    /// request to neuron, ensuring the daemon fetches from the right
    /// registry regardless of which entry happens to match `id`.
    ///
    /// `None` lets neuron substitute its own `default_source` (typically
    /// `huggingface`). Set to `"helexa"` when the model is hosted in
    /// the helexa registry — operator-procurement-grade audit relies
    /// on this being explicit per model rather than implicit.
    #[serde(default)]
    pub source: Option<String>,
}

fn default_min_devices() -> u32 {
    1
}

/// The full model catalogue.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCatalogue {
    #[serde(default)]
    pub models: Vec<ModelProfile>,
    /// Tier aliases — clients can send a request with `model: "helexa/small"`
    /// and the gateway transparently rewrites + routes to the concrete
    /// model id this maps to. Lets operators define latency/quality
    /// tiers (`small`/`balanced`/`large`, `fast`/`thinking`, etc.)
    /// without imposing knowledge of specific model ids on clients.
    /// Loaded from the `[aliases]` table in models.toml.
    #[serde(default)]
    pub aliases: HashMap<String, String>,
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

    /// Resolve an alias to its concrete model id. Returns `id` verbatim
    /// when it isn't an alias. Aliases never chain — operator config
    /// is treated as flat — so this is a single lookup.
    pub fn resolve_alias<'a>(&'a self, id: &'a str) -> &'a str {
        self.aliases.get(id).map(String::as_str).unwrap_or(id)
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
            source: None,
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

    #[test]
    fn resolve_alias_returns_target_when_alias_present() {
        let mut cat = ModelCatalogue::default();
        cat.aliases
            .insert("helexa/small".into(), "Qwen/Qwen3-1.7B".into());
        assert_eq!(cat.resolve_alias("helexa/small"), "Qwen/Qwen3-1.7B");
    }

    #[test]
    fn resolve_alias_passes_through_when_not_an_alias() {
        let mut cat = ModelCatalogue::default();
        cat.aliases
            .insert("helexa/small".into(), "Qwen/Qwen3-1.7B".into());
        assert_eq!(cat.resolve_alias("Qwen/Qwen3-8B"), "Qwen/Qwen3-8B");
    }

    #[test]
    fn source_defaults_to_none_when_absent_from_toml() {
        let src = r#"
[[models]]
id = "Qwen/Qwen3-30B"
harness = "candle"
"#;
        let cat: ModelCatalogue = toml::from_str(src).expect("parse models table");
        assert!(cat.models[0].source.is_none());
    }

    #[test]
    fn source_round_trips_through_toml() {
        let src = r#"
[[models]]
id = "Helexa/Qwen3.6-27B-Uncensored"
harness = "candle"
source = "helexa"
"#;
        let cat: ModelCatalogue = toml::from_str(src).expect("parse models table");
        assert_eq!(cat.models[0].source.as_deref(), Some("helexa"));
    }

    #[test]
    fn aliases_table_round_trips_through_toml() {
        let src = r#"
[aliases]
"helexa/small" = "Qwen/Qwen3-1.7B"
"helexa/large" = "Qwen/Qwen3.6-27B"
"#;
        let cat: ModelCatalogue = toml::from_str(src).expect("parse aliases table");
        assert_eq!(cat.resolve_alias("helexa/small"), "Qwen/Qwen3-1.7B");
        assert_eq!(cat.resolve_alias("helexa/large"), "Qwen/Qwen3.6-27B");
    }
}
