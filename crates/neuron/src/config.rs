//! Neuron configuration loaded from neuron.toml.

use cortex_core::harness::{HarnessConfig, ModelSpec};
use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub harnesses: Vec<HarnessConfig>,
    /// Per-harness configuration. Currently only `candle` is recognised.
    #[serde(default)]
    pub harness: HarnessSettings,
    /// Models to auto-load when the neuron service activates. Each entry
    /// is loaded sequentially before the HTTP listener binds. A failure
    /// on any single entry logs a warning and proceeds — broken entries
    /// don't prevent the rest of the fleet from starting.
    #[serde(default)]
    pub default_models: Vec<ModelSpec>,
}

/// Settings for individual harness implementations. Each harness owns
/// its own sub-table so users only configure the harnesses they enable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarnessSettings {
    #[serde(default)]
    pub candle: CandleHarnessConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandleHarnessConfig {
    /// HuggingFace cache directory for model weights.
    /// When unset, defers to hf-hub's default (~/.cache/huggingface).
    #[serde(default)]
    pub hf_cache: Option<PathBuf>,
}

fn default_port() -> u16 {
    13131
}

impl NeuronConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("NEURON_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            port: 13131,
            harnesses: vec![],
            harness: HarnessSettings::default(),
            default_models: vec![],
        }
    }
}
