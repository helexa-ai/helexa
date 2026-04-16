//! Neuron configuration loaded from neuron.toml.

use cortex_core::harness::HarnessConfig;
use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub harnesses: Vec<HarnessConfig>,
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
        }
    }
}
