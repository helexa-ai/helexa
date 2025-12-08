use anyhow::Result;
use serde::Deserialize;

/// placeholder for a root configuration structure.
/// this can be expanded as the project grows.
#[derive(Debug, Deserialize)]
pub struct HelexaConfig {
    pub node_id: Option<String>,
}

pub fn load_from_file(_path: &str) -> Result<HelexaConfig> {
    // TODO: real loading + error messages
    Ok(HelexaConfig { node_id: None })
}
