use std::net::SocketAddr;

use anyhow::Result;
use tracing::info;

pub mod runtime;
pub mod control_plane;
pub mod registry;

pub struct Config {
    pub control_socket: SocketAddr,
    pub api_socket: SocketAddr,
    pub models_dir: Option<String>,
    pub node_id: Option<String>,
}

pub async fn run(config: Config) -> Result<()> {
    info!("starting neuron node: {:?}", config.node_id);

    let registry = registry::ModelRegistry::new(config.models_dir.clone());
    let runtime = runtime::RuntimeManager::new(registry);

    control_plane::spawn(config.control_socket, runtime.clone());

    runtime::spawn_api_server(config.api_socket, runtime).await?;

    Ok(())
}
