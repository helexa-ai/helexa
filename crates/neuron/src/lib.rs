// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use anyhow::Result;
use tracing::info;

pub mod control_plane;
pub mod process;
pub mod registry;
pub mod runtime;

pub struct Config {
    pub control_socket: SocketAddr,
    pub api_socket: SocketAddr,
    pub models_dir: Option<String>,
    pub node_id: Option<String>,
}

pub async fn run(config: Config) -> Result<()> {
    info!("starting neuron node: {:?}", config.node_id);

    let registry = registry::ModelRegistry::new(config.models_dir.clone());
    let process_manager = process::ProcessManager::new();
    let runtime = runtime::RuntimeManager::new(registry, process_manager);

    control_plane::spawn(config.control_socket, runtime.clone());

    runtime::spawn_api_server(config.api_socket, runtime).await?;

    Ok(())
}
