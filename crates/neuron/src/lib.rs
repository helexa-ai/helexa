// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use anyhow::Result;
use tracing::info;

pub mod control_plane;
pub mod process;
pub mod registry;
pub mod runtime;

#[derive(Clone)]
pub struct Config {
    pub control_socket: SocketAddr,
    pub api_socket: SocketAddr,
    pub models_dir: Option<String>,
    pub node_id: Option<String>,
    /// URL of the cortex control-plane websocket endpoint this neuron should
    /// connect to for registration, heartbeats and provisioning commands.
    pub cortex_control_endpoint: String,
}

pub async fn run(config: Config) -> Result<()> {
    info!("starting neuron node: {:?}", config.node_id);

    let registry = registry::ModelRegistry::new(config.models_dir.clone());
    let process_manager = process::ProcessManager::new();
    let runtime = runtime::RuntimeManager::new(registry, process_manager, config.clone());

    control_plane::spawn(config.control_socket, runtime.clone());

    runtime::spawn_api_server(config.api_socket, runtime).await?;

    // keep the neuron process alive until a shutdown signal, mirroring cortex
    tokio::signal::ctrl_c().await?;
    info!("neuron node shutting down");

    Ok(())
}
