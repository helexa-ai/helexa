use std::net::SocketAddr;

use anyhow::Result;
use tracing::info;

pub mod gateway;
pub mod mesh;
pub mod orchestrator;
pub mod portal;
pub mod shutdown;

pub struct Config {
    pub orchestrator_socket: Option<SocketAddr>,
    pub gateway_socket: Option<SocketAddr>,
    pub portal_sockets: Vec<SocketAddr>,
    pub node_id: Option<String>,
}

pub async fn run(config: Config) -> Result<()> {
    info!("starting cortex node: {:?}", config.node_id);

    let mesh_handle = mesh::start_mesh(config.node_id.clone()).await?;

    if let Some(addr) = config.orchestrator_socket {
        orchestrator::spawn(addr, mesh_handle.clone());
    }

    if let Some(addr) = config.gateway_socket {
        gateway::spawn(addr, mesh_handle.clone());
    }

    for addr in &config.portal_sockets {
        portal::spawn(*addr, mesh_handle.clone());
    }

    shutdown::wait_for_signal().await;
    info!("cortex node shutting down");

    Ok(())
}
