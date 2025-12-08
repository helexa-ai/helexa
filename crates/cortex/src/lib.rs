// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use anyhow::Result;
use tracing::info;

pub mod control_plane;
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
    /// optional address for the cortex control-plane websocket listener that
    /// neurons will connect to for registration, heartbeats, and provisioning.
    pub control_plane_socket: Option<SocketAddr>,
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

    if let Some(addr) = config.control_plane_socket {
        let registry = control_plane::NeuronRegistry::new();
        let mesh_for_control = mesh_handle.clone();
        tokio::spawn(async move {
            if let Err(e) =
                control_plane::start_control_plane_server(addr, mesh_for_control, registry).await
            {
                tracing::error!("control-plane server failed on {}: {:?}", addr, e);
            }
        });
    }

    for addr in &config.portal_sockets {
        portal::spawn(*addr, mesh_handle.clone());
    }

    shutdown::wait_for_signal().await;
    info!("cortex node shutting down");

    Ok(())
}
