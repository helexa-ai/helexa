// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::control_plane::ModelProvisioningStore;
use crate::observe::ObserveBus;
use anyhow::Result;
use tracing::info;

pub mod control_plane;
pub mod gateway;
pub mod mesh;
pub mod observe;
pub mod orchestrator;
pub mod portal;
pub mod shutdown;
pub mod spec;

pub struct Config {
    pub orchestrator_socket: Option<SocketAddr>,
    pub gateway_socket: Option<SocketAddr>,
    pub portal_sockets: Vec<SocketAddr>,
    pub node_id: Option<String>,
    /// Optional path to a cortex spec file used to bootstrap model configs
    /// and demand hints at startup.
    pub spec_path: Option<PathBuf>,
    /// Optional address for the cortex control-plane websocket listener that
    /// neurons will connect to for registration, heartbeats, and provisioning.
    pub control_plane_socket: Option<SocketAddr>,
    /// Optional address for the cortex dashboard / observe websocket listener
    /// that operator dashboards (e.g. Vite/React SPA) will connect to.
    pub dashboard_socket: Option<SocketAddr>,
}

pub async fn run(config: Config) -> Result<()> {
    info!("starting cortex node: {:?}", config.node_id);

    // Load demand/spec state if provided. The resulting state can be consumed
    // by the future orchestrator/provisioner and is also used to seed
    // bootstrap provisioning for newly connected neurons.
    let demand_store = crate::spec::DemandStore::new()?;
    let demand_state: crate::spec::ModelDemandState =
        crate::spec::load_combined_demand_state(config.spec_path.clone(), &demand_store)?;

    let mesh_handle = mesh::start_mesh(config.node_id.clone()).await?;

    if let Some(addr) = config.orchestrator_socket {
        orchestrator::spawn(addr, mesh_handle.clone());
    }

    if let Some(addr) = config.gateway_socket {
        gateway::spawn(addr, mesh_handle.clone());
    }

    // Shared neuron registry for both control-plane and dashboard observers.
    let registry = control_plane::NeuronRegistry::new();
    let model_store = ModelProvisioningStore::new();
    let observe_bus = ObserveBus::new(1024);
    let observe_publisher = observe_bus.publisher();

    if let Some(addr) = config.control_plane_socket {
        let registry_for_control = registry.clone();
        let mesh_for_control = mesh_handle.clone();
        let demand_state_for_control = demand_state.clone();
        let observe_for_control = observe_publisher.clone();
        let model_store_for_control = model_store.clone();
        tokio::spawn(async move {
            if let Err(e) = control_plane::start_control_plane_server(
                addr,
                mesh_for_control,
                registry_for_control,
                demand_state_for_control,
                observe_for_control,
                model_store_for_control,
            )
            .await
            {
                tracing::error!("control-plane server failed on {}: {:?}", addr, e);
            }
        });
    }

    if let Some(addr) = config.dashboard_socket {
        let registry_for_dashboard = registry.clone();
        let events_rx = observe_bus.subscribe();
        let model_store_for_dashboard = model_store.clone();

        tokio::spawn(async move {
            if let Err(e) = observe::start_observe_server(
                addr,
                registry_for_dashboard,
                model_store_for_dashboard,
                events_rx,
            )
            .await
            {
                tracing::error!("dashboard/observe server failed on {}: {:?}", addr, e);
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
