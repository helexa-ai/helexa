// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use mesh::MeshHandle;
use protocol::{ModelCapability, RoutingDecision, WorkloadClass};
use tracing::info;

/// trait implemented by orchestrators that make scheduling decisions.
pub trait Scheduler: Send + Sync {
    fn schedule(&self, workload: WorkloadClass) -> RoutingDecision;
}

/// trait responsible for ensuring models are loaded on target neurons.
pub trait Provisioner: Send + Sync {
    fn ensure_model_loaded(&self, model: &ModelCapability);
}

/// simple placeholder scheduler that picks the first available neuron.
pub struct BasicScheduler {
    mesh: MeshHandle,
}

impl BasicScheduler {
    pub fn new(mesh: MeshHandle) -> Self {
        info!(
            "basic scheduler initialised for mesh node {}",
            mesh.node_id()
        );
        Self { mesh }
    }
}

impl Scheduler for BasicScheduler {
    fn schedule(&self, workload: WorkloadClass) -> RoutingDecision {
        // Explicitly use mesh in the scheduling path so the field is clearly intentional.
        info!(
            "scheduling workload {:?} using mesh node {} (placeholder implementation)",
            workload,
            self.mesh.node_id()
        );

        // TODO: replace this with real scheduling logic based on mesh state and neuron capabilities.
        // For now, we still use a trivial default routing decision to keep behaviour predictable.
        RoutingDecision::default_for(workload)
    }
}

pub fn spawn(_addr: SocketAddr, mesh: MeshHandle) {
    info!("starting orchestrator role");
    let _scheduler = BasicScheduler::new(mesh);
    // TODO: listen for control-plane requests from gateway and peers.
}
