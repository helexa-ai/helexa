use std::net::SocketAddr;

use protocol::{ModelCapability, RoutingDecision, WorkloadClass};
use mesh::MeshHandle;
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
        Self { mesh }
    }
}

impl Scheduler for BasicScheduler {
    fn schedule(&self, workload: WorkloadClass) -> RoutingDecision {
        // TODO: use mesh state + neuron capabilities
        RoutingDecision::default_for(workload)
    }
}

pub fn spawn(_addr: SocketAddr, mesh: MeshHandle) {
    info!("starting orchestrator role");
    let _scheduler = BasicScheduler::new(mesh);
    // TODO: listen for control-plane requests from gateway and peers.
}
