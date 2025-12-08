use std::net::SocketAddr;

use mesh::MeshHandle;
use protocol::{RoutingDecision, WorkloadClass};
use tracing::info;

use crate::orchestrator::{BasicScheduler, Scheduler};

pub fn spawn(addr: SocketAddr, mesh: MeshHandle) {
    info!("starting gateway role on {}", addr);

    let scheduler = BasicScheduler::new(mesh);

    // TODO: replace with real http server
    tokio::spawn(async move {
        // placeholder to illustrate the flow:
        // - classify a request into a WorkloadClass
        // - ask the scheduler for a RoutingDecision
        // - (eventually) dispatch to neuron(s) and stream responses back
        let workload = WorkloadClass::ChatInteractive;

        let routing: RoutingDecision = scheduler.schedule(workload);

        // TODO:
        // - use routing decision to contact neuron(s)
        // - forward responses back to client
        let _ = routing;
    });
}
