use std::net::SocketAddr;

use mesh::MeshHandle;
use protocol::{WorkloadClass, RoutingDecision};
use tracing::info;

use crate::orchestrator::{BasicScheduler, Scheduler};

pub fn spawn(addr: SocketAddr, mesh: MeshHandle) {
    info!("starting gateway role on {}", addr);

    let scheduler = BasicScheduler::new(mesh);

    // TODO: replace with real http server
    tokio::spawn(async move {
        // placeholder pseudo-loop to illustrate the flow
        loop {
            // pretend we received a request and classified it
            let workload = WorkloadClass::ChatInteractive;

            let routing: RoutingDecision = scheduler.schedule(workload);

            // TODO:
            // - use routing decision to contact neuron(s)
            // - forward responses back to client
            let _ = routing;
            break;
        }
    });
}
