use std::net::SocketAddr;

use tracing::info;

use protocol::NeuronControl;
use crate::runtime::RuntimeManager;

/// neuron implements the control-plane interface expected by cortex.
/// for now this is just a placeholder that logs its start.
pub fn spawn(addr: SocketAddr, _runtime: RuntimeManager) {
    info!("starting neuron control-plane on {}", addr);

    // TODO: start transport server and expose NeuronControl implementation
}

/// example struct that will eventually implement the NeuronControl trait.
pub struct NeuronControlImpl {
    runtime: RuntimeManager,
}

impl NeuronControlImpl {
    pub fn new(runtime: RuntimeManager) -> Self {
        Self { runtime }
    }
}

impl NeuronControl for NeuronControlImpl {
    // TODO: add async methods once the trait is async-friendly via a macro or async-trait
}
