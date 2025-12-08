use std::net::SocketAddr;

use tracing::info;

use crate::runtime::RuntimeManager;
use protocol::NeuronControl;

/// neuron implements the control-plane interface expected by cortex.
/// for now this is just a placeholder that logs its start.
pub fn spawn(addr: SocketAddr, runtime: RuntimeManager) {
    info!("starting neuron control-plane on {}", addr);

    // Use the runtime so it is not considered dead code yet.
    let control = NeuronControlImpl::new(runtime);

    // Exercise the placeholder handler so the runtime field is clearly intentional.
    // This will panic at runtime until a real implementation is provided.
    control.handle_placeholder();

    // TODO: start transport server and expose NeuronControl implementation
}

/// example struct that will eventually implement the NeuronControl trait.
pub struct NeuronControlImpl {
    runtime: RuntimeManager,
}

impl NeuronControlImpl {
    pub fn new(runtime: RuntimeManager) -> Self {
        info!("initialising NeuronControlImpl");
        Self { runtime }
    }

    pub fn handle_placeholder(&self) {
        // Explicitly use the runtime so its presence is intentional and obvious.
        info!(
            "placeholder handler using neuron runtime (unimplemented) for control-plane on runtime pointer {:p}",
            &self.runtime as *const RuntimeManager
        );
        unimplemented!("NeuronControlImpl::handle_placeholder is not implemented yet");
    }
}

impl NeuronControl for NeuronControlImpl {
    // TODO: add async methods once the trait is async-friendly via a macro or async-trait
}
