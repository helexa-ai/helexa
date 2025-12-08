// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use tracing::info;

use crate::runtime::RuntimeManager;
use protocol::{ModelConfig, NeuronControl, ProvisioningCommand, ProvisioningResponse};

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

    /// Apply an updated model configuration by recording it in the in-memory
    /// model config state and returning a generic success response.
    ///
    /// This does not yet spawn or tear down any backend processes; it only
    /// updates configuration state so that later `LoadModel` / `UnloadModel`
    /// commands can make use of it.
    fn handle_upsert_model_config(&self, cfg: ModelConfig) -> ProvisioningResponse {
        let model_id = cfg.id.clone();
        let configs = self.runtime.model_configs();
        {
            // Update the in-memory configuration map.
            let mut state = futures::executor::block_on(configs.write());
            state.upsert(cfg);
        }

        // For now we do not persist immediately; higher layers can call
        // `persist_model_config_state` at appropriate times (e.g. shutdown).
        ProvisioningResponse::Ok {
            model_id,
            message: Some("configuration updated (no runtime changes yet)".to_string()),
        }
    }
}

impl NeuronControl for NeuronControlImpl {
    /// Apply a provisioning command such as model configuration updates or
    /// load/unload requests.
    ///
    /// `UpsertModelConfig` is partially implemented to update the in-memory
    /// model configuration state; `LoadModel` and `UnloadModel` remain
    /// placeholders that should be wired into process management and the
    /// model registry in subsequent steps.
    fn apply_provisioning(&self, cmd: ProvisioningCommand) -> ProvisioningResponse {
        match cmd {
            ProvisioningCommand::UpsertModelConfig(cfg) => {
                info!("received UpsertModelConfig for model_id={:?}", cfg.id);
                self.handle_upsert_model_config(cfg)
            }
            ProvisioningCommand::LoadModel { model_id } => {
                info!(
                    "received LoadModel for model_id={:?} (placeholder handler)",
                    model_id
                );
                unimplemented!("LoadModel handling is not implemented yet")
            }
            ProvisioningCommand::UnloadModel { model_id } => {
                info!(
                    "received UnloadModel for model_id={:?} (placeholder handler)",
                    model_id
                );
                unimplemented!("UnloadModel handling is not implemented yet")
            }
        }
    }
}
