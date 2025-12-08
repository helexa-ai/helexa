// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use tracing::info;

use crate::runtime::RuntimeManager;
use protocol::{ModelConfig, ModelId, NeuronControl, ProvisioningCommand, ProvisioningResponse};

/// neuron implements the control-plane interface expected by cortex.
/// for now this is just a placeholder that logs its start.
pub fn spawn(addr: SocketAddr, runtime: RuntimeManager) {
    info!("starting neuron control-plane on {}", addr);

    let _control = NeuronControlImpl::new(runtime);

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
        info!(
            "placeholder handler using neuron runtime for control-plane on runtime pointer {:p}",
            &self.runtime as *const RuntimeManager
        );
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

        ProvisioningResponse::Ok {
            model_id,
            message: Some("configuration updated".to_string()),
        }
    }

    /// Handle a request to load a model by:
    /// - looking up its configuration from ModelConfigState
    /// - spawning a backend process via ProcessManager (placeholder)
    /// - registering a runtime handle in the ModelRegistry (placeholder)
    fn handle_load_model(&self, model_id: ModelId) -> ProvisioningResponse {
        let configs = self.runtime.model_configs();
        let cfg_opt = {
            let state = futures::executor::block_on(configs.read());
            state.get(&model_id).cloned()
        };

        let Some(cfg) = cfg_opt else {
            return ProvisioningResponse::Error {
                model_id,
                error: "no configuration found for model; send UpsertModelConfig first".to_string(),
            };
        };

        info!(
            "handle_load_model: would spawn backend for model_id={:?} with backend_kind={} command={:?} args={:?}",
            cfg.id, cfg.backend_kind, cfg.command, cfg.args
        );

        // TODO:
        // - use self.runtime.process_manager() to spawn a worker with cfg.command/args/env
        // - construct a ProcessRuntime pointing at cfg.listen_endpoint
        // - wrap it in ChatRuntimeHandle and register via ModelRegistry

        ProvisioningResponse::Ok {
            model_id: cfg.id,
            message: Some("load requested (runtime wiring not implemented yet)".to_string()),
        }
    }

    /// Handle a request to unload a model by:
    /// - instructing the process manager to terminate workers
    /// - removing the model from the registry and config state (placeholders)
    fn handle_unload_model(&self, model_id: ModelId) -> ProvisioningResponse {
        info!(
            "handle_unload_model: would terminate backend workers and unregister model_id={:?}",
            model_id
        );

        // TODO:
        // - call self.runtime.process_manager().terminate_workers_for_model(...)
        // - remove from ModelRegistry and ModelConfigState

        ProvisioningResponse::Ok {
            model_id,
            message: Some("unload requested (runtime teardown not implemented yet)".to_string()),
        }
    }
}

impl NeuronControl for NeuronControlImpl {
    /// Apply a provisioning command such as model configuration updates or
    /// load/unload requests.
    ///
    /// `UpsertModelConfig` updates the in-memory model configuration state.
    /// `LoadModel` and `UnloadModel` are wired to dedicated handlers that
    /// currently log intent and return success responses; the actual process
    /// management and registry wiring will be implemented next.
    fn apply_provisioning(&self, cmd: ProvisioningCommand) -> ProvisioningResponse {
        match cmd {
            ProvisioningCommand::UpsertModelConfig(cfg) => {
                info!("received UpsertModelConfig for model_id={:?}", cfg.id);
                self.handle_upsert_model_config(cfg)
            }
            ProvisioningCommand::LoadModel { model_id } => {
                info!("received LoadModel for model_id={:?}", model_id);
                self.handle_load_model(model_id)
            }
            ProvisioningCommand::UnloadModel { model_id } => {
                info!("received UnloadModel for model_id={:?}", model_id);
                self.handle_unload_model(model_id)
            }
        }
    }
}
