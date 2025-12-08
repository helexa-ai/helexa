// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::info;

use crate::process::ProcessManager;
use crate::registry::ModelRegistry;
use model_runtime::{ChatRequest, ChatResponse};

#[derive(Clone)]
pub struct RuntimeManager {
    registry: Arc<RwLock<ModelRegistry>>,
    process_manager: Arc<ProcessManager>,
}

impl RuntimeManager {
    /// Create a new runtime manager with an associated model registry and
    /// process manager.
    ///
    /// The process manager is responsible for spawning and tracking external
    /// backend processes (e.g. vLLM or llama.cpp instances), while the
    /// registry owns logical model → runtime bindings.
    pub fn new(registry: ModelRegistry, process_manager: ProcessManager) -> Self {
        Self {
            registry: Arc::new(RwLock::new(registry)),
            process_manager: Arc::new(process_manager),
        }
    }

    /// Access the underlying process manager.
    ///
    /// This is primarily intended for future control-plane operations such as
    /// explicit model load/unload directives that need to spawn or terminate
    /// backend workers.
    pub fn process_manager(&self) -> &Arc<ProcessManager> {
        &self.process_manager
    }

    pub async fn execute_chat(&self, model_id: &str, request: ChatRequest) -> Result<ChatResponse> {
        let registry = self.registry.read().await;
        let runtime = registry.get_runtime_for_model(model_id)?;
        runtime.chat(request).await
    }
}

pub async fn spawn_api_server(_addr: SocketAddr, _runtime: RuntimeManager) -> Result<()> {
    info!("starting neuron api server on {}", _addr);
    // TODO: implement local api server (http/grpc/etc)
    Ok(())
}
