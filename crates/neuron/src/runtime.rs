// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::info;

use crate::process::ProcessManager;
use crate::registry::ModelRegistry;
use cache::JsonStore;
use model_runtime::{ChatRequest, ChatResponse};

#[derive(Clone)]
pub struct RuntimeManager {
    registry: Arc<RwLock<ModelRegistry>>,
    process_manager: Arc<ProcessManager>,
    /// JSON-backed cache store for model configuration state as learned from cortex.
    ///
    /// On startup, this is used to hydrate in-memory configuration from the last
    /// successful shutdown. On shutdown, higher layers should persist the current
    /// configuration back to disk via this store.
    model_config_store: Arc<JsonStore>,
}

impl RuntimeManager {
    /// Create a new runtime manager with an associated model registry and
    /// process manager.
    ///
    /// The process manager is responsible for spawning and tracking external
    /// backend processes (e.g. vLLM or llama.cpp instances), while the
    /// registry owns logical model → runtime bindings.
    ///
    /// This constructor also initialises a JSON-backed configuration store
    /// for model definitions under the helexa cache root. The store itself
    /// does not load or persist any data automatically; higher layers are
    /// responsible for calling into it during startup and shutdown.
    pub fn new(registry: ModelRegistry, process_manager: ProcessManager) -> Self {
        let store = JsonStore::new("neuron-model-configs")
            .expect("failed to initialise neuron model config cache store");
        Self {
            registry: Arc::new(RwLock::new(registry)),
            process_manager: Arc::new(process_manager),
            model_config_store: Arc::new(store),
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

    /// Access the JSON-backed model configuration store.
    ///
    /// Callers can use this to:
    /// - hydrate in-memory model configuration state at startup, and
    /// - persist the latest configuration snapshot during shutdown.
    pub fn model_config_store(&self) -> &Arc<JsonStore> {
        &self.model_config_store
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
