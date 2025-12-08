// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::info;

use crate::process::ProcessManager;
use crate::registry::ModelRegistry;
use cache::JsonStore;
use model_runtime::{ChatRequest, ChatResponse};
use protocol::{ModelConfig, ModelId};

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
    /// In-memory map of model id → configuration payload as last supplied by cortex.
    ///
    /// This is the authoritative runtime view of model configuration on the neuron.
    /// It is hydrated from `model_config_store` at startup and should be persisted
    /// back to disk whenever configuration changes.
    model_configs: Arc<RwLock<ModelConfigState>>,
}

/// Persistent, cacheable state describing model configurations known to a neuron.
///
/// This is stored as JSON under the helexa cache root and reloaded on startup so
/// that neurons can recover model configuration knowledge across restarts without
/// requiring a static on-disk catalog.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelConfigState {
    /// Mapping from logical model id to its last known configuration.
    pub configs: HashMap<String, ModelConfig>,
}

impl ModelConfigState {
    /// Convert a `ModelId` into the key used in the configs map.
    fn key_for(model_id: &ModelId) -> String {
        model_id.0.clone()
    }

    /// Upsert a configuration for the given model id.
    pub fn upsert(&mut self, cfg: ModelConfig) {
        self.configs.insert(cfg.id.0.clone(), cfg);
    }

    /// Look up a configuration for the given model id.
    pub fn get(&self, model_id: &ModelId) -> Option<&ModelConfig> {
        self.configs.get(&Self::key_for(model_id))
    }

    /// Remove the configuration for the given model id.
    pub fn remove(&mut self, model_id: &ModelId) {
        self.configs.remove(&Self::key_for(model_id));
    }
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
    /// for model definitions under the helexa cache root and eagerly hydrates
    /// the in-memory `ModelConfigState` from that store. The store itself
    /// does not automatically persist changes; higher layers are responsible
    /// for calling [`persist_model_config_state`] or equivalent during
    /// shutdown or after configuration updates.
    pub fn new(registry: ModelRegistry, process_manager: ProcessManager) -> Self {
        let store = JsonStore::new("neuron-model-configs")
            .expect("failed to initialise neuron model config cache store");
        let initial_state: ModelConfigState = store
            .load_or_default()
            .expect("failed to load neuron model config state from cache");
        Self {
            registry: Arc::new(RwLock::new(registry)),
            process_manager: Arc::new(process_manager),
            model_config_store: Arc::new(store),
            model_configs: Arc::new(RwLock::new(initial_state)),
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
    /// - hydrate in-memory model configuration state at startup (already done
    ///   by `new`), and
    /// - persist the latest configuration snapshot during shutdown.
    pub fn model_config_store(&self) -> &Arc<JsonStore> {
        &self.model_config_store
    }

    /// Access the in-memory `ModelConfigState` map.
    ///
    /// This is the primary in-process representation of which models the
    /// neuron knows how to run and how they should be wired. It is backed
    /// by the `model_config_store` on disk.
    pub fn model_configs(&self) -> &Arc<RwLock<ModelConfigState>> {
        &self.model_configs
    }

    /// Persist the current in-memory model configuration state to the cache
    /// store.
    ///
    /// This is intentionally fallible so that callers can log or react to
    /// failures when shutting down.
    pub async fn persist_model_config_state(&self) -> Result<()> {
        let state = self.model_configs.read().await;
        self.model_config_store.save(&*state)?;
        Ok(())
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
