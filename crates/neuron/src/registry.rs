// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use model_runtime::ChatRuntimeHandle;

/// tracks locally available models and their runtime bindings.
pub struct ModelRegistry {
    models_dir: Option<String>,
    /// In-memory mapping from model id → runtime handle and optional worker handle.
    ///
    /// This will eventually be populated by control-plane directives (e.g.
    /// "load model") that spawn backend processes via the process manager and
    /// then bind them to logical model identifiers.
    entries: HashMap<String, ModelEntry>,
}

/// describes a single model entry known to the registry.
pub struct ModelEntry {
    /// Handle to a chat-capable runtime for this model.
    pub runtime: ChatRuntimeHandle,
    /// Future extension point: track which backend worker(s) serve this model.
    /// For now this is left as an opaque string or identifier; it can be
    /// widened to a concrete handle type when the process manager integration
    /// is fully wired.
    pub worker_id: Option<String>,
}

impl ModelRegistry {
    pub fn new(models_dir: Option<String>) -> Self {
        Self {
            models_dir,
            entries: HashMap::new(),
        }
    }

    /// Register a chat-capable model runtime under the given identifier.
    ///
    /// This is a preparatory API that allows control-plane or startup code to
    /// bind logical model ids to concrete runtime handles. For now it simply
    /// records the mapping in memory; persistence and richer metadata can be
    /// added later.
    pub fn register_chat_model(
        &mut self,
        model_id: String,
        runtime: ChatRuntimeHandle,
        worker_id: Option<String>,
    ) {
        let entry = ModelEntry { runtime, worker_id };
        self.entries.insert(model_id, entry);
    }

    pub fn get_runtime_for_model(&self, model_id: &str) -> Result<ChatRuntimeHandle> {
        // Use models_dir in a clearly intentional way while the real lookup is not implemented.
        if let Some(dir) = &self.models_dir {
            tracing::info!(
                "lookup runtime for model {} in configured models_dir {}",
                model_id,
                dir
            );
        } else {
            tracing::info!(
                "lookup runtime for model {} with no models_dir configured",
                model_id
            );
        }

        // Look up a registered runtime for this model id.
        if let Some(entry) = self.entries.get(model_id) {
            return Ok(entry.runtime.clone());
        }

        Err(anyhow!(
            "no runtime registered for model_id={}; did you forget to call register_chat_model?",
            model_id
        ))
    }
}
