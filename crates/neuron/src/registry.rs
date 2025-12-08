// SPDX-License-Identifier: PolyForm-Shield-1.0

use anyhow::Result;
use model_runtime::ChatRuntimeHandle;

/// tracks locally available models and their runtime bindings.
pub struct ModelRegistry {
    models_dir: Option<String>,
    // TODO: add internal maps from model id to runtime handles
}

impl ModelRegistry {
    pub fn new(models_dir: Option<String>) -> Self {
        Self { models_dir }
    }

    pub fn get_runtime_for_model(&self, model_id: &str) -> Result<ChatRuntimeHandle> {
        // Use models_dir in a clearly intentional way while the real lookup is not implemented.
        if let Some(dir) = &self.models_dir {
            tracing::info!(
                "placeholder lookup for model {} in configured models_dir {}",
                model_id,
                dir
            );
        } else {
            tracing::info!(
                "placeholder lookup for model {} with no models_dir configured",
                model_id
            );
        }

        // TODO: replace this with a real lookup that returns a bound ChatRuntimeHandle.
        unimplemented!("ModelRegistry::get_runtime_for_model is not implemented yet");
    }
}
