use anyhow::{Result, anyhow};
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

    pub fn get_runtime_for_model(&self, _model_id: &str) -> Result<ChatRuntimeHandle> {
        // TODO: lookup real runtime by id
        Err(anyhow!("no runtime registered for model"))
    }
}
