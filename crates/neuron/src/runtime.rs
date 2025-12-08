use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::info;

use crate::registry::ModelRegistry;
use model_runtime::{ChatInference, ChatRequest, ChatResponse};

#[derive(Clone)]
pub struct RuntimeManager {
    registry: Arc<RwLock<ModelRegistry>>,
}

impl RuntimeManager {
    pub fn new(registry: ModelRegistry) -> Self {
        Self {
            registry: Arc::new(RwLock::new(registry)),
        }
    }

    pub async fn execute_chat(
        &self,
        model_id: &str,
        request: ChatRequest,
    ) -> Result<ChatResponse> {
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
