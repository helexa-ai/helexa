use std::sync::Arc;

use tracing::info;

/// handle representing participation in the mesh network.
/// this is a placeholder for now.
#[derive(Clone)]
pub struct MeshHandle {
    node_id: Arc<String>,
}

impl MeshHandle {
    pub fn new(node_id: String) -> Self {
        info!("creating mesh handle for {}", node_id);
        Self {
            node_id: Arc::new(node_id),
        }
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    // TODO: message sending/receiving apis
}
