//! Model-to-node routing logic.
//!
//! Given a model ID from an inbound request, determine which node should
//! handle it. Priority:
//!   1. Node where the model is currently `Loaded`
//!   2. Node where the model is `Unloaded` (will lazy-load on request)
//!   3. Error: model not found on any node

use crate::state::CortexState;
use cortex_core::node::ModelStatus;
use std::sync::Arc;

/// The routing decision: which node endpoint to proxy the request to.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub node_name: String,
    pub endpoint: String,
    /// Whether the model will need to load (cold start).
    pub cold_start: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("model '{0}' not found on any node")]
    ModelNotFound(String),
    #[error("no healthy nodes available")]
    NoHealthyNodes,
}

/// Resolve which node should serve a request for the given model.
pub async fn resolve(
    fleet: &Arc<CortexState>,
    model_id: &str,
) -> Result<RouteDecision, RouteError> {
    let nodes = fleet.nodes.read().await;

    // Pass 1: find a node where the model is already loaded.
    let mut loaded_candidate = None;
    let mut unloaded_candidate = None;

    for node in nodes.values() {
        if !node.healthy {
            continue;
        }
        if let Some(entry) = node.models.get(model_id) {
            match entry.status {
                ModelStatus::Loaded | ModelStatus::Reloading => {
                    loaded_candidate = Some(RouteDecision {
                        node_name: node.name.clone(),
                        endpoint: node.endpoint.clone(),
                        cold_start: false,
                    });
                    break; // loaded is best, stop searching
                }
                ModelStatus::Unloaded => {
                    if unloaded_candidate.is_none() {
                        unloaded_candidate = Some(RouteDecision {
                            node_name: node.name.clone(),
                            endpoint: node.endpoint.clone(),
                            cold_start: true,
                        });
                    }
                }
            }
        }
    }

    loaded_candidate.or(unloaded_candidate).ok_or_else(|| {
        if nodes.values().any(|n| n.healthy) {
            RouteError::ModelNotFound(model_id.to_string())
        } else {
            RouteError::NoHealthyNodes
        }
    })
}
