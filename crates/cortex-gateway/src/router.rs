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
    /// The inference endpoint to proxy to (from neuron's /models/{id}/endpoint).
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
    #[error("failed to resolve inference endpoint for model '{0}' on node '{1}'")]
    EndpointResolveFailed(String, String),
}

/// Resolve which node should serve a request for the given model.
/// Asks the neuron for the inference endpoint after selecting a node.
pub async fn resolve(
    fleet: &Arc<CortexState>,
    model_id: &str,
) -> Result<RouteDecision, RouteError> {
    let (node_name, neuron_endpoint, cold_start) = {
        let nodes = fleet.nodes.read().await;

        let mut loaded_candidate = None;
        let mut unloaded_candidate = None;

        for node in nodes.values() {
            if !node.healthy {
                continue;
            }
            if let Some(entry) = node.models.get(model_id) {
                match entry.status {
                    ModelStatus::Loaded | ModelStatus::Reloading => {
                        loaded_candidate = Some((node.name.clone(), node.endpoint.clone(), false));
                        break;
                    }
                    ModelStatus::Unloaded => {
                        if unloaded_candidate.is_none() {
                            unloaded_candidate =
                                Some((node.name.clone(), node.endpoint.clone(), true));
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
        })?
    };

    // Ask the neuron for the inference endpoint for this model.
    let endpoint_url = format!(
        "{}/models/{}/endpoint",
        neuron_endpoint,
        urlencoding::encode(model_id)
    );

    let inference_endpoint = match fleet.http_client.get(&endpoint_url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(body) => body
                .get("url")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            Err(_) => None,
        },
        _ => None,
    };

    let endpoint = inference_endpoint.ok_or_else(|| {
        RouteError::EndpointResolveFailed(model_id.to_string(), node_name.clone())
    })?;

    Ok(RouteDecision {
        node_name,
        endpoint,
        cold_start,
    })
}
