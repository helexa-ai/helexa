// SPDX-License-Identifier: PolyForm-Shield-1.0

use serde::{Deserialize, Serialize};

/// logical identifier for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// workload classification used by the orchestrator and gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadClass {
    ChatInteractive,
    ChatBulk,
    Embedding,
    VisionCaption,
    Other(String),
}

/// description of a model's capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapability {
    pub id: ModelId,
    pub supports_chat: bool,
    pub supports_embeddings: bool,
    pub supports_vision: bool,
    pub max_context_tokens: u32,
}

/// describes a neuron node as seen from the control-plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronDescriptor {
    pub node_id: String,
    pub operator: Option<String>,
    pub cost_hint: Option<f64>,
}

/// routing decision returned by a scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub model: ModelId,
    pub target_neurons: Vec<NeuronDescriptor>,
}

impl RoutingDecision {
    pub fn default_for(workload: WorkloadClass) -> Self {
        let model = match workload {
            WorkloadClass::ChatInteractive => ModelId("default-chat".into()),
            WorkloadClass::ChatBulk => ModelId("bulk-chat".into()),
            WorkloadClass::Embedding => ModelId("default-embedding".into()),
            WorkloadClass::VisionCaption => ModelId("default-vision".into()),
            WorkloadClass::Other(s) => ModelId(s),
        };

        Self {
            model,
            target_neurons: Vec::new(),
        }
    }
}

/// trait implemented by neurons to expose control-plane operations.
/// methods are intentionally left unimplemented in this scaffold and will be
/// defined concretely once the transport is chosen.
pub trait NeuronControl {}
