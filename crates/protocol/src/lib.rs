// SPDX-License-Identifier: PolyForm-Shield-1.0

use serde::{Deserialize, Serialize};

/// logical identifier for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// configuration payload for a model as understood by neurons.
///
/// this is sent from cortex to neuron over the control channel when
/// provisioning or reconfiguring a model. it is intentionally transport-agnostic
/// and does not assume any particular backend implementation, though some
/// fields (like `backend_kind`) are hints used by `neuron::runtime` to decide
/// which process runner or adapter to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// logical model identifier, typically matching the external name or slug.
    pub id: ModelId,
    /// human-readable description or display name for operators.
    pub display_name: Option<String>,
    /// opaque backend kind hint (e.g. "vllm", "llama_cpp", "openai_proxy").
    pub backend_kind: String,
    /// command used to spawn the backend process, if applicable.
    pub command: Option<String>,
    /// arguments passed to the backend process.
    pub args: Vec<String>,
    /// additional environment variables for the backend process.
    pub env: Vec<EnvVar>,
    /// address or url where the backend will listen for requests (e.g. http endpoint).
    pub listen_endpoint: Option<String>,
    /// optional free-form metadata for future extension (e.g. quantisation, tags).
    pub metadata: serde_json::Value,
}

/// a single environment variable entry for backend processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    pub key: String,
    pub value: String,
}

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

/// provisioning commands sent from cortex to neuron over the control plane.
///
/// these commands allow cortex to drive dynamic model configuration and
/// lifecycle without requiring neurons to have a static on-disk model catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProvisioningCommand {
    /// inform a neuron of a new or updated model configuration.
    ///
    /// this does *not* necessarily mean the model is loaded yet; it just
    /// makes the configuration available so that a subsequent `LoadModel`
    /// can use it.
    UpsertModelConfig(ModelConfig),

    /// request that a neuron loads a model into an active serving state.
    ///
    /// neurons interpret this using their current configuration for the model
    /// (e.g. spawning a backend process via `ProcessManager` and wiring a
    /// corresponding runtime handle into the registry).
    LoadModel { model_id: ModelId },

    /// request that a neuron gracefully unloads a model.
    ///
    /// this typically implies terminating associated backend processes and
    /// removing the model from the registry of available runtimes.
    UnloadModel { model_id: ModelId },
}

/// responses from neuron to cortex acknowledging provisioning commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProvisioningResponse {
    /// generic success for a provisioning operation.
    Ok {
        model_id: ModelId,
        message: Option<String>,
    },
    /// provisioning failed for the given model.
    Error { model_id: ModelId, error: String },
}

/// trait implemented by neurons to expose control-plane operations.
/// methods are intentionally left unimplemented in this scaffold and will be
/// defined concretely once the transport is chosen.
///
/// higher layers (e.g. grpc/http/websocket servers) will translate from
/// concrete transport messages into calls on this trait.
pub trait NeuronControl {
    /// apply a provisioning command such as `UpsertModelConfig`, `LoadModel`,
    /// or `UnloadModel`.
    fn apply_provisioning(&self, cmd: ProvisioningCommand) -> ProvisioningResponse;
}
