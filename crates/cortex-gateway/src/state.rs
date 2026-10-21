use crate::entitlements_chain::ChainedEntitlementProvider;
use crate::entitlements_local::LocalEntitlementProvider;
use crate::entitlements_upstream::UpstreamEntitlementProvider;
use cortex_core::catalogue::ModelCatalogue;
use cortex_core::config::{EvictionSettings, GatewayConfig, NeuronEndpoint};
use cortex_core::entitlements::EntitlementProvider;
use cortex_core::node::NodeState;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared fleet state, protected by a RwLock for concurrent reader access.
pub struct CortexState {
    pub nodes: RwLock<HashMap<String, NodeState>>,
    pub neuron_configs: Vec<NeuronEndpoint>,
    pub eviction: EvictionSettings,
    pub catalogue: ModelCatalogue,
    pub http_client: reqwest::Client,
    /// Resolves bearer keys to principals and enforces token budgets (#47).
    /// A local/static provider today (#50); the upstream client later (#57).
    pub entitlements: Arc<dyn EntitlementProvider>,
    /// Whether to reject unauthenticated requests (#49). Read by the auth
    /// middleware once it lands.
    pub require_auth: bool,
    /// Per-principal served-token tally (#58), reported to upstream for
    /// operator reconciliation by the flush task when upstream is enabled.
    pub served_usage: Arc<crate::served_usage::ServedUsage>,
    /// Modalities discovered for models that are not currently loaded
    /// (#241), keyed by model id.
    ///
    /// A loaded model reports its own capabilities every poll, so this
    /// only carries the cold ones — which is the case that mattered,
    /// since the image model spends nearly all its time evicted and was
    /// therefore advertising nothing. Neurons derive these from their
    /// local model cache; cortex asks once and keeps the answer, because
    /// what a given model id can do does not change.
    pub discovered_capabilities: RwLock<HashMap<String, Vec<String>>>,
    /// When each `(neuron, model id)` pair was last probed for
    /// capabilities.
    ///
    /// A model no node has cached can never be answered, and the poll
    /// loop runs every few seconds — without this it would ask every
    /// node about every such model forever, turning a one-off lookup
    /// into steady background traffic and a stream of 404s in the logs.
    /// Resolved ids stop being probed entirely.
    ///
    /// Keyed per node, not per model: nodes are polled in config order,
    /// so a model-only key let whichever node came first record the
    /// attempt and then, if it answered 404, suppress the question to
    /// every other node for the whole back-off window. Weights are not
    /// distributed evenly across the fleet, so that is the common case
    /// rather than an edge — it hid a GGUF that two of three nodes could
    /// describe perfectly well.
    pub capability_probe_attempts: RwLock<HashMap<(String, String), std::time::Instant>>,
    /// Async audio generation jobs (#382).
    pub audio_jobs: crate::audio_jobs::AudioJobs,
    /// Per-model rates for non-token units (#386), from upstream when it
    /// is enabled, else the local default.
    pub rates: Arc<crate::rates::Rates>,
    /// One lock per node, held across a cold-load's evict → load → fresh
    /// `/health` sequence. Placement reads VRAM from the last poll, which
    /// cannot see a load still in progress; two cold-loads deciding at
    /// once both saw the same free card and loaded side by side into an
    /// out-of-memory.
    cold_load_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// When cortex last cold-loaded each `(node, model)`. A model loaded
    /// for a request is protected from eviction for [`COLD_LOAD_GRACE`],
    /// long enough for that request to reach it; otherwise the next
    /// cold-load in line evicts it before it has served anything.
    recent_cold_loads: std::sync::Mutex<HashMap<(String, String), std::time::Instant>>,
}

/// How long a freshly cold-loaded model is protected from eviction.
pub const COLD_LOAD_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

impl CortexState {
    pub fn from_config(config: &GatewayConfig) -> Self {
        let mut nodes = HashMap::new();
        for nc in &config.neurons {
            nodes.insert(
                nc.name.clone(),
                NodeState {
                    device_health: Vec::new(),
                    name: nc.name.clone(),
                    endpoint: nc.endpoint.clone(),
                    healthy: false,
                    models: HashMap::new(),
                    // Filled from the neuron's /models reply at the
                    // first poll (#223).
                    reasoning_budget: Vec::new(),
                    lifecycle_cycles: 0,
                    last_poll: None,
                    discovery: None,
                    activation: None,
                    model_load: HashMap::new(),
                    consecutive_poll_failures: 0,
                    next_dark: None,
                    in_scheduled_absence: false,
                },
            );
        }

        let catalogue = ModelCatalogue::load(&config.models_config);

        // Local provider always handles operator + infra keys. When the
        // upstream client is enabled (#57), wrap it in the chain so locally
        // unknown keys fall through to the mesh authority; otherwise stay
        // purely local.
        let local = LocalEntitlementProvider::from_config(&config.entitlements);
        let entitlements: Arc<dyn EntitlementProvider> = if config.upstream.enabled {
            tracing::info!(url = %config.upstream.url, "upstream entitlement client enabled");
            Arc::new(ChainedEntitlementProvider::new(
                local,
                UpstreamEntitlementProvider::new(&config.upstream),
            ))
        } else {
            Arc::new(local)
        };

        Self {
            nodes: RwLock::new(nodes),
            neuron_configs: config.neurons.clone(),
            eviction: config.eviction.clone(),
            catalogue,
            // READ timeout, not a total one. reqwest's `.timeout()` is a
            // total deadline — "from when the request starts connecting
            // until the response body has finished" — which for a streamed
            // completion is a hard cap on how long a model may generate,
            // not a liveness check. At 300s and ~13 tok/s that capped every
            // stream at roughly 3.9k tokens: cortex severed the connection
            // mid-answer, the client reported the stream as terminated, and
            // neuron carried on generating into a socket nobody was reading.
            // A long agentic turn could therefore never complete through the
            // gateway however healthy the fleet was, and the failure looked
            // like a model fault because the model was blameless.
            //
            // `.read_timeout()` resets on every successful read, so it times
            // out a stalled upstream while letting a producing one run as
            // long as it needs. The same 300s is kept deliberately: it is
            // the headroom a cold model load needs before its first byte,
            // which is what the original total timeout was reaching for.
            //
            // This pairs with neuron's 1s SSE keep-alive: a healthy stream
            // puts bytes on the wire every second, so idle here means idle,
            // and the deadline finally measures what it was meant to.
            http_client: reqwest::Client::builder()
                .read_timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            entitlements,
            require_auth: config.entitlements.require_auth,
            served_usage: Arc::new(crate::served_usage::ServedUsage::new()),
            discovered_capabilities: RwLock::new(HashMap::new()),
            capability_probe_attempts: RwLock::new(HashMap::new()),
            audio_jobs: crate::audio_jobs::AudioJobs::new(&config.audio_jobs),
            rates: Arc::new(if config.upstream.enabled {
                crate::rates::Rates::upstream()
            } else {
                crate::rates::Rates::local()
            }),
            cold_load_locks: std::sync::Mutex::new(HashMap::new()),
            recent_cold_loads: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// The lock that serialises cold-loads on `node`.
    pub fn cold_load_lock(&self, node: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .cold_load_locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        Arc::clone(locks.entry(node.to_string()).or_default())
    }

    /// Record that cortex has just cold-loaded `model` on `node`.
    pub fn note_cold_load(&self, node: &str, model: &str) {
        let mut recent = self
            .recent_cold_loads
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        recent.retain(|_, at| at.elapsed() < COLD_LOAD_GRACE);
        recent.insert(
            (node.to_string(), model.to_string()),
            std::time::Instant::now(),
        );
    }

    /// Whether `model` on `node` is inside its post-cold-load grace.
    pub fn recently_cold_loaded(&self, node: &str, model: &str) -> bool {
        let recent = self
            .recent_cold_loads
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        recent
            .get(&(node.to_string(), model.to_string()))
            .is_some_and(|at| at.elapsed() < COLD_LOAD_GRACE)
    }

    /// Whether a resident model must not be evicted right now: it has
    /// requests running or queued (as of the last `/health`), or cortex
    /// cold-loaded it moments ago for a request that has yet to arrive.
    /// Evicting either kills work in progress.
    pub fn eviction_blocked(&self, node: &cortex_core::node::NodeState, model: &str) -> bool {
        let busy = node
            .model_load
            .get(model)
            .is_some_and(|l| l.in_flight + l.queue_depth > 0);
        busy || self.recently_cold_loaded(&node.name, model)
    }
}
