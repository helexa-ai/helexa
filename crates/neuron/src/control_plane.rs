// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json;
use tokio::sync::mpsc;
use tokio::time;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};

use crate::runtime::RuntimeManager;
use model_runtime::{ChatRuntimeHandle, ProcessRuntime};
use protocol::{ModelConfig, ModelId, NeuronControl, ProvisioningCommand, ProvisioningResponse};

/// Simple exponential backoff helper for reconnect attempts.
struct Backoff {
    current: Duration,
    initial: Duration,
    max: Duration,
}

impl Backoff {
    fn new(initial_secs: u64, max_secs: u64) -> Self {
        let initial = Duration::from_secs(initial_secs);
        let max = Duration::from_secs(max_secs);
        Self {
            current: initial,
            initial,
            max,
        }
    }

    fn next_delay(&mut self) -> Duration {
        let delay = self.current;
        let next = self.current * 2;
        self.current = if next > self.max { self.max } else { next };
        delay
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.current = self.initial;
    }
}

/// messages sent from neuron to cortex over the websocket.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum NeuronToCortex {
    /// initial registration when the websocket connection is established.
    Register { neuron: NeuronDescriptor },
    /// periodic heartbeat including optional lightweight metrics.
    Heartbeat {
        neuron_id: String,
        metrics: serde_json::Value,
    },
    /// provisioning response for a command previously sent by cortex.
    ProvisioningResponse {
        neuron_id: String,
        response: ProvisioningResponse,
    },
    /// explicit shutdown notification indicating that this neuron is exiting
    /// gracefully and will no longer send heartbeats or accept work.
    Shutdown {
        neuron_id: String,
        reason: Option<String>,
    },
}

/// messages sent from cortex to neuron over the websocket.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CortexToNeuron {
    Provisioning {
        cmd: ProvisioningCommand,
    },
    RequestCapabilities,
    /// planned shutdown notification from cortex. neurons should not shut
    /// themselves down; they should keep serving in-flight work and rely on
    /// their reconnect logic to resume control-plane connectivity once cortex
    /// comes back.
    ShutdownNotice {
        reason: Option<String>,
    },
}

/// minimal descriptor for this neuron as reported to cortex.
#[derive(Debug, Serialize)]
struct NeuronDescriptor {
    node_id: Option<String>,
    hostname: String,
    domain: Option<String>,
    label: Option<String>,
    metadata: serde_json::Value,
}

/// neuron implements the control-plane client logic expected by cortex.
///
/// it connects to the configured cortex websocket endpoint, registers itself,
/// sends periodic heartbeats, and listens for provisioning commands.
///
/// this function now supervises the control-plane client with an exponential
/// backoff loop so that neurons can survive cortex outages and restarts
/// without requiring manual intervention.
pub fn spawn(_addr: SocketAddr, runtime: RuntimeManager) {
    info!("starting neuron control-plane websocket client");

    let control = Arc::new(NeuronControlImpl::new(runtime));
    let endpoint = control.runtime.cortex_control_endpoint().to_string();

    tokio::spawn(async move {
        let mut backoff = Backoff::new(30, 3600); // 30s initial, up to 1h
        loop {
            match run_control_plane_client(endpoint.clone(), Arc::clone(&control)).await {
                Ok(()) => {
                    info!("neuron control-plane client exited cleanly");
                    // Treat a clean exit as process-level shutdown and stop
                    // supervising reconnects.
                    break;
                }
                Err(e) => {
                    warn!(
                        "neuron control-plane client disconnected or failed: {:?}",
                        e
                    );
                    let delay = backoff.next_delay();
                    warn!("will retry cortex control-plane connection in {:?}", delay);
                    time::sleep(delay).await;
                }
            }
        }
    });
}

/// example struct that will eventually implement the NeuronControl trait.
pub struct NeuronControlImpl {
    pub(crate) runtime: RuntimeManager,
}

impl NeuronControlImpl {
    pub fn new(runtime: RuntimeManager) -> Self {
        info!("initialising NeuronControlImpl");
        Self { runtime }
    }

    pub fn handle_placeholder(&self) {
        info!(
            "placeholder handler using neuron runtime for control-plane on runtime pointer {:p}",
            &self.runtime as *const RuntimeManager
        );
    }

    /// Apply an updated model configuration by recording it in the in-memory
    /// model config state and returning a generic success response.
    ///
    /// This does not yet spawn or tear down any backend processes; it only
    /// updates configuration state so that later `LoadModel` / `UnloadModel`
    /// commands can make use of it.
    fn handle_upsert_model_config(&self, cfg: ModelConfig) -> ProvisioningResponse {
        let model_id = cfg.id.clone();
        let configs = self.runtime.model_configs();
        {
            // Update the in-memory configuration map.
            let mut state = futures::executor::block_on(configs.write());
            state.upsert(cfg);
        }

        ProvisioningResponse::Ok {
            model_id,
            message: Some("configuration updated".to_string()),
        }
    }

    /// Handle a request to load a model by:
    /// - looking up its configuration from ModelConfigState
    /// - spawning a backend process via ProcessManager
    /// - registering a runtime handle in the ModelRegistry
    fn handle_load_model(&self, model_id: ModelId) -> ProvisioningResponse {
        let configs = self.runtime.model_configs();
        let cfg_opt = {
            let state = futures::executor::block_on(configs.read());
            state.get(&model_id).cloned()
        };

        let Some(cfg) = cfg_opt else {
            return ProvisioningResponse::Error {
                model_id,
                error: "no configuration found for model; send UpsertModelConfig first".to_string(),
            };
        };

        // Determine the listen endpoint; if none is explicitly provided, derive
        // it from backend kind and internal port allocation.
        let listen = match futures::executor::block_on(self.runtime.derive_listen_endpoint(&cfg)) {
            Ok(url) => url,
            Err(e) => {
                return ProvisioningResponse::Error {
                    model_id: cfg.id,
                    error: format!("failed to derive listen endpoint: {e}"),
                }
            }
        };

        // Spawn the backend process exactly as described in the configuration.
        let cmd = match cfg.command.as_deref() {
            Some(c) => c,
            None => {
                return ProvisioningResponse::Error {
                    model_id: cfg.id,
                    error: "missing command in ModelConfig; cortex must supply it".to_string(),
                }
            }
        };

        let args_ref: Vec<&str> = cfg.args.iter().map(String::as_str).collect();
        let env_pairs: Vec<(String, String)> = cfg
            .env
            .iter()
            .map(|e| (e.key.clone(), e.value.clone()))
            .collect();

        let process_manager = self.runtime.process_manager();
        let worker = match process_manager.spawn_worker_with_env(
            cmd,
            &args_ref[..],
            &cfg.id.0,
            &env_pairs[..],
        ) {
            Ok(w) => w,
            Err(e) => {
                return ProvisioningResponse::Error {
                    model_id: cfg.id,
                    error: format!("failed to spawn backend process: {e}"),
                }
            }
        };

        info!(
            "loaded model_id={:?} with backend_kind={} on worker pid={}",
            cfg.id, cfg.backend_kind, worker.pid
        );

        // Construct a ProcessRuntime pointing at the derived listen endpoint
        // and register it in the model registry.
        let timeout = std::time::Duration::from_secs(30);
        let runtime = ProcessRuntime::new(listen.clone(), timeout, Some(cfg.id.0.clone()));
        let handle = ChatRuntimeHandle::new(Arc::new(runtime));

        let registry_arc = self.runtime.registry();
        {
            let mut registry = futures::executor::block_on(registry_arc.write());
            registry.register_chat_model(cfg.id.0.clone(), handle, Some(worker.pid.to_string()));
        }

        ProvisioningResponse::Ok {
            model_id: cfg.id,
            message: Some(format!("model loaded and serving at {}", listen)),
        }
    }

    /// Handle a request to unload a model by:
    /// - instructing the process manager to terminate workers
    /// - removing the model from the registry
    fn handle_unload_model(&self, model_id: ModelId) -> ProvisioningResponse {
        info!(
            "handle_unload_model: terminating backend workers and unregistering model_id={:?}",
            model_id
        );

        // Terminate all backend workers associated with this model.
        let process_manager = self.runtime.process_manager();
        process_manager.terminate_workers_for_model(&model_id.0);

        // Remove the model from the registry so that new requests cannot be
        // scheduled to it. Existing in-flight requests that already hold a
        // handle will continue to complete as long as the backend cooperates.
        let registry_arc = self.runtime.registry();
        {
            let mut registry = futures::executor::block_on(registry_arc.write());
            registry.unregister_chat_model(&model_id.0);
        }

        ProvisioningResponse::Ok {
            model_id,
            message: Some(
                "unload requested; backend workers terminated and model unregistered".to_string(),
            ),
        }
    }
}

impl NeuronControl for NeuronControlImpl {
    /// Apply a provisioning command such as model configuration updates or
    /// load/unload requests.
    ///
    /// `UpsertModelConfig` updates the in-memory model configuration state.
    /// `LoadModel` and `UnloadModel` are wired to dedicated handlers that
    /// now spawn/terminate backend processes and update the model registry.
    fn apply_provisioning(&self, cmd: ProvisioningCommand) -> ProvisioningResponse {
        match cmd {
            ProvisioningCommand::UpsertModelConfig(cfg) => {
                info!("received UpsertModelConfig for model_id={:?}", cfg.id);
                self.handle_upsert_model_config(cfg)
            }
            ProvisioningCommand::LoadModel { model_id } => {
                info!("received LoadModel for model_id={:?}", model_id);
                self.handle_load_model(model_id)
            }
            ProvisioningCommand::UnloadModel { model_id } => {
                info!("received UnloadModel for model_id={:?}", model_id);
                self.handle_unload_model(model_id)
            }
        }
    }
}

/// run the neuron-side websocket control-plane client loop.
///
/// this connects to the given `endpoint`, registers the neuron, sends
/// heartbeats, and dispatches provisioning commands from cortex into the
/// local `NeuronControlImpl`.
async fn run_control_plane_client(
    endpoint: String,
    control: Arc<NeuronControlImpl>,
) -> anyhow::Result<()> {
    info!("neuron connecting to cortex control-plane at {}", endpoint);

    // Add detailed logging around the websocket handshake so that failures are
    // explicit in the neuron logs (in addition to the server-side errors).
    let (ws_stream, _resp) = match connect_async(&endpoint).await {
        Ok(ok) => {
            info!(
                "neuron successfully completed websocket handshake with cortex at {}",
                endpoint
            );
            ok
        }
        Err(e) => {
            error!(
                "neuron failed websocket handshake with cortex at {}: {:?}",
                endpoint, e
            );
            return Err(e.into());
        }
    };

    info!("neuron websocket connected to cortex control-plane");

    let (tx, mut rx) = ws_stream.split();

    // channel for all outbound messages (heartbeats + provisioning responses)
    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<Message>();

    // spawn single writer task owning the websocket sink
    tokio::spawn(async move {
        let mut sink = tx;
        while let Some(msg) = msg_rx.recv().await {
            if let Err(e) = sink.send(msg).await {
                warn!("failed to send message to cortex: {:?}", e);
                break;
            }
        }
        info!("neuron control-plane writer task exiting");
    });

    // send initial registration
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()))
        .unwrap_or_else(|_| "unknown".to_string());

    let domain = std::fs::read_to_string("/etc/resolv.conf")
        .ok()
        .and_then(|c| {
            c.lines()
                .find(|l| l.starts_with("search ") || l.starts_with("domain "))
                .and_then(|l| l.split_whitespace().nth(1).map(String::from))
        });

    let descriptor = NeuronDescriptor {
        node_id: control.runtime.node_id().clone(),
        hostname,
        domain,
        label: control.runtime.node_id().clone(),
        metadata: serde_json::json!({
            "backend": "neuron",
        }),
    };
    let register_msg = NeuronToCortex::Register { neuron: descriptor };
    let register_text = match serde_json::to_string(&register_msg) {
        Ok(text) => text,
        Err(e) => {
            error!(
                "neuron failed to serialise Register message for cortex control-plane at {}: {:?}",
                endpoint, e
            );
            return Err(e.into());
        }
    };
    if let Err(e) = msg_tx.send(Message::Text(register_text)) {
        error!(
            "neuron failed to enqueue initial Register message to cortex at {}: {:?}",
            endpoint, e
        );
        return Err(e.into());
    }

    // derive neuron id string for heartbeats and responses
    let neuron_id = control
        .runtime
        .node_id()
        .clone()
        .unwrap_or_else(|| "anonymous-neuron".to_string());

    // spawn heartbeat task that pushes messages into the writer channel
    {
        let neuron_id = neuron_id.clone();
        let hb_tx = msg_tx.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs(15);
            loop {
                time::sleep(interval).await;
                let hb = NeuronToCortex::Heartbeat {
                    neuron_id: neuron_id.clone(),
                    metrics: serde_json::json!({}),
                };
                match serde_json::to_string(&hb) {
                    Ok(text) => {
                        if let Err(e) = hb_tx.send(Message::Text(text)) {
                            warn!("failed to enqueue heartbeat to cortex: {:?}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("failed to serialise heartbeat: {:?}", e);
                    }
                }
            }
        });
    }

    // spawn shutdown signal handler that will notify cortex before exit
    {
        let neuron_id = neuron_id.clone();
        let shutdown_tx = msg_tx.clone();
        tokio::spawn(async move {
            // Prefer SIGTERM on Unix; fall back to Ctrl+C elsewhere.
            #[cfg(unix)]
            {
                let mut sigterm = match signal(SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("failed to register SIGTERM handler: {:?}", e);
                        return;
                    }
                };

                sigterm.recv().await;
                info!("neuron received SIGTERM; notifying cortex of shutdown");
            }

            #[cfg(not(unix))]
            {
                if let Err(e) = tokio::signal::ctrl_c().await {
                    warn!("failed to await ctrl_c for shutdown: {:?}", e);
                    return;
                }
                info!("neuron received ctrl_c; notifying cortex of shutdown");
            }

            let msg = NeuronToCortex::Shutdown {
                neuron_id: neuron_id.clone(),
                reason: Some("process exiting".to_string()),
            };
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = shutdown_tx.send(Message::Text(text));
            }
        });
    }

    // main receive loop: handle cortex → neuron messages
    while let Some(msg) = rx.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<CortexToNeuron>(&text) {
                    Ok(CortexToNeuron::Provisioning { cmd }) => {
                        let response = control.apply_provisioning(cmd);
                        let resp_msg = NeuronToCortex::ProvisioningResponse {
                            neuron_id: neuron_id.clone(),
                            response,
                        };
                        if let Ok(text) = serde_json::to_string(&resp_msg) {
                            if let Err(e) = msg_tx.send(Message::Text(text)) {
                                warn!("failed to enqueue provisioning response to cortex: {:?}", e);
                                break;
                            }
                        } else if let Err(e) = serde_json::to_string(&resp_msg) {
                            warn!(
                                "failed to serialise provisioning response for neuron_id={}: {:?}",
                                neuron_id, e
                            );
                        }
                    }
                    Ok(CortexToNeuron::RequestCapabilities) => {
                        // TODO: implement capability reporting once the protocol
                        // has concrete capability structures.
                        info!("received RequestCapabilities from cortex (not yet implemented)");
                    }
                    Ok(CortexToNeuron::ShutdownNotice { reason }) => {
                        // planned cortex shutdown; treat subsequent disconnect as
                        // a planned outage so that higher-level reconnect logic
                        // can avoid unloading models aggressively.
                        info!(
                            "received ShutdownNotice from cortex control-plane: {:?}",
                            reason
                        );
                        // in a follow-up change, this method can accept a shared
                        // flag (e.g. Arc<AtomicBool>) to record the planned
                        // shutdown state for the reconnect supervisor.
                    }
                    Err(e) => {
                        warn!("failed to parse CortexToNeuron message: {:?}", e);
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                warn!("ignoring unexpected binary websocket frame from cortex");
            }
            Ok(Message::Close(_)) => {
                info!("cortex closed control-plane websocket connection");
                break;
            }
            Ok(other) => {
                warn!("unexpected websocket message from cortex: {:?}", other);
            }
            Err(e) => {
                warn!("websocket error in neuron control-plane client: {:?}", e);
                break;
            }
        }
    }

    info!("neuron control-plane websocket client loop exiting");
    Ok(())
}
