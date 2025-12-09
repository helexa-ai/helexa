#![allow(clippy::unused_async)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, RwLock};
use tokio::time;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::observe::{ObserveBus, ObserveEvent};
use mesh::MeshHandle;
use protocol::ProvisioningCommand;

/// Describes a neuron as seen from cortex over the control-plane websocket.
///
/// This is essentially the same descriptor that neurons send in their
/// initial `Register` message. In future revisions this may be promoted
/// to the shared `protocol` crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronDescriptor {
    /// Opaque id the neuron reports for itself (e.g. machine-id or CLI node-id).
    pub node_id: Option<String>,
    /// Optional human-readable label or hostname.
    pub label: Option<String>,
    /// Free-form metadata provided by neuron (os, arch, gpu summary, etc).
    pub metadata: serde_json::Value,
}

/// Messages sent from neuron to cortex over the websocket.
///
/// These are control-plane messages used for registration, heartbeats
/// and provisioning responses. The websocket transport carries these
/// as JSON-encoded frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NeuronToCortex {
    /// Initial registration message sent when a neuron connects.
    Register { neuron: NeuronDescriptor },

    /// Periodic heartbeat containing liveness and lightweight metrics.
    Heartbeat {
        neuron_id: String,
        /// Optional summary of current load/utilisation as free-form JSON.
        metrics: serde_json::Value,
    },

    /// Acknowledgement or error for a provisioning command previously sent
    /// from cortex.
    ProvisioningResponse {
        neuron_id: String,
        response: protocol::ProvisioningResponse,
    },
}

/// Messages sent from cortex to neuron over the websocket.
///
/// For now this is focussed on provisioning commands and a simple
/// capabilities request. Future revisions may extend this with new
/// control-plane operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CortexToNeuron {
    /// Provisioning command such as UpsertModelConfig, LoadModel, UnloadModel.
    Provisioning { cmd: ProvisioningCommand },

    /// Request for the neuron to publish an updated capabilities snapshot.
    RequestCapabilities,
}

/// Internal representation of a connected neuron in cortex.
///
/// This is a simple in-memory structure used to keep track of neurons
/// that have registered via the control-plane websocket.
#[derive(Debug, Clone)]
pub struct ConnectedNeuron {
    pub descriptor: NeuronDescriptor,
    /// Last time we received a heartbeat from this neuron.
    pub last_heartbeat: std::time::Instant,
    /// Sender used to push control-plane messages from cortex to this neuron.
    pub outbound_tx: Option<mpsc::UnboundedSender<CortexToNeuron>>,
}

/// Shared state tracking neurons connected over the control-plane websocket.
///
/// This type is intentionally minimal and focussed on neuron tracking and
/// outbound message routing. Higher-level orchestration and observability
/// concerns should be built on top of this registry rather than embedded
/// directly.
#[derive(Debug, Default, Clone)]
pub struct NeuronRegistry {
    inner: Arc<RwLock<Vec<ConnectedNeuron>>>,
}

impl NeuronRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Insert or update a neuron descriptor in the registry.
    pub async fn upsert_neuron(&self, descriptor: NeuronDescriptor) {
        let mut neurons = self.inner.write().await;
        if let Some(existing) = neurons
            .iter_mut()
            .find(|n| n.descriptor.node_id == descriptor.node_id)
        {
            existing.descriptor = descriptor;
            existing.last_heartbeat = std::time::Instant::now();
        } else {
            neurons.push(ConnectedNeuron {
                descriptor,
                last_heartbeat: std::time::Instant::now(),
                outbound_tx: None,
            });
        }
    }

    /// Attach an outbound sender for the given neuron id so that cortex can
    /// push `CortexToNeuron` messages (e.g. provisioning commands).
    pub async fn set_sender_for_neuron(
        &self,
        neuron_id: &str,
        tx: mpsc::UnboundedSender<CortexToNeuron>,
    ) {
        let mut neurons = self.inner.write().await;
        if let Some(existing) = neurons
            .iter_mut()
            .find(|n| n.descriptor.node_id.as_deref() == Some(neuron_id))
        {
            existing.outbound_tx = Some(tx);
        }
    }

    /// Attempt to send a control-plane message to a specific neuron by id.
    ///
    /// This is a low-level helper; higher-level code should prefer the
    /// `send_provisioning_to_neuron` wrapper below.
    pub async fn send_to_neuron(&self, neuron_id: &str, msg: CortexToNeuron) -> Result<(), String> {
        let neurons = self.inner.read().await;
        if let Some(existing) = neurons
            .iter()
            .find(|n| n.descriptor.node_id.as_deref() == Some(neuron_id))
        {
            if let Some(ref tx) = existing.outbound_tx {
                tx.send(msg).map_err(|e| {
                    format!(
                        "failed to enqueue message for neuron_id={}: {:?}",
                        neuron_id, e
                    )
                })
            } else {
                Err(format!(
                    "no outbound sender registered for neuron_id={}",
                    neuron_id
                ))
            }
        } else {
            Err(format!("no neuron registered with id={}", neuron_id))
        }
    }

    /// Update heartbeat timestamp for a neuron and keep the registry fresh.
    pub async fn update_heartbeat(&self, neuron_id: &str, _metrics: serde_json::Value) {
        let mut neurons = self.inner.write().await;
        if let Some(existing) = neurons
            .iter_mut()
            .find(|n| n.descriptor.node_id.as_deref() == Some(neuron_id))
        {
            existing.last_heartbeat = std::time::Instant::now();
        }
    }

    /// Periodically prune neurons that have not sent a heartbeat within
    /// the given timeout.
    pub async fn prune_stale(&self, timeout: Duration) {
        let mut neurons = self.inner.write().await;
        let now = std::time::Instant::now();
        neurons.retain(|n| now.duration_since(n.last_heartbeat) <= timeout);
    }

    /// List all known neurons by descriptor.
    pub async fn list(&self) -> Vec<NeuronDescriptor> {
        let neurons = self.inner.read().await;
        neurons.iter().map(|n| n.descriptor.clone()).collect()
    }
}

/// Start the cortex-side control-plane websocket server.
///
/// This listener accepts websocket connections from neuron nodes. Each
/// neuron is expected to:
///
/// - Immediately send a `NeuronToCortex::Register` message.
/// - Periodically send `NeuronToCortex::Heartbeat`.
/// - Accept `CortexToNeuron::Provisioning` commands.
///
/// The `mesh` handle is currently unused but included so that future
/// revisions can integrate neuron descriptors into the distributed
/// topology (e.g. advertising neuron presence over the mesh).
///
/// The `observe_publisher` is used to emit `ObserveEvent`s for the
/// dashboard/observe websocket server.
pub async fn start_control_plane_server(
    addr: SocketAddr,
    mesh: MeshHandle,
    registry: NeuronRegistry,
    demand_state: crate::spec::ModelDemandState,
    observe_publisher: tokio::sync::broadcast::Sender<ObserveEvent>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("cortex control-plane websocket listening on {}", addr);

    // Spawn a background task to periodically prune stale neurons.
    let prune_registry = registry_list_clone(&registry);
    tokio::spawn(async move {
        let interval = Duration::from_secs(30);
        let timeout = Duration::from_secs(90);
        loop {
            time::sleep(interval).await;
            prune_registry.prune_stale(timeout).await;
        }
    });

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        info!(
            "control-plane accepted TCP connection from {} on {}",
            peer_addr, addr
        );
        let registry_clone = registry_list_clone(&registry);
        let mesh_clone = mesh.clone();
        let demand_state_clone = demand_state.clone();
        let observe_for_connection = observe_publisher.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_neuron_connection(
                stream,
                peer_addr,
                registry_clone,
                mesh_clone,
                demand_state_clone,
                observe_for_connection,
            )
            .await
            {
                warn!(
                    "control-plane connection from {} ended with error: {:?}",
                    peer_addr, e
                );
            }
        });
    }
}

async fn handle_neuron_connection(
    stream: tokio::net::TcpStream,
    peer_addr: SocketAddr,
    registry: NeuronRegistry,
    _mesh: MeshHandle,
    demand_state: crate::spec::ModelDemandState,
    observe_publisher: tokio::sync::broadcast::Sender<ObserveEvent>,
) -> Result<()> {
    info!(
        "attempting websocket upgrade for neuron control-plane connection from {}",
        peer_addr
    );
    let ws_stream = accept_async(stream)
        .await
        .map_err(|e| anyhow!("failed to upgrade websocket from {}: {e}", peer_addr))?;
    info!(
        "neuron connection successfully upgraded to websocket from {}",
        peer_addr
    );

    let (tx, mut rx) = ws_stream.split();

    // Expect an initial Register message.
    let first_msg = rx.next().await.ok_or_else(|| {
        anyhow!(
            "neuron {} closed websocket before sending initial Register message",
            peer_addr
        )
    })??;
    info!(
        "cortex received first websocket message from neuron peer {}: {:?}",
        peer_addr, first_msg
    );

    let register: NeuronToCortex = parse_ws_json(first_msg)?;
    let neuron_id = match register {
        NeuronToCortex::Register { neuron } => {
            let id = neuron
                .node_id
                .clone()
                .unwrap_or_else(|| format!("peer-{}", peer_addr));
            info!("registered neuron_id={} from {}", id, peer_addr);
            registry.upsert_neuron(neuron.clone()).await;

            // Publish registration event for dashboards.
            let _ = observe_publisher.send(ObserveEvent::NeuronRegistered { neuron });

            // create an outbound channel + writer task for this neuron
            let (out_tx, mut out_rx) = mpsc::unbounded_channel::<CortexToNeuron>();
            registry.set_sender_for_neuron(&id, out_tx.clone()).await;

            // writer task logs and sends control-plane messages to this neuron
            let writer_id = id.clone();
            tokio::spawn(async move {
                use futures::SinkExt;
                let mut sink = tx;
                while let Some(msg) = out_rx.recv().await {
                    match serde_json::to_string(&msg) {
                        Ok(text) => {
                            if let Err(e) = sink.send(Message::Text(text)).await {
                                warn!(
                                    "failed to send control-plane message to neuron_id={} / {}: {:?}",
                                    writer_id, peer_addr, e
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            warn!(
                                "failed to serialise CortexToNeuron message for neuron_id={}: {:?}",
                                writer_id, e
                            );
                        }
                    }
                }
                info!(
                    "control-plane writer task exiting for neuron_id={} / {}",
                    writer_id, peer_addr
                );
            });

            // On first connection, opportunistically upsert all models from the
            // current demand/spec state into this neuron to exercise the
            // provisioning path.
            if let Err(e) =
                bootstrap_upsert_for_neuron(&id, &registry, &demand_state, out_tx.clone()).await
            {
                warn!(
                    "failed to bootstrap UpsertModelConfig for neuron_id={}: {:?}",
                    id, e
                );
            }

            id
        }
        other => {
            return Err(anyhow!(
                "expected Register message from neuron {}, got {:?}",
                peer_addr,
                other
            ));
        }
    };

    // Spawn a task to process subsequent messages from this neuron.
    let registry_clone = registry_list_clone(&registry);
    let neuron_id_clone = neuron_id.clone();
    let observe_for_messages = observe_publisher.clone();
    tokio::spawn(async move {
        while let Some(msg) = rx.next().await {
            match msg {
                Ok(message) => {
                    if let Err(e) = handle_neuron_message(
                        &neuron_id_clone,
                        &registry_clone,
                        message,
                        &observe_for_messages,
                    )
                    .await
                    {
                        warn!(
                            "error handling message from neuron_id={}: {:?}",
                            neuron_id_clone, e
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "websocket error from neuron_id={} / {}: {:?}",
                        neuron_id_clone, peer_addr, e
                    );
                    break;
                }
            }
        }

        info!(
            "neuron websocket connection closed for neuron_id={} / {}",
            neuron_id_clone, peer_addr
        );
    });

    // Keep the connection alive; all work happens in spawned tasks.
    loop {
        time::sleep(Duration::from_secs(3600)).await;
    }
}

async fn handle_neuron_message(
    neuron_id: &str,
    registry: &NeuronRegistry,
    message: Message,
    observe_publisher: &tokio::sync::broadcast::Sender<ObserveEvent>,
) -> Result<()> {
    let msg: NeuronToCortex = parse_ws_json(message)?;
    match msg {
        NeuronToCortex::Register { neuron } => {
            // Allow re-registration to refresh metadata.
            info!(
                "received re-register from neuron_id={:?}; updating descriptor",
                neuron.node_id
            );
            registry.upsert_neuron(neuron).await;
        }
        NeuronToCortex::Heartbeat {
            neuron_id: hb_id,
            metrics,
        } => {
            info!("heartbeat from neuron_id={} metrics={}", hb_id, metrics);
            registry.update_heartbeat(&hb_id, metrics.clone()).await;

            // Emit heartbeat event for dashboards.
            let _ = observe_publisher.send(ObserveEvent::NeuronHeartbeat {
                neuron_id: hb_id,
                metrics,
            });
        }
        NeuronToCortex::ProvisioningResponse {
            neuron_id: resp_id,
            response,
        } => {
            info!(
                "provisioning response from neuron_id={}: {:?}",
                resp_id, response
            );
            // Emit provisioning response event for dashboards.
            let _ = observe_publisher.send(ObserveEvent::ProvisioningResponse {
                neuron_id: resp_id,
                response: response.clone(),
            });
            // TODO: integrate with orchestrator/provisioner once those traits have
            // async entrypoints for tracking provisioning results.
        }
    }
    Ok(())
}

/// Send a provisioning command to a specific neuron (by `node_id`) over the
/// established websocket control-plane connection.
///
/// This is a low-level helper intended for admin tooling and, eventually,
/// the orchestrator/provisioner. It returns a simple `Result` with a string
/// error for ease of use in higher layers.
pub async fn send_provisioning_to_neuron(
    registry: &NeuronRegistry,
    neuron_id: &str,
    cmd: ProvisioningCommand,
    observe_publisher: &tokio::sync::broadcast::Sender<crate::observe::ObserveEvent>,
) -> Result<(), String> {
    let msg = CortexToNeuron::Provisioning { cmd: cmd.clone() };
    // Emit a ProvisioningSent event for dashboards before enqueuing the command.
    let _ = observe_publisher.send(crate::observe::ObserveEvent::ProvisioningSent {
        neuron_id: neuron_id.to_string(),
        cmd,
    });
    registry.send_to_neuron(neuron_id, msg).await
}

fn parse_ws_json<T: for<'de> Deserialize<'de>>(message: Message) -> Result<T> {
    let text = match message {
        Message::Text(t) => t,
        Message::Binary(b) => String::from_utf8(b)
            .map_err(|e| anyhow!("expected UTF-8 websocket text frame, got binary: {e}"))?,
        Message::Close(_) => return Err(anyhow!("websocket closed unexpectedly")),
        other => {
            return Err(anyhow!(
                "unexpected websocket message type when expecting JSON: {:?}",
                other
            ))
        }
    };

    let parsed = serde_json::from_str::<T>(&text)
        .map_err(|e| anyhow!("failed to parse websocket JSON payload: {e}"))?;
    Ok(parsed)
}

/// Lightweight clone helper to avoid deriving Clone for the entire registry,
/// which would encourage copying potentially large state.
///
/// For now `NeuronRegistry` is small (a Vec under a lock), so this is fine.
/// If it grows more complex, consider switching to an `Arc<NeuronRegistry>`.
fn registry_list_clone(registry: &NeuronRegistry) -> NeuronRegistry {
    NeuronRegistry {
        inner: registry.inner.clone(),
    }
}

/// Bootstrap helper: send UpsertModelConfig commands for all models in the
/// current demand/spec state to the newly connected neuron. This is a
/// temporary harness to exercise provisioning; future versions will move
/// this logic into a dedicated provisioner/orchestrator component.
async fn bootstrap_upsert_for_neuron(
    neuron_id: &str,
    _registry: &NeuronRegistry,
    demand_state: &crate::spec::ModelDemandState,
    tx: mpsc::UnboundedSender<CortexToNeuron>,
) -> Result<()> {
    if demand_state.models.is_empty() {
        info!(
            "no models found in demand/spec state; skipping bootstrap UpsertModelConfig for neuron_id={}",
            neuron_id
        );
        return Ok(());
    }

    info!(
        "bootstrapping {} model(s) to neuron_id={} via UpsertModelConfig",
        demand_state.models.len(),
        neuron_id
    );

    for entry in &demand_state.models {
        let cmd = ProvisioningCommand::UpsertModelConfig(entry.config.clone());
        let msg = CortexToNeuron::Provisioning { cmd };
        tx.send(msg).map_err(|e| {
            anyhow!(
                "failed to enqueue bootstrap UpsertModelConfig for neuron_id={}: {:?}",
                neuron_id,
                e
            )
        })?;
    }

    Ok(())
}
