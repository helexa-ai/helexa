// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use mesh::MeshHandle;
use protocol::ProvisioningCommand;

/// Describes a neuron as seen from cortex over the control-plane websocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronDescriptor {
    /// Opaque id the neuron reports for itself (e.g. host label + uuid).
    pub node_id: Option<String>,
    /// Optional human-readable label or hostname.
    pub label: Option<String>,
    /// Free-form metadata provided by neuron (os, arch, gpu summary, etc).
    pub metadata: serde_json::Value,
}

/// Messages sent from neuron to cortex over the websocket.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CortexToNeuron {
    /// Provisioning command such as UpsertModelConfig, LoadModel, UnloadModel.
    Provisioning { cmd: ProvisioningCommand },
    /// Request for the neuron to publish an updated capabilities snapshot.
    RequestCapabilities,
}

/// Internal representation of a connected neuron in cortex.
#[derive(Debug, Clone)]
pub struct ConnectedNeuron {
    pub descriptor: NeuronDescriptor,
    /// Last time we received a heartbeat from this neuron.
    pub last_heartbeat: std::time::Instant,
}

/// Shared state tracking neurons connected over the control-plane websocket.
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
            });
        }
    }

    pub async fn update_heartbeat(&self, neuron_id: &str) {
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
pub async fn start_control_plane_server(
    addr: SocketAddr,
    mesh: MeshHandle,
    registry: NeuronRegistry,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("cortex control-plane websocket listening on {}", addr);

    // Spawn a background task to periodically prune stale neurons.
    let prune_registry = registry.clone();
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
        let registry_clone = registry.clone();
        let mesh_clone = mesh.clone();
        tokio::spawn(async move {
            if let Err(e) =
                handle_neuron_connection(stream, peer_addr, registry_clone, mesh_clone).await
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
) -> Result<()> {
    let ws_stream = accept_async(stream)
        .await
        .map_err(|e| anyhow!("failed to upgrade websocket from {}: {e}", peer_addr))?;
    info!("neuron connection upgraded to websocket from {}", peer_addr);

    let (_tx, mut rx) = ws_stream.split();

    // Expect an initial Register message.
    let first_msg = rx
        .next()
        .await
        .ok_or_else(|| anyhow!("neuron {} closed before sending register", peer_addr))??;

    let register: NeuronToCortex = parse_ws_json(first_msg)?;
    let neuron_id = match register {
        NeuronToCortex::Register { neuron } => {
            let id = neuron
                .node_id
                .clone()
                .unwrap_or_else(|| format!("peer-{}", peer_addr));
            info!("registered neuron_id={} from {}", id, peer_addr);
            registry.upsert_neuron(neuron).await;
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
    let registry_clone = registry.clone();
    let neuron_id_clone = neuron_id.clone();
    tokio::spawn(async move {
        while let Some(msg) = rx.next().await {
            match msg {
                Ok(message) => {
                    if let Err(e) =
                        handle_neuron_message(&neuron_id_clone, &registry_clone, message).await
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

    // For now we keep the sender half idle; future revisions will use `tx` to
    // push provisioning commands and capability requests. We retain the sink
    // in case we want to implement simple broadcast/testing behaviour here.
    // To keep the connection alive, just await on an infinite sleep.
    loop {
        time::sleep(Duration::from_secs(3600)).await;
    }
}

async fn handle_neuron_message(
    _neuron_id: &str,
    registry: &NeuronRegistry,
    message: Message,
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
            registry.update_heartbeat(&hb_id).await;
        }
        NeuronToCortex::ProvisioningResponse {
            neuron_id: resp_id,
            response,
        } => {
            info!(
                "provisioning response from neuron_id={}: {:?}",
                resp_id, response
            );
            // TODO: integrate with orchestrator/provisioner once those traits have
            // async entrypoints for tracking provisioning results.
        }
    }
    Ok(())
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
