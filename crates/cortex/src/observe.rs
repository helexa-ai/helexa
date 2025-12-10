// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Result};
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::control_plane::{ModelProvisioningStatus, NeuronDescriptor, NeuronView};
use crate::ModelProvisioningStore;
use protocol::{ProvisioningCommand, ProvisioningResponse};

/// Lightweight view of a neuron for dashboards, enriched with live health
/// information derived from the control-plane registry.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ObserveNeuron {
    pub descriptor: NeuronDescriptor,
    /// Best-effort timestamp of the last heartbeat observed for this neuron.
    /// This is derived from the internal `ConnectedNeuron::last_heartbeat`
    /// instant and converted to a wall-clock time where possible.
    pub last_heartbeat_at: Option<SystemTime>,
    /// Simple health classification derived from heartbeat recency.
    /// - "healthy"  => recent heartbeat within `healthy_threshold_secs`
    /// - "stale"    => no heartbeat yet or outside healthy window
    pub health: String,
    /// Whether cortex currently considers this neuron online. This will be set
    /// to `false` when the neuron has been explicitly removed (e.g. via a
    /// clean Shutdown message) or pruned due to missing heartbeats.
    pub offline: bool,
    pub models: Vec<ModelProvisioningStatus>,
}

/// Events published onto the observe bus for dashboard consumption.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObserveEvent {
    NeuronRegistered {
        neuron: NeuronDescriptor,
    },
    /// Emitted when cortex considers a neuron to have left the cluster. This is
    /// typically triggered when the neuron is pruned due to missing heartbeats.
    NeuronRemoved {
        neuron_id: String,
    },
    NeuronHeartbeat {
        neuron_id: String,
        metrics: serde_json::Value,
    },
    ProvisioningSent {
        neuron_id: String,
        cmd: ProvisioningCommand,
    },
    ProvisioningResponse {
        neuron_id: String,
        response: ProvisioningResponse,
    },
    /// Emitted whenever cortex updates its internal view of model provisioning
    /// state for a given neuron. This is typically triggered after processing
    /// a provisioning response, and mirrors the data that will be reflected in
    /// the next snapshot under `ObserveNeuron.models`.
    ModelStateChanged {
        neuron_id: String,
        models: Vec<ModelProvisioningStatus>,
    },
}

/// Simple broadcast-based bus for dashboard/observer subscriptions.
#[derive(Debug, Clone)]
pub struct ObserveBus {
    tx: broadcast::Sender<ObserveEvent>,
}

impl ObserveBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn publisher(&self) -> broadcast::Sender<ObserveEvent> {
        self.tx.clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ObserveEvent> {
        self.tx.subscribe()
    }
}

/// Initial snapshot payload sent to dashboard clients on connection.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ObserveSnapshot {
    /// Enriched neuron views that include both the static descriptor (as
    /// reported by the neuron itself) and derived health metadata such as
    /// last heartbeat time and a coarse health classification.
    pub neurons: Vec<ObserveNeuron>,
    // In future we can include:
    // - model demand summaries
    // - per-model/per-neuron state
    // - cluster-level health indicators
}

/// Top-level message wrapper sent to dashboard clients.
///
/// This keeps the protocol extensible and unambiguous:
/// - `kind = "snapshot"` for the initial state
/// - `kind = "event"` for streaming updates
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObserveMessage {
    Snapshot { snapshot: ObserveSnapshot },
    Event { event: ObserveEvent },
}

/// Start the dashboard/observer websocket server.
///
/// This server is intended for cortex operators and dashboards. It is
/// **read-only** from the perspective of cortex: clients connecting
/// here only receive:
///
/// - an initial snapshot of cortex state relevant to operators
///   (currently just the neuron list with health),
/// - a continuous stream of `ObserveEvent` values representing:
///   - neuron registrations,
///   - heartbeats,
///   - provisioning commands and responses.
///
/// In the future this endpoint may also accept operator commands to
/// adjust configuration, weights and policies. For now, it is a pure
/// observe channel.
pub async fn start_observe_server(
    addr: SocketAddr,
    registry: crate::control_plane::NeuronRegistry,
    model_store: ModelProvisioningStore,
    events_rx: broadcast::Receiver<ObserveEvent>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("cortex observe/dashboard websocket listening on {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        info!(
            "observe: accepted TCP connection from {} on {}",
            peer_addr, addr
        );

        // Clone the shared registry handle for this connection; the underlying
        // inner state is already behind an Arc/RwLock so this is cheap.
        let registry_for_connection = registry.clone();
        let model_store_for_connection = model_store.clone();
        let mut client_events_rx = events_rx.resubscribe();

        tokio::spawn(async move {
            // Build an enriched snapshot with last-heartbeat and health
            // classification for each known neuron at the time of connection.
            //
            // `list_with_health` exposes a `NeuronView` that includes both the
            // descriptor and a `Duration` since last heartbeat, which we map
            // into a coarse health bucket and an optional wall-clock timestamp.
            let neuron_views: Vec<NeuronView> = registry_for_connection.list_with_health().await;

            // Thresholds for health classification.
            let healthy_threshold = Duration::from_secs(60);
            let degraded_threshold = Duration::from_secs(5 * 60);

            let now = SystemTime::now();

            let mut neurons: Vec<ObserveNeuron> = Vec::new();
            for view in neuron_views {
                let (last_heartbeat_at, health) = match view.last_heartbeat_age {
                    None => (None, "stale".to_string()),
                    Some(age) => {
                        let health = if age <= healthy_threshold {
                            "healthy".to_string()
                        } else if age <= degraded_threshold {
                            "degraded".to_string()
                        } else {
                            "stale".to_string()
                        };
                        let last_heartbeat_at = now.checked_sub(age);
                        (last_heartbeat_at, health)
                    }
                };

                // Pull model provisioning state for this neuron_id, if we know it.
                let neuron_id = view
                    .descriptor
                    .node_id
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                let models = model_store_for_connection.list_for_neuron(&neuron_id).await;

                // For now, any neuron present in the registry snapshot is treated
                // as online; neurons that have been explicitly removed or pruned
                // will not appear here and will instead be represented via
                // `neuron_removed` events.
                let offline = false;

                neurons.push(ObserveNeuron {
                    descriptor: view.descriptor,
                    last_heartbeat_at,
                    health,
                    offline,
                    models,
                });
            }

            if let Err(e) =
                handle_observer_connection(stream, peer_addr, neurons, &mut client_events_rx).await
            {
                warn!(
                    "observe connection from {} ended with error: {:?}",
                    peer_addr, e
                );
            }
        });
    }
}

async fn handle_observer_connection(
    stream: tokio::net::TcpStream,
    peer_addr: SocketAddr,
    neurons: Vec<ObserveNeuron>,
    events_rx: &mut broadcast::Receiver<ObserveEvent>,
) -> Result<()> {
    let ws_stream = accept_async(stream).await.map_err(|e| {
        anyhow!(
            "failed to upgrade observe websocket from {}: {e}",
            peer_addr
        )
    })?;
    info!(
        "observe connection successfully upgraded to websocket from {}",
        peer_addr
    );

    let (mut tx, mut rx) = ws_stream.split();

    // 1. Send initial snapshot to the dashboard client.
    let snapshot = ObserveSnapshot { neurons };
    let snapshot_msg = ObserveMessage::Snapshot { snapshot };

    let snapshot_text = serde_json::to_string(&snapshot_msg).map_err(|e| {
        anyhow!(
            "failed to serialise observe snapshot for {}: {e}",
            peer_addr
        )
    })?;
    tx.send(Message::Text(snapshot_text))
        .await
        .map_err(|e| anyhow!("failed to send observe snapshot to {}: {e}", peer_addr))?;

    // 2. Stream events from the observe bus.
    //
    // We ignore anything the client sends for now; future versions may
    // use client messages to drive operator actions (e.g. config edits).
    loop {
        tokio::select! {
            biased;

            // Server-side events → dashboard.
            evt = events_rx.recv() => {
                match evt {
                    Ok(event) => {
                        let msg = ObserveMessage::Event { event };
                        match serde_json::to_string(&msg) {
                            Ok(text) => {
                                if let Err(e) = tx.send(Message::Text(text)).await {
                                    warn!(
                                        "failed to send observe event to {}: {:?}",
                                        peer_addr, e
                                    );
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "failed to serialise observe event for {}: {:?}",
                                    peer_addr, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "observe bus receiver for {} closed or errored: {:?}",
                            peer_addr, e
                        );
                        break;
                    }
                }
            }

            // Client → server messages (currently ignored, but we keep
            // the receive half alive to detect client disconnects).
            msg = rx.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) => {
                        info!("observe client {} closed websocket", peer_addr);
                        break;
                    }
                    Some(Ok(_other)) => {
                        // Ignore other message types for now.
                    }
                    Some(Err(e)) => {
                        warn!("observe websocket error from {}: {:?}", peer_addr, e);
                        break;
                    }
                    None => {
                        info!("observe websocket stream ended for {}", peer_addr);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}
