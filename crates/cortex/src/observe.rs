// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use anyhow::{anyhow, Result};
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::control_plane::NeuronDescriptor;
use protocol::{ProvisioningCommand, ProvisioningResponse};

/// Events published onto the observe bus for dashboard consumption.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObserveEvent {
    NeuronRegistered {
        neuron: NeuronDescriptor,
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
    pub neurons: Vec<NeuronDescriptor>,
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
///   (currently just the neuron list),
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
    neurons_snapshot: Vec<NeuronDescriptor>,
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

        let neurons = neurons_snapshot.clone();
        let mut client_events_rx = events_rx.resubscribe();

        tokio::spawn(async move {
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
    neurons: Vec<NeuronDescriptor>,
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
