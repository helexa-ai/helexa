//! Cold-load placement under concurrency.
//!
//! The mock neuron keeps real VRAM books for one 24 GB card: loads take
//! time, `/health` reports what is actually resident, and the card notes
//! if it is ever asked to hold more than it has. That last flag is the
//! failure being guarded against — two cold-loads deciding from the same
//! stale poll, both landing, and every request on the card then failing
//! out of memory.

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::discovery::{DeviceInfo, DiscoveryResponse};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::router::RouteError;
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CARD_MB: u64 = 24_000;

const CATALOGUE: &str = r#"
[[models]]
id = "music-a"
harness = "candle"
vram_mb = 14000

[[models]]
id = "music-b"
harness = "candle"
vram_mb = 12000
"#;

#[derive(Default)]
struct Card {
    resident: HashMap<String, u64>,
    in_flight: HashMap<String, usize>,
    overcommitted: bool,
    unloads: Vec<String>,
}

fn footprint(model: &str) -> u64 {
    match model {
        "music-a" => 14_000,
        "music-b" => 12_000,
        _ => 1_000,
    }
}

async fn spawn_neuron(card: Arc<Mutex<Card>>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let inference = url.clone();
    let app = Router::new()
        .route(
            "/models/load",
            post(
                |State(card): State<Arc<Mutex<Card>>>, Json(spec): Json<Value>| async move {
                    let id = spec["model_id"].as_str().unwrap().to_string();
                    // A load takes a while: long enough for a second
                    // request to decide while this one is in progress.
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    let mut card = card.lock().unwrap();
                    card.resident.insert(id.clone(), footprint(&id));
                    if card.resident.values().sum::<u64>() > CARD_MB {
                        card.overcommitted = true;
                    }
                    Json(json!({"status": "loaded"}))
                },
            ),
        )
        .route(
            "/models/unload",
            post(
                |State(card): State<Arc<Mutex<Card>>>, Json(body): Json<Value>| async move {
                    let id = body["model_id"].as_str().unwrap().to_string();
                    let mut card = card.lock().unwrap();
                    card.resident.remove(&id);
                    card.unloads.push(id);
                    Json(json!({"status": "unloaded"}))
                },
            ),
        )
        .route(
            "/health",
            get(|State(card): State<Arc<Mutex<Card>>>| async move {
                let card = card.lock().unwrap();
                let used: u64 = card.resident.values().sum();
                let models: Vec<Value> = card
                    .resident
                    .keys()
                    .map(|id| {
                        json!({
                            "id": id,
                            "in_flight": card.in_flight.get(id).copied().unwrap_or(0),
                            "queue_depth": 0
                        })
                    })
                    .collect();
                Json(json!({
                    "uptime_secs": 1,
                    "devices": [{
                        "index": 0,
                        "vram_used_mb": used,
                        "vram_free_mb": CARD_MB.saturating_sub(used),
                        "utilization_pct": 0,
                        "temp_c": 40
                    }],
                    "models": models
                }))
            }),
        )
        .route(
            "/models/{id}/endpoint",
            get(move |Path(_id): Path<String>| {
                let url = inference.clone();
                async move { Json(json!({"url": url})) }
            }),
        )
        .with_state(card);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

/// A gateway with one healthy 24 GB node, its first `/health` already in.
async fn gateway(neuron: &str, tag: &str) -> Arc<CortexState> {
    let path = std::env::temp_dir().join(format!("cortex-cold-load-{tag}.toml"));
    std::fs::write(&path, CATALOGUE).unwrap();
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![NeuronEndpoint {
            name: "gpu".into(),
            endpoint: neuron.to_string(),
        }],
        models_config: path.to_string_lossy().into_owned(),
        entitlements: Default::default(),
        upstream: Default::default(),
        audio_jobs: Default::default(),
    };
    let fleet = Arc::new(CortexState::from_config(&config));
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("gpu").unwrap();
        node.healthy = true;
        node.discovery = Some(DiscoveryResponse {
            hostname: "gpu".into(),
            os: "linux".into(),
            kernel: "6".into(),
            cuda_version: None,
            driver_version: None,
            devices: vec![DeviceInfo {
                index: 0,
                name: "RTX 4090".into(),
                vram_total_mb: CARD_MB,
                compute_capability: "8.9".into(),
            }],
            harnesses: vec!["candle".into()],
            cuda_unavailable_reason: None,
            max_prompt_tokens: 0,
        });
    }
    cortex_gateway::poller::poll_health(&fleet, "gpu", neuron).await;
    fleet
}

/// Mark `id` resident in the mock and in cortex's view, last used long
/// ago — as the poller would have seen it, not as a fresh cold-load.
async fn seed_resident(fleet: &CortexState, card: &Mutex<Card>, neuron: &str, id: &str) {
    card.lock()
        .unwrap()
        .resident
        .insert(id.into(), footprint(id));
    {
        let mut nodes = fleet.nodes.write().await;
        nodes.get_mut("gpu").unwrap().models.insert(
            id.into(),
            ModelEntry {
                id: id.into(),
                status: ModelStatus::Loaded,
                last_accessed: Some(chrono::Utc::now() - chrono::Duration::minutes(10)),
                vram_estimate_mb: Some(footprint(id)),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
                servable: None,
                reasoning_budget: Vec::new(),
            },
        );
    }
    cortex_gateway::poller::poll_health(fleet, "gpu", neuron).await;
}

#[tokio::test]
async fn concurrent_cold_loads_never_overcommit_the_card() {
    let card = Arc::new(Mutex::new(Card::default()));
    let neuron = spawn_neuron(Arc::clone(&card)).await;
    let fleet = gateway(&neuron, "concurrent").await;

    // Both decide while the card is empty; together they do not fit.
    let (a, b) = tokio::join!(cortex_gateway::router::resolve(&fleet, "music-a"), async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cortex_gateway::router::resolve(&fleet, "music-b").await
    });

    let card = card.lock().unwrap();
    assert!(!card.overcommitted, "both models were resident at once");
    assert!(a.is_ok(), "{a:?}");
    // music-a was cold-loaded for a request that has yet to reach it, so
    // music-b may not take its place yet: retry shortly, not a failure.
    match b {
        Err(e @ RouteError::NodeBusy { .. }) => {
            assert_eq!(e.http_status(), 503);
            assert!(e.retry_after_secs().is_some());
        }
        other => panic!("expected NodeBusy, got {other:?}"),
    }
    assert_eq!(card.resident.keys().collect::<Vec<_>>(), ["music-a"]);
    assert!(card.unloads.is_empty());
}

#[tokio::test]
async fn a_model_with_requests_in_flight_is_not_evicted() {
    let card = Arc::new(Mutex::new(Card::default()));
    let neuron = spawn_neuron(Arc::clone(&card)).await;
    let fleet = gateway(&neuron, "in-flight").await;
    card.lock().unwrap().in_flight.insert("music-a".into(), 1);
    seed_resident(&fleet, &card, &neuron, "music-a").await;

    let b = cortex_gateway::router::resolve(&fleet, "music-b").await;

    assert!(matches!(b, Err(RouteError::NodeBusy { .. })), "{b:?}");
    let card = card.lock().unwrap();
    assert!(card.unloads.is_empty(), "evicted a busy model");
    assert!(!card.overcommitted);
}

#[tokio::test]
async fn an_idle_model_is_swapped_out() {
    let card = Arc::new(Mutex::new(Card::default()));
    let neuron = spawn_neuron(Arc::clone(&card)).await;
    let fleet = gateway(&neuron, "idle").await;
    seed_resident(&fleet, &card, &neuron, "music-a").await;

    let b = cortex_gateway::router::resolve(&fleet, "music-b").await;

    assert!(b.is_ok(), "{b:?}");
    let card = card.lock().unwrap();
    assert_eq!(card.unloads, ["music-a"]);
    assert_eq!(card.resident.keys().collect::<Vec<_>>(), ["music-b"]);
    assert!(!card.overcommitted);
}
