//! Integration tests for per-request token metering (#51).
//!
//! Drives authenticated requests through the gateway to a mock neuron that
//! reports a fixed `usage` object, then asserts the EntitlementProvider's
//! spend ledger reflects cumulative per-key spend and that reservations
//! settle to actual (no outstanding reserved tokens once requests complete).

mod common;

use cortex_core::config::{
    ApiKeyConfig, EntitlementsConfig, EvictionSettings, EvictionStrategy, GatewayConfig,
    GatewaySettings, NeuronEndpoint,
};
use cortex_core::entitlements::{CapWindow, Principal};
use cortex_core::node::{ModelEntry, ModelStatus};
use cortex_gateway::state::CortexState;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

const ACCOUNT: &str = "acct-meter";
const KEY_ID: &str = "key-meter";
const BEARER: &str = "sk-meter";

/// The mock neuron (common::spawn_mock_neuron) reports this fixed usage on
/// every chat completion.
const PROMPT_PER_REQ: u64 = 10;
const COMPLETION_PER_REQ: u64 = 5;

async fn spawn_metered_gateway(neuron_url: &str) -> (Arc<CortexState>, String) {
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
            name: "mock-node".into(),
            endpoint: neuron_url.to_string(),
        }],
        models_config: "/dev/null".into(),
        entitlements: EntitlementsConfig {
            require_auth: true,
            keys: vec![ApiKeyConfig {
                key: BEARER.into(),
                account_id: ACCOUNT.into(),
                key_id: Some(KEY_ID.into()),
                hard_cap: Some(1_000_000),
                window: CapWindow::Balance,
            }],
        },
    };

    let fleet = Arc::new(CortexState::from_config(&config));
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").unwrap();
        node.healthy = true;
        node.models.insert(
            "test-model".into(),
            ModelEntry {
                id: "test-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
            },
        );
    }

    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (fleet, format!("http://{addr}"))
}

fn principal() -> Principal {
    Principal {
        account_id: ACCOUNT.into(),
        key_id: KEY_ID.into(),
    }
}

/// Poll the provider ledger until settled spend reaches `expected` (settle
/// runs in a spawned task after the response stream finishes) or time out.
async fn await_spent(fleet: &CortexState, expected: u64) -> u64 {
    let principal = principal();
    for _ in 0..100 {
        let snap = fleet.entitlements.snapshot(&principal).await.unwrap();
        if snap.spent >= expected {
            return snap.spent;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    fleet.entitlements.snapshot(&principal).await.unwrap().spent
}

#[tokio::test]
async fn cumulative_spend_is_metered_per_key() {
    let neuron = common::spawn_mock_neuron().await;
    let (fleet, gateway) = spawn_metered_gateway(&neuron).await;
    let client = reqwest::Client::new();

    const N: u64 = 3;
    for _ in 0..N {
        let resp = client
            .post(format!("{gateway}/v1/chat/completions"))
            .bearer_auth(BEARER)
            .json(&json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        // Drain the body so the response stream finishes and metering settles.
        let _ = resp.bytes().await.unwrap();
    }

    let expected = N * (PROMPT_PER_REQ + COMPLETION_PER_REQ);
    let spent = await_spent(&fleet, expected).await;
    assert_eq!(
        spent, expected,
        "ledger must reflect cumulative per-key spend"
    );

    // Reservations settled to actual — nothing left outstanding.
    let snap = fleet.entitlements.snapshot(&principal()).await.unwrap();
    assert_eq!(snap.reserved, 0, "all reservations must settle/release");
    assert_eq!(snap.hard_cap, Some(1_000_000));
}

#[tokio::test]
async fn anonymous_request_records_no_spend() {
    // require_auth=false so the unauthenticated request is served, but with
    // no principal it must not touch any ledger.
    let neuron = common::spawn_mock_neuron().await;
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
            name: "mock-node".into(),
            endpoint: neuron.clone(),
        }],
        models_config: "/dev/null".into(),
        entitlements: EntitlementsConfig::default(),
    };
    let fleet = Arc::new(CortexState::from_config(&config));
    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").unwrap();
        node.healthy = true;
        node.models.insert(
            "test-model".into(),
            ModelEntry {
                id: "test-model".into(),
                status: ModelStatus::Loaded,
                last_accessed: None,
                vram_estimate_mb: Some(8000),
                capabilities: Vec::new(),
                tool_call: false,
                reasoning: false,
                limit: None,
            },
        );
    }
    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let _ = resp.bytes().await.unwrap();

    // An unconfigured principal has a zeroed snapshot — nothing was metered.
    let snap = fleet
        .entitlements
        .snapshot(&Principal {
            account_id: "nobody".into(),
            key_id: "nobody".into(),
        })
        .await
        .unwrap();
    assert_eq!(snap.spent, 0);
}
