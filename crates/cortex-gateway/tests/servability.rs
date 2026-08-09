//! A node that cannot serve is not a routing target (#245).
//!
//! The incident: quadbrat held `helexa/small` with 1405 MiB free against
//! a 1500 MiB prefill floor, so every request was rejected before any
//! device work. The model was `loaded`, the node answered every poll,
//! and cortex reported `3/3 healthy` — while the whole small tier
//! returned 503 to clients. Liveness was true and useless.
//!
//! Two properties are pinned here, and the second matters as much as the
//! first: an unservable location must be skipped, and a *silent* neuron
//! must not be mistaken for an unservable one. Reading absent evidence
//! as "broken" would let a version skew empty the routing table and take
//! the fleet down far more thoroughly than the fault being guarded.

mod common;

use cortex_core::harness::ModelServability;
use cortex_core::node::{ModelEntry, ModelStatus};

fn entry(id: &str, servable: Option<ModelServability>) -> ModelEntry {
    ModelEntry {
        id: id.into(),
        status: ModelStatus::Loaded,
        last_accessed: None,
        vram_estimate_mb: Some(4000),
        capabilities: vec!["text".into()],
        tool_call: false,
        reasoning: false,
        limit: None,
        servable,
    }
}

fn squeezed() -> ModelServability {
    ModelServability {
        ok: false,
        reason: Some("insufficient_vram".into()),
        detail: Some("1405 MiB free, need at least 1500 MiB to start a prefill".into()),
    }
}

/// The safety property. A neuron that says nothing — an older build, a
/// CPU load, an image model, a cache not yet seeded — stays eligible.
#[test]
fn silence_means_servable() {
    assert!(
        entry("m", None).is_servable(),
        "absent evidence must not be read as unservable"
    );
}

#[test]
fn an_explicit_ok_is_servable() {
    assert!(
        entry(
            "m",
            Some(ModelServability {
                ok: true,
                reason: None,
                detail: None
            })
        )
        .is_servable()
    );
}

/// The incident condition itself.
#[test]
fn a_squeezed_node_is_not_servable() {
    let e = entry("helexa/small", Some(squeezed()));
    assert!(!e.is_servable());
    assert_eq!(
        e.servable.as_ref().and_then(|s| s.reason.as_deref()),
        Some("insufficient_vram"),
        "the reason should match the code the request itself would fail with"
    );
}

/// `/health` must stop calling the fleet `ok` while a resident model
/// cannot be served. Reporting healthy through a tier outage is what
/// made the incident invisible for as long as it was.
#[tokio::test]
async fn health_reports_impaired_when_a_model_cannot_be_served() {
    let neuron = common::spawn_mock_neuron().await;
    let (fleet, gateway) = common::spawn_gateway_with_state(&neuron).await;

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").expect("seeded node");
        node.healthy = true;
        node.models
            .insert("test-model".into(), entry("test-model", Some(squeezed())));
    }

    let body: serde_json::Value = reqwest::get(format!("{gateway}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(
        body["status"], "impaired",
        "a reachable node that cannot serve is not 'ok'"
    );
    let listed = body["unservable"].as_array().expect("unservable listed");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["node"], "mock-node");
    assert_eq!(listed[0]["model"], "test-model");
    assert_eq!(listed[0]["reason"], "insufficient_vram");
}

/// The control: the same fleet with nothing squeezed reports `ok`, so
/// the assertion above is detecting the condition rather than always
/// firing.
#[tokio::test]
async fn health_reports_ok_when_everything_can_serve() {
    let neuron = common::spawn_mock_neuron().await;
    let (fleet, gateway) = common::spawn_gateway_with_state(&neuron).await;

    {
        let mut nodes = fleet.nodes.write().await;
        let node = nodes.get_mut("mock-node").expect("seeded node");
        node.healthy = true;
        node.models
            .insert("test-model".into(), entry("test-model", None));
    }

    let body: serde_json::Value = reqwest::get(format!("{gateway}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["status"], "ok");
    assert!(body.get("unservable").is_none());
}

// ── Routing ──────────────────────────────────────────────────────────

/// The behavioural fix. Two replicas, one squeezed: the request must go
/// to the one that can actually answer it.
///
/// This is the shape the incident took, minus the detail that made it
/// total — quadbrat was the *only* node holding `helexa/small`, so there
/// was no second replica and the request 503'd. With the location
/// skipped, that case falls through to the catalogue cold-load path
/// instead, which places the model somewhere it fits.
#[tokio::test]
async fn routing_skips_a_loaded_but_unservable_replica() {
    let neuron_a = common::spawn_mock_neuron().await;
    let neuron_b = common::spawn_mock_neuron().await;
    let fleet = common::two_neuron_fleet(&neuron_a, &neuron_b).await;

    {
        let mut nodes = fleet.nodes.write().await;
        for name in ["node-a", "node-b"] {
            let n = nodes.get_mut(name).expect("node exists");
            n.healthy = true;
        }
        // node-a is squeezed; node-b is fine. node-a sorts first by
        // name, so a router that ignored servability would pick it.
        nodes
            .get_mut("node-a")
            .unwrap()
            .models
            .insert("test-model".into(), entry("test-model", Some(squeezed())));
        nodes.get_mut("node-b").unwrap().models.insert(
            "test-model".into(),
            entry(
                "test-model",
                Some(ModelServability {
                    ok: true,
                    reason: None,
                    detail: None,
                }),
            ),
        );
    }

    let route = cortex_gateway::router::resolve(&fleet, "test-model")
        .await
        .expect("one replica can serve");
    assert_eq!(
        route.node_name, "node-b",
        "must route past the squeezed replica, not to it"
    );
}
