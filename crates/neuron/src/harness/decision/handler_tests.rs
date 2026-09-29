//! `CandleHarness::systemone` end to end: request in, routed, validated,
//! scored and shaped response out (#338).
//!
//! The tiny checkpoint (`testdata/laya/tiny/`) exercises the plumbing in
//! CI. The released-weights test replays every reference case through
//! the same entry point the HTTP handler calls and compares the whole
//! response — answers, usage and routing — with what the reference
//! server returned; it needs `LAYA_REFERENCE_DIR`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cortex_core::decisions::{Json, SystemOneRequest, limits};

use crate::harness::candle::{CandleHarness, SystemOneError};
use crate::harness::decision_model::{CheckpointFiles, DecisionCheckpoint, LoadedDecisionModel};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/harness/testdata/laya")
}

fn harness() -> Arc<CandleHarness> {
    CandleHarness::new(
        "http://localhost:13131".into(),
        &crate::config::CandleHarnessConfig::default(),
    )
}

/// A decision model made of the given `(routing name, directory)`
/// checkpoints, loaded on the CPU in f32.
async fn decision_model(model_id: &str, checkpoints: &[(&str, PathBuf)]) -> LoadedDecisionModel {
    let mut loaded = Vec::new();
    for (name, dir) in checkpoints {
        let files = CheckpointFiles {
            tokenizer: if dir.join("tokenizer/tokenizer.json").exists() {
                dir.join("tokenizer/tokenizer.json")
            } else {
                dir.join("tokenizer.json")
            },
            dir: dir.clone(),
        };
        loaded.push(
            DecisionCheckpoint::load(name.to_string(), files, model_id, None)
                .await
                .unwrap_or_else(|e| panic!("load {name}: {e:#}")),
        );
    }
    LoadedDecisionModel {
        model_id: model_id.to_string(),
        spec: serde_json::from_value(serde_json::json!({
            "model_id": model_id,
            "harness": "candle",
        }))
        .expect("model spec"),
        devices: vec![0],
        checkpoints: loaded,
        poisoned: Arc::new(AtomicBool::new(false)),
        admission: crate::harness::admission::AdmissionController::new(
            &crate::config::AdmissionConfig::default(),
        ),
    }
}

fn request(body: &str) -> SystemOneRequest {
    SystemOneRequest::parse(body.as_bytes(), limits::DEFAULT_MAX_TOKEN_BUDGET)
        .expect("valid request")
}

async fn tiny_harness() -> Arc<CandleHarness> {
    let h = harness();
    h.register_decision_model(
        decision_model("test/tiny", &[("english", testdata().join("tiny"))]).await,
    )
    .await;
    h
}

#[tokio::test]
async fn answers_every_question_in_order_with_exact_usage() {
    let h = tiny_harness().await;
    let req = request(
        r#"{"state": "a b a", "questions": {
            "z_last_alphabetically": {"type": "noul", "instructions": "a?"},
            "a_first": {"type": "choice", "instructions": "b?", "criteria": ["a", "b", "a b"]}
        }}"#,
    );
    let resp = h.systemone(None, req, None).await.expect("answered");
    let json = serde_json::to_value(&resp).unwrap();
    // Answers come back in request order, not sorted.
    let keys: Vec<&str> = resp.answers.0.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["z_last_alphabetically", "a_first"]);
    assert_eq!(json["answers"]["z_last_alphabetically"]["type"], "noul");
    let probs = json["answers"]["a_first"]["probabilities"]
        .as_object()
        .unwrap();
    let total: f64 = probs.values().map(|p| p.as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-3, "{probs:?}");
    // Usage is the exact number of tokens encoded, output tokens zero.
    assert!(json["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(json["usage"]["output_tokens"], 0);
    assert_eq!(json["routing"]["model"], "english");
    assert_eq!(json["routing"]["repo"], "test/tiny");
}

#[tokio::test]
async fn pinning_a_checkpoint_the_model_lacks_is_404() {
    let h = tiny_harness().await;
    let req = request(
        r#"{"state": "a", "model": "multilingual",
            "questions": {"q": {"type": "noul", "instructions": "a?"}}}"#,
    );
    match h.systemone(None, req, None).await {
        Err(SystemOneError::Request(e)) => assert_eq!(e.status, 404, "{}", e.detail),
        other => panic!("expected a 404, got {other:?}"),
    }
}

#[tokio::test]
async fn routing_to_a_missing_checkpoint_falls_back_and_says_so() {
    // Devanagari routes to `multilingual`, which this model lacks.
    let h = tiny_harness().await;
    let req = request(
        r#"{"state": "मुझे पैसे वापस चाहिए",
            "questions": {"q": {"type": "noul", "instructions": "a?"}}}"#,
    );
    let resp = h.systemone(None, req, None).await.expect("answered");
    let routing = resp.routing.expect("routing");
    assert_eq!(routing.model, "english");
    assert!(
        routing
            .reason
            .contains("multilingual checkpoint is not loaded"),
        "{}",
        routing.reason
    );
}

#[tokio::test]
async fn invalid_questions_are_rejected_before_admission() {
    let h = tiny_harness().await;
    let req =
        request(r#"{"state": "a", "questions": {"q": {"type": "maybe", "instructions": "a?"}}}"#);
    match h.systemone(None, req, None).await {
        Err(SystemOneError::Request(e)) => assert_eq!(e.status, 422, "{}", e.detail),
        other => panic!("expected a 422, got {other:?}"),
    }
}

#[tokio::test]
async fn a_gateway_routed_model_that_is_not_a_decision_model_is_refused() {
    let h = tiny_harness().await;
    let req = request(r#"{"state": "a", "questions": {}}"#);
    match h.systemone(Some("not/loaded"), req, None).await {
        Err(SystemOneError::Inference(crate::harness::candle::InferenceError::ModelNotLoaded(
            id,
        ))) => {
            assert_eq!(id, "not/loaded")
        }
        other => panic!("expected ModelNotLoaded, got {other:?}"),
    }
}

/// Every reference case through `systemone`, compared with the reference
/// server's whole response. Routing and usage must match exactly;
/// probabilities to within the engine's measured f32 distance from the
/// reference (a last-digit wobble after 4-decimal rounding); the chosen
/// answer exactly.
#[tokio::test]
#[ignore = "needs the released weights: set LAYA_REFERENCE_DIR"]
async fn released_family_answers_like_the_reference_server() {
    let root = PathBuf::from(std::env::var("LAYA_REFERENCE_DIR").expect("LAYA_REFERENCE_DIR"));
    let h = harness();
    h.register_decision_model(
        decision_model(
            "convaiinnovations/laya",
            &[
                ("english", root.clone()),
                ("multilingual", root.join("multilingual")),
                ("typed-decisions", root.join("typed-decisions")),
            ],
        )
        .await,
    )
    .await;

    let text = std::fs::read_to_string(testdata().join("reference.json")).unwrap();
    let reference = Json::parse(text.as_bytes()).unwrap();
    let Some(Json::Array(cases)) = reference.get("cases") else {
        panic!("no cases")
    };
    let mut max_dp = 0.0f64;
    for case in cases {
        let name = case.get("name").and_then(Json::as_str).unwrap();
        let body = serde_json::to_vec(case.get("request").unwrap()).unwrap();
        let req = SystemOneRequest::parse(&body, limits::DEFAULT_MAX_TOKEN_BUDGET).unwrap();
        let resp = h
            .systemone(Some("convaiinnovations/laya"), req, None)
            .await
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let want_json = case.get("response").unwrap();
        // Routing and usage compared as serialized text through `Json`:
        // exact, key order included. serde_json would re-round the
        // reference's floats on the way in.
        for key in ["routing", "usage", "model"] {
            let got = match key {
                "routing" => serde_json::to_string(&resp.routing).unwrap(),
                "usage" => serde_json::to_string(&resp.usage).unwrap(),
                _ => serde_json::to_string(&resp.model).unwrap(),
            };
            let want = serde_json::to_string(want_json.get(key).unwrap()).unwrap();
            assert_eq!(got, want, "{name}: {key}");
        }
        let got = serde_json::to_value(&resp).unwrap();
        let want: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(want_json).unwrap()).unwrap();
        let (ga, wa) = (
            got["answers"].as_object().unwrap(),
            want["answers"].as_object().unwrap(),
        );
        assert_eq!(
            ga.keys().collect::<Vec<_>>(),
            wa.keys().collect::<Vec<_>>(),
            "{name}: answer ids"
        );
        for (qid, w) in wa {
            let g = &ga[qid];
            for key in ["choice", "type", "legend"] {
                assert_eq!(g.get(key), w.get(key), "{name}/{qid}: {key}");
            }
            if let Some(wp) = w["probabilities"].as_object() {
                for (opt, p) in wp {
                    let d = (g["probabilities"][opt].as_f64().unwrap() - p.as_f64().unwrap()).abs();
                    max_dp = max_dp.max(d);
                }
            }
            for key in ["noul", "score", "confidence", "answer_confidence"] {
                if let (Some(gv), Some(wv)) = (
                    g.get(key).and_then(|v| v.as_f64()),
                    w.get(key).and_then(|v| v.as_f64()),
                ) {
                    max_dp = max_dp.max((gv - wv).abs());
                }
            }
        }
    }
    eprintln!("released family: max |Δ| over probabilities and confidences {max_dp:.1e}");
    assert!(max_dp <= 5e-4, "max |Δ| {max_dp:e}");
}
