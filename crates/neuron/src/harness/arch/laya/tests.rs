//! Parity against the upstream reference.
//!
//! `testdata/laya/tiny/` is a toy-sized checkpoint built and scored by
//! the SDK's own `DecisionModel` (`script/generate_laya_fixture.py`);
//! it runs in CI. `testdata/laya/reference.json` holds the released
//! checkpoints' outputs over a real corpus (`script/laya-reference.py`);
//! scoring it needs the weights, so that test is ignored unless
//! `LAYA_REFERENCE_DIR` names a snapshot of `convaiinnovations/laya`.

use super::*;
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/harness/testdata/laya")
}

#[derive(Deserialize)]
struct TinyExpected {
    seqs: Vec<TinySeq>,
}

#[derive(Deserialize)]
struct TinySeq {
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: u8,
    logits: Vec<f32>,
    act: Vec<f32>,
}

fn load_tiny() -> (LayaModel, TinyExpected) {
    let dir = testdata().join("tiny");
    let model = LayaModel::load(&dir, &Device::Cpu, DType::F32).expect("load tiny checkpoint");
    let expected: TinyExpected = serde_json::from_str(
        &std::fs::read_to_string(dir.join("expected.json")).expect("read expected.json"),
    )
    .expect("parse expected.json");
    (model, expected)
}

fn batch_of(seqs: &[TinySeq]) -> DecisionBatch {
    DecisionBatch {
        seqs: seqs
            .iter()
            .map(|s| DecisionSeq {
                input_ids: s.input_ids.clone(),
                markers: s.markers.clone(),
                qtype: s.qtype,
            })
            .collect(),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The whole forward — encoder with both attention kinds and rotary
/// bases, padding, type embedding, head layers, marker gather, scorer
/// and act head — against the reference, in f32.
#[test]
fn tiny_checkpoint_matches_reference() {
    let (model, expected) = load_tiny();
    let rows = model.forward(&batch_of(&expected.seqs)).expect("forward");
    assert_eq!(rows.len(), expected.seqs.len());
    for (i, (got, want)) in rows.iter().zip(&expected.seqs).enumerate() {
        let dl = max_abs_diff(&got.logits, &want.logits);
        let da = max_abs_diff(&got.act, &want.act);
        assert!(
            dl < 1e-5 && da < 1e-5,
            "row {i}: logit diff {dl:e}, act diff {da:e}\n got {:?} / {:?}\nwant {:?} / {:?}",
            got.logits,
            got.act,
            want.logits,
            want.act
        );
    }
}

/// A row's result must not depend on what it was batched with: padded
/// keys are masked out of every attention, including the head's.
#[test]
fn padding_does_not_change_a_row() {
    let (model, expected) = load_tiny();
    let batched = model.forward(&batch_of(&expected.seqs)).expect("batched");
    for (i, seq) in expected.seqs.iter().enumerate() {
        let alone = model
            .forward(&batch_of(std::slice::from_ref(seq)))
            .expect("single");
        let dl = max_abs_diff(&alone[0].logits, &batched[i].logits);
        let da = max_abs_diff(&alone[0].act, &batched[i].act);
        assert!(dl < 1e-5 && da < 1e-5, "row {i}: {dl:e} / {da:e}");
    }
}

#[test]
fn malformed_batches_are_refused() {
    let (model, _) = load_tiny();
    let seq = |ids: Vec<u32>, markers: Vec<usize>, qtype| DecisionBatch {
        seqs: vec![DecisionSeq {
            input_ids: ids,
            markers,
            qtype,
        }],
    };
    assert!(model.forward(&DecisionBatch::default()).is_err());
    assert!(model.forward(&seq(vec![1, 2, 3], vec![], 0)).is_err());
    assert!(model.forward(&seq(vec![1, 2, 3], vec![3], 0)).is_err());
    assert!(model.forward(&seq(vec![1, 2, 3], vec![1], 3)).is_err());
    assert!(model.forward(&seq(vec![1, 500, 3], vec![1], 0)).is_err());
}

#[test]
fn act_features_match_the_reference_definition() {
    // Two equal logits: top1 = top2 = 0.5, entropy exactly 1.
    let f = model::act_features(&[0.3, 0.3]);
    assert!((f[0] - 0.5).abs() < 1e-6 && f[1].abs() < 1e-6 && (f[2] - 1.0).abs() < 1e-6);
    assert!((f[3] - 2.0 / 255.0).abs() < 1e-7);
    // One option: probability 1, no second, k clamps to 2.
    let f = model::act_features(&[4.0]);
    assert_eq!(f[0], 1.0);
    assert_eq!(f[1], 1.0);
    assert!(f[2].abs() < 1e-6);
    assert!((f[3] - 2.0 / 255.0).abs() < 1e-7);
}

// ---------------------------------------------------------------- released checkpoints

#[derive(Deserialize)]
struct Reference {
    cases: Vec<RefCase>,
}

#[derive(Deserialize)]
struct RefCase {
    name: String,
    items: Vec<RefItem>,
    response: serde_json::Value,
}

#[derive(Deserialize)]
struct RefItem {
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: u8,
    logits: Vec<f32>,
    act: Vec<f32>,
}

/// Every case in `reference.json`, scored by the checkpoint the
/// reference routed it to, in f32 on the CPU.
///
/// `LAYA_REFERENCE_DIR` is a snapshot of `convaiinnovations/laya` at the
/// revision recorded in the fixture header: the root holds the English
/// checkpoint, `multilingual/` and `typed-decisions/` the other two.
#[test]
#[ignore = "needs the released weights: set LAYA_REFERENCE_DIR"]
fn released_checkpoints_match_reference() {
    let Ok(root) = std::env::var("LAYA_REFERENCE_DIR") else {
        panic!("LAYA_REFERENCE_DIR is not set");
    };
    let root = PathBuf::from(root);
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(testdata().join("reference.json")).expect("read reference.json"),
    )
    .expect("parse reference.json");

    let mut models = std::collections::BTreeMap::new();
    let mut stats: std::collections::BTreeMap<String, (usize, f32, f32, usize)> =
        Default::default();
    // Forward wall time and tokens encoded, per checkpoint: a first
    // CPU number, not a benchmark (debug builds are far slower).
    let mut timing: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
    for case in &reference.cases {
        let ckpt = case.response["routing"]["model"]
            .as_str()
            .expect("routing.model")
            .to_string();
        let model = models.entry(ckpt.clone()).or_insert_with(|| {
            let dir = match ckpt.as_str() {
                "english" => root.clone(),
                other => root.join(other),
            };
            LayaModel::load(&dir, &Device::Cpu, DType::F32)
                .unwrap_or_else(|e| panic!("load {ckpt}: {e:#}"))
        });
        let batch = DecisionBatch {
            seqs: case
                .items
                .iter()
                .map(|it| DecisionSeq {
                    input_ids: it.input_ids.clone(),
                    markers: it.markers.clone(),
                    qtype: it.qtype,
                })
                .collect(),
        };
        let started = std::time::Instant::now();
        let rows = model
            .forward(&batch)
            .unwrap_or_else(|e| panic!("{}: {e:#}", case.name));
        let t = timing.entry(ckpt.clone()).or_default();
        t.0 += started.elapsed().as_secs_f64() * 1e3;
        t.1 += batch.seqs.iter().map(|s| s.input_ids.len()).sum::<usize>();
        let entry = stats.entry(ckpt.clone()).or_default();
        for (got, want) in rows.iter().zip(&case.items) {
            let dl = max_abs_diff(&got.logits, &want.logits);
            let da = max_abs_diff(&got.act, &want.act);
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
            };
            entry.0 += 1;
            entry.1 = entry.1.max(dl);
            entry.2 = entry.2.max(da);
            if argmax(&got.logits) == argmax(&want.logits) {
                entry.3 += 1;
            }
            assert!(
                dl < 1e-3,
                "{} ({ckpt}): logit diff {dl:e}\n got {:?}\nwant {:?}",
                case.name,
                got.logits,
                want.logits
            );
            assert!(da < 1e-3, "{} ({ckpt}): act diff {da:e}", case.name);
        }
    }
    for (ckpt, (n, dl, da, agree)) in &stats {
        let (ms, tokens) = timing[ckpt];
        eprintln!(
            "{ckpt}: {n} questions, max |Δlogit| {dl:.2e}, max |Δact| {da:.2e}, argmax {agree}/{n}; \
             {tokens} tokens in {ms:.0} ms ({:.2} ms/token)",
            ms / tokens as f64
        );
        assert_eq!(agree, n, "{ckpt}: argmax disagreement");
    }
}
