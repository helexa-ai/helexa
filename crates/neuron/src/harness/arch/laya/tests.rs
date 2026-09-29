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

/// Device and precision for the weight-backed tests: `LAYA_DEVICE`
/// (`cpu`, or `cuda:N` in a build with the `cuda` feature) and
/// `LAYA_DTYPE` (`f32`, `bf16` or `f16`). Defaults: CPU, f32.
fn test_device() -> (Device, DType) {
    let device = match std::env::var("LAYA_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(s) => {
            let ordinal = s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("LAYA_DEVICE={s}: expected cpu or cuda:N"));
            Device::new_cuda(ordinal).unwrap_or_else(|e| panic!("LAYA_DEVICE={s}: {e}"))
        }
    };
    let dtype = match std::env::var("LAYA_DTYPE").as_deref() {
        Ok("f32") | Err(_) => DType::F32,
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        Ok(s) => panic!("LAYA_DTYPE={s}: expected f32, bf16 or f16"),
    };
    (device, dtype)
}

/// Logit tolerance against the f32 reference for a given precision.
/// Reduced precision is judged mainly on argmax agreement; the bound
/// only catches a broken kernel, not rounding.
fn logit_tolerance(dtype: DType) -> f32 {
    if dtype == DType::F32 { 1e-3 } else { 1.0 }
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
/// reference routed it to, on the device and precision
/// [`test_device`] selects (f32 on the CPU by default).
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
    let (device, dtype) = test_device();
    let tolerance = logit_tolerance(dtype);
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(testdata().join("reference.json")).expect("read reference.json"),
    )
    .expect("parse reference.json");

    // The reference's own GPU run (bf16 autocast), for judging reduced
    // precision against the drift upstream itself serves with.
    let torch_bf16: Option<Reference> =
        std::fs::read_to_string(testdata().join("reference_cuda_bf16.json"))
            .ok()
            .map(|t| serde_json::from_str(&t).expect("parse reference_cuda_bf16.json"));
    // Per checkpoint: summed centred drift of ours and of the reference's
    // bf16 run, their maxima, and the row count.
    let mut drift: std::collections::BTreeMap<String, (f32, f32, f32, f32, usize)> =
        Default::default();

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
            LayaModel::load(&dir, &device, dtype).unwrap_or_else(|e| panic!("load {ckpt}: {e:#}"))
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
        let torch_case = torch_bf16
            .as_ref()
            .and_then(|r| r.cases.iter().find(|c| c.name == case.name));
        for (i, (got, want)) in rows.iter().zip(&case.items).enumerate() {
            let d = drift.entry(ckpt.clone()).or_default();
            let ours = centred_diff(&got.logits, &want.logits);
            d.0 += ours;
            d.2 = d.2.max(ours);
            if let Some(t) = torch_case.and_then(|c| c.items.get(i)) {
                let theirs = centred_diff(&t.logits, &want.logits);
                d.1 += theirs;
                d.3 = d.3.max(theirs);
            }
            d.4 += 1;
        }
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
            if dtype != DType::F32 && std::env::var_os("LAYA_VERBOSE").is_some() {
                eprintln!("  {:<40} {ckpt:<16} |Δlogit| {dl:.3}", case.name);
            }
            assert!(
                dl < tolerance,
                "{} ({ckpt}): logit diff {dl:e}\n got {:?}\nwant {:?}",
                case.name,
                got.logits,
                want.logits
            );
            assert!(da < tolerance, "{} ({ckpt}): act diff {da:e}", case.name);
        }
    }
    for (ckpt, (n, dl, da, agree)) in &stats {
        let (ms, tokens) = timing[ckpt];
        eprintln!(
            "[{device:?} {dtype:?}] {ckpt}: {n} questions, max |Δlogit| {dl:.2e}, max |Δact| {da:.2e}, argmax {agree}/{n}; \
             {tokens} tokens in {ms:.0} ms ({:.2} ms/token)",
            ms / tokens as f64
        );
        assert_eq!(agree, n, "{ckpt}: argmax disagreement");
    }
    // Reduced precision: centred drift (softmax ignores a constant shift)
    // against f32, next to the reference's own bf16 drift. Per-case drift
    // is chaotic, so the gate is the mean: no worse than twice upstream's,
    // plus 0.01 — about one bf16 ulp at these logits' magnitude — so a
    // checkpoint with a single fixture case whose reference drift happens
    // to round to nothing is not held to zero.
    for (ckpt, (ours, theirs, ours_max, theirs_max, n)) in &drift {
        let (ours, theirs) = (ours / *n as f32, theirs / *n as f32);
        eprintln!(
            "[{device:?} {dtype:?}] {ckpt}: centred drift mean {ours:.4} (reference bf16 {theirs:.4}), \
             max {ours_max:.3} (reference bf16 {theirs_max:.3})"
        );
        if dtype != DType::F32 && torch_bf16.is_some() {
            assert!(
                ours <= 2.0 * theirs + 0.01,
                "{ckpt}: mean drift {ours:.4} exceeds twice the reference's own bf16 drift {theirs:.4}"
            );
        }
    }
}

/// Largest difference between two logit rows after removing each row's
/// mean — the part of a difference a softmax can see.
fn centred_diff(a: &[f32], b: &[f32]) -> f32 {
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (ma, mb) = (mean(a), mean(b));
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - ma) - (y - mb)).abs())
        .fold(0.0, f32::max)
}

/// Forward latency for the four workloads #337 benchmarks against the
/// PyTorch reference, plus batch throughput, on the English checkpoint.
///
/// Sequences are real ones from `reference.json`: `choice_basic` is the
/// short state (41 tokens), `multi_question` five questions over it, and
/// `state_right_truncated` a full 512-token sequence (repeated for the
/// five-question and throughput batches). The timed span is the whole
/// `forward` call — host-to-device copy, forward, and the CPU readback
/// that ends it — so it is directly comparable to the reference's
/// end-to-end `predict()` minus tokenisation.
///
/// Run with `cargo test --release -p neuron [--features cuda] --
/// --ignored --nocapture laya_forward_timing`, with `LAYA_REFERENCE_DIR`
/// and optionally `LAYA_DEVICE` / `LAYA_DTYPE` set.
#[test]
#[ignore = "benchmark: needs the released weights (LAYA_REFERENCE_DIR)"]
fn laya_forward_timing() {
    let root = PathBuf::from(std::env::var("LAYA_REFERENCE_DIR").expect("LAYA_REFERENCE_DIR"));
    let (device, dtype) = test_device();
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(testdata().join("reference.json")).expect("read reference.json"),
    )
    .expect("parse reference.json");
    let seqs_of = |name: &str| -> Vec<DecisionSeq> {
        reference
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case {name}"))
            .items
            .iter()
            .map(|it| DecisionSeq {
                input_ids: it.input_ids.clone(),
                markers: it.markers.clone(),
                qtype: it.qtype,
            })
            .collect()
    };
    let long = seqs_of("state_right_truncated").remove(0);
    let workloads: Vec<(String, DecisionBatch)> = vec![
        (
            "short x1q".into(),
            DecisionBatch {
                seqs: seqs_of("choice_basic"),
            },
        ),
        (
            "short x5q".into(),
            DecisionBatch {
                seqs: seqs_of("multi_question"),
            },
        ),
        (
            "512tok x1q".into(),
            DecisionBatch {
                seqs: vec![long.clone()],
            },
        ),
        (
            "512tok x5q".into(),
            DecisionBatch {
                seqs: vec![long.clone(); 5],
            },
        ),
        (
            "512tok x16".into(),
            DecisionBatch {
                seqs: vec![long.clone(); 16],
            },
        ),
        (
            "512tok x64".into(),
            DecisionBatch {
                seqs: vec![long.clone(); 64],
            },
        ),
    ];

    let model = LayaModel::load(&root, &device, dtype).expect("load English checkpoint");
    // `LAYA_TIMING_ONLY` keeps the workloads whose name contains it, and
    // `LAYA_TIMING_ITERS` sets the timed iterations — for profiling one
    // workload under a tracer.
    let only = std::env::var("LAYA_TIMING_ONLY").ok();
    let iters: usize = std::env::var("LAYA_TIMING_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    for (name, batch) in &workloads {
        if only.as_deref().is_some_and(|o| !name.contains(o)) {
            continue;
        }
        for _ in 0..3 {
            model.forward(batch).expect("warm-up forward");
        }
        let mut ms: Vec<f64> = (0..iters)
            .map(|_| {
                let started = std::time::Instant::now();
                model.forward(batch).expect("forward");
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        let median = ms[ms.len() / 2];
        let tokens: usize = batch.seqs.iter().map(|s| s.input_ids.len()).sum();
        eprintln!(
            "[{device:?} {dtype:?}] {name:>11}: median {median:7.1} ms, min {:7.1} ms, \
             {tokens} tokens, {:.0} seq/s",
            ms[0],
            batch.seqs.len() as f64 / (median / 1e3)
        );
    }
}
