//! Replays the reference fixtures (`testdata/laya/reference.json`, recorded
//! by `script/laya-reference.py`) through [`prepare`] and [`shape`].
//!
//! Token ids and marker positions must match exactly. The answers and usage
//! must serialize to exactly the same JSON — same keys, same order, same
//! rounded values — when fed the reference's own raw logits.
//!
//! The English tokenizer is committed. The multilingual one is 34 MB, so the
//! multilingual cases run only when `LAYA_ML_TOKENIZER_DIR` names a copy of
//! the checkpoint's `multilingual/tokenizer/` directory.

use super::*;
use cortex_core::decisions::{Json, SystemOneRequest, limits};
use std::path::{Path, PathBuf};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/harness/testdata/laya")
}

fn reference() -> Json {
    Json::parse(&std::fs::read(testdata().join("reference.json")).expect("fixtures")).expect("json")
}

fn to_value(j: &Json) -> serde_json::Value {
    serde_json::to_value(j).expect("serializable")
}

fn tokenizer_for(checkpoint: &str) -> Option<DecisionTokenizer> {
    let dir = match checkpoint {
        // typed-decisions ships the English tokenizer, byte for byte.
        "english" | "typed-decisions" => testdata().join("tokenizer"),
        "multilingual" => PathBuf::from(std::env::var_os("LAYA_ML_TOKENIZER_DIR")?),
        other => panic!("unknown checkpoint {other}"),
    };
    Some(DecisionTokenizer::from_dir(&dir).expect("tokenizer"))
}

fn config_for(reference: &Json, checkpoint: &str) -> DecisionConfig {
    let cfg = reference
        .get("header")
        .and_then(|h| h.get("checkpoints"))
        .and_then(|c| c.get(checkpoint))
        .and_then(|c| c.get("cfg"))
        .expect("checkpoint cfg");
    DecisionConfig::from_json(&serde_json::to_string(cfg).unwrap()).unwrap()
}

fn array(j: &Json) -> &[Json] {
    match j {
        Json::Array(a) => a,
        other => panic!("expected a list, got {other:?}"),
    }
}

fn num(j: &Json) -> f64 {
    match j {
        Json::Int(d) => d.parse().unwrap(),
        Json::Float(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// Replay every case whose tokenizer is available; return how many ran per
/// checkpoint.
fn replay(filter: impl Fn(&str) -> bool) -> Vec<(String, usize)> {
    let reference = reference();
    let mut ran: Vec<(String, usize)> = Vec::new();
    for case in array(reference.get("cases").unwrap()) {
        let name = case.get("name").and_then(Json::as_str).unwrap();
        if !filter(name) {
            continue;
        }
        let response = case.get("response").unwrap();
        let routing = response.get("routing").unwrap();
        let checkpoint = routing.get("model").and_then(Json::as_str).unwrap();
        let Some(tok) = tokenizer_for(checkpoint) else {
            continue;
        };
        let cfg = config_for(&reference, checkpoint);

        let body = serde_json::to_vec(case.get("request").unwrap()).unwrap();
        let req = SystemOneRequest::parse(&body, limits::DEFAULT_MAX_TOKEN_BUDGET)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let budget = Budget {
            max_len: req.max_len,
            head_max_len: req.head_max_len,
        };
        let prepared = prepare(&req.state, &req.questions, &cfg, &tok, budget)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        let items = array(case.get("items").unwrap());
        assert_eq!(prepared.rows.len(), items.len(), "{name}: row count");
        let mut outputs = Vec::new();
        for (i, (row, item)) in prepared.rows.iter().zip(items).enumerate() {
            let ids: Vec<u32> = array(item.get("input_ids").unwrap())
                .iter()
                .map(|v| num(v) as u32)
                .collect();
            let markers: Vec<usize> = array(item.get("markers").unwrap())
                .iter()
                .map(|v| num(v) as usize)
                .collect();
            assert_eq!(row.input_ids, ids, "{name}: row {i} input_ids");
            assert_eq!(row.markers, markers, "{name}: row {i} markers");
            assert_eq!(row.qtype as usize, num(item.get("qtype").unwrap()) as usize);
            let opts = item.get("options").unwrap();
            let s = prepared.stats[i];
            assert_eq!(
                (s.options, s.options_distinct, s.tokens_per_option),
                (
                    num(opts.get("options").unwrap()) as usize,
                    num(opts.get("options_distinct").unwrap()) as usize,
                    match opts.get("tokens_per_option").unwrap() {
                        Json::Null => None,
                        v => Some(num(v) as usize),
                    }
                ),
                "{name}: row {i} option stats"
            );
            outputs.push(RowOutput {
                logits: array(item.get("logits").unwrap())
                    .iter()
                    .map(|v| num(v) as f32)
                    .collect(),
                act_probability: num(&array(item.get("act").unwrap())[0]) as f32,
            });
        }

        let lang = routing
            .get("detection")
            .and_then(|d| d.get("language"))
            .and_then(Json::as_str);
        let (answers, usage) = shape(&prepared, &outputs, &cfg, lang);
        // Serialized strings, not values: key order is part of the contract.
        assert_eq!(
            serde_json::to_string(&answers).unwrap(),
            serde_json::to_string(response.get("answers").unwrap()).unwrap(),
            "{name}: answers"
        );
        assert_eq!(
            serde_json::to_value(&usage).unwrap(),
            to_value(response.get("usage").unwrap()),
            "{name}: usage"
        );
        match ran.iter_mut().find(|(c, _)| c == checkpoint) {
            Some((_, n)) => *n += 1,
            None => ran.push((checkpoint.to_string(), 1)),
        }
    }
    ran
}

#[test]
fn every_english_fixture_matches_the_reference() {
    let ran = replay(|_| true);
    let count = |c: &str| ran.iter().find(|(n, _)| n == c).map_or(0, |(_, n)| *n);
    // The corpus routes 19 cases to English and 1 to typed-decisions; a
    // drop means a case silently stopped running.
    assert_eq!(count("english"), 19, "{ran:?}");
    assert_eq!(count("typed-decisions"), 1, "{ran:?}");
    if std::env::var_os("LAYA_ML_TOKENIZER_DIR").is_some() {
        assert_eq!(count("multilingual"), 8, "{ran:?}");
    }
}

#[test]
fn empty_questions_need_no_forward() {
    let reference = reference();
    let tok = tokenizer_for("english").unwrap();
    let cfg = config_for(&reference, "english");
    let prepared = prepare(
        &Json::String("x".into()),
        &[],
        &cfg,
        &tok,
        Budget::default(),
    )
    .unwrap();
    assert!(prepared.rows.is_empty());
    let (answers, usage) = shape(&prepared, &[], &cfg, None);
    assert_eq!(serde_json::to_string(&answers).unwrap(), "{}");
    assert_eq!(
        serde_json::to_string(&usage).unwrap(),
        r#"{"input_tokens":0,"output_tokens":0}"#
    );
}

#[test]
fn options_that_cannot_all_fit_are_refused() {
    let reference = reference();
    let tok = tokenizer_for("english").unwrap();
    let cfg = config_for(&reference, "english");
    let questions = Json::parse(
        br#"{"q": {"type": "choice", "instructions": "i", "criteria": ["a", "b", "c", "d"]}}"#,
    )
    .unwrap();
    let Json::Object(questions) = questions else {
        unreachable!()
    };
    let budget = Budget {
        max_len: Some(10),
        head_max_len: None,
    };
    let err = prepare(&Json::String("x".into()), &questions, &cfg, &tok, budget).unwrap_err();
    assert_eq!(
        (err.status, err.detail.as_str()),
        (422, "question 'q' options exceed head_max_len=192")
    );
}
