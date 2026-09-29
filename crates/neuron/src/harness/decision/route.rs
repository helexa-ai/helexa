//! Choosing which checkpoint of a Laya family answers a request (#339).
//!
//! A released Laya repo carries three checkpoints: the English one at
//! its root, `multilingual/` and `typed-decisions/`. The English
//! checkpoint cannot read other scripts — upstream measured Khmer at
//! 0.000 accuracy with 0.952 confidence — so an unpinned request is
//! routed by the language of its state. This follows the reference's
//! `Router._route` (`laya/router.py`) and its HTTP server's
//! `_resolve_model` (`laya/serve.py`), including the `reason` strings.
//!
//! Precedence, as upstream: a checkpoint the caller names in `model`,
//! then the detected script and language, then the default (English).
//! Upstream's opt-in typed-decisions workflow routing and caller
//! language hints are not exposed over HTTP by the reference, so they
//! are not here either; the matched workflow is still reported.

use cortex_core::decisions::{DecisionRouting, Json};

use crate::harness::lang_detect;

/// The root checkpoint's routing name.
pub const ENGLISH: &str = "english";
pub const MULTILINGUAL: &str = "multilingual";
pub const TYPED_DECISIONS: &str = "typed-decisions";

/// Where an unpinned request with no usable language signal goes.
pub const DEFAULT_CHECKPOINT: &str = ENGLISH;

/// Every checkpoint a released family carries, root first.
pub const FAMILY: [&str; 3] = [ENGLISH, MULTILINGUAL, TYPED_DECISIONS];

/// The reference's `_ALIASES`: names people are likely to type.
const ALIASES: [(&str, &str); 10] = [
    ("en", ENGLISH),
    ("laya", ENGLISH),
    ("default", ENGLISH),
    ("multi", MULTILINGUAL),
    ("ml", MULTILINGUAL),
    ("laya-multilingual", MULTILINGUAL),
    ("typed", TYPED_DECISIONS),
    ("typed_decisions", TYPED_DECISIONS),
    ("laya-typed-decisions", TYPED_DECISIONS),
    ("decisions", TYPED_DECISIONS),
];

/// The reference server's `_PUBLISHED_MODEL_IDS`: the Hugging Face ids
/// the non-root checkpoints are published under.
const PUBLISHED: [(&str, &str); 2] = [
    ("convaiinnovations/laya-multilingual", MULTILINGUAL),
    ("convaiinnovations/laya-typed-decisions", TYPED_DECISIONS),
];

/// Question-id signatures of the four typed-decisions workflows.
const WORKFLOWS: [(&str, [&str; 5]); 4] = [
    (
        "agent_trace_observability",
        ["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        ["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        [
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        [
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

/// The checkpoint a request's `model` field pins, if it names one.
///
/// Anything else — a Jev service name like `jev-1`, the family's own
/// repo id, an empty string — pins nothing and the request is routed.
pub fn resolve_pin(model: Option<&str>) -> Option<&'static str> {
    let model = model?;
    if model.is_empty() {
        return None;
    }
    let key = py_strip(model).to_lowercase();
    if let Some((_, ckpt)) = PUBLISHED.iter().find(|(id, _)| *id == key) {
        return Some(ckpt);
    }
    let key = ALIASES
        .iter()
        .find(|(alias, _)| *alias == key)
        .map(|(_, ckpt)| *ckpt)
        .unwrap_or(key.as_str());
    FAMILY.iter().copied().find(|c| *c == key)
}

/// The typed-decisions workflow whose question ids these are exactly.
pub fn match_workflow(questions: &[(String, Json)]) -> Option<&'static str> {
    let ids: std::collections::BTreeSet<&str> = questions.iter().map(|(q, _)| q.as_str()).collect();
    WORKFLOWS.iter().find_map(|(name, sig)| {
        let sig: std::collections::BTreeSet<&str> = sig.iter().copied().collect();
        (ids == sig).then_some(*name)
    })
}

/// Decide which checkpoint answers, and say why.
///
/// `model_id` is the served family's repo id, used for the `repo` field.
/// The result names a checkpoint whether or not this node has it
/// loaded; the caller decides what to do about one it lacks.
pub fn route(
    model_id: &str,
    state: &Json,
    questions: &[(String, Json)],
    pin: Option<&'static str>,
) -> DecisionRouting {
    if let Some(key) = pin {
        return DecisionRouting {
            model: key.to_string(),
            repo: repo_of(model_id, key),
            reason: format!("explicit model={}", py_repr(key)),
            detection: None,
            workflow: None,
        };
    }
    let workflow = match_workflow(questions).map(str::to_string);
    let det = lang_detect::analyse(state);
    let (key, reason) = if det.script == "unknown" {
        (
            DEFAULT_CHECKPOINT,
            format!("no letters detected in state; using default ({DEFAULT_CHECKPOINT})"),
        )
    } else if det.script != "latin" {
        (
            MULTILINGUAL,
            format!(
                "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                det.script,
                100.0 * det.non_latin_fraction
            ),
        )
    } else if !det.is_english {
        let reason = if let Some(segment) = &det.mixed_segment {
            format!(
                "Latin script, mostly English, but a line or field reads as {} ({}); \
                 the English checkpoint cannot read it",
                det.language.map(py_repr).unwrap_or_else(|| "None".into()),
                py_repr(&segment.chars().take(60).collect::<String>()),
            )
        } else if let Some(lang) = det.language {
            format!(
                "Latin script but language looks like {}, not English",
                py_repr(lang)
            )
        } else {
            format!(
                "Latin script, language not identified but {:.0}% non-English letters; \
                 not safe for the English checkpoint",
                100.0 * det.diacritic_rate
            )
        };
        (MULTILINGUAL, reason)
    } else if det.language_undecided {
        (
            DEFAULT_CHECKPOINT,
            format!(
                "Latin script, language not identified and no non-English letters; \
                 using default ({DEFAULT_CHECKPOINT})"
            ),
        )
    } else {
        (ENGLISH, "English Latin text".to_string())
    };
    DecisionRouting {
        model: key.to_string(),
        repo: repo_of(model_id, key),
        reason,
        detection: Some(analysis_json(&det)),
        workflow,
    }
}

/// The language per-language calibration keys on: the detected one.
pub fn detected_language(routing: &DecisionRouting) -> Option<&str> {
    routing
        .detection
        .as_ref()
        .and_then(|d| d.get("language"))
        .and_then(Json::as_str)
}

/// `<repo>` for the root checkpoint, `<repo>/<subfolder>` otherwise.
pub fn repo_of(model_id: &str, checkpoint: &str) -> String {
    if checkpoint == ENGLISH {
        model_id.to_string()
    } else {
        format!("{model_id}/{checkpoint}")
    }
}

/// The analysis as an order-preserving [`Json`], so the response keeps
/// the reference's field order.
fn analysis_json(det: &lang_detect::Analysis) -> Json {
    let text = serde_json::to_string(det).unwrap_or_else(|_| "null".into());
    Json::parse(text.as_bytes()).unwrap_or(Json::Null)
}

fn py_repr(s: &str) -> String {
    Json::String(s.to_string()).py_repr()
}

/// Python's `str.strip()`: Unicode whitespace, which includes the
/// U+001C–U+001F separators Rust's `trim` leaves alone.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_follow_the_reference_server() {
        assert_eq!(resolve_pin(Some("multilingual")), Some(MULTILINGUAL));
        assert_eq!(resolve_pin(Some(" ML ")), Some(MULTILINGUAL));
        assert_eq!(
            resolve_pin(Some("convaiinnovations/laya-typed-decisions")),
            Some(TYPED_DECISIONS)
        );
        assert_eq!(resolve_pin(Some("laya")), Some(ENGLISH));
        // A Jev service name, the family's own id, and nothing: no pin.
        assert_eq!(resolve_pin(Some("jev-1")), None);
        assert_eq!(resolve_pin(Some("convaiinnovations/laya")), None);
        assert_eq!(resolve_pin(Some("")), None);
        assert_eq!(resolve_pin(None), None);
    }

    #[test]
    fn workflow_needs_the_exact_id_set() {
        let qs = |ids: &[&str]| -> Vec<(String, Json)> {
            ids.iter().map(|i| (i.to_string(), Json::Null)).collect()
        };
        assert_eq!(
            match_workflow(&qs(&[
                "urgency",
                "action",
                "category",
                "churn_risk",
                "needs_human"
            ])),
            Some("customer_service")
        );
        assert_eq!(match_workflow(&qs(&["urgency"])), None);
    }

    /// Every case in the reference fixtures routes to the checkpoint, with
    /// the reason and detection, the reference reported.
    #[test]
    fn routes_every_reference_case_like_the_reference() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/harness/testdata/laya/reference.json"),
        )
        .expect("read reference.json");
        let reference = Json::parse(text.as_bytes()).expect("parse reference.json");
        let Some(Json::Array(cases)) = reference.get("cases") else {
            panic!("no cases");
        };
        for case in cases {
            let name = case.get("name").and_then(Json::as_str).unwrap();
            let request = case.get("request").unwrap();
            let body = serde_json::to_vec(request).unwrap();
            let req = cortex_core::decisions::SystemOneRequest::parse(
                &body,
                cortex_core::decisions::limits::DEFAULT_MAX_TOKEN_BUDGET,
            )
            .unwrap();
            let routing = route(
                "convaiinnovations/laya",
                &req.state,
                &req.questions,
                resolve_pin(req.model.as_deref()),
            );
            let want = case.get("response").unwrap().get("routing").unwrap();
            assert_eq!(
                serde_json::to_string(&routing).unwrap(),
                serde_json::to_string(want).unwrap(),
                "{name}"
            );
        }
    }
}
