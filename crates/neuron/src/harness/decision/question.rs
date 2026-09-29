//! Question validation, normalisation and option rendering.
//!
//! Every rejection here is a 422 whose message names the question and what
//! to fix — the same text the reference returns, so a client written
//! against it sees the same errors.

use super::{DecisionRequestError, unprocessable};
use cortex_core::decisions::{HashKey, Json};
use std::collections::HashMap;

/// The three decision primitives. The discriminant is the index the model's
/// type embedding and the calibration tables use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    /// Pick one of the caller's labels.
    Choice = 0,
    /// An ordinal level on the caller's rubric.
    Score = 1,
    /// Yes/no: the probability that a statement holds.
    Noul = 2,
}

impl QType {
    pub const NAMES: [&'static str; 3] = ["choice", "score", "noul"];

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }
}

/// A validated question, normalised the way the reference normalises it.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub kind: QType,
    /// The instruction text: the caller's string, or the JSON rendering of
    /// a structured value.
    pub instructions: String,
    /// Each option's rendered text, in answer-index order. For `noul` this
    /// is always `[false, true]`, so index 1 is P(true).
    pub options: Vec<String>,
    /// `choice`: each option's label as the caller wrote it (the answer key).
    /// `score`: each level's description (the legend). `noul`: empty.
    pub labels: Vec<Json>,
}

impl Question {
    /// Validate one question definition and normalise it.
    pub fn from_definition(id: &str, def: &Json) -> Result<Self, DecisionRequestError> {
        check_question(id, def)?;
        Ok(to_internal(id, def))
    }
}

/// Validate and normalise every question, in request order. The first
/// invalid question is reported; nothing is normalised until all pass.
pub fn questions_from_request(
    questions: &[(String, Json)],
) -> Result<Vec<Question>, DecisionRequestError> {
    for (id, def) in questions {
        check_question(id, def)?;
    }
    Ok(questions
        .iter()
        .map(|(id, def)| to_internal(id, def))
        .collect())
}

fn q(id: &str) -> String {
    Json::String(id.to_string()).py_repr()
}

fn check_question(id: &str, def: &Json) -> Result<(), DecisionRequestError> {
    let qid = q(id);
    if !matches!(def, Json::Object(_)) {
        return Err(unprocessable(format!(
            "question {qid}: definition must be a dict, got {}",
            def.py_type_name()
        )));
    }
    let t = def.get("type").unwrap_or(&Json::Null);
    let Some(kind) = t.as_str().and_then(QType::parse) else {
        return Err(unprocessable(format!(
            "question {qid}: unknown type {}; use one of ['choice', 'noul', 'score']",
            t.py_repr()
        )));
    };
    if def.get("instructions").is_none() {
        return Err(unprocessable(format!(
            "question {qid}: no 'instructions'; add the text the model should answer"
        )));
    }
    let crit = def.get("criteria");
    match kind {
        QType::Choice => check_choice(&qid, crit)?,
        QType::Score => {
            let Some(Json::Array(levels)) = crit else {
                return Err(unprocessable(format!(
                    "question {qid}: a score question takes 'criteria' as a list of level \
                     descriptions, index 0 first"
                )));
            };
            if levels.is_empty() {
                return Err(unprocessable(format!(
                    "question {qid}: a score question needs at least one level"
                )));
            }
            if let Some(i) = levels.iter().position(|l| *l == Json::Null) {
                return Err(unprocessable(format!(
                    "question {qid}: score level {i} is null; give every level a description, \
                     index 0 first"
                )));
            }
        }
        QType::Noul => match crit {
            None | Some(Json::Null) => {}
            Some(Json::Object(m)) => {
                let mut keys: Vec<String> = m.iter().map(|(k, _)| k.to_lowercase()).collect();
                keys.sort();
                keys.dedup();
                if keys.iter().any(|k| k != "true" && k != "false") {
                    let listed = Json::Array(keys.into_iter().map(Json::String).collect());
                    return Err(unprocessable(format!(
                        "question {qid}: a noul question takes 'criteria' keyed only \
                         'true'/'false' (either or both, and omitted is fine), got {}. Those \
                         keys are the option texts the model reads; any other key was silently \
                         dropped and replaced with the defaults. If you want the answer worded \
                         differently, keep 'criteria' keyed 'true'/'false' and set 'labels' \
                         instead.",
                        listed.py_repr()
                    )));
                }
            }
            Some(_) => {
                return Err(unprocessable(format!(
                    "question {qid}: a noul question takes 'criteria' as a dict with optional \
                     'true'/'false' descriptions, or omits it"
                )));
            }
        },
    }
    if let Some(labels) = def.get("labels") {
        if kind != QType::Noul {
            return Err(unprocessable(format!(
                "question {qid}: 'labels' is only supported for noul questions"
            )));
        }
        resolve_noul_labels(labels).map_err(|e| unprocessable(format!("question {qid}: {e}")))?;
    }
    Ok(())
}

fn check_choice(qid: &str, crit: Option<&Json>) -> Result<(), DecisionRequestError> {
    let labels: Vec<Json> = match crit {
        Some(Json::Object(m)) => m.iter().map(|(k, _)| Json::String(k.clone())).collect(),
        Some(Json::Array(a)) => a.clone(),
        _ => {
            return Err(unprocessable(format!(
                "question {qid}: a choice question takes 'criteria' as a dict of label -> \
                 description, or a list of labels"
            )));
        }
    };
    if labels.is_empty() {
        return Err(unprocessable(format!(
            "question {qid}: a choice question needs at least one criterion"
        )));
    }
    for (i, label) in labels.iter().enumerate() {
        match label {
            Json::Array(_) | Json::Object(_) => {
                return Err(unprocessable(format!(
                    "question {qid}: choice label {i} is a {}; a label is rendered as option \
                     text and used as the answer key, so it must be a scalar (a string, number \
                     or bool), got {}",
                    label.py_type_name(),
                    label.py_repr()
                )));
            }
            Json::Null => {
                return Err(unprocessable(format!(
                    "question {qid}: choice label {i} is null; a label is rendered as option \
                     text and used as the answer key, so it must be a string, number or bool \
                     -- a null label renders as the text \"None\" while its answer key is \
                     \"null\""
                )));
            }
            _ => {}
        }
    }
    // Only the list form can repeat a key: an object's keys are unique.
    if matches!(crit, Some(Json::Array(_))) {
        let mut seen: HashMap<HashKey, usize> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            let key = label.hash_key().expect("scalars are hashable");
            if key == HashKey::Unique {
                continue;
            }
            if let Some(first) = seen.get(&key) {
                return Err(unprocessable(format!(
                    "question {qid}: choice label {i} ({}) repeats label {first}; the labels are \
                     the answer keys, so every option needs its own (1, 1.0 and True are one key)",
                    label.py_repr()
                )));
            }
            seen.insert(key, i);
        }
    }
    Ok(())
}

const DEFAULT_NOUL_FALSE: &str = "no, the statement does not hold";
const DEFAULT_NOUL_TRUE: &str = "yes, the statement holds";

/// The words a noul question shows the model for its two outcomes.
fn resolve_noul_labels(labels: &Json) -> Result<(String, String), &'static str> {
    const BAD: &str =
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
    let Json::Object(m) = labels else {
        return if *labels == Json::Null {
            Ok(("false".into(), "true".into()))
        } else {
            Err(BAD)
        };
    };
    let mut keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
    keys.sort_unstable();
    if keys != ["false", "true"] {
        return Err(BAD);
    }
    let (Some(Json::String(f)), Some(Json::String(t))) = (labels.get("false"), labels.get("true"))
    else {
        return Err(BAD);
    };
    let (f, t) = (py_strip(f), py_strip(t));
    if f.is_empty() || t.is_empty() || f == t {
        return Err(BAD);
    }
    Ok((f.to_string(), t.to_string()))
}

/// Python's `str.strip()` with no argument: Unicode whitespace.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// `render_criterion`: a string as itself, anything else as JSON.
fn render_criterion(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        other => other.dumps(),
    }
}

/// Only `None` and `""` mean "no description"; `0` and `false` are values.
fn is_blank(v: &Json) -> bool {
    matches!(v, Json::Null) || matches!(v, Json::String(s) if s.is_empty())
}

fn to_internal(id: &str, def: &Json) -> Question {
    let kind = def
        .get("type")
        .and_then(Json::as_str)
        .and_then(QType::parse)
        .expect("validated");
    let instructions = match def.get("instructions").expect("validated") {
        Json::String(s) => s.clone(),
        other => other.dumps(),
    };
    let crit = def.get("criteria");
    let (options, labels) = match kind {
        QType::Choice => {
            let pairs: Vec<(Json, Json)> = match crit {
                Some(Json::Object(m)) => m
                    .iter()
                    .map(|(k, v)| (Json::String(k.clone()), v.clone()))
                    .collect(),
                Some(Json::Array(a)) => a.iter().map(|l| (l.clone(), Json::Null)).collect(),
                _ => unreachable!("validated"),
            };
            let options = pairs
                .iter()
                .map(|(k, v)| {
                    if is_blank(v) {
                        k.py_str()
                    } else {
                        format!("{}: {}", k.py_str(), render_criterion(v))
                    }
                })
                .collect();
            (options, pairs.into_iter().map(|(k, _)| k).collect())
        }
        QType::Score => {
            let Some(Json::Array(levels)) = crit else {
                unreachable!("validated")
            };
            let options = levels
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect();
            (options, levels.clone())
        }
        QType::Noul => {
            // Keys are matched case-insensitively; a repeat after
            // lower-casing keeps the later value, as a dict would.
            let mut desc: HashMap<String, Json> = HashMap::new();
            if let Some(Json::Object(m)) = crit {
                for (k, v) in m {
                    desc.insert(k.to_lowercase(), v.clone());
                }
            }
            let (fl, tl) =
                resolve_noul_labels(def.get("labels").unwrap_or(&Json::Null)).expect("validated");
            let render = |key: &str, default: &str| match desc.get(key) {
                Some(v) if !is_blank(v) => render_criterion(v),
                _ => default.to_string(),
            };
            (
                vec![
                    format!("{fl}: {}", render("false", DEFAULT_NOUL_FALSE)),
                    format!("{tl}: {}", render("true", DEFAULT_NOUL_TRUE)),
                ],
                Vec::new(),
            )
        }
    };
    Question {
        id: id.to_string(),
        kind,
        instructions,
        options,
        labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(s: &str) -> Json {
        Json::parse(s.as_bytes()).unwrap()
    }

    fn err(s: &str) -> String {
        Question::from_definition("q", &def(s)).unwrap_err().detail
    }

    #[test]
    fn options_render_as_the_reference_renders_them() {
        let c = Question::from_definition(
            "q",
            &def(r#"{"type": "choice", "instructions": {"a": 1},
                     "criteria": {"x": "", "y": null, "z": 0, "w": {"d": [1, 2.0]}, "v": "text"}}"#),
        )
        .unwrap();
        assert_eq!(c.instructions, r#"{"a": 1}"#);
        assert_eq!(
            c.options,
            ["x", "y", "z: 0", r#"w: {"d": [1, 2.0]}"#, "v: text"]
        );

        let l = Question::from_definition(
            "q",
            &def(r#"{"type": "choice", "instructions": "i", "criteria": [1, 2.5, false, "s"]}"#),
        )
        .unwrap();
        assert_eq!(l.options, ["1", "2.5", "False", "s"]);

        let s = Question::from_definition(
            "q",
            &def(r#"{"type": "score", "instructions": "i", "criteria": ["calm", {"k": false}]}"#),
        )
        .unwrap();
        assert_eq!(s.options, ["level 0: calm", r#"level 1: {"k": false}"#]);

        let n = Question::from_definition(
            "q",
            &def(
                r#"{"type": "noul", "instructions": "i", "criteria": {"TRUE": "held", "false": ""},
                     "labels": {"false": " no ", "true": "yes"}}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            n.options,
            ["no: no, the statement does not hold", "yes: held"]
        );
    }

    #[test]
    fn invalid_questions_name_the_question_and_the_fix() {
        assert_eq!(
            err("[]"),
            "question 'q': definition must be a dict, got list"
        );
        assert_eq!(
            err(r#"{"type": "pick"}"#),
            "question 'q': unknown type 'pick'; use one of ['choice', 'noul', 'score']"
        );
        assert_eq!(
            err(r#"{"type": "noul"}"#),
            "question 'q': no 'instructions'; add the text the model should answer"
        );
        assert!(
            err(r#"{"type": "choice", "instructions": "i", "criteria": []}"#)
                .ends_with("needs at least one criterion")
        );
        assert!(
            err(r#"{"type": "choice", "instructions": "i", "criteria": [[1]]}"#)
                .starts_with("question 'q': choice label 0 is a list;")
        );
        assert!(
            err(r#"{"type": "choice", "instructions": "i", "criteria": ["a", null]}"#)
                .starts_with("question 'q': choice label 1 is null;")
        );
        assert!(
            err(r#"{"type": "choice", "instructions": "i", "criteria": [1, "x", true]}"#)
                .starts_with("question 'q': choice label 2 (True) repeats label 0;")
        );
        assert_eq!(
            err(r#"{"type": "score", "instructions": "i", "criteria": ["a", null]}"#),
            "question 'q': score level 1 is null; give every level a description, index 0 first"
        );
        assert!(
            err(r#"{"type": "noul", "instructions": "i", "criteria": {"yes": "x"}}"#)
                .contains("got ['yes'].")
        );
        assert_eq!(
            err(r#"{"type": "score", "instructions": "i", "criteria": ["a"], "labels": {}}"#),
            "question 'q': 'labels' is only supported for noul questions"
        );
        assert_eq!(
            err(r#"{"type": "noul", "instructions": "i", "labels": {"false": "x", "true": " x"}}"#),
            "question 'q': noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        );
    }

    #[test]
    fn nan_labels_never_collide() {
        assert!(
            Question::from_definition(
                "q",
                &def(r#"{"type": "choice", "instructions": "i", "criteria": [NaN, NaN]}"#),
            )
            .is_ok()
        );
    }
}
