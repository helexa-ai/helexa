//! Per-checkpoint decision settings, read from `rl_agent_config.json`.
//!
//! Parsed with [`Json`], not `serde_json`: the fitted temperatures are
//! written with 17 significant digits, and `serde_json`'s default float
//! parser is not correctly rounded (it reads `1.7601518630981445` as
//! `…443`).

use cortex_core::decisions::Json;
use std::collections::HashMap;

/// Default token budgets when a checkpoint's config omits them.
pub const DEFAULT_MAX_LEN: usize = 512;
pub const DEFAULT_HEAD_MAX_LEN: usize = 192;

/// Bounds on a usable calibration temperature. A fitted value below 1
/// sharpens instead of softening; the shipped `choice:11+` bucket is 0.1006,
/// which would publish a coin flip as a certainty, so the reference refuses
/// to apply anything outside this range.
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// A temperature confined to [`TEMP_MIN`, `TEMP_MAX`]; a non-finite or
/// non-numeric value falls back to 1.0.
pub fn clamp_temperature(t: Option<f64>) -> f64 {
    match t {
        Some(t) if t.is_finite() => t.clamp(TEMP_MIN, TEMP_MAX),
        _ => 1.0,
    }
}

/// Python's `float(v)` for a JSON value: numbers, bools, and numeric strings.
fn py_float(v: &Json) -> Option<f64> {
    match v {
        Json::Int(d) => d.parse().ok(),
        Json::Float(x) => Some(*x),
        Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Json::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// One set of calibration temperatures: per question type, overridden per
/// `(type, option-count bucket)` (see [`temp_bucket`]). Already clamped.
#[derive(Debug, Clone, PartialEq)]
pub struct Temperatures {
    /// Indexed by [`super::QType`] (`choice`, `score`, `noul`).
    pub by_type: [f64; 3],
    /// `"choice:3-5"` → temperature, and so on.
    pub by_options: HashMap<String, f64>,
}

impl Temperatures {
    /// The temperature for a question of type index `qtype` with `k` options.
    pub fn for_question(&self, qtype: usize, k: usize) -> f64 {
        self.by_options
            .get(&temp_bucket(qtype, k))
            .copied()
            .unwrap_or(self.by_type[qtype])
    }

    fn from_json(
        temperature: Option<&Json>,
        by_options: Option<&Json>,
        fallback: Option<[f64; 3]>,
    ) -> Result<Self, String> {
        let raw = match temperature {
            None => fallback.unwrap_or([1.0; 3]).map(Some),
            Some(Json::Array(a)) if a.len() == 3 => {
                [py_float(&a[0]), py_float(&a[1]), py_float(&a[2])]
            }
            Some(other) => {
                return Err(format!(
                    "temperature must be a list of 3 floats, got {}",
                    other.py_repr()
                ));
            }
        };
        let by_options = match by_options {
            Some(Json::Object(m)) => m
                .iter()
                .map(|(k, v)| (k.clone(), clamp_temperature(py_float(v))))
                .collect(),
            _ => HashMap::new(),
        };
        Ok(Self {
            by_type: raw.map(clamp_temperature),
            by_options,
        })
    }
}

/// The calibration bucket of a question: its type and option count.
pub fn temp_bucket(qtype: usize, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", super::QType::NAMES[qtype])
}

/// What the sequence builder and the calibration need from a checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionConfig {
    /// Total sequence budget.
    pub max_len: usize,
    /// Budget for the question text and every option together.
    pub head_max_len: usize,
    pub temperatures: Temperatures,
    /// Per-language overrides, keyed by the lower-cased primary subtag
    /// (`"de"` for `de-AT`). Empty for the shipped checkpoints; an operator
    /// may fit them per language.
    pub lang_temperatures: HashMap<String, Temperatures>,
}

impl DecisionConfig {
    /// Parse a checkpoint's `rl_agent_config.json`.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v = Json::parse(text.as_bytes()).map_err(|e| format!("rl_agent_config.json: {e}"))?;
        let usize_or = |key: &str, default: usize| match v.get(key) {
            Some(Json::Int(d)) => d.parse().unwrap_or(default),
            _ => default,
        };
        Ok(Self {
            max_len: usize_or("max_len", DEFAULT_MAX_LEN),
            head_max_len: usize_or("head_max_len", DEFAULT_HEAD_MAX_LEN),
            temperatures: Temperatures::from_json(
                v.get("temperature"),
                v.get("temperature_by_options"),
                None,
            )?,
            lang_temperatures: HashMap::new(),
        })
    }

    /// Add per-language temperatures, the shape the reference takes them in:
    /// `{"de": {"temperature": [..3], "temperature_by_options": {..}}}`. A
    /// language that omits `temperature` inherits the checkpoint's raw
    /// per-type values.
    pub fn with_lang_temperatures(mut self, spec: &Json) -> Result<Self, String> {
        let Json::Object(m) = spec else {
            return Err("lang_temperatures must be an object".into());
        };
        for (lang, cfg) in m {
            let t = Temperatures::from_json(
                cfg.get("temperature"),
                cfg.get("temperature_by_options"),
                Some(self.temperatures.by_type),
            )
            .map_err(|e| format!("language override {lang:?}: {e}"))?;
            let primary = lang.split('-').next().unwrap_or(lang).to_lowercase();
            self.lang_temperatures.insert(primary, t);
        }
        Ok(self)
    }

    /// The temperatures to apply for a request detected (or declared) as
    /// `lang`, which may carry a region (`de-AT`).
    pub fn temperatures_for(&self, lang: Option<&str>) -> &Temperatures {
        lang.and_then(|l| {
            let primary = l.split('-').next().unwrap_or(l).to_lowercase();
            self.lang_temperatures.get(&primary)
        })
        .unwrap_or(&self.temperatures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENGLISH: &str = r#"{"encoder": "answerdotai/ModernBERT-large", "head_layers": 2,
        "max_len": 512, "head_max_len": 192,
        "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
        "temperature_by_options": {"choice:3-5": 1.7601518630981445, "choice:11+": 0.10058280825614929}}"#;

    #[test]
    fn shipped_choice_11_bucket_is_clamped() {
        let c = DecisionConfig::from_json(ENGLISH).unwrap();
        assert_eq!(c.temperatures.for_question(0, 12), TEMP_MIN);
        assert_eq!(c.temperatures.for_question(0, 4), 1.7601518630981445);
        // No bucket: the per-type value.
        assert_eq!(c.temperatures.for_question(1, 7), 1.2514300346374512);
    }

    #[test]
    fn buckets_split_where_the_reference_splits() {
        let b: Vec<_> = [1, 2, 3, 5, 6, 10, 11]
            .iter()
            .map(|&k| temp_bucket(2, k))
            .collect();
        assert_eq!(
            b,
            [
                "noul:2",
                "noul:2",
                "noul:3-5",
                "noul:3-5",
                "noul:6-10",
                "noul:6-10",
                "noul:11+"
            ]
        );
    }

    #[test]
    fn language_overrides_match_on_primary_subtag() {
        let c = DecisionConfig::from_json(ENGLISH)
            .unwrap()
            .with_lang_temperatures(
                &Json::parse(br#"{"DE": {"temperature": [1.0, 1.4, 9.0]}}"#).unwrap(),
            )
            .unwrap();
        assert_eq!(
            c.temperatures_for(Some("de-AT")).by_type,
            [1.0, 1.4, TEMP_MAX]
        );
        assert_eq!(c.temperatures_for(Some("fr")), &c.temperatures);
        assert_eq!(c.temperatures_for(None), &c.temperatures);
    }

    #[test]
    fn bad_temperatures_are_refused_or_neutralised() {
        assert!(DecisionConfig::from_json(r#"{"temperature": [1.0]}"#).is_err());
        let c = DecisionConfig::from_json(r#"{"temperature": [" 2.5 ", "x", null]}"#).unwrap();
        assert_eq!(c.temperatures.by_type, [2.5, 1.0, 1.0]);
        assert_eq!(
            (c.max_len, c.head_max_len),
            (DEFAULT_MAX_LEN, DEFAULT_HEAD_MAX_LEN)
        );
    }
}
