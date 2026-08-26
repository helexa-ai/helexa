//! Which reasoning-effort rungs a model actually supports, discovered
//! from the model itself (#290).
//!
//! ## Why this is discovered rather than configured
//!
//! `Qwen/Qwen3.8-27B` accepts exactly three efforts and hard-fails on
//! anything else:
//!
//! ```jinja
//! {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
//! {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
//!     {{- raise_exception('Unexpected reasoning effort ...') }}
//! ```
//!
//! We previously advertised an invented ladder — `minimal`, `low`,
//! `medium`, `high` — of which two values the template rejects outright
//! and which omits `xhigh`, the model's own default and strongest rung.
//! Clients read that advertisement: pi-ai's `getSupportedThinkingLevels`
//! offers a level only when the model declares it, so publishing a
//! ladder that does not exist made the model's default *unreachable*.
//!
//! A hardcoded per-model table would drift the moment a model ships a
//! new template, which is exactly the class of defect #280 fixed by
//! fetching the template instead of assuming it. So the rungs come from
//! the artifact.
//!
//! ## How the probe works
//!
//! `raise_exception` is already wired to a render error, so the template
//! answers the question directly: render it once per candidate effort
//! and see which ones survive.
//!
//! Two ambiguities the probe has to resolve, or it would report
//! confident nonsense:
//!
//! 1. **A template that ignores `reasoning_effort` entirely** renders
//!    successfully for every candidate. Supporting "all six" would be a
//!    lie. So a candidate only counts when the rendered prompt actually
//!    *differs* across efforts — if every render is identical the model
//!    has no effort control and we advertise none.
//! 2. **The default** is whichever supported candidate renders
//!    identically to a render with no effort supplied. Qwen3.8 defaults
//!    to `xhigh`; without detecting that, a caller who says nothing gets
//!    the strongest rung while we believe it got nothing.

use anyhow::Result;
use cortex_core::openai::{ChatMessage, MessageContent};
use serde_json::{Value, json};

/// Effort names in ascending order of thinking, as clients spell them.
///
/// Deliberately a superset of what any one model supports: this is the
/// *candidate* set the probe tries, and the conventional vocabulary
/// OpenAI-compatible clients use (pi-ai's `EXTENDED_THINKING_LEVELS`
/// minus `off`, which is not an effort but a separate switch handled by
/// `enable_thinking`).
pub const CANDIDATE_EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// What one model's template accepts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SupportedEfforts {
    /// Accepted efforts, ascending, as the template spells them.
    pub levels: Vec<String>,
    /// The effort applied when the caller names none — the template's
    /// own `|default(...)`. `None` when it cannot be determined.
    pub default: Option<String>,
}

impl SupportedEfforts {
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn supports(&self, effort: &str) -> bool {
        self.levels.iter().any(|l| l == effort)
    }

    /// Map a caller's effort onto the nearest rung this model has.
    ///
    /// Exact match wins. Otherwise walk **down** from the requested
    /// level first, then up — a caller asking for more thinking than the
    /// model offers should get the most it has, and a caller asking for
    /// less should not be silently upgraded into a slower, costlier
    /// response than they asked for.
    ///
    /// Descending-first is the opposite of pi-ai's `clampThinkingLevel`,
    /// which searches upward first. That is deliberate: pi is choosing
    /// among levels it believes exist, whereas we are the ones who know
    /// what exists, and over-delivering thinking is the failure that
    /// costs the caller money and latency.
    ///
    /// `None` when the model has no rungs at all.
    pub fn nearest(&self, requested: &str) -> Option<&str> {
        if self.levels.is_empty() {
            return None;
        }
        if let Some(exact) = self.levels.iter().find(|l| *l == requested) {
            return Some(exact.as_str());
        }
        let idx = CANDIDATE_EFFORTS.iter().position(|c| *c == requested)?;
        for c in CANDIDATE_EFFORTS[..idx].iter().rev() {
            if let Some(hit) = self.levels.iter().find(|l| l.as_str() == *c) {
                return Some(hit.as_str());
            }
        }
        for c in &CANDIDATE_EFFORTS[idx + 1..] {
            if let Some(hit) = self.levels.iter().find(|l| l.as_str() == *c) {
                return Some(hit.as_str());
            }
        }
        None
    }
}

/// Render a probe prompt at `effort`, or `None` if the template rejects
/// it.
fn probe_render(template: &str, effort: Option<&str>) -> Option<String> {
    let messages = [ChatMessage {
        role: "user".into(),
        content: MessageContent::Text("probe".into()),
        extra: Value::Null,
    }];
    let kwargs = match effort {
        Some(e) => json!({ "reasoning_effort": e }),
        None => Value::Null,
    };
    super::chat_template::render_chat_template(template, &messages, &Value::Null, &kwargs).ok()
}

/// Discover which efforts `template` accepts (#290).
///
/// Errors are not propagated: a template that cannot render even the
/// baseline probe yields no rungs, and the caller behaves as it did
/// before this existed.
pub fn probe(template: &str, model_id: &str) -> Result<SupportedEfforts> {
    let Some(baseline) = probe_render(template, None) else {
        tracing::debug!(
            model = %model_id,
            "reasoning effort: template does not render a baseline probe; advertising no rungs"
        );
        return Ok(SupportedEfforts::default());
    };

    let mut rendered: Vec<(&str, String)> = Vec::new();
    for candidate in CANDIDATE_EFFORTS {
        if let Some(out) = probe_render(template, Some(candidate)) {
            rendered.push((candidate, out));
        }
    }

    // A template that never reads `reasoning_effort` renders the same
    // bytes whatever we pass. Advertising every candidate in that case
    // would be the invented-ladder mistake again, in a new place.
    let effort_aware = rendered.windows(2).any(|w| w[0].1 != w[1].1)
        || rendered.iter().any(|(_, out)| *out != baseline);
    if !effort_aware {
        tracing::debug!(
            model = %model_id,
            "reasoning effort: template ignores reasoning_effort; advertising no rungs"
        );
        return Ok(SupportedEfforts::default());
    }

    let default = rendered
        .iter()
        .find(|(_, out)| *out == baseline)
        .map(|(name, _)| (*name).to_string());
    let levels: Vec<String> = rendered
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();

    tracing::info!(
        model = %model_id,
        levels = %levels.join(","),
        default = ?default,
        "reasoning effort: rungs discovered from the model's chat template (#290)"
    );
    Ok(SupportedEfforts { levels, default })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe against the **real** deployed Qwen3.8-27B template,
    /// not a hand-written approximation of it. A synthetic template
    /// proves the probe logic; only the shipped artifact proves the
    /// answer, and the artifact is what #290 is about.
    #[test]
    fn the_real_qwen38_template_yields_low_medium_xhigh_defaulting_to_xhigh() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/qwen3.8-27b.chat_template.jinja"
        );
        let Ok(template) = std::fs::read_to_string(path) else {
            // The fixture is checked in; if it is ever removed this test
            // should fail loudly rather than silently pass.
            panic!("missing fixture {path}");
        };
        let s = probe(&template, "Qwen/Qwen3.8-27B").expect("probe");
        assert_eq!(
            s.levels,
            vec!["low", "medium", "xhigh"],
            "the shipped template accepts exactly these"
        );
        assert_eq!(s.default.as_deref(), Some("xhigh"));
        // The two rungs we used to advertise are rejected outright.
        assert!(!s.supports("minimal"));
        assert!(!s.supports("high"));
    }

    /// The shape Qwen3.8-27B actually ships: three rungs, `xhigh`
    /// default, `raise_exception` on anything else.
    const QWEN38: &str = r#"
{%- set resolved = reasoning_effort|default('xhigh') %}
{%- if resolved not in ('xhigh', 'medium', 'low') %}
{{- raise_exception('Unexpected reasoning effort ' ~ resolved) }}
{%- endif %}
{%- if resolved == 'xhigh' %}EFFORT-XHIGH{% elif resolved == 'low' %}EFFORT-LOW{% else %}EFFORT-MEDIUM{% endif %}
{%- for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{%- endfor %}
"#;

    /// A template with no effort control at all.
    const PLAIN: &str = r#"
{%- for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{%- endfor %}
"#;

    #[test]
    fn discovers_exactly_the_rungs_the_template_accepts() {
        let s = probe(QWEN38, "test").expect("probe");
        assert_eq!(s.levels, vec!["low", "medium", "xhigh"]);
        assert!(!s.supports("minimal"), "template raises on minimal");
        assert!(!s.supports("high"), "template raises on high");
    }

    /// Without this, a caller who names no effort silently receives the
    /// strongest rung while we believe it received nothing — which is
    /// how the xhigh instruction ended up on every request in #290.
    #[test]
    fn detects_the_templates_own_default() {
        let s = probe(QWEN38, "test").expect("probe");
        assert_eq!(s.default.as_deref(), Some("xhigh"));
    }

    /// A template that ignores the kwarg must advertise nothing rather
    /// than appearing to support all six.
    #[test]
    fn a_template_without_effort_control_advertises_no_rungs() {
        let s = probe(PLAIN, "test").expect("probe");
        assert!(s.is_empty(), "got {:?}", s.levels);
        assert_eq!(s.nearest("medium"), None);
    }

    #[test]
    fn an_exact_match_is_returned_unchanged() {
        let s = probe(QWEN38, "test").expect("probe");
        assert_eq!(s.nearest("medium"), Some("medium"));
        assert_eq!(s.nearest("xhigh"), Some("xhigh"));
    }

    /// `high` does not exist on this model. Descending first means it
    /// resolves to `medium`, not `xhigh`: a caller is never given more
    /// thinking (slower, dearer) than it asked for.
    #[test]
    fn an_unsupported_level_resolves_downward_not_upward() {
        let s = probe(QWEN38, "test").expect("probe");
        assert_eq!(s.nearest("high"), Some("medium"));
    }

    /// `minimal` is below every rung the model has, so there is nothing
    /// to descend to and the nearest is the weakest available.
    #[test]
    fn a_level_below_every_rung_resolves_up_to_the_weakest() {
        let s = probe(QWEN38, "test").expect("probe");
        assert_eq!(s.nearest("minimal"), Some("low"));
    }

    #[test]
    fn an_unknown_level_name_resolves_to_nothing_rather_than_a_guess() {
        let s = probe(QWEN38, "test").expect("probe");
        assert_eq!(s.nearest("turbo"), None);
    }
}
