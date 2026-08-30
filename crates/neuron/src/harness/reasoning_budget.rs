//! Bounding how long a model may think before it must answer (#223).
//!
//! A reasoning model with no stopping criterion will fill whatever
//! output budget it is given. Measured on Qwen3.8-27B: given 16,384
//! tokens it used 16,384; given 32,768 it used 32,768; both times it was
//! still opening new topics when the cap bit, and both times the caller
//! received no answer at all.
//!
//! The fix is not to stop the model thinking — the run that did produce
//! files designed for 27,157 tokens first, and that design was good. It
//! is to bound the thinking so the rest of the output budget survives
//! for the answer.
//!
//! Two halves live here:
//!
//! - [`ReasoningBudget`] — how many reasoning tokens this request may
//!   spend, resolved from whichever control the caller's wire format
//!   gave them.
//! - [`ReasoningGovernor`] — the per-request state that enforces it, by
//!   substituting the model's own `</think>` token when the budget runs
//!   out so generation transitions to answering instead of being cut off.

use serde::{Deserialize, Serialize};

/// Per-model mapping from an OpenAI effort level to a token budget.
///
/// OpenAI-shaped clients cannot send a token count — pi-ai's vocabulary
/// is the ladder `off | minimal | low | medium | high | xhigh | max` and
/// nothing else — so the numbers have to live on the server. Operator
/// configurable, because the right values depend on the model and the
/// hardware; a derived default would be a guess baked into the binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningBudgetLadder {
    pub minimal: usize,
    pub low: usize,
    pub medium: usize,
    pub high: usize,
    /// `xhigh` and `max` are only offered by a client when the model
    /// declares them; absent here means "not offered", and a request
    /// naming one falls back to `high` rather than being rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xhigh: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<usize>,
    /// Tokens of the caller's **own** output budget held back for the
    /// answer, when the caller names no explicit `reasoning.max_tokens`.
    ///
    /// This is what an OpenAI-shaped client gets instead of the rungs
    /// above, and it is deliberately not derived from effort. Effort is
    /// a prompt instruction the model is trained to follow (#290);
    /// making it *also* set a guillotine is what told Qwen3.8-27B to
    /// reason at `xhigh` and then cut it off at `medium`'s 12,288.
    ///
    /// A reserve has no such double meaning. The model thinks as long
    /// as it likes within the budget the caller declared, minus enough
    /// to answer — so the caller always gets an answer, and the size of
    /// the think block stays the model's decision rather than ours.
    ///
    /// Set to `0` to disable, restoring fully unbounded reasoning.
    #[serde(default = "default_answer_reserve")]
    pub answer_reserve_tokens: usize,
}

/// Enough for a substantive answer with tool calls, not so much that it
/// meaningfully shortens a think block: at the 32,768 budget pi sends,
/// this leaves ~28.7 k for reasoning — above the ~27.2 k a full agentic
/// design turn was measured to use.
fn default_answer_reserve() -> usize {
    4_096
}

impl Default for ReasoningBudgetLadder {
    /// Rungs sized against observed behaviour rather than round numbers:
    /// a long agentic design turn on this class of model runs to ~27k
    /// tokens, so `high` sits above that (bounded but rarely binding),
    /// `medium` at roughly half (enough to plan a whole implementation),
    /// and `low` at the scale of a focused single-file change.
    fn default() -> Self {
        Self {
            minimal: 1_024,
            low: 4_096,
            medium: 12_288,
            high: 32_768,
            xhigh: None,
            max: None,
            answer_reserve_tokens: default_answer_reserve(),
        }
    }
}

impl ReasoningBudgetLadder {
    /// Resolve an effort level to a token budget.
    ///
    /// `none`/`off` are handled before this — they disable thinking
    /// outright via the chat template — so they have no budget here.
    /// An unrecognised level resolves to `None` (unbounded) rather than
    /// to a guess: silently substituting a budget nobody asked for is
    /// the failure mode this whole area has been full of.
    pub fn for_effort(&self, effort: &str) -> Option<usize> {
        match effort.trim().to_ascii_lowercase().as_str() {
            "minimal" => Some(self.minimal),
            "low" => Some(self.low),
            "medium" => Some(self.medium),
            "high" => Some(self.high),
            "xhigh" => Some(self.xhigh.unwrap_or(self.high)),
            "max" => Some(self.max.or(self.xhigh).unwrap_or(self.high)),
            _ => None,
        }
    }

    /// The mapping as a client should see it on `/v1/models` — the rungs
    /// this deployment actually offers, so a caller picking `low` knows
    /// what it bought instead of guessing (#274).
    pub fn advertised(&self) -> Vec<(String, usize)> {
        let mut out = vec![
            ("minimal".to_string(), self.minimal),
            ("low".to_string(), self.low),
            ("medium".to_string(), self.medium),
            ("high".to_string(), self.high),
        ];
        if let Some(v) = self.xhigh {
            out.push(("xhigh".to_string(), v));
        }
        if let Some(v) = self.max {
            out.push(("max".to_string(), v));
        }
        out
    }
}

/// Enforces a resolved budget over one request's token stream.
///
/// Fed every sampled token; returns the token that should actually be
/// emitted. While the model is inside a think block and over budget, it
/// returns the close marker instead, exactly once — after which the
/// model is answering and the governor is inert.
#[derive(Debug, Clone)]
pub struct ReasoningGovernor {
    limit: usize,
    open_id: u32,
    close_id: u32,
    inside: bool,
    used: usize,
    forced: bool,
}

impl ReasoningGovernor {
    /// `None` when the request named no budget, or the model has no
    /// reasoning markers to enforce one with — in both cases generation
    /// proceeds exactly as it did before this existed.
    pub fn new(
        limit: Option<usize>,
        markers: Option<(u32, u32)>,
        prompt_opened_reasoning: bool,
    ) -> Option<Self> {
        let limit = limit?;
        let (open_id, close_id) = markers?;
        Some(Self {
            limit,
            open_id,
            close_id,
            inside: prompt_opened_reasoning,
            used: 0,
            forced: false,
        })
    }

    /// Whether the forced close has already been spent.
    pub fn forced(&self) -> bool {
        self.forced
    }

    /// Reasoning tokens counted so far.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Govern one sampled token, returning the token to emit.
    pub fn govern(&mut self, sampled: u32) -> u32 {
        if sampled == self.open_id {
            self.inside = true;
            return sampled;
        }
        if sampled == self.close_id {
            self.inside = false;
            return sampled;
        }
        if !self.inside {
            return sampled;
        }
        if self.used >= self.limit {
            // Budget spent: hand the model its own close marker so it
            // transitions to answering. Cutting the stream here instead
            // would reproduce the very failure this exists to prevent.
            self.inside = false;
            self.forced = true;
            return self.close_id;
        }
        self.used += 1;
        sampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: u32 = 100;
    const CLOSE: u32 = 101;

    fn governor(limit: usize, opened: bool) -> ReasoningGovernor {
        ReasoningGovernor::new(Some(limit), Some((OPEN, CLOSE)), opened).expect("governor")
    }

    #[test]
    fn a_request_with_no_budget_is_ungoverned() {
        assert!(ReasoningGovernor::new(None, Some((OPEN, CLOSE)), true).is_none());
    }

    /// A model with no detected markers cannot be governed — better
    /// unbounded than truncated at an arbitrary token.
    #[test]
    fn a_model_without_markers_is_ungoverned() {
        assert!(ReasoningGovernor::new(Some(10), None, true).is_none());
    }

    /// The case that motivated this: the chat template left the think
    /// block open, so the model is reasoning from the first token.
    #[test]
    fn a_prompt_opened_block_is_bounded_and_force_closed() {
        let mut g = governor(3, true);
        assert_eq!(g.govern(7), 7);
        assert_eq!(g.govern(8), 8);
        assert_eq!(g.govern(9), 9);
        assert_eq!(g.govern(10), CLOSE, "the fourth token must close the block");
        assert!(g.forced());
        assert_eq!(g.used(), 3);
        // Answering now — the governor is inert.
        assert_eq!(g.govern(11), 11);
        assert_eq!(g.govern(12), 12);
    }

    /// Content before the block opens is never counted against the
    /// reasoning budget.
    #[test]
    fn only_tokens_inside_the_block_are_counted() {
        let mut g = governor(2, false);
        assert_eq!(g.govern(5), 5);
        assert_eq!(g.govern(6), 6);
        assert_eq!(g.used(), 0);
        assert_eq!(g.govern(OPEN), OPEN);
        assert_eq!(g.govern(7), 7);
        assert_eq!(g.govern(8), 8);
        assert_eq!(g.govern(9), CLOSE);
        assert_eq!(g.used(), 2);
    }

    /// A model that finishes thinking on its own is untouched — the
    /// budget is a ceiling, not a target.
    #[test]
    fn a_model_that_stops_on_its_own_is_not_forced() {
        let mut g = governor(100, true);
        for t in 20..30 {
            assert_eq!(g.govern(t), t);
        }
        assert_eq!(g.govern(CLOSE), CLOSE);
        assert!(!g.forced());
        for t in 40..50 {
            assert_eq!(g.govern(t), t);
        }
        assert!(!g.forced(), "content after the block must not be governed");
    }

    #[test]
    fn the_ladder_resolves_the_levels_clients_can_send() {
        let l = ReasoningBudgetLadder::default();
        assert_eq!(l.for_effort("minimal"), Some(1_024));
        assert_eq!(l.for_effort("LOW"), Some(4_096));
        assert_eq!(l.for_effort(" medium "), Some(12_288));
        assert_eq!(l.for_effort("high"), Some(32_768));
        // Not offered by default → falls back to high rather than
        // rejecting a level the client was told it could send.
        assert_eq!(l.for_effort("xhigh"), Some(32_768));
        assert_eq!(l.for_effort("max"), Some(32_768));
        // Unknown → unbounded, never a substituted guess.
        assert_eq!(l.for_effort("enthusiastic"), None);
        assert_eq!(l.for_effort("none"), None);
    }

    #[test]
    fn the_advertised_ladder_lists_only_what_is_offered() {
        let l = ReasoningBudgetLadder::default();
        let names: Vec<_> = l.advertised().into_iter().map(|(k, _)| k).collect();
        assert_eq!(names, ["minimal", "low", "medium", "high"]);
        let l2 = ReasoningBudgetLadder {
            xhigh: Some(49_152),
            ..ReasoningBudgetLadder::default()
        };
        assert!(
            l2.advertised()
                .iter()
                .any(|(k, v)| k == "xhigh" && *v == 49_152)
        );
    }
}

/// Whether the longest reasoning rung should be offered on a node
/// running `max_in_flight` concurrent slots.
///
/// Operator-reported (2026-08-29): `xhigh` "hits issues" when this host
/// runs eight slots. The mechanism is not yet isolated, so this is a
/// precaution rather than a diagnosis, and it is written down as one —
/// naming it a fix would stop anyone looking.
///
/// The plausible account: `xhigh` instructs the model to deliberate
/// exhaustively, so a turn can spend tens of thousands of tokens
/// reasoning. KV is a shared pool (#291), and a slot holds its
/// reservation for the request's whole life, so several concurrent
/// `xhigh` turns can hold most of the pool for minutes while shorter
/// requests queue on bytes. That is consistent with "issues" but is not
/// evidence of them; #305's turn is a candidate to instrument.
///
/// **2026-08-30: that account is now contradicted, and the constant
/// below did not hold.** A controlled A/B/C on this host reproduced the
/// defect at `max_in_flight = 2` with `in_flight == 1` — the very
/// configuration this function calls safe — while the `medium` arm at
/// `max_in_flight = 8` was clean. Concurrency does not predict it.
/// Neither does the GatedDeltaNet bf16 state round-trip (#284):
/// `NEURON_GDN_STATE_F32=1` changed neither the defect rate nor its
/// class on a run matched to within 0.2% on the opening generation.
/// What both `xhigh` arms produced was a read against the wrong
/// receiver with the correct one in scope on an adjacent line
/// (`tw.damage` for `def.damage`; `1 - slowPct` for `slowPct / 100`) —
/// the same shape as #305's `<parameter=` for `<function=`. The
/// mechanism remains unisolated; this is still a precaution, and the
/// evidence now says it is very likely guarding the wrong variable.
/// Do not read the number below as a diagnosis.
///
/// Availability is advertised, never enforced. The request path does not
/// reject an unavailable rung — a caller that insists still gets it. The
/// point is to keep a UI from offering an option that will serve someone
/// badly, not to add a refusal path to every existing API client.
pub fn long_rung_available(max_in_flight: usize) -> bool {
    max_in_flight <= LONG_RUNG_MAX_IN_FLIGHT
}

/// The slot count above which the longest rung stops being offered.
///
/// Two, because that is the concurrency at which a single deep session
/// still has the pool substantially to itself. A named constant rather
/// than a literal at the call site so the number is findable when the
/// mechanism above is finally isolated and it turns out to be wrong.
pub const LONG_RUNG_MAX_IN_FLIGHT: usize = 2;

/// The rung a deployment considers "longest", if it offers one.
///
/// Taken as the last entry rather than by matching on `"xhigh"`: the
/// rungs come from the model's own template (#290), so the name of the
/// deepest one is the model's to choose and a future model may spell it
/// differently.
pub fn longest_rung(levels: &[String]) -> Option<&String> {
    levels.last()
}

#[cfg(test)]
mod availability_tests {
    use super::*;

    #[test]
    fn the_long_rung_is_offered_only_at_low_concurrency() {
        assert!(long_rung_available(1));
        assert!(long_rung_available(2));
        assert!(!long_rung_available(3));
        assert!(!long_rung_available(8));
    }

    #[test]
    fn the_longest_rung_is_the_last_one_the_template_offers() {
        // Not a match on "xhigh": the ladder is the model's, and the
        // deepest rung is whatever it puts last.
        let levels = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        assert_eq!(longest_rung(&levels).map(String::as_str), Some("xhigh"));

        let other = vec!["brief".to_string(), "exhaustive".to_string()];
        assert_eq!(longest_rung(&other).map(String::as_str), Some("exhaustive"));
    }

    #[test]
    fn a_model_with_no_rungs_has_no_longest_one() {
        assert_eq!(longest_rung(&[]), None);
    }

    /// A single-rung model must not have its only option withdrawn: the
    /// rule exists to steer a choice, and there is no choice to steer.
    #[test]
    fn a_single_rung_is_never_withdrawn() {
        let levels = vec!["only".to_string()];
        assert_eq!(longest_rung(&levels).map(String::as_str), Some("only"));
        // The caller is responsible for not withdrawing a lone rung;
        // pinned here so the intent survives a refactor of that caller.
        assert_eq!(levels.len(), 1);
    }
}
