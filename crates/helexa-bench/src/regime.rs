//! Measurement-regime boundaries — builds where a number changed
//! meaning rather than value.
//!
//! A trend line silently spans them, and every reader re-derives the
//! same thing: that a cliff was the instrument, not the engine. Three
//! have already cost real investigation time:
//!
//! - beast's `concurrency:8` fell 168.9 → 104.6 tok/s with `ttft_p95`
//!   rising 0.53 → 11.77 s, and it read as a 38% throughput regression
//!   until the one commit in the window turned out to be an admission
//!   *policy* change that bench, being anonymous, was subject to (#288).
//! - quadbrat's `capability:debug-reason` TTFT fell 88.8 → 0.1 s, which
//!   read as a latency win and was a redefinition of when the clock
//!   stops.
//! - the same redefinition put decode rates at ~250 tok/s on a 3060
//!   beforehand, by dividing reasoning-inclusive token counts by a
//!   visible-content-only window.
//!
//! So the boundaries are declared, not remembered. Declared here rather
//! than in the frontend because `report` renders the same series into
//! `benchmarks.md` and needs the same caveats.
//!
//! ## What belongs here
//!
//! Only changes to **what a metric means**. A change that made the
//! server genuinely faster, slower, or differently behaved does *not*
//! belong — that is the signal these lines exist to keep readable. #280
//! (fetch the model's own chat template) moved quadbrat's
//! `completion_tokens` from 4096 to 1053 by ending a truncation loop;
//! that is a real behavioural change and stays unmarked.
//!
//! The identified/anonymous split is deliberately absent too: it is
//! recorded per-row in `runs.principal`, so the UI derives it from the
//! data instead of trusting a constant to stay true.

use serde::Serialize;

/// A build at which one or more metrics changed meaning.
#[derive(Debug, Clone, Serialize)]
pub struct MeasurementRegime {
    /// Short SHA of the first build measured under the new semantics.
    /// Matched against `SeriesPoint::git_sha`, so a boundary whose build
    /// never appears in a selection simply does not draw.
    pub first_sha: &'static str,
    /// Terse label for the chart rule.
    pub label: &'static str,
    /// What changed, and why the step is not a performance result.
    pub detail: &'static str,
    /// Series keys the change affects, as the API/UI name them. A
    /// boundary only draws on panels carrying one of these — marking
    /// every panel would train people to ignore the rules.
    pub affects: &'static [&'static str],
}

/// The declared boundaries, oldest first.
pub const REGIMES: &[MeasurementRegime] = &[
    MeasurementRegime {
        first_sha: "92dddc0",
        label: "#262 anonymous callers yield",
        detail: "Anonymous traffic became servable only from capacity left over \
                 once identified traffic is satisfied — capped below max_in_flight \
                 and parked at the class gate. An anonymous bench therefore \
                 characterises the yield policy rather than serving capacity. \
                 Steps here are admission policy, not the engine (#288).",
        affects: &["decode", "ttft", "ttftP95", "queueWait", "rejected"],
    },
    MeasurementRegime {
        first_sha: "0fe7aa3",
        label: "#117 reasoning counts as liveness",
        detail: "TTFT previously waited for the first *visible content* chunk, so \
                 on a thinking model it measured the whole reasoning span (88.8 s \
                 on quadbrat). It now stops at the first delta of any kind. The \
                 same change fixed decode rates computed by dividing \
                 reasoning-inclusive token counts by a content-only window, which \
                 had produced ~250 tok/s on a 3060.",
        affects: &["decode", "ttft"],
    },
];

/// Boundaries affecting `metric`, in declaration order.
pub fn regimes_for(metric: &str) -> Vec<&'static MeasurementRegime> {
    REGIMES
        .iter()
        .filter(|r| r.affects.contains(&metric))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A boundary that names no metric would draw nowhere and mislead by
    /// omission; one naming an unknown key would draw nowhere and look
    /// fine. Both are typos this catches.
    #[test]
    fn every_regime_affects_a_known_series_key() {
        // The keys `Trends.tsx` maps `SeriesPoint` onto.
        const KNOWN: &[&str] = &[
            "decode",
            "ttft",
            "ttftP95",
            "queueWait",
            "rejected",
            "prefillTps",
            "reasoning",
            "cached",
            "completion",
            "tpot",
        ];
        for r in REGIMES {
            assert!(
                !r.affects.is_empty(),
                "{} affects nothing, so it would never draw",
                r.first_sha
            );
            for key in r.affects {
                assert!(
                    KNOWN.contains(key),
                    "{} names unknown series key {key:?}",
                    r.first_sha
                );
            }
        }
    }

    #[test]
    fn regimes_for_selects_by_metric() {
        assert!(regimes_for("ttft").iter().any(|r| r.first_sha == "0fe7aa3"));
        assert!(
            regimes_for("completion").is_empty(),
            "#280 changed behaviour, not meaning — it must not be marked"
        );
    }

    /// Short SHAs are matched against `SeriesPoint::git_sha`, which the
    /// store records at the width `/version` reports.
    #[test]
    fn declared_shas_are_short_form() {
        for r in REGIMES {
            assert_eq!(
                r.first_sha.len(),
                7,
                "{} is not a 7-char short sha; it would never match a series point",
                r.first_sha
            );
        }
    }
}
