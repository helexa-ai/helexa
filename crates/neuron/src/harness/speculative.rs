//! Speculative decoding (#25) — a small same-family drafter proposes
//! tokens that the large target verifies in one forward pass.
//!
//! batch-1 decode is exactly the regime where speculation wins, and
//! the regime helexa lives in. A cheap drafter (Qwen3.5-0.8B) proposes
//! K tokens for the 27B target; the target verifies all K in a single
//! forward and the longest agreeing prefix is committed for free.
//!
//! ## What lives here
//!
//! This module is the **acceptance core** plus config — the pure,
//! state-free heart of the algorithm, where off-by-ones live. The
//! draft/verify loop and the GDN-state rollback (which reuses #11's
//! snapshot/restore — see the issue) wire this into the generation
//! path in later phases.
//!
//! ## Greedy acceptance
//!
//! Per round, with the target's greedy token already known at the
//! committed boundary and at each speculative position, the longest
//! drafter-matching prefix is accepted and one **bonus** token is
//! always committed on top (the target's own token at the first
//! mismatch, or a free extra token when every draft matched). So a
//! round commits between 1 and K+1 tokens — never zero, which
//! guarantees forward progress even when the drafter is useless.
//!
//! Greedy (argmax) acceptance is exact for temperature-0 sampling —
//! the fleet's probe + #22 bench regime. Stochastic acceptance that
//! preserves the target distribution at temperature > 0 is a later
//! phase.

use serde::{Deserialize, Serialize};

/// Per-target speculative-decoding settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Drafter model id. MUST share the target's tokenizer/vocabulary
    /// (e.g. `Qwen/Qwen3.5-0.8B` for a `Qwen/Qwen3.6-27B` target — both
    /// `qwen3_5`, byte-identical tokenizer). `None` disables
    /// speculation for the target.
    #[serde(default)]
    pub drafter: Option<String>,

    /// Tokens the drafter proposes per round (K). Larger K wins more
    /// when acceptance is high and loses more (wasted target compute on
    /// rejected tail) when it's low. 4 is a conservative default.
    #[serde(default = "default_draft_len")]
    pub draft_len: usize,
}

fn default_draft_len() -> usize {
    4
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            drafter: None,
            draft_len: default_draft_len(),
        }
    }
}

impl SpeculativeConfig {
    /// Speculation is active only when a drafter is named and K ≥ 1.
    pub fn is_enabled(&self) -> bool {
        self.drafter.is_some() && self.draft_len >= 1
    }
}

/// Outcome of verifying one speculative round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecAccept {
    /// Number of drafter-proposed tokens accepted (the matching
    /// prefix length), in `0..=draft.len()`.
    pub accepted: usize,
    /// The target's own next token, always committed after the
    /// accepted prefix — the correction at the first mismatch, or a
    /// free extra token when the whole draft matched.
    pub bonus: u32,
}

impl SpecAccept {
    /// The tokens this round commits: the accepted draft prefix
    /// followed by the bonus. Always non-empty (≥ the bonus).
    pub fn committed(&self, draft: &[u32]) -> Vec<u32> {
        let mut out = draft[..self.accepted].to_vec();
        out.push(self.bonus);
        out
    }
}

/// Greedy speculative acceptance.
///
/// - `draft`: the K tokens the drafter proposed this round.
/// - `target_greedy`: the target's greedy (argmax) token at each of
///   the K+1 positions — `target_greedy[j]` is what the target would
///   emit given the committed prefix plus `draft[..j]`. So
///   `target_greedy[0]` is checked against `draft[0]`, and
///   `target_greedy[K]` is the free bonus available when the whole
///   draft is accepted.
///
/// Accepts the longest prefix where the target agrees with the drafter
/// and returns the bonus token at the boundary. `target_greedy` must
/// have exactly `draft.len() + 1` entries.
pub fn greedy_accept(draft: &[u32], target_greedy: &[u32]) -> SpecAccept {
    debug_assert_eq!(
        target_greedy.len(),
        draft.len() + 1,
        "target_greedy must carry one distribution per draft position plus the bonus"
    );
    let mut accepted = 0;
    while accepted < draft.len() && target_greedy[accepted] == draft[accepted] {
        accepted += 1;
    }
    // `accepted` is in 0..=draft.len(), and target_greedy has
    // draft.len()+1 entries, so this index is always in bounds: it's
    // the target's correction at the first mismatch, or the free token
    // past the end when everything matched.
    SpecAccept {
        accepted,
        bonus: target_greedy[accepted],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_accept_commits_k_plus_one() {
        // Target agrees with every draft; the K+1-th greedy token is a
        // free bonus.
        let draft = [10, 11, 12, 13];
        let target = [10, 11, 12, 13, 99];
        let a = greedy_accept(&draft, &target);
        assert_eq!(
            a,
            SpecAccept {
                accepted: 4,
                bonus: 99
            }
        );
        assert_eq!(a.committed(&draft), vec![10, 11, 12, 13, 99]);
    }

    #[test]
    fn partial_accept_takes_prefix_plus_correction() {
        // Drafter right for two tokens, wrong on the third; commit the
        // two + the target's correction, drop the rest of the draft.
        let draft = [10, 11, 12, 13];
        let target = [10, 11, 7, 13, 99];
        let a = greedy_accept(&draft, &target);
        assert_eq!(
            a,
            SpecAccept {
                accepted: 2,
                bonus: 7
            }
        );
        assert_eq!(a.committed(&draft), vec![10, 11, 7]);
    }

    #[test]
    fn zero_accept_still_commits_the_target_token() {
        // First draft already wrong → accept nothing, but the target's
        // own token is committed, so the round always makes progress
        // (degrades to one plain decode step, never a stall).
        let draft = [10, 11, 12, 13];
        let target = [42, 11, 12, 13, 99];
        let a = greedy_accept(&draft, &target);
        assert_eq!(
            a,
            SpecAccept {
                accepted: 0,
                bonus: 42
            }
        );
        assert_eq!(a.committed(&draft), vec![42]);
    }

    #[test]
    fn mismatch_at_last_position() {
        let draft = [10, 11, 12, 13];
        let target = [10, 11, 12, 8, 99];
        let a = greedy_accept(&draft, &target);
        assert_eq!(
            a,
            SpecAccept {
                accepted: 3,
                bonus: 8
            }
        );
        assert_eq!(a.committed(&draft), vec![10, 11, 12, 8]);
    }

    #[test]
    fn single_token_draft() {
        let draft = [10];
        assert_eq!(
            greedy_accept(&draft, &[10, 55]),
            SpecAccept {
                accepted: 1,
                bonus: 55
            }
        );
        assert_eq!(
            greedy_accept(&draft, &[9, 55]),
            SpecAccept {
                accepted: 0,
                bonus: 9
            }
        );
    }

    #[test]
    fn config_enabled_gating() {
        assert!(!SpeculativeConfig::default().is_enabled());
        assert!(
            !SpeculativeConfig {
                drafter: Some("d".into()),
                draft_len: 0,
            }
            .is_enabled()
        );
        assert!(
            SpeculativeConfig {
                drafter: Some("d".into()),
                draft_len: 4,
            }
            .is_enabled()
        );
    }
}
