//! Which experts the VRAM tier should hold, from decayed routing counts.
//!
//! Admitting every miss (LRU) lets an expert routed once displace one
//! that is routed every few tokens. Strata's adaptive tier instead counts
//! routings per (layer, expert), multiplies the counts by a decay after
//! each round, and every round swaps a layer's most-routed missing
//! experts for its least-routed resident ones, when the candidate's count
//! beats the victim's by a margin. This is that rule, as a pure function
//! of the counts and the residency.

/// The rule's knobs. Defaults are Strata's for a tier that holds 15-65%
/// of the experts (its #4870 mid-coverage values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayedPolicy {
    /// Decode steps between rounds.
    pub every: usize,
    /// What the counts are multiplied by after each round.
    pub decay: f32,
    /// A missing expert must have been routed at least this much.
    pub min_count: f32,
    /// ... and beat its victim by this much.
    pub min_gain: f32,
    /// Swaps per round, across all layers, best gain first.
    pub max_swaps: usize,
}

impl Default for DecayedPolicy {
    fn default() -> Self {
        Self {
            every: 2,
            decay: 0.92,
            min_count: 2.0,
            min_gain: 1.5,
            max_swaps: 16,
        }
    }
}

/// One swap: `admit` replaces `evict` in `layer`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swap {
    pub layer: usize,
    pub admit: u32,
    pub evict: u32,
    pub gain: f32,
}

/// The round's swaps. `usage` is `[layer][expert]`; `resident(layer)`
/// lists the experts the tier holds there.
pub fn plan_swaps(
    policy: &DecayedPolicy,
    usage: &[f32],
    experts: usize,
    resident: impl Fn(usize) -> Vec<u32>,
) -> Vec<Swap> {
    let layers = usage.len() / experts.max(1);
    let mut swaps = Vec::new();
    let mut held = vec![false; experts];
    for layer in 0..layers {
        let u = &usage[layer * experts..(layer + 1) * experts];
        held.fill(false);
        let mut victims: Vec<(f32, u32)> = Vec::new();
        for e in resident(layer) {
            if let Some(h) = held.get_mut(e as usize) {
                *h = true;
                victims.push((u[e as usize], e));
            }
        }
        if victims.is_empty() {
            continue;
        }
        let mut candidates: Vec<(f32, u32)> = u
            .iter()
            .enumerate()
            .filter(|&(e, &c)| !held[e] && c >= policy.min_count)
            .map(|(e, &c)| (c, e as u32))
            .collect();
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        victims.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (c, v) in candidates.iter().zip(&victims) {
            if c.0 < v.0 + policy.min_gain {
                break;
            }
            swaps.push(Swap {
                layer,
                admit: c.1,
                evict: v.1,
                gain: c.0 - v.0,
            });
        }
    }
    swaps.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    swaps.truncate(policy.max_swaps);
    swaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> DecayedPolicy {
        DecayedPolicy {
            max_swaps: 100,
            ..DecayedPolicy::default()
        }
    }

    /// The hottest missing expert replaces the coldest resident one, and
    /// a candidate that does not clear the margin swaps nothing.
    #[test]
    fn the_hottest_miss_replaces_the_coldest_resident() {
        // One layer of 6: 0 and 1 resident (counts 5 and 1), 2..6 missing.
        let usage = [5.0, 1.0, 9.0, 2.0, 0.0, 2.4];
        let swaps = plan_swaps(&policy(), &usage, 6, |_| vec![0, 1]);
        // 9 beats 1 (gain 8); 2.4 against 5 does not clear 1.5.
        assert_eq!(
            swaps,
            [Swap {
                layer: 0,
                admit: 2,
                evict: 1,
                gain: 8.0
            }]
        );
    }

    #[test]
    fn a_rarely_routed_miss_never_displaces_anything() {
        // 1.9 is below the count floor even though the resident is at 0.
        let usage = [0.0, 1.9];
        assert!(plan_swaps(&policy(), &usage, 2, |_| vec![0]).is_empty());
    }

    /// Swaps are taken best gain first across layers, up to the cap.
    #[test]
    fn the_cap_keeps_the_best_gains_across_layers() {
        let usage = [
            0.0, 4.0, // layer 0: gain 4
            1.0, 9.0, // layer 1: gain 8
            0.0, 3.0, // layer 2: gain 3
        ];
        let capped = DecayedPolicy {
            max_swaps: 2,
            ..policy()
        };
        let swaps = plan_swaps(&capped, &usage, 2, |_| vec![0]);
        let picked: Vec<(usize, u32, u32)> =
            swaps.iter().map(|s| (s.layer, s.admit, s.evict)).collect();
        assert_eq!(picked, [(1, 1, 0), (0, 1, 0)]);
    }

    /// Several swaps in one layer pair hottest with coldest, in order,
    /// and stop at the first pair that does not clear the margin.
    #[test]
    fn pairs_go_hottest_with_coldest_until_the_margin_fails() {
        // Resident 0..3 at 0, 3, 6; missing 3..6 at 10, 7, 5.
        let usage = [0.0, 3.0, 6.0, 10.0, 7.0, 5.0];
        let swaps = plan_swaps(&policy(), &usage, 6, |_| vec![0, 1, 2]);
        let picked: Vec<(u32, u32)> = swaps.iter().map(|s| (s.admit, s.evict)).collect();
        // 10/0 and 7/3 clear 1.5; 5 against 6 does not.
        assert_eq!(picked, [(3, 0), (4, 1)]);
    }
}
