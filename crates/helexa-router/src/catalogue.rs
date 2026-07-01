//! Federation catalogue (#75) — the router's aggregate `/v1/models`.
//!
//! Presents the **deduped union** of every reachable cortex's `/v1/models`
//! as the router's own catalogue, so an opencode client doing discovery
//! against the router resolves the whole federation without knowing about
//! operators or cortexes (resolves #61's "Router/discovery contract").
//!
//! Re-tiering: the fractal design is neuron ← cortex ← router. At the
//! router tier the "nodes" are **cortexes**, so the merged entry's
//! `feasible_on` / `locations` are rewritten to **operator names**, not the
//! neuron names a cortex reports. That keeps the federation view honest
//! ("served by these operators") without leaking each operator's internal
//! topology (neuron names, per-device VRAM) to end users.
//!
//! Conflict resolution when operators advertise the same model with
//! different enrichment:
//! - **`limit`** → the *tightest* (smallest `context`), so a client never
//!   overflows the most-constrained operator that might serve it (same rule
//!   cortex uses across its neurons).
//! - **`cost`** → the *cheapest* (lowest input, then output), the
//!   federation "from" price. Richer policy (a range, region/price-aware
//!   selection) couples to #68 and is left as a follow-up.

use crate::state::{CortexTopology, entry_feasible};
use cortex_core::harness::{ModelCost, ModelLimit};
use cortex_core::node::{CortexModelEntry, ModelLocation, ModelStatus};
use std::collections::HashMap;

/// Build the federation catalogue: the deduped union of every reachable
/// cortex's serveable models, merged across operators and sorted by id.
pub fn aggregate_models(topology: &HashMap<String, CortexTopology>) -> Vec<CortexModelEntry> {
    // Iterate cortexes in name order so `feasible_on` / `locations` and the
    // limit/cost tie-breaks are deterministic regardless of map ordering.
    let mut cortexes: Vec<(&String, &CortexTopology)> = topology.iter().collect();
    cortexes.sort_by(|a, b| a.0.cmp(b.0));

    let mut merged: HashMap<String, CortexModelEntry> = HashMap::new();
    for (cortex_name, t) in cortexes {
        if !t.reachable {
            continue;
        }
        for entry in t.models.values() {
            // Only surface models the cortex can actually serve — a
            // catalogue-only entry no neuron can host shouldn't appear in
            // the federation view.
            if !entry_feasible(entry) {
                continue;
            }
            merged
                .entry(entry.id.clone())
                .and_modify(|acc| merge_into(acc, cortex_name, entry))
                .or_insert_with(|| router_entry(cortex_name, entry));
        }
    }

    let mut out: Vec<CortexModelEntry> = merged.into_values().collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    // Re-derive the flat ecosystem fields (#78) from the merged (tightest)
    // limit — the values deserialized from each cortex are per-operator and
    // may not match the federation-wide merge.
    for e in &mut out {
        e.sync_flat_limit();
    }
    out
}

/// Seed a federation entry from the first cortex that serves the model,
/// re-tiering `feasible_on` / `locations` to the operator name.
fn router_entry(cortex: &str, e: &CortexModelEntry) -> CortexModelEntry {
    CortexModelEntry {
        id: e.id.clone(),
        object: "model".into(),
        created: e.created,
        owned_by: e.owned_by.clone(),
        loaded: e.loaded,
        feasible_on: vec![cortex.to_string()],
        locations: loaded_location(cortex, e),
        capabilities: e.capabilities.clone(),
        limit: e.limit.clone(),
        cost: e.cost.clone(),
        tool_call: e.tool_call,
        reasoning: e.reasoning,
        // Derived from `limit` by the final sync pass in aggregate_models.
        max_model_len: None,
        max_input_tokens: None,
        max_output_tokens: None,
    }
}

/// Fold another cortex's view of the same model into the merged entry.
fn merge_into(acc: &mut CortexModelEntry, cortex: &str, e: &CortexModelEntry) {
    acc.loaded |= e.loaded;
    acc.feasible_on.push(cortex.to_string());
    acc.locations.extend(loaded_location(cortex, e));
    for cap in &e.capabilities {
        if !acc.capabilities.contains(cap) {
            acc.capabilities.push(cap.clone());
        }
    }
    acc.tool_call |= e.tool_call;
    acc.reasoning |= e.reasoning;
    acc.limit = tightest_limit(acc.limit.take(), e.limit.clone());
    acc.cost = cheapest_cost(acc.cost.take(), e.cost.clone());
}

/// A single cortex-tier location when the model is loaded at that operator;
/// empty when only cold-loadable. Neuron-level VRAM is deliberately dropped.
fn loaded_location(cortex: &str, e: &CortexModelEntry) -> Vec<ModelLocation> {
    if e.loaded {
        vec![ModelLocation {
            node: cortex.to_string(),
            status: ModelStatus::Loaded,
            vram_estimate_mb: None,
        }]
    } else {
        Vec::new()
    }
}

/// Smaller `context` wins — never advertise more headroom than the
/// most-constrained operator can honour.
fn tightest_limit(a: Option<ModelLimit>, b: Option<ModelLimit>) -> Option<ModelLimit> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => Some(if b.context < a.context { b } else { a }),
    }
}

/// Cheapest by (input, output) price — the federation "from" price.
fn cheapest_cost(a: Option<ModelCost>, b: Option<ModelCost>) -> Option<ModelCost> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => Some(if (b.input, b.output) < (a.input, a.output) {
            b
        } else {
            a
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CortexTopology;

    fn entry(id: &str, loaded: bool, feasible: bool) -> CortexModelEntry {
        CortexModelEntry {
            id: id.into(),
            object: "model".into(),
            created: 0,
            owned_by: "helexa".into(),
            loaded,
            feasible_on: if feasible || loaded {
                vec!["some-neuron".into()]
            } else {
                vec![]
            },
            locations: vec![],
            capabilities: vec![],
            limit: None,
            cost: None,
            tool_call: false,
            reasoning: false,
            max_model_len: None,
            max_input_tokens: None,
            max_output_tokens: None,
        }
    }

    fn cortex(reachable: bool, entries: Vec<CortexModelEntry>) -> CortexTopology {
        CortexTopology {
            reachable,
            consecutive_failures: 0,
            last_poll: None,
            healthy_nodes: 1,
            total_nodes: 1,
            models: entries.into_iter().map(|e| (e.id.clone(), e)).collect(),
        }
    }

    #[test]
    fn dedupes_and_merges_availability_across_cortexes() {
        let mut topo = HashMap::new();
        // c-a: model loaded. c-b: same model only cold-loadable.
        topo.insert("c-a".into(), cortex(true, vec![entry("m", true, true)]));
        topo.insert("c-b".into(), cortex(true, vec![entry("m", false, true)]));

        let out = aggregate_models(&topo);
        assert_eq!(out.len(), 1, "duplicate model id collapses to one");
        let m = &out[0];
        assert!(m.loaded, "loaded somewhere → loaded");
        // feasible_on re-tiered to operator names, both present, sorted.
        assert_eq!(m.feasible_on, vec!["c-a".to_string(), "c-b".to_string()]);
        // Only the loaded operator contributes a location, named by operator.
        assert_eq!(m.locations.len(), 1);
        assert_eq!(m.locations[0].node, "c-a");
        assert_eq!(m.locations[0].vram_estimate_mb, None);
    }

    #[test]
    fn unreachable_cortex_is_excluded() {
        let mut topo = HashMap::new();
        topo.insert("up".into(), cortex(true, vec![entry("m", true, true)]));
        topo.insert(
            "down".into(),
            cortex(false, vec![entry("other", true, true)]),
        );
        let out = aggregate_models(&topo);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "m");
    }

    #[test]
    fn catalogue_only_infeasible_entries_are_hidden() {
        let mut topo = HashMap::new();
        topo.insert("c".into(), cortex(true, vec![entry("ghost", false, false)]));
        assert!(aggregate_models(&topo).is_empty());
    }

    #[test]
    fn preserves_tightest_limit_and_cheapest_cost() {
        let mut a = entry("m", true, true);
        a.limit = Some(ModelLimit {
            context: 32_768,
            input: None,
            output: 4096,
        });
        a.cost = Some(ModelCost {
            input: 0.50,
            output: 1.50,
            cache_read: None,
            cache_write: None,
        });
        let mut b = entry("m", true, true);
        b.limit = Some(ModelLimit {
            context: 16_384, // tighter
            input: None,
            output: 4096,
        });
        b.cost = Some(ModelCost {
            input: 0.20, // cheaper
            output: 0.80,
            cache_read: None,
            cache_write: None,
        });

        let mut topo = HashMap::new();
        topo.insert("c-a".into(), cortex(true, vec![a]));
        topo.insert("c-b".into(), cortex(true, vec![b]));

        let out = aggregate_models(&topo);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].limit.as_ref().unwrap().context, 16_384);
        assert_eq!(out[0].cost.as_ref().unwrap().input, 0.20);
        // Flat #78 fields re-derived from the merged (tightest) limit.
        assert_eq!(out[0].max_model_len, Some(16_384));
        assert_eq!(out[0].max_input_tokens, None);
        assert_eq!(out[0].max_output_tokens, Some(4096));
    }
}
