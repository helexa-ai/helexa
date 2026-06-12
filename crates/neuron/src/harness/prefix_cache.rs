//! Prefix-cache registry: which cache snapshots exist for a loaded
//! model, which one matches an incoming prompt, and which to evict.
//!
//! Pure bookkeeping — no tensors live here. Each entry pairs the exact
//! token sequence a snapshot was captured at with an opaque snapshot
//! reference `R` (a worker-side snapshot id for CUDA loads, the
//! snapshot itself for CPU loads) and its byte size for the VRAM
//! budget. The caller owns actually dropping evicted snapshots.
//!
//! ## Matching policy
//!
//! A snapshot is reusable only when its **entire** token sequence is a
//! strict prefix of the incoming prompt (`entry.len() < prompt.len()`
//! — at least one suffix token must be forwarded to produce logits).
//! The GatedDeltaNet recurrent state cannot be rewound, so partial
//! matches are unusable; see `arch/qwen3_5/snapshot.rs`.
//!
//! ## Insertion policy
//!
//! Inserting an entry drops existing entries that are strict prefixes
//! of it: the append-only agent loop (turn N+1 = turn N + new text)
//! keeps exactly one entry per conversation thread that way, instead
//! of one per turn. Eviction beyond that is LRU over total bytes
//! against the configured budget, plus a max-entries cap.

/// One cached snapshot: the token sequence it was captured at, the
/// opaque snapshot reference, and bookkeeping for eviction.
struct Entry<R> {
    tokens: Vec<u32>,
    snapshot: R,
    bytes: u64,
    last_used: u64,
}

/// A match returned by [`PrefixCache::longest_match`].
pub struct PrefixMatch<R> {
    /// Clone of the matched snapshot reference.
    pub snapshot: R,
    /// Number of prompt tokens the snapshot covers (the entry's full
    /// token count). Prefill resumes at this offset.
    pub tokens: usize,
}

/// LRU prefix-snapshot registry for one loaded model.
pub struct PrefixCache<R> {
    entries: Vec<Entry<R>>,
    budget_bytes: u64,
    max_entries: usize,
    /// Monotonic access clock for LRU ordering.
    seq: u64,
}

impl<R: Clone> PrefixCache<R> {
    pub fn new(budget_bytes: u64, max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            max_entries,
            seq: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn used_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// Longest entry whose token sequence is a strict prefix of
    /// `prompt`. Touches the entry's LRU clock on hit.
    pub fn longest_match(&mut self, prompt: &[u32]) -> Option<PrefixMatch<R>> {
        let idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tokens.len() < prompt.len() && prompt.starts_with(&e.tokens))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)?;
        let now = self.tick();
        let entry = &mut self.entries[idx];
        entry.last_used = now;
        Some(PrefixMatch {
            snapshot: entry.snapshot.clone(),
            tokens: entry.tokens.len(),
        })
    }

    /// Remove the entry whose tokens exactly prefix-match what
    /// `longest_match` just returned. Called when restoring its
    /// snapshot failed; returns the reference so the caller can drop
    /// the underlying snapshot.
    pub fn remove_covering(&mut self, prompt: &[u32], tokens: usize) -> Option<R> {
        let idx = self
            .entries
            .iter()
            .position(|e| e.tokens.len() == tokens && prompt.starts_with(&e.tokens))?;
        Some(self.entries.swap_remove(idx).snapshot)
    }

    /// Insert a fresh snapshot captured at exactly `tokens`. Returns
    /// every snapshot reference the caller must now drop: replaced
    /// duplicates, strict prefixes of the new entry, LRU evictions to
    /// fit the byte budget / entry cap — and the new snapshot itself
    /// when it alone exceeds the budget (in which case it is not
    /// inserted).
    pub fn insert(&mut self, tokens: Vec<u32>, snapshot: R, bytes: u64) -> Vec<R> {
        let mut dropped = Vec::new();
        if bytes > self.budget_bytes || self.max_entries == 0 || tokens.is_empty() {
            dropped.push(snapshot);
            return dropped;
        }
        // Drop entries the new one supersedes: exact duplicates and
        // strict prefixes (the conversation they belong to has moved
        // on; the new entry matches everything they would have).
        let mut i = 0;
        while i < self.entries.len() {
            if tokens.starts_with(&self.entries[i].tokens) {
                dropped.push(self.entries.swap_remove(i).snapshot);
            } else {
                i += 1;
            }
        }
        let now = self.tick();
        self.entries.push(Entry {
            tokens,
            snapshot,
            bytes,
            last_used: now,
        });
        // LRU-evict to budget and cap. The just-inserted entry has the
        // freshest clock, so it is only evicted if it is the last one
        // standing — and it fits the budget alone (checked above).
        while self.used_bytes() > self.budget_bytes || self.entries.len() > self.max_entries {
            let lru = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .expect("eviction loop runs only while entries is non-empty");
            dropped.push(self.entries.swap_remove(lru).snapshot);
        }
        dropped
    }

    /// Number of live entries (test/log helper).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(budget: u64, max: usize) -> PrefixCache<u64> {
        PrefixCache::new(budget, max)
    }

    #[test]
    fn longest_strict_prefix_wins() {
        let mut c = cache(1000, 8);
        assert!(c.insert(vec![1, 2], 10, 1).is_empty());
        // [1,2,3] is NOT a prefix of [1,2] superseding chain — diverge
        // it so both stay live.
        assert!(c.insert(vec![1, 9, 9, 9], 11, 1).is_empty());
        let m = c.longest_match(&[1, 2, 3, 4]).expect("hit");
        assert_eq!(m.snapshot, 10);
        assert_eq!(m.tokens, 2);
    }

    #[test]
    fn exact_length_match_is_rejected() {
        // A snapshot covering the whole prompt leaves no suffix token
        // to forward — must not match.
        let mut c = cache(1000, 8);
        c.insert(vec![1, 2, 3], 10, 1);
        assert!(c.longest_match(&[1, 2, 3]).is_none());
        assert!(c.longest_match(&[1, 2, 3, 4]).is_some());
    }

    #[test]
    fn divergent_prompt_misses() {
        let mut c = cache(1000, 8);
        c.insert(vec![1, 2, 3], 10, 1);
        assert!(c.longest_match(&[1, 2, 4, 5]).is_none());
    }

    #[test]
    fn insert_supersedes_prefix_entries() {
        let mut c = cache(1000, 8);
        c.insert(vec![1, 2], 10, 1);
        let dropped = c.insert(vec![1, 2, 3, 4], 11, 1);
        assert_eq!(dropped, vec![10]);
        assert_eq!(c.len(), 1);
        // The longer entry still matches its own continuations.
        assert_eq!(c.longest_match(&[1, 2, 3, 4, 5]).unwrap().snapshot, 11);
    }

    #[test]
    fn insert_replaces_exact_duplicate() {
        let mut c = cache(1000, 8);
        c.insert(vec![1, 2], 10, 1);
        let dropped = c.insert(vec![1, 2], 11, 1);
        assert_eq!(dropped, vec![10]);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn byte_budget_evicts_lru() {
        let mut c = cache(10, 8);
        c.insert(vec![1], 10, 4);
        c.insert(vec![2], 11, 4);
        // Touch [1] so [2] becomes LRU.
        assert!(c.longest_match(&[1, 5]).is_some());
        let dropped = c.insert(vec![3], 12, 4);
        assert_eq!(dropped, vec![11]);
        assert_eq!(c.len(), 2);
        assert!(c.longest_match(&[1, 5]).is_some());
        assert!(c.longest_match(&[2, 5]).is_none());
    }

    #[test]
    fn max_entries_cap_evicts_lru() {
        let mut c = cache(1000, 2);
        c.insert(vec![1], 10, 1);
        c.insert(vec![2], 11, 1);
        let dropped = c.insert(vec![3], 12, 1);
        assert_eq!(dropped, vec![10]);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn oversized_snapshot_is_rejected_back() {
        let mut c = cache(10, 8);
        let dropped = c.insert(vec![1, 2], 10, 11);
        assert_eq!(dropped, vec![10]);
        assert!(c.is_empty());
    }

    #[test]
    fn remove_covering_drops_the_matched_entry() {
        let mut c = cache(1000, 8);
        c.insert(vec![1, 2], 10, 1);
        let m = c.longest_match(&[1, 2, 3]).unwrap();
        let removed = c.remove_covering(&[1, 2, 3], m.tokens);
        assert_eq!(removed, Some(10));
        assert!(c.is_empty());
        assert_eq!(c.remove_covering(&[1, 2, 3], m.tokens), None);
    }

    #[test]
    fn empty_tokens_never_stored() {
        let mut c = cache(1000, 8);
        let dropped = c.insert(vec![], 10, 1);
        assert_eq!(dropped, vec![10]);
        assert!(c.is_empty());
    }
}
