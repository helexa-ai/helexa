//! A byte-budgeted VRAM tier for host-resident routed experts (#318).
//!
//! A model whose experts exceed VRAM keeps all of them in host memory.
//! Routing is skewed, so a modest share of them serves most tokens, and
//! each expert held on the device is one the CPU does not have to
//! compute. The tier is shared by every MoE block of a model because the
//! budget is a property of the card and heat varies between layers.
//!
//! Slots are sized in bytes rather than counted: expert sizes differ
//! between layers and quantisation types, and a slot count sized for the
//! largest wastes VRAM on the rest.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

/// The VRAM budget from `NEURON_EXPERT_CACHE_MB`, when one is set.
pub fn configured_budget_mb() -> Option<u64> {
    std::env::var("NEURON_EXPERT_CACHE_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&mb| mb > 0)
}

/// `(block, expert)`. Blocks are numbered as they register.
type Key = (u32, u32);

struct Entry<T> {
    value: Arc<T>,
    bytes: u64,
    stamp: u64,
}

/// Least-recently-used map with a byte budget.
struct ByteLru<T> {
    capacity: u64,
    used: u64,
    clock: u64,
    entries: HashMap<Key, Entry<T>>,
    by_age: BTreeMap<u64, Key>,
}

impl<T> ByteLru<T> {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            clock: 0,
            entries: HashMap::new(),
            by_age: BTreeMap::new(),
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn get(&mut self, key: Key) -> Option<Arc<T>> {
        let stamp = self.tick();
        let entry = self.entries.get_mut(&key)?;
        self.by_age.remove(&entry.stamp);
        entry.stamp = stamp;
        self.by_age.insert(stamp, key);
        Some(entry.value.clone())
    }

    /// Bytes that must be released before `bytes` more can be held.
    /// Returns how many entries were evicted.
    fn make_room(&mut self, bytes: u64) -> u64 {
        let mut evicted = 0;
        while self.used + bytes > self.capacity {
            let Some((_, key)) = self.by_age.pop_first() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&key) {
                self.used -= entry.bytes;
                evicted += 1;
            }
        }
        evicted
    }

    fn insert(&mut self, key: Key, value: Arc<T>, bytes: u64) {
        let stamp = self.tick();
        self.used += bytes;
        self.by_age.insert(stamp, key);
        self.entries.insert(
            key,
            Entry {
                value,
                bytes,
                stamp,
            },
        );
    }
}

/// Cumulative counters, for measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Lookups that may not admit: a prompt chunk's.
    pub prompt_hits: u64,
    pub prompt_misses: u64,
    pub used_bytes: u64,
    pub entries: u64,
}

struct State<T> {
    lru: ByteLru<T>,
    hits: u64,
    misses: u64,
    evictions: u64,
    prompt_hits: u64,
    prompt_misses: u64,
    next_block: u32,
}

/// The tier itself. `T` is the device-resident expert.
pub struct ExpertTier<T> {
    state: Mutex<State<T>>,
}

impl<T> ExpertTier<T> {
    pub fn new(capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                lru: ByteLru::new(capacity_bytes),
                hits: 0,
                misses: 0,
                evictions: 0,
                prompt_hits: 0,
                prompt_misses: 0,
                next_block: 0,
            }),
        })
    }

    /// A number identifying one MoE block's experts within the tier.
    pub fn register_block(&self) -> u32 {
        let mut state = self.state.lock().expect("expert tier lock");
        let block = state.next_block;
        state.next_block += 1;
        block
    }

    /// The expert, from the tier or from `load`.
    ///
    /// Room is made *before* `load` runs, so the outgoing expert's VRAM
    /// is available to the incoming one. An expert larger than the whole
    /// budget is returned without being retained.
    pub fn get_or_load<E>(
        &self,
        block: u32,
        expert: u32,
        bytes: u64,
        load: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        let key = (block, expert);
        let retain = {
            let mut state = self.state.lock().expect("expert tier lock");
            if let Some(value) = state.lru.get(key) {
                state.hits += 1;
                return Ok(value);
            }
            state.misses += 1;
            let retain = bytes <= state.lru.capacity;
            if retain {
                let evicted = state.lru.make_room(bytes);
                state.evictions += evicted;
            }
            retain
        };
        let value = Arc::new(load()?);
        if retain {
            let mut state = self.state.lock().expect("expert tier lock");
            state.lru.insert(key, value.clone(), bytes);
        }
        Ok(value)
    }

    /// Counts a use of the expert and refreshes it if held. Returns
    /// whether it was held. A caller that admits on a budget uses this
    /// for the uses it will not admit, so they still count as misses.
    pub fn touch(&self, block: u32, expert: u32) -> bool {
        let mut state = self.state.lock().expect("expert tier lock");
        let held = state.lru.get((block, expert)).is_some();
        if held {
            state.hits += 1;
        } else {
            state.misses += 1;
        }
        held
    }

    /// The expert if it is held, without admitting it or refreshing it.
    ///
    /// A prompt chunk touches more experts than the tier holds, once
    /// each. Letting that scan admit or reorder would replace the set
    /// decode keeps returning to with experts it may never use again.
    pub fn peek(&self, block: u32, expert: u32) -> Option<Arc<T>> {
        let mut state = self.state.lock().expect("expert tier lock");
        let value = state
            .lru
            .entries
            .get(&(block, expert))
            .map(|entry| entry.value.clone());
        if value.is_some() {
            state.prompt_hits += 1;
        } else {
            state.prompt_misses += 1;
        }
        value
    }

    /// The byte budget.
    pub fn capacity_bytes(&self) -> u64 {
        self.state.lock().expect("expert tier lock").lru.capacity
    }

    pub fn stats(&self) -> TierStats {
        let state = self.state.lock().expect("expert tier lock");
        TierStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            prompt_hits: state.prompt_hits,
            prompt_misses: state.prompt_misses,
            used_bytes: state.lru.used,
            entries: state.lru.entries.len() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn fetch(tier: &ExpertTier<u32>, expert: u32, bytes: u64, loads: &Cell<u32>) -> u32 {
        *tier
            .get_or_load::<()>(0, expert, bytes, || {
                loads.set(loads.get() + 1);
                Ok(expert)
            })
            .unwrap()
    }

    #[test]
    fn a_hit_does_not_load_again() {
        let tier = ExpertTier::new(100);
        let loads = Cell::new(0);
        assert_eq!(fetch(&tier, 7, 10, &loads), 7);
        assert_eq!(fetch(&tier, 7, 10, &loads), 7);
        assert_eq!(loads.get(), 1);
        let stats = tier.stats();
        assert_eq!((stats.hits, stats.misses, stats.used_bytes), (1, 1, 10));
    }

    #[test]
    fn the_least_recently_used_expert_is_evicted() {
        let tier = ExpertTier::new(30);
        let loads = Cell::new(0);
        fetch(&tier, 1, 10, &loads);
        fetch(&tier, 2, 10, &loads);
        fetch(&tier, 3, 10, &loads);
        // Touching 1 leaves 2 as the oldest.
        fetch(&tier, 1, 10, &loads);
        fetch(&tier, 4, 10, &loads);
        assert_eq!(loads.get(), 4);
        fetch(&tier, 1, 10, &loads);
        assert_eq!(loads.get(), 4, "1 was recently used and must survive");
        fetch(&tier, 2, 10, &loads);
        assert_eq!(loads.get(), 5, "2 was the oldest and must have gone");
    }

    #[test]
    fn the_budget_is_in_bytes_and_is_never_exceeded() {
        let tier = ExpertTier::new(50);
        let loads = Cell::new(0);
        for (expert, bytes) in [(1, 20), (2, 20), (3, 30), (4, 5), (5, 45)] {
            fetch(&tier, expert, bytes, &loads);
            assert!(tier.stats().used_bytes <= 50);
        }
        assert_eq!(tier.stats().used_bytes, 50);
    }

    #[test]
    fn an_expert_larger_than_the_budget_is_served_but_not_kept() {
        let tier = ExpertTier::new(10);
        let loads = Cell::new(0);
        fetch(&tier, 1, 5, &loads);
        assert_eq!(fetch(&tier, 2, 11, &loads), 2);
        let stats = tier.stats();
        assert_eq!((stats.used_bytes, stats.entries), (5, 1));
        fetch(&tier, 1, 5, &loads);
        assert_eq!(loads.get(), 2, "the oversized expert must not evict 1");
    }

    #[test]
    fn blocks_do_not_share_experts() {
        let tier: Arc<ExpertTier<u32>> = ExpertTier::new(100);
        let (a, b) = (tier.register_block(), tier.register_block());
        assert_ne!(a, b);
        let first = tier.get_or_load::<()>(a, 3, 10, || Ok(30)).unwrap();
        let second = tier.get_or_load::<()>(b, 3, 10, || Ok(40)).unwrap();
        assert_eq!((*first, *second), (30, 40));
    }

    #[test]
    fn a_prompt_scan_neither_admits_nor_reorders() {
        let tier = ExpertTier::new(20);
        let loads = Cell::new(0);
        fetch(&tier, 1, 10, &loads);
        fetch(&tier, 2, 10, &loads);
        // Peeking at 1 must not make it newer than 2.
        assert_eq!(tier.peek(0, 1).as_deref(), Some(&1));
        assert!(tier.peek(0, 9).is_none());
        let stats = tier.stats();
        assert_eq!(
            (stats.prompt_hits, stats.prompt_misses, stats.entries),
            (1, 1, 2)
        );
        fetch(&tier, 3, 10, &loads);
        assert!(tier.peek(0, 1).is_none(), "1 was still the oldest");
        assert!(tier.peek(0, 2).is_some());
    }

    /// A touch refreshes what is held, so it survives the next eviction,
    /// and counts what is not as a miss without admitting it.
    #[test]
    fn a_touch_refreshes_without_admitting() {
        let tier = ExpertTier::new(20);
        let loads = Cell::new(0);
        fetch(&tier, 1, 10, &loads);
        fetch(&tier, 2, 10, &loads);
        assert!(tier.touch(0, 1));
        assert!(!tier.touch(0, 9));
        fetch(&tier, 3, 10, &loads);
        assert!(tier.peek(0, 1).is_some(), "the touch made 1 newer than 2");
        assert!(tier.peek(0, 2).is_none());
        assert!(tier.peek(0, 9).is_none(), "a touch never admits");
        assert_eq!(loads.get(), 3);
        let stats = tier.stats();
        assert_eq!((stats.hits, stats.misses), (1, 4));
    }

    #[test]
    fn a_failed_load_retains_nothing() {
        let tier: Arc<ExpertTier<u32>> = ExpertTier::new(100);
        assert!(tier.get_or_load(0, 1, 10, || Err("oom")).is_err());
        let stats = tier.stats();
        assert_eq!((stats.used_bytes, stats.entries), (0, 0));
    }
}
