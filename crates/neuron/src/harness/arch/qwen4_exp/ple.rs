//! PLE — per-layer embedding over hashed n-grams.
//!
//! `qwen4_exp` puts one of these on zero-indexed layer **1**
//! (`ple_layer_ids: [2]` is one-indexed upstream), and it accounts for
//! **28.44% of the model's parameters** — a 320,001,536-row table that
//! each token touches 16 rows of. This module covers the *addressing*:
//! turning token ids into row indices. The lookup itself is a gather
//! from a table that will not be device-resident, and the consumption
//! (projections, gate, dilated conv) is separate.
//!
//! ## Geometry
//!
//! `ngram_heads = (ngram_size - 1) * heads_per_ngram` — for the shipped
//! config, `(3 - 1) * 8 = 16` heads of `ple_embed_dim / 16 = 160` dims.
//! There is **no unigram head**: with `ngram_size = 3` the orders are
//! bigram and trigram only. `layer_multipliers` still has three entries
//! because a trigram needs three shift multipliers.
//!
//! ## The hash
//!
//! For n-gram order *n*, over heads `[(n-2)*heads_per_ngram, +8)`:
//!
//! ```text
//! mixed = tok[0] * m[0]
//! for p in 1..n:  mixed ^= tok[p] * m[p]
//! row[head] = (mixed mod vocab_sizes[head]) + offsets[head]
//! ```
//!
//! where `tok[s]` is the id stream shifted right by `s`.
//!
//! Three things worth stating precisely:
//!
//! 1. **The arithmetic is int64 and stays non-negative by
//!    construction.** Upstream derives `multiplier_max` as
//!    `i64::MAX / vocab_size` specifically so that `token_id *
//!    multiplier` cannot overflow: the largest possible product is
//!    9,223,334,893,764,784,449 against an `i64::MAX` of
//!    9,223,372,036,854,775,807. Every product therefore has its sign
//!    bit clear, so their XOR does too and `mixed` is never negative.
//!    The products still overflow `i32` by four orders of magnitude, so
//!    the width matters.
//! 2. **The modulo is nonetheless Euclidean.** Given (1) it agrees with
//!    a truncated `%` on every value this checkpoint can produce, but
//!    torch's `remainder` follows Python's sign convention, and a
//!    checkpoint whose constants were derived differently would
//!    silently produce negative row indices under Rust's `%`.
//!    `rem_euclid` makes the addressing correct rather than
//!    incidentally correct.
//! 3. **Shifts do not cross an EOS.** Positions closer to the start of
//!    their EOS-delimited segment than `shift` read `eos_token_id`
//!    rather than borrowing the previous document's tokens. Invisible
//!    on a single-turn prompt, wrong on anything batched or
//!    multi-document.
//!
//! The three derived quantities — `layer_multipliers`,
//! `ngram_heads_vocab_sizes`, `ngram_heads_offsets` — all ship in the
//! checkpoint as I64 buffers, so the prime search and the splitmix64
//! multiplier derivation are load-time trivia we deliberately do not
//! reimplement. See `doc/qwen4_exp-port-spec.md` §2.

use anyhow::{Result, ensure};

/// Addressing for the n-gram table.
pub struct NGramHasher {
    ngram_size: usize,
    heads_per_ngram: usize,
    /// One per shift, `len == ngram_size`.
    multipliers: Vec<i64>,
    /// Per-head, `len == ngram_heads()`.
    head_vocab_sizes: Vec<i64>,
    head_offsets: Vec<i64>,
    eos_token_id: i64,
}

impl NGramHasher {
    /// Build from the checkpoint's own buffers.
    pub fn new(
        ngram_size: usize,
        heads_per_ngram: usize,
        multipliers: Vec<i64>,
        head_vocab_sizes: Vec<i64>,
        head_offsets: Vec<i64>,
        eos_token_id: i64,
    ) -> Result<Self> {
        ensure!(ngram_size >= 2, "ngram_size must be >= 2, got {ngram_size}");
        ensure!(
            multipliers.len() == ngram_size,
            "expected {ngram_size} layer_multipliers, got {}",
            multipliers.len()
        );
        let heads = (ngram_size - 1) * heads_per_ngram;
        ensure!(
            head_vocab_sizes.len() == heads && head_offsets.len() == heads,
            "expected {heads} per-head vocab sizes and offsets, got {} and {}",
            head_vocab_sizes.len(),
            head_offsets.len()
        );
        ensure!(
            head_vocab_sizes.iter().all(|v| *v > 0),
            "per-head vocab sizes must be positive"
        );
        Ok(Self {
            ngram_size,
            heads_per_ngram,
            multipliers,
            head_vocab_sizes,
            head_offsets,
            eos_token_id,
        })
    }

    /// `(ngram_size - 1) * heads_per_ngram` — 16 for the shipped config.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// How many previous ids must be carried across a decode step for
    /// the shifts to be correct: `ngram_size - 1`.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// Row indices for the last `seq_len` positions of `history`.
    ///
    /// `history` is the carried context followed by this step's ids, so
    /// `history.len() >= seq_len`. Returns one row per head per position,
    /// flattened as `[position][head]`.
    pub fn rows(&self, history: &[i64], seq_len: usize) -> Result<Vec<Vec<i64>>> {
        ensure!(
            history.len() >= seq_len,
            "history ({}) shorter than seq_len ({seq_len})",
            history.len()
        );
        // tok[s] for s in 0..ngram_size, each segment-aware.
        let shifted: Vec<Vec<i64>> = (0..self.ngram_size)
            .map(|s| self.shift_right_ignore_eos(history, s))
            .collect();

        // Transpose to [position][shift] so each position's n-gram is a
        // contiguous little slice rather than a stride across vectors.
        let start = history.len() - seq_len;
        let per_position =
            (start..history.len()).map(|pos| shifted.iter().map(|s| s[pos]).collect::<Vec<i64>>());

        let mut out = Vec::with_capacity(seq_len);
        for toks in per_position {
            let mut row = Vec::with_capacity(self.ngram_heads());
            for order in 2..=self.ngram_size {
                // Products are bounded by construction: upstream caps the
                // multiplier at i64::MAX / vocab_size, so a token id times
                // its multiplier cannot overflow. wrapping_mul keeps that
                // a documented assumption rather than a debug panic.
                let mut mixed = toks[0].wrapping_mul(self.multipliers[0]);
                for (tok, mult) in toks.iter().zip(&self.multipliers).take(order).skip(1) {
                    mixed ^= tok.wrapping_mul(*mult);
                }
                let head_start = (order - 2) * self.heads_per_ngram;
                let head_end = head_start + self.heads_per_ngram;
                let vocabs = &self.head_vocab_sizes[head_start..head_end];
                let offsets = &self.head_offsets[head_start..head_end];
                for (vocab, offset) in vocabs.iter().zip(offsets) {
                    // Euclidean, not truncated — see the module note.
                    row.push(mixed.rem_euclid(*vocab) + *offset);
                }
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Shift ids right by `shift` without crossing an EOS boundary.
    /// Positions with fewer than `shift` predecessors inside their own
    /// segment read `eos_token_id`.
    fn shift_right_ignore_eos(&self, ids: &[i64], shift: usize) -> Vec<i64> {
        if shift == 0 {
            return ids.to_vec();
        }
        let mut out = Vec::with_capacity(ids.len());
        // Position of the most recent EOS strictly before i.
        let mut previous_eos: i64 = -1;
        for (i, &id) in ids.iter().enumerate() {
            let segment_start = previous_eos + 1;
            let position_in_segment = i as i64 - segment_start;
            let source = i as i64 - shift as i64;
            let valid = position_in_segment >= shift as i64 && source >= 0;
            out.push(if valid {
                ids[source as usize]
            } else {
                self.eos_token_id
            });
            if id == self.eos_token_id {
                previous_eos = i as i64;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: i64 = 248044;

    /// The shipped geometry, with the checkpoint's real derivation for
    /// the head vocab sizes (primes just above 20,000,000) approximated
    /// by distinct nearby primes — only their distinctness and
    /// positivity matter to the addressing.
    fn hasher(ngram_size: usize, heads_per_ngram: usize) -> NGramHasher {
        let heads = (ngram_size - 1) * heads_per_ngram;
        let vocabs: Vec<i64> = (0..heads).map(|i| 20_000_003 + 2 * i as i64).collect();
        let mut offsets = Vec::with_capacity(heads);
        let mut acc = 0i64;
        for v in &vocabs {
            offsets.push(acc);
            acc += v;
        }
        // Realistic magnitude: odd, ~1e13, as _build_layer_multipliers
        // produces.
        let mults: Vec<i64> = (0..ngram_size)
            .map(|i| 12_345_678_901_235 + 2 * i as i64)
            .collect();
        NGramHasher::new(ngram_size, heads_per_ngram, mults, vocabs, offsets, EOS).unwrap()
    }

    #[test]
    fn geometry_matches_the_shipped_config() {
        let h = hasher(3, 8);
        assert_eq!(h.ngram_heads(), 16, "(ngram_size - 1) * heads_per_ngram");
        assert_eq!(h.context_len(), 2);
    }

    /// Every row must land inside its own head's slice of the table.
    /// This is what a truncated remainder would break, and it is also
    /// the invariant that keeps a gather in bounds.
    #[test]
    fn rows_stay_within_their_head_slice() {
        let h = hasher(3, 8);
        let ids: Vec<i64> = (0..64).map(|i| (i * 3907) % 248_320).collect();
        let rows = h.rows(&ids, ids.len()).unwrap();
        assert_eq!(rows.len(), ids.len());
        for row in &rows {
            assert_eq!(row.len(), 16);
            for (head, &r) in row.iter().enumerate() {
                let lo = h.head_offsets[head];
                let hi = lo + h.head_vocab_sizes[head];
                assert!(r >= lo && r < hi, "head {head}: {r} outside [{lo}, {hi})");
            }
        }
    }

    /// Upstream's multiplier bound guarantees products fit i64. Assert
    /// it rather than trusting the comment, since the whole hash relies
    /// on the sign bit staying clear.
    #[test]
    fn largest_possible_product_does_not_overflow_i64() {
        let vocab_size: i64 = 248_320;
        let multiplier_max = i64::MAX / vocab_size;
        let largest_multiplier = 2 * (multiplier_max / 2 - 1) + 1;
        let product = (vocab_size - 1).checked_mul(largest_multiplier);
        assert!(product.is_some(), "token id * multiplier must fit i64");
        assert!(product.unwrap() >= 0, "and stay non-negative");
    }

    /// A shift must not read across an EOS: the token after a document
    /// boundary sees EOS as its predecessor, not the previous
    /// document's last token.
    #[test]
    fn shifts_do_not_cross_an_eos_boundary() {
        let h = hasher(3, 8);
        //        0    1    2     3(eos)  4    5
        let ids = [11i64, 12, 13, EOS, 21, 22];

        let by_one = h.shift_right_ignore_eos(&ids, 1);
        // Position 4 is the first of a new segment — its predecessor
        // must be EOS, not the boundary token itself.
        assert_eq!(
            by_one[4], EOS,
            "first token of a segment has no predecessor"
        );
        assert_eq!(by_one[5], 21, "second token reads within its segment");
        assert_eq!(by_one[0], EOS, "very first position has no predecessor");
        assert_eq!(by_one[1], 11);

        let by_two = h.shift_right_ignore_eos(&ids, 2);
        assert_eq!(by_two[4], EOS);
        assert_eq!(by_two[5], EOS, "only one predecessor inside the segment");
        assert_eq!(by_two[2], 11);
    }

    #[test]
    fn shift_zero_is_the_identity() {
        let h = hasher(3, 8);
        let ids = [11i64, 12, EOS, 21];
        assert_eq!(h.shift_right_ignore_eos(&ids, 0), ids.to_vec());
    }

    /// The bigram heads see two shifts, the trigram heads three, so a
    /// token whose grandparent differs must change only the trigram
    /// half of the row.
    #[test]
    fn bigram_heads_ignore_the_third_token_but_trigram_heads_do_not() {
        let h = hasher(3, 4);
        // Same last two tokens, different one before that.
        let a = [7i64, 8, 9];
        let b = [99i64, 8, 9];
        let ra = &h.rows(&a, 1).unwrap()[0];
        let rb = &h.rows(&b, 1).unwrap()[0];

        assert_eq!(&ra[0..4], &rb[0..4], "bigram heads depend on 2 tokens only");
        assert_ne!(&ra[4..8], &rb[4..8], "trigram heads must see the third");
    }

    /// Carried context makes a decode step agree with the prefill that
    /// would have produced the same position.
    #[test]
    fn carried_context_reproduces_the_prefill_row() {
        let h = hasher(3, 8);
        let full = [11i64, 12, 13, 14];
        let prefill = h.rows(&full, full.len()).unwrap();
        // Decode step for the last token, carrying ngram_size - 1 ids.
        let step = h.rows(&full[1..], 1).unwrap();
        assert_eq!(step[0], prefill[3]);
    }

    #[test]
    fn rejects_buffers_that_disagree_with_the_geometry() {
        // 3 multipliers expected for ngram_size 3.
        assert!(
            NGramHasher::new(3, 2, vec![1, 3], vec![7; 4], vec![0; 4], EOS).is_err(),
            "wrong multiplier count must be rejected at load"
        );
        // (3-1)*2 = 4 heads expected.
        assert!(
            NGramHasher::new(3, 2, vec![1, 3, 5], vec![7; 3], vec![0; 3], EOS).is_err(),
            "wrong head count must be rejected at load"
        );
        assert!(
            NGramHasher::new(3, 2, vec![1, 3, 5], vec![0; 4], vec![0; 4], EOS).is_err(),
            "a zero vocab size would divide by zero"
        );
    }
}
