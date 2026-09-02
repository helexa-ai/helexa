//! PLE — per-layer embedding over hashed n-grams.
//!
//! `qwen4_exp` puts one of these on zero-indexed layer **1**
//! (`ple_layer_ids: [2]` is one-indexed upstream), and it accounts for
//! **28.44% of the model's parameters** — a 320,001,536-row table that
//! each token touches 16 rows of.
//!
//! PLE is three separable pieces. [`NGramHasher`] is the *addressing*:
//! token ids to row indices. [`PleBlock`] is the *consumption*: the
//! gated mix of the gathered embedding into the four residual streams.
//! Between them sits the gather, behind the [`NGramTable`] trait —
//! *where* those bytes live is #310's decision, and the trait is the
//! seam so that decision changes one implementation and nothing else.
//! [`ShardedNGramTable`] is the naive reading of the checkpoint, which
//! is the baseline that issue measures against rather than its
//! answer.
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

use anyhow::{Context, Result, ensure};
use candle_core::{D, Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use crate::harness::arch::qwen3_5::rmsnorm::Qwen3_5RmsNorm;

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

/// Where the n-gram table's rows come from.
///
/// 320,001,536 rows of 160, and every token needs sixteen of them —
/// a few KB per token out of 27 GB quantised. That ratio is why the
/// table can live off-device at almost no cost, and why #310 is a
/// residency decision rather than an algorithm. Nothing above this
/// trait depends on the answer.
pub trait NGramTable: Send + Sync {
    /// Gather `rows`, **in the order given**, as
    /// `[rows.len(), head_dim]`.
    fn gather(&self, rows: &[i64]) -> Result<Tensor>;

    /// Width of one row — `ple_embed_dim / ngram_heads`, 160.
    fn head_dim(&self) -> usize;
}

/// The checkpoint's 128 shards, held wherever they were loaded.
///
/// The shards are a **flat split of one concatenated table**, not a
/// split per head: shard `s` holds rows `[s * rows_per_shard,
/// (s+1) * rows_per_shard)`, so a global row index from the hash
/// addresses straight into it. They are kept separate rather than
/// concatenated because joining 128 tensors of 2,500,012 x 160 would
/// need the whole table free a second time.
///
/// This is the naive residency — the bytes end up wherever the
/// VarBuilder put them, which for a 27 GB table is exactly the problem
/// #310 exists to solve. It is the control that issue measures its
/// host- and NVMe-resident variants against.
pub struct ShardedNGramTable {
    shards: Vec<Tensor>,
    rows_per_shard: usize,
    head_dim: usize,
}

impl ShardedNGramTable {
    /// `vb` should be `.pp(...)`-ed to `ple_embedding.ngram_embedding`.
    pub fn load(
        vb: &ShardedVarBuilder,
        parts: usize,
        rows: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(parts > 0, "split_ngram_parts must be > 0");
        ensure!(
            rows.is_multiple_of(parts),
            "{rows} rows do not divide evenly into {parts} shards"
        );
        let rows_per_shard = rows / parts;
        let mut shards = Vec::with_capacity(parts);
        for s in 0..parts {
            shards.push(
                vb.pp(format!("shard_{s}"))
                    .get((rows_per_shard, head_dim), "weight")
                    .with_context(|| format!("load '{}/shard_{s}/weight'", vb.prefix()))?,
            );
        }
        Ok(Self {
            shards,
            rows_per_shard,
            head_dim,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_shards(shards: Vec<Tensor>) -> Result<Self> {
        ensure!(!shards.is_empty(), "need at least one shard");
        let (rows_per_shard, head_dim) = shards[0].dims2()?;
        Ok(Self {
            shards,
            rows_per_shard,
            head_dim,
        })
    }

    fn shard_of(&self, row: i64) -> usize {
        row as usize / self.rows_per_shard
    }
}

impl NGramTable for ShardedNGramTable {
    fn gather(&self, rows: &[i64]) -> Result<Tensor> {
        let total = (self.shards.len() * self.rows_per_shard) as i64;
        if let Some(bad) = rows.iter().find(|r| **r < 0 || **r >= total) {
            anyhow::bail!("row {bad} is outside the table's 0..{total}");
        }
        if rows.is_empty() {
            let device = self.shards[0].device();
            return Ok(Tensor::zeros(
                (0, self.head_dim),
                self.shards[0].dtype(),
                device,
            )?);
        }

        // Visit each shard once: sort the requests by shard, gather, then
        // undo the permutation. Getting that inverse wrong hands every
        // token somebody else's n-gram, which is not a crash.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| (self.shard_of(rows[i]), i));

        let device = self.shards[0].device();
        let mut parts = Vec::new();
        let mut i = 0;
        while i < order.len() {
            let shard = self.shard_of(rows[order[i]]);
            let mut within = Vec::new();
            while i < order.len() && self.shard_of(rows[order[i]]) == shard {
                within.push((rows[order[i]] as usize - shard * self.rows_per_shard) as u32);
                i += 1;
            }
            let idx = Tensor::from_vec(within.clone(), within.len(), device)?;
            parts.push(self.shards[shard].index_select(&idx, 0)?);
        }
        let gathered = Tensor::cat(&parts, 0)?;

        let mut inverse = vec![0u32; rows.len()];
        for (position, &original) in order.iter().enumerate() {
            inverse[original] = position as u32;
        }
        let inverse = Tensor::from_vec(inverse, rows.len(), device)?;
        Ok(gathered.index_select(&inverse, 0)?)
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// The consumption half of PLE: `Qwen4ExpTextPLE` minus the table.
///
/// Given the gathered n-gram embedding `e` and the four-stream residual
/// `h`, this produces the `hidden * hc_count` tensor that layer 1 adds
/// back into `h`. It is a gated read: the embedding proposes a value,
/// the residual streams decide per-stream how much of it to take.
///
/// ```text
/// k    = norm_key(key_proj(e))      -> [.., hc, hidden]
/// q    = norm_query(h)              -> [.., hc, hidden]
/// v    = value_proj(e)              -> [.., hidden]
/// g    = (k * q).sum(-1) / sqrt(hidden)               [.., hc, 1]
/// g    = sign(g) * sqrt(max(|g|, 1e-6))               signed sqrt
/// gv   = sigmoid(g) * v                               [.., hc, hidden]
/// out  = gv.flatten(-2) + short_conv(norm_conv(gv.flatten(-2)))
/// ```
///
/// Two things here are not what they look like:
///
/// 1. **`short_conv` is dilated.** `dilation = ngram_size = 3`, so its
///    four taps reach 9 positions back, not 3, and the cached left
///    context is 9 wide rather than `kernel_size`. A plain causal conv
///    runs, trains nothing, and mixes the wrong neighbours — the test
///    below moves a single position and asserts where it lands.
/// 2. **The conv is a residual branch, not the output.** `gv` passes
///    through whether or not the conv contributes.
///
/// All three norms are grouped over `hidden_size` (§8): four
/// independent normalisations of a 10240-wide vector, not one.
/// See `doc/qwen4_exp-port-spec.md` §2.
pub struct PleBlock {
    key_proj: Linear,
    value_proj: Linear,
    norm_key: Qwen3_5RmsNorm,
    norm_query: Qwen3_5RmsNorm,
    norm_conv: Qwen3_5RmsNorm,
    /// Depthwise: `(hidden * hc_count, 1, kernel_size)`.
    conv1d_weight: Tensor,
    /// Rolling left context for the dilated conv,
    /// `(B, hidden * hc_count, (kernel_size - 1) * dilation)`. Upstream
    /// keeps this as `conv_states[1]` on layer 1.
    conv_state: Option<Tensor>,
    hidden_size: usize,
    hc_count: usize,
    kernel_size: usize,
    dilation: usize,
}

impl PleBlock {
    /// `vb` should be `.pp(...)`-ed to the layer's `ple` prefix.
    pub fn load(
        vb: &ShardedVarBuilder,
        hidden_size: usize,
        hc_count: usize,
        kernel_size: usize,
        dilation: usize,
        eps: f64,
    ) -> Result<Self> {
        ensure!(kernel_size >= 1, "ple conv kernel_size must be >= 1");
        ensure!(dilation >= 1, "ple conv dilation must be >= 1");
        let wide = hidden_size * hc_count;
        let conv1d_weight = vb
            .pp("conv1d")
            .get((wide, 1, kernel_size), "weight")
            .with_context(|| format!("load '{}/conv1d/weight'", vb.prefix()))?;
        Ok(Self {
            key_proj: linear_no_bias(vb, "key_proj", hidden_size, wide)?,
            value_proj: linear_no_bias(vb, "value_proj", hidden_size, hidden_size)?,
            norm_key: Qwen3_5RmsNorm::load_grouped(&vb.pp("norm_key"), wide, hidden_size, eps)?,
            norm_query: Qwen3_5RmsNorm::load_grouped(&vb.pp("norm_query"), wide, hidden_size, eps)?,
            norm_conv: Qwen3_5RmsNorm::load_grouped(&vb.pp("norm_conv"), wide, hidden_size, eps)?,
            conv1d_weight,
            conv_state: None,
            hidden_size,
            hc_count,
            kernel_size,
            dilation,
        })
    }

    /// `h`: `[B, L, hidden * hc]` — the residual streams.
    /// `ngram_embed`: `[B, L, hidden]` — the gathered n-gram rows.
    /// Returns `[B, L, hidden * hc]`, to be **added** to `h`.
    pub fn forward(&mut self, h: &Tensor, ngram_embed: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, wide) = h.dims3()?;
        debug_assert_eq!(wide, self.hidden_size * self.hc_count);
        let streams = (b, l, self.hc_count, self.hidden_size);

        let k = self
            .norm_key
            .forward(&self.key_proj.forward(ngram_embed)?)?
            .reshape(streams)?;
        let q = self.norm_query.forward(h)?.reshape(streams)?;
        let v = self.value_proj.forward(ngram_embed)?;

        // Per-stream agreement between the proposed key and the stream's
        // own state, scaled like an attention logit.
        let g = (k.mul(&q)?.sum_keepdim(D::Minus1)? / (self.hidden_size as f64).sqrt())?;
        // Signed sqrt: compresses the magnitude without losing the sign,
        // so a strongly disagreeing stream still shuts the gate.
        let g = g.sign()?.mul(&g.abs()?.maximum(1e-6)?.sqrt()?)?;

        let gv = candle_nn::ops::sigmoid(&g)?
            .broadcast_mul(&v.unsqueeze(D::Minus2)?)?
            .flatten_from(D::Minus2)?;

        let conv = self.short_conv(&self.norm_conv.forward(&gv)?)?;
        gv + conv
    }

    /// Depthwise dilated causal conv + SiLU, with rolling left context.
    ///
    /// Unlike the GatedDeltaNet short conv this cannot use
    /// `run_causal_conv1d`: that path (and the fused cuda kernel behind
    /// it) is hard-wired to `dilation = 1`.
    fn short_conv(&mut self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, wide) = x.dims3()?;
        let reach = self.left_context();
        // (B, wide, L) for the conv.
        let x = x.transpose(1, 2)?.contiguous()?;

        let prepended = match self.conv_state.take() {
            Some(prev) => Tensor::cat(&[&prev, &x], 2)?,
            None => x,
        };
        let prep_len = prepended.dim(2)?;

        self.conv_state = Some(if prep_len >= reach {
            prepended.narrow(2, prep_len - reach, reach)?.contiguous()?
        } else {
            let pad = Tensor::zeros(
                (b, wide, reach - prep_len),
                prepended.dtype(),
                prepended.device(),
            )?;
            Tensor::cat(&[&pad, &prepended], 2)?
        });

        // Pad both sides by the full reach, then keep the left-aligned
        // window: output t sees inputs t - j*dilation only.
        let out = prepended.conv1d(&self.conv1d_weight, reach, 1, self.dilation, wide)?;
        let out = candle_nn::ops::silu(&out.narrow(2, 0, prep_len)?)?;
        out.narrow(2, prep_len - l, l)?
            .transpose(1, 2)?
            .contiguous()
    }

    /// How many positions of left context the dilated conv reaches back
    /// over — `(kernel_size - 1) * dilation`, 9 for the shipped config.
    /// This, not `kernel_size`, is the cached state width.
    pub fn left_context(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    /// Capture the rolling conv context for a prefix snapshot.
    ///
    /// Deep-copied rather than shared. The buffer is replaced by
    /// assignment today, so a shallow clone would be sound — but
    /// `narrow(..).contiguous()` on an already-contiguous slice is a
    /// view, so whether the snapshot shares storage depends on a
    /// property of the *data*, not of the code. That is exactly the
    /// kind of conditional aliasing the GatedDeltaNet states are
    /// deep-copied to avoid, and nine positions of a 10240-wide signal
    /// is not worth being clever about.
    pub fn snapshot_state(&self) -> candle_core::Result<Option<Tensor>> {
        self.conv_state.as_ref().map(Tensor::copy).transpose()
    }

    /// Replace the rolling conv context from a snapshot.
    pub fn restore_state(&mut self, conv_state: Option<&Tensor>) -> candle_core::Result<()> {
        self.conv_state = conv_state.map(Tensor::copy).transpose()?;
        Ok(())
    }

    /// Drop the rolling conv context. Call alongside a KV-cache clear;
    /// the carried n-gram ids ([`NGramHasher::context_len`]) reset with
    /// it.
    pub fn clear_state(&mut self) {
        self.conv_state = None;
    }
}

fn linear_no_bias(
    vb: &ShardedVarBuilder,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
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

    // ---- consumption ----

    use candle_core::{DType, Device};

    /// A `PleBlock` whose projections and norms are supplied directly,
    /// so each test can make the parts it is not measuring analytic.
    /// All three norm weights are zero, which makes the `(1 + w)` scale
    /// exactly 1 and leaves a pure RMS normalisation.
    fn block(
        hidden: usize,
        hc: usize,
        kernel_size: usize,
        dilation: usize,
        key_w: Tensor,
        value_w: Tensor,
        conv_w: Tensor,
    ) -> PleBlock {
        let dev = Device::Cpu;
        let wide = hidden * hc;
        let norm = || {
            Qwen3_5RmsNorm::from_weight(
                Tensor::zeros(wide, DType::F32, &dev).unwrap(),
                1e-6,
                Some(hidden),
            )
        };
        PleBlock {
            key_proj: Linear::new(key_w, None),
            value_proj: Linear::new(value_w, None),
            norm_key: norm(),
            norm_query: norm(),
            norm_conv: norm(),
            conv1d_weight: conv_w,
            conv_state: None,
            hidden_size: hidden,
            hc_count: hc,
            kernel_size,
            dilation,
        }
    }

    fn eye(n: usize) -> Tensor {
        let dev = Device::Cpu;
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        Tensor::from_vec(v, (n, n), &dev).unwrap()
    }

    /// The gate, end to end, against arithmetic done by hand.
    ///
    /// With identity projections and zero norm weights, `k` and `q` both
    /// normalise to all-ones over four dims, so the dot product is 4 and
    /// the scale divides it by sqrt(4) = 2. The signed sqrt then takes
    /// that 2 to sqrt(2) — the step a plain `g` would skip, and 2 vs
    /// sqrt(2) is 0.88 vs 0.80 through the sigmoid, which is exactly the
    /// kind of difference that reads as "slightly worse model".
    #[test]
    fn gate_matches_hand_computed_forward() {
        let dev = Device::Cpu;
        let hidden = 4;
        let mut ple = block(
            hidden,
            1,
            2,
            3,
            eye(hidden),
            eye(hidden),
            // Zero conv weight: silu(0) = 0, so the residual branch
            // contributes nothing and `out` is the gate alone.
            Tensor::zeros((hidden, 1, 2), DType::F32, &dev).unwrap(),
        );

        let e = Tensor::new(&[[[1.0f32, 1.0, 1.0, 1.0]]], &dev).unwrap();
        // Magnitude is irrelevant — norm_query normalises it away, which
        // is why the gate is bounded no matter how large the streams get.
        let h = Tensor::new(&[[[2.0f32, 2.0, 2.0, 2.0]]], &dev).unwrap();

        let got: Vec<f32> = ple
            .forward(&h, &e)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let want = 1.0 / (1.0 + (-(2.0f32.sqrt())).exp());
        assert_eq!(got.len(), hidden);
        for g in &got {
            assert!(
                (g - want).abs() < 1e-5,
                "gate: got {got:?} want {want} in every channel"
            );
        }
    }

    /// The conv is a residual branch. Zeroing the key projection shuts
    /// the gate to sigmoid(0) = 0.5 and the output is then exactly half
    /// the value projection, broadcast across every stream — which also
    /// pins the flatten order (stream-major, hidden-minor).
    #[test]
    fn value_is_broadcast_across_streams_and_survives_a_dead_conv() {
        let dev = Device::Cpu;
        let (hidden, hc) = (2, 2);
        let mut ple = block(
            hidden,
            hc,
            2,
            3,
            Tensor::zeros((hidden * hc, hidden), DType::F32, &dev).unwrap(),
            eye(hidden),
            Tensor::zeros((hidden * hc, 1, 2), DType::F32, &dev).unwrap(),
        );

        let e = Tensor::new(&[[[3.0f32, 5.0]]], &dev).unwrap();
        let h = Tensor::new(&[[[1.0f32, 1.0, 7.0, 7.0]]], &dev).unwrap();

        let got: Vec<f32> = ple
            .forward(&h, &e)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want = [1.5f32, 2.5, 1.5, 2.5];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
        }
    }

    /// The conv reaches `(kernel_size - 1) * dilation` back, not
    /// `kernel_size - 1`. A single impulse is placed at position 0 and
    /// the far tap is the only live weight, so the response must appear
    /// at position `dilation`. With the dilation dropped it would land
    /// at position 1 — the whole difference between mixing an n-gram's
    /// neighbours and mixing the token next door.
    #[test]
    fn short_conv_is_dilated_not_merely_causal() {
        let dev = Device::Cpu;
        let (wide, dilation) = (1, 3);
        // Taps are [far, near]: output t = w[0]*x[t - dilation] + w[1]*x[t].
        let conv_w = Tensor::from_vec(vec![1.0f32, 0.0], (wide, 1, 2), &dev).unwrap();
        let mut ple = block(1, 1, 2, dilation, eye(1), eye(1), conv_w);

        let x = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 0.0], (1, 5, wide), &dev).unwrap();
        let got: Vec<f32> = ple
            .short_conv(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let want = [0.0, 0.0, 0.0, silu1, 0.0];
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "position {i}: got {got:?} want {want:?}"
            );
        }
        assert_eq!(ple.left_context(), dilation);
    }

    /// Decoding one token at a time must give the same answer as running
    /// the whole sequence at once — the property the rolling state
    /// exists for, and the one that breaks silently when the cached
    /// context is `kernel_size` wide instead of the full reach.
    #[test]
    fn decode_step_by_step_matches_prefill() {
        let dev = Device::Cpu;
        let (wide, dilation) = (2, 3);
        let conv_w = Tensor::from_vec(vec![0.5f32, -0.25, 1.5, 0.75], (wide, 1, 2), &dev).unwrap();
        let mk = || block(2, 1, 2, dilation, eye(2), eye(2), conv_w.clone());

        let seq: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let x = Tensor::from_vec(seq.clone(), (1, 8, wide), &dev).unwrap();

        let mut whole = mk();
        let want: Vec<f32> = whole
            .short_conv(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut stepwise = mk();
        let mut got = Vec::new();
        for t in 0..8 {
            let step = x.narrow(1, t, 1).unwrap().contiguous().unwrap();
            got.extend(
                stepwise
                    .short_conv(&step)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
        }

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "position {i}: got {g} want {w}");
        }

        // And a cleared block starts over rather than reading the tail of
        // the previous request.
        stepwise.clear_state();
        let first = stepwise
            .short_conv(&x.narrow(1, 0, 1).unwrap().contiguous().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            (first[0] - want[0]).abs() < 1e-5,
            "clear_state should return the block to a fresh prefill"
        );
    }

    // ---- the table ----

    /// Rows come back in the order asked for, not the order they were
    /// fetched in. Two shards of three rows, requested out of order and
    /// across the boundary: the shard-major gather returns them sorted,
    /// so the inverse permutation is the whole test. Getting it wrong
    /// hands every token somebody else's n-gram, silently.
    #[test]
    fn a_gather_preserves_the_order_it_was_asked_for() {
        let dev = Device::Cpu;
        // Row r holds [r, r] so a row is identifiable by its contents.
        let shard = |base: f32| {
            Tensor::from_vec(
                vec![base, base, base + 1.0, base + 1.0, base + 2.0, base + 2.0],
                (3, 2),
                &dev,
            )
            .unwrap()
        };
        let table = ShardedNGramTable::from_shards(vec![shard(0.0), shard(3.0)]).unwrap();
        assert_eq!(table.head_dim(), 2);

        let got: Vec<f32> = table
            .gather(&[4, 1, 3, 0, 5])
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got, vec![4.0, 4.0, 1.0, 1.0, 3.0, 3.0, 0.0, 0.0, 5.0, 5.0]);
    }

    /// Repeated rows are legitimate — sixteen heads can collide — and
    /// must each come back.
    #[test]
    fn a_repeated_row_is_returned_once_per_request() {
        let dev = Device::Cpu;
        let shard = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (2, 2), &dev).unwrap();
        let table = ShardedNGramTable::from_shards(vec![shard]).unwrap();
        let got: Vec<f32> = table
            .gather(&[1, 1, 0])
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got, vec![2.0, 3.0, 2.0, 3.0, 0.0, 1.0]);
    }

    /// A row index past the end is a hashing bug upstream, and must not
    /// become a panic in the middle of a forward.
    #[test]
    fn a_row_outside_the_table_is_an_error() {
        let dev = Device::Cpu;
        let shard = Tensor::zeros((2, 2), DType::F32, &dev).unwrap();
        let table = ShardedNGramTable::from_shards(vec![shard]).unwrap();
        assert!(table.gather(&[0, 1]).is_ok());
        assert!(table.gather(&[2]).is_err());
        assert!(table.gather(&[-1]).is_err());
        assert_eq!(table.gather(&[]).unwrap().dims(), &[0, 2]);
    }

    /// The hasher and the table meet here: sixteen row indices per
    /// position, gathered and flattened into the 2560-wide embedding the
    /// block consumes. The widths have to agree or the projections do
    /// not fit.
    #[test]
    fn hashed_rows_gather_into_one_embedding_per_position() {
        let dev = Device::Cpu;
        let heads = 4;
        let head_dim = 2;
        let rows = 64;
        let shard = Tensor::from_vec(
            (0..rows * head_dim).map(|v| v as f32).collect::<Vec<_>>(),
            (rows, head_dim),
            &dev,
        )
        .unwrap();
        let table = ShardedNGramTable::from_shards(vec![shard]).unwrap();

        let hasher = hasher(3, heads / 2);
        let ids = [7i64, 11, 13];
        let per_position = hasher.rows(&ids, ids.len()).unwrap();
        assert_eq!(per_position.len(), 3);
        assert_eq!(per_position[0].len(), heads);

        let flat: Vec<i64> = per_position
            .iter()
            .flatten()
            .map(|r| r.rem_euclid(rows as i64))
            .collect();
        let gathered = table.gather(&flat).unwrap();
        assert_eq!(gathered.dims(), &[ids.len() * heads, head_dim]);
        // Reshaped per position, that is the ple_embed_dim-wide row.
        assert_eq!(
            gathered
                .reshape((ids.len(), heads * head_dim))
                .unwrap()
                .dims(),
            &[3, heads * head_dim]
        );
    }
}
