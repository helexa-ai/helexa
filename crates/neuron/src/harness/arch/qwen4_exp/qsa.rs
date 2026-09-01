//! QSA — the sparse-attention indexer's block selection.
//!
//! The 12 full-attention layers (and the MTP layer) do not attend over
//! every visible position. A small side-channel — 4 query heads and a
//! single key head of 128 dims, projected by `index_qk_proj` — scores
//! *blocks* of 4 consecutive visible positions, and only the best 512
//! blocks are attended. That is a 2048-token budget over a 262,144-token
//! window, which is why decode degrades only ~20% from 0 to 64k.
//!
//! This module is the selection itself: pooling the visible keys into
//! blocks, scoring them, and turning the scores into the set of
//! positions the main attention may see. It deliberately does not own
//! the indexer's KV cache, the RoPE application, or the mask — those
//! belong with the attention layer, and the selection is the part with
//! an exact test.
//!
//! ```text
//! keys [n_visible, 128]  (raw: pre-norm, pre-RoPE, straight off the cache)
//!   │
//!   ├─ complete blocks of 4 ─► mean in f32 ─► k_layernorm ─► RoPE(block start)
//!   │                                                          │
//!   │   q = RoPE(q_layernorm(q), pos)   [4, 128] ──────────────┤
//!   │                                                          ▼
//!   ├─ score = relu(q @ pooled^T).sum(heads) / sqrt(128)   [n_blocks]
//!   ├─ top-512 blocks ─► expand each to its 4 positions
//!   └─ + the n_visible mod 4 tail positions, never scored, always kept
//! ```
//!
//! Two of those steps are the ones to get right:
//!
//! 1. **The ReLU is before the head sum.** Summing signed scores lets a
//!    head that strongly rejects a block cancel one that wants it, which
//!    is a different model that still produces text.
//! 2. **The tail is unconditional.** The `n_visible mod 4` most recent
//!    positions never form a complete block, are never scored, and are
//!    always attended — including the query's own position, which is
//!    why dropping them is not a subtle degradation but a broken
//!    causal chain.
//!
//! Below the budget the selection is a no-op — every complete block is
//! selected and the tail is appended, so the result is dense causal
//! attention. That makes short-prompt parity against a dense
//! implementation an exact test rather than a tolerance, and it is the
//! cheapest correctness gate this architecture offers.
//!
//! See `doc/qwen4_exp-port-spec.md` §3.

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

/// The indexer's geometry, from `config.json`.
pub struct BlockSelector {
    n_heads: usize,
    head_dim: usize,
    /// Positions per block — `indexer_compress_ratio`, 4.
    block_size: usize,
    /// `indexer_budget` in **tokens**, not blocks.
    budget: usize,
}

impl BlockSelector {
    pub fn new(n_heads: usize, head_dim: usize, block_size: usize, budget: usize) -> Result<Self> {
        ensure!(n_heads >= 1, "indexer_n_heads must be >= 1");
        ensure!(head_dim >= 1, "indexer_head_dim must be >= 1");
        ensure!(block_size >= 1, "indexer_compress_ratio must be >= 1");
        ensure!(
            budget.is_multiple_of(block_size),
            "indexer_budget ({budget}) must be a whole number of blocks of {block_size}"
        );
        Ok(Self {
            n_heads,
            head_dim,
            block_size,
            budget,
        })
    }

    /// How many blocks may be selected: `indexer_budget /
    /// indexer_compress_ratio`, 512 for the shipped config.
    pub fn block_topk(&self) -> usize {
        self.budget / self.block_size
    }

    /// Complete blocks over `n_visible` causally visible positions. The
    /// remainder is the tail, and is not a block.
    pub fn n_blocks(&self, n_visible: usize) -> usize {
        n_visible / self.block_size
    }

    /// The position each block starts at — where its pooled key is
    /// RoPE'd, **not** the position it is scored against.
    pub fn block_starts(&self, n_visible: usize) -> Vec<usize> {
        (0..self.n_blocks(n_visible))
            .map(|b| b * self.block_size)
            .collect()
    }

    /// Mean-pool the visible keys into blocks.
    ///
    /// `keys`: `[n_visible, head_dim]`, raw off the indexer cache. The
    /// tail is dropped — it is never pooled and never scored.
    ///
    /// The mean is taken in f32 whatever the cache dtype, and the result
    /// **stays f32**. A bf16 accumulation of four keys loses the small
    /// ones outright, and casting the f32 mean straight back is no
    /// better: one bf16 tick near 0.25 is 0.002, wider than the
    /// difference the mean exists to carry. The caller normalises and
    /// RoPEs this in f32 and casts once, at the matmul.
    pub fn pool(&self, keys: &Tensor) -> candle_core::Result<Tensor> {
        let (n_visible, dim) = keys.dims2()?;
        debug_assert_eq!(dim, self.head_dim);
        let blocks = self.n_blocks(n_visible);
        if blocks == 0 {
            return Tensor::zeros((0, dim), DType::F32, keys.device());
        }
        keys.narrow(0, 0, blocks * self.block_size)?
            .to_dtype(DType::F32)?
            .reshape((blocks, self.block_size, dim))?
            .mean(1)
    }

    /// Score every block for one query position.
    ///
    /// `q`: `[n_heads, head_dim]`, already layer-normed and RoPE'd at
    /// the query's own position. `pooled`: `[n_blocks, head_dim]`,
    /// already layer-normed and RoPE'd at each block's start.
    /// Returns `[n_blocks]`.
    pub fn scores(&self, q: &Tensor, pooled: &Tensor) -> candle_core::Result<Tensor> {
        debug_assert_eq!(q.dims2()?, (self.n_heads, self.head_dim));
        let per_head = q.matmul(&pooled.t()?)?.relu()?;
        per_head.sum(0)? / (self.head_dim as f64).sqrt()
    }

    /// Turn block scores into the visible positions the main attention
    /// may see, ascending.
    ///
    /// Ties are broken towards the earlier block, which matters only for
    /// reproducibility — the reference's `topk` does the same.
    pub fn select(&self, scores: &[f32], n_visible: usize) -> Result<Vec<usize>> {
        let blocks = self.n_blocks(n_visible);
        ensure!(
            scores.len() == blocks,
            "expected {blocks} block scores for {n_visible} visible positions, got {}",
            scores.len()
        );

        let mut order: Vec<usize> = (0..blocks).collect();
        let keep = self.block_topk().min(blocks);
        if keep < blocks {
            order.sort_by(|a, b| {
                scores[*b]
                    .partial_cmp(&scores[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            order.truncate(keep);
            order.sort_unstable();
        }

        let tail_start = blocks * self.block_size;
        let mut positions = Vec::with_capacity(keep * self.block_size + (n_visible - tail_start));
        for b in order {
            let start = b * self.block_size;
            positions.extend(start..start + self.block_size);
        }
        // The tail never competes for the budget. It holds the query's
        // own position.
        positions.extend(tail_start..n_visible);
        Ok(positions)
    }

    /// Whether the selection can bind at all at this depth. Below the
    /// budget it cannot, and the layer is free to skip scoring entirely
    /// and attend densely — same answer, no indexer work.
    pub fn is_dense(&self, n_visible: usize) -> bool {
        self.n_blocks(n_visible) <= self.block_topk()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The shipped geometry, shrunk where the numbers do not matter:
    /// 4 heads of 8 dims, blocks of 4, and a budget of 3 blocks so the
    /// selection actually has to choose.
    fn small() -> BlockSelector {
        BlockSelector::new(4, 8, 4, 12).unwrap()
    }

    fn shipped() -> BlockSelector {
        BlockSelector::new(4, 128, 4, 2048).unwrap()
    }

    #[test]
    fn shipped_geometry_selects_512_blocks() {
        let qsa = shipped();
        assert_eq!(qsa.block_topk(), 512);
        // 2048 tokens of budget over the full window.
        assert_eq!(qsa.block_topk() * 4, 2048);
    }

    /// The cheapest gate in the architecture: below the budget, the
    /// selection returns every visible position, so a short prompt must
    /// be bit-comparable to dense causal attention.
    #[test]
    fn below_budget_selection_is_dense() {
        let qsa = shipped();
        for n_visible in [1usize, 3, 4, 7, 2047, 2048] {
            assert!(qsa.is_dense(n_visible), "{n_visible} should be dense");
            let scores = vec![0.0f32; qsa.n_blocks(n_visible)];
            let got = qsa.select(&scores, n_visible).unwrap();
            assert_eq!(
                got,
                (0..n_visible).collect::<Vec<_>>(),
                "{n_visible} visible positions should all be attended"
            );
        }
        // One block past the budget it can bind.
        assert!(!qsa.is_dense(2052));
    }

    /// The tail is not scored and not budgeted. Here every block is
    /// beaten out of a full budget, and the three tail positions —
    /// including the query's own — must still come back.
    #[test]
    fn the_tail_is_always_attended() {
        let qsa = small();
        // 19 visible: 4 complete blocks (16 positions) + a 3-position tail.
        let n_visible = 19;
        assert_eq!(qsa.n_blocks(n_visible), 4);
        // Block 2 is worthless; with topk 3 it is the one dropped.
        let scores = [0.9f32, 0.8, 0.1, 0.7];
        let got = qsa.select(&scores, n_visible).unwrap();
        let want: Vec<usize> = (0..8).chain(12..19).collect();
        assert_eq!(got, want);
        assert!(got.contains(&(n_visible - 1)), "the query's own position");
    }

    /// ReLU before the head sum. Head 0 likes block 0 a little; head 1
    /// hates it a lot. A signed sum ranks block 0 last; the reference
    /// ranks it first, because rejection is clamped to indifference.
    #[test]
    fn relu_precedes_the_head_sum() {
        let dev = Device::Cpu;
        let qsa = BlockSelector::new(2, 1, 4, 8).unwrap();
        let q = Tensor::from_vec(vec![1.0f32, 1.0], (2, 1), &dev).unwrap();
        // block 0: +2 for head 0, -5 for head 1.  block 1: +1, +1.
        let pooled = Tensor::from_vec(vec![2.0f32, 1.0], (2, 1), &dev).unwrap();
        let scores: Vec<f32> = qsa.scores(&q, &pooled).unwrap().to_vec1().unwrap();
        // Both heads share the query here, so the point is made by the
        // per-head clamp: 2 + 2 = 4 beats 1 + 1 = 2.
        assert!(scores[0] > scores[1]);

        // Now give head 1 the opposite sign and check the clamp holds.
        let q = Tensor::from_vec(vec![1.0f32, -2.5], (2, 1), &dev).unwrap();
        let scores: Vec<f32> = qsa.scores(&q, &pooled).unwrap().to_vec1().unwrap();
        // head 0: relu(2) = 2, head 1: relu(-5) = 0  ->  2
        // signed would be 2 - 5 = -3, which loses to block 1's -1.5.
        assert!(
            (scores[0] - 2.0).abs() < 1e-6,
            "expected the negative head clamped away, got {scores:?}"
        );
        assert!(scores[0] > scores[1], "got {scores:?}");
    }

    /// Pooling accumulates in f32 even when the cache is bf16, and keeps
    /// the result there. Four keys whose sum a bf16 accumulator cannot
    /// represent: 1.0 + 3 x 0.001 rounds straight back to 1.0, so a bf16
    /// mean is exactly 0.25 — and so is an f32 mean cast back to bf16,
    /// because the true 0.2507 sits below the first bf16 tick above
    /// 0.25. Both mistakes land on the same wrong number, which is why
    /// this asserts the value rather than the dtype.
    #[test]
    fn pooling_accumulates_in_f32() {
        let dev = Device::Cpu;
        let qsa = BlockSelector::new(4, 1, 4, 8).unwrap();
        let keys = Tensor::from_vec(vec![1.0f32, 0.001, 0.001, 0.001], (4, 1), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let pooled_t = qsa.pool(&keys).unwrap();
        assert_eq!(pooled_t.dtype(), DType::F32);
        let pooled: Vec<f32> = pooled_t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            (pooled[0] - 0.25).abs() > 1e-4,
            "a bf16 accumulation, or an f32 one cast back, gives exactly \
             0.25; got {pooled:?}"
        );
        assert!((pooled[0] - 0.2507).abs() < 1e-3, "got {pooled:?}");
    }

    /// The tail is dropped from pooling, and a prompt shorter than one
    /// block pools nothing at all.
    #[test]
    fn pooling_drops_the_tail() {
        let dev = Device::Cpu;
        let qsa = small();
        let keys = Tensor::zeros((11, 8), DType::BF16, &dev).unwrap();
        assert_eq!(qsa.pool(&keys).unwrap().dims(), &[2, 8]);
        assert_eq!(qsa.block_starts(11), vec![0, 4]);

        let short = Tensor::zeros((3, 8), DType::F32, &dev).unwrap();
        assert_eq!(qsa.pool(&short).unwrap().dims(), &[0, 8]);
        assert!(qsa.block_starts(3).is_empty());
    }

    #[test]
    fn selection_is_ascending_and_rejects_a_wrong_score_count() {
        let qsa = small();
        let got = qsa.select(&[0.1f32, 0.9, 0.5, 0.7, 0.2], 23).unwrap();
        assert!(got.windows(2).all(|w| w[0] < w[1]), "got {got:?}");
        // blocks 1, 3, 2 win; block 0 and 4 lose; tail 20..23 stays.
        assert_eq!(
            got,
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22]
        );
        assert!(qsa.select(&[0.1f32], 23).is_err());
    }
}
