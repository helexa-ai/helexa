//! QSA — the sparse-attention indexer's block selection.
//!
//! The 12 full-attention layers (and the MTP layer) do not attend over
//! every visible position. A small side-channel — 4 query heads and a
//! single key head of 128 dims, projected by `index_qk_proj` — scores
//! *blocks* of 4 consecutive visible positions, and only the best 512
//! blocks are attended. That is a 2048-token budget over a 262,144-token
//! window, which is why decode degrades only ~20% from 0 to 64k.
//!
//! [`BlockSelector`] is the selection arithmetic: pooling visible keys
//! into blocks, scoring them, and turning the scores into a position
//! set. [`Indexer`] wraps it in the layer's own machinery — the fused
//! `index_qk_proj`, the two layer norms, the rotation, and the second
//! KV cache the epic's per-token arithmetic omitted. The mask itself
//! stays with the attention layer.
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

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, IndexOp, Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use crate::harness::arch::qwen3_5::rmsnorm::Qwen3_5RmsNorm;
use crate::harness::arch::qwen3_5::rope::{RotaryEmbedding, apply_partial_rotary};

use super::config::TextConfig as Qwen4ExpTextConfig;

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

/// The indexer as a layer: projection, norms, rotation, and its own KV
/// cache, wrapped around [`BlockSelector`].
///
/// ```text
/// hidden [B,L,2560] ─ index_qk_proj ─┬─ q  [B,L,4,128] ─ q_layernorm ─ RoPE(own pos)
///                                    └─ k  [B,L,128] ─────────────────► cache (raw)
///                                                                        │
///        pooled blocks ◄─ RoPE(block start) ◄─ k_layernorm ◄─ mean(f32) ◄┘
/// ```
///
/// Three orderings matter and none of them are guessable from the
/// shapes:
///
/// 1. **The cached key is raw** — pre-norm and pre-RoPE. Normalising or
///    rotating before the cache would rotate each key at its own
///    position, but a pooled block is rotated once at the block's
///    *first* position. Caching post-RoPE quietly makes every block's
///    position wrong except its first member's.
/// 2. **Pooling comes before the norm**, not after: `mean` then
///    `k_layernorm`, so the block's key is normalised as a block.
/// 3. **`index_qk_proj` is fused, and the split is not a halving.** The
///    first `n_heads * head_dim` output channels are the queries and
///    the last `kv_heads * head_dim` are the single index key — 512 and
///    128 of 640, not 320 and 320.
///
/// This is the second KV cache: 12 layers x 1 head x 128 dims, about
/// 3 KiB/token at bf16 on top of the main 24 KiB. llama.cpp measures
/// 2304 MB of it beside 6144 MB of main cache at the full 262k window,
/// so it is a ~27% surcharge and belongs in any `kv_budget_mb`
/// arithmetic (#310, #315).
pub struct Indexer {
    /// Fused `index_qk_proj`: `[n_heads * head_dim + kv_heads * head_dim, hidden]`.
    qk_proj: Linear,
    q_layernorm: Qwen3_5RmsNorm,
    k_layernorm: Qwen3_5RmsNorm,
    selector: BlockSelector,
    n_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Raw index keys, `(B, T, kv_heads * head_dim)`. Pre-norm,
    /// pre-RoPE — see the note above.
    key_cache: Option<Tensor>,
}

impl Indexer {
    /// `vb` should be `.pp(...)`-ed to the attention layer's prefix, so
    /// the tensors resolve as `index_qk_proj`, `q_layernorm`,
    /// `k_layernorm`. Every dimension comes from the config rather than
    /// the caller, so a layer cannot be built against a geometry the
    /// checkpoint does not declare.
    pub fn load(vb: &ShardedVarBuilder, cfg: &Qwen4ExpTextConfig) -> Result<Self> {
        let (n_heads, kv_heads) = (cfg.indexer_n_heads, cfg.indexer_kv_heads);
        let head_dim = cfg.indexer_head_dim;
        let fused = (n_heads + kv_heads) * head_dim;
        let weight = vb
            .pp("index_qk_proj")
            .get((fused, cfg.hidden_size), "weight")
            .with_context(|| format!("load '{}/index_qk_proj/weight'", vb.prefix()))?;
        Ok(Self {
            qk_proj: Linear::new(weight, None),
            q_layernorm: Qwen3_5RmsNorm::load(&vb.pp("q_layernorm"), head_dim, cfg.rms_norm_eps)?,
            k_layernorm: Qwen3_5RmsNorm::load(&vb.pp("k_layernorm"), head_dim, cfg.rms_norm_eps)?,
            selector: BlockSelector::new(
                n_heads,
                head_dim,
                cfg.indexer_compress_ratio,
                cfg.indexer_budget,
            )?,
            n_heads,
            kv_heads,
            head_dim,
            key_cache: None,
        })
    }

    pub fn selector(&self) -> &BlockSelector {
        &self.selector
    }

    /// Drop the indexer's cache. Must happen with the main KV clear —
    /// a stale indexer cache selects blocks of another request's text.
    pub fn clear_cache(&mut self) {
        self.key_cache = None;
    }

    /// How many positions the indexer has seen.
    pub fn cached_len(&self) -> usize {
        self.key_cache.as_ref().map_or(0, |c| c.dims()[1])
    }

    /// Split the fused projection into normed queries and raw keys.
    ///
    /// Returns `q` as `(B, n_heads, L, head_dim)` — normed, unrotated —
    /// and `k` as `(B, L, kv_heads * head_dim)`, untouched.
    fn project(&self, hidden: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let (b, l, _) = hidden.dims3()?;
        let qk = self.qk_proj.forward(hidden)?;
        let q_width = self.n_heads * self.head_dim;
        let q = qk
            .narrow(D::Minus1, 0, q_width)?
            .reshape((b, l, self.n_heads, self.head_dim))?;
        let q = self
            .q_layernorm
            .forward(&q)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qk
            .narrow(D::Minus1, q_width, self.kv_heads * self.head_dim)?
            .contiguous()?;
        Ok((q, k))
    }

    /// Pool the cache into blocks, norm them, and rotate each at its own
    /// block's first position. `(B, 1, n_blocks, head_dim)`.
    fn block_keys(
        &self,
        cache: &Tensor,
        rope: &RotaryEmbedding,
        total: usize,
    ) -> candle_core::Result<Tensor> {
        let b = cache.dims()[0];
        let starts = self.selector.block_starts(total);
        let mut pooled = Vec::with_capacity(b);
        for row in 0..b {
            pooled.push(self.selector.pool(&cache.i(row)?)?);
        }
        let pooled = Tensor::stack(&pooled, 0)?.to_dtype(cache.dtype())?;
        let normed = self.k_layernorm.forward(&pooled)?.unsqueeze(1)?;
        let (cos, sin) = rope.cos_sin_at(&starts)?;
        apply_partial_rotary(&normed, &cos, &sin, rope.rotary_dim())
    }

    /// The positions each query may attend, as `[batch][query]`.
    ///
    /// `past_len` is the query's sequence offset, so query `i` sits at
    /// absolute position `past_len + i` and sees `past_len + i + 1`
    /// positions.
    ///
    /// The pooling and scoring are batched — one matmul covers every
    /// query against every block — because a block's pooled key does not
    /// depend on which query is looking at it. Only the top-k runs per
    /// query, over the prefix of blocks that query can see. The
    /// reference is a double loop over batch x query position; it is an
    /// oracle, not a design.
    pub fn visible_positions(
        &mut self,
        hidden: &Tensor,
        rope: &RotaryEmbedding,
        past_len: usize,
    ) -> Result<Vec<Vec<Vec<usize>>>> {
        let (batch, seq_len, _) = hidden.dims3()?;
        let (q, k) = self.project(hidden)?;

        let cache = match self.key_cache.take() {
            Some(prev) => Tensor::cat(&[&prev, &k], 1)?,
            None => k,
        };
        let total = cache.dims()[1];
        ensure!(
            total == past_len + seq_len,
            "indexer cache holds {total} positions but the queries start at {past_len} \
             and run for {seq_len} — the cache and the main KV have diverged"
        );
        self.key_cache = Some(cache.clone());

        // Queries rotate at their own positions, which are contiguous.
        let (cos, sin) = rope.plain_cos_sin(past_len, seq_len)?;
        let q = apply_partial_rotary(&q, &cos, &sin, rope.rotary_dim())?;

        let n_blocks = self.selector.n_blocks(total);
        if n_blocks == 0 {
            // Nothing is poolable yet; every query is all tail.
            return Ok((0..batch)
                .map(|_| (0..seq_len).map(|i| (0..=past_len + i).collect()).collect())
                .collect());
        }
        let blocks = self.block_keys(&cache, rope, total)?;

        // relu(q . k) summed over heads, for every (query, block) pair
        // in one batched matmul: every head and query shares the same
        // block keys, so fold them into the row axis rather than
        // broadcasting the keys across heads.
        let q_rows = q.reshape((batch, self.n_heads * seq_len, self.head_dim))?;
        let keys = blocks.squeeze(1)?.transpose(1, 2)?.contiguous()?;
        let per_head = q_rows
            .matmul(&keys)?
            .reshape((batch, self.n_heads, seq_len, n_blocks))?
            .relu()?;
        let scores = (per_head.sum(1)? / (self.head_dim as f64).sqrt())?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;

        let mut out = Vec::with_capacity(batch);
        for row in scores.iter().take(batch) {
            let mut per_query = Vec::with_capacity(seq_len);
            for (i, all_blocks) in row.iter().enumerate().take(seq_len) {
                let visible = past_len + i + 1;
                // Only blocks wholly behind this query exist for it.
                let visible_blocks = self.selector.n_blocks(visible);
                per_query.push(
                    self.selector
                        .select(&all_blocks[..visible_blocks], visible)?,
                );
            }
            out.push(per_query);
        }
        Ok(out)
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

    // ---- the indexer as a layer ----

    use crate::harness::arch::qwen3_5::TextConfig;

    /// A rope whose `head_dim` matches the indexer's, with half of it
    /// rotating — the shipped model rotates 64 of the indexer's 128,
    /// which is the same fraction as the main attention's 64 of 256.
    fn rope_for(head_dim: usize) -> RotaryEmbedding {
        let json = format!(
            r#"{{"vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
                 "num_hidden_layers": 1, "num_attention_heads": 1,
                 "num_key_value_heads": 1, "head_dim": {head_dim},
                 "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
                 "rope_parameters": {{"rope_theta": 10000.0,
                                      "partial_rotary_factor": 0.5}}}}"#
        );
        let cfg: TextConfig = serde_json::from_str(&json).unwrap();
        RotaryEmbedding::new(DType::F32, &cfg, &Device::Cpu).unwrap()
    }

    /// Deterministic, well-spread values — the selection is a top-k, so
    /// tied scores would make a test flaky rather than wrong.
    fn spread(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.7 + seed) * 1.3).sin() + 0.1 * (i as f32).cos())
            .collect()
    }

    fn indexer(
        hidden_size: usize,
        n_heads: usize,
        head_dim: usize,
        block_size: usize,
        budget: usize,
    ) -> Indexer {
        let dev = Device::Cpu;
        let fused = (n_heads + 1) * head_dim;
        let qk =
            Tensor::from_vec(spread(fused * hidden_size, 0.3), (fused, hidden_size), &dev).unwrap();
        let norm = || {
            Qwen3_5RmsNorm::from_weight(
                Tensor::zeros(head_dim, DType::F32, &dev).unwrap(),
                1e-6,
                None,
            )
        };
        Indexer {
            qk_proj: Linear::new(qk, None),
            q_layernorm: norm(),
            k_layernorm: norm(),
            selector: BlockSelector::new(n_heads, head_dim, block_size, budget).unwrap(),
            n_heads,
            kv_heads: 1,
            head_dim,
            key_cache: None,
        }
    }

    fn hidden(batch: usize, seq: usize, width: usize, seed: f32) -> Tensor {
        Tensor::from_vec(
            spread(batch * seq * width, seed),
            (batch, seq, width),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// `index_qk_proj` is fused 4:1, not halved. With 4 query heads and
    /// 1 key head of 8 dims the split is 32 and 8 of 40 — a halving
    /// would take 20 and 20 and put half a query head in the key.
    #[test]
    fn the_fused_projection_splits_by_head_count_not_in_half() {
        let (hidden_size, n_heads, head_dim) = (8, 4, 8);
        let ix = indexer(hidden_size, n_heads, head_dim, 4, 16);
        let h = hidden(1, 3, hidden_size, 1.1);

        let (q, k) = ix.project(&h).unwrap();
        assert_eq!(q.dims(), &[1, n_heads, 3, head_dim]);
        assert_eq!(k.dims(), &[1, 3, head_dim]);

        // The key is the LAST head_dim channels of the 40-wide fused
        // output, and it is untouched by the norms.
        let fused = ix.qk_proj.forward(&h).unwrap();
        let want: Vec<f32> = fused
            .narrow(D::Minus1, n_heads * head_dim, head_dim)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, want);
    }

    /// The cached key is raw: pre-norm and pre-RoPE. A block is rotated
    /// once at its first position, so a cache of per-position-rotated
    /// keys would put three of every four keys at the wrong angle — and
    /// still return a plausible selection.
    #[test]
    fn the_cache_holds_raw_keys() {
        let (hidden_size, head_dim) = (8, 8);
        let mut ix = indexer(hidden_size, 4, head_dim, 4, 4096);
        let h = hidden(1, 6, hidden_size, 2.2);

        let (_, raw) = ix.project(&h).unwrap();
        ix.visible_positions(&h, &rope_for(head_dim), 0).unwrap();

        let cached: Vec<f32> = ix
            .key_cache
            .as_ref()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want: Vec<f32> = raw.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(cached, want, "the cache must not be normed or rotated");
        assert_eq!(ix.cached_len(), 6);

        ix.clear_cache();
        assert_eq!(ix.cached_len(), 0);
    }

    /// Gate 4, through the whole layer rather than the bare selector:
    /// below the budget every query sees every position it causally
    /// could, so the layer is exactly dense causal attention.
    #[test]
    fn below_budget_every_query_sees_its_whole_past() {
        let (hidden_size, head_dim) = (8, 8);
        // Budget of 512 blocks, as shipped — 11 positions cannot reach it.
        let mut ix = indexer(hidden_size, 4, head_dim, 4, 2048);
        let h = hidden(2, 11, hidden_size, 3.3);

        let got = ix.visible_positions(&h, &rope_for(head_dim), 0).unwrap();
        assert_eq!(got.len(), 2);
        for row in &got {
            assert_eq!(row.len(), 11);
            for (i, positions) in row.iter().enumerate() {
                assert_eq!(
                    *positions,
                    (0..=i).collect::<Vec<_>>(),
                    "query {i} should attend its whole past"
                );
            }
        }
    }

    /// Decoding a token must select what a prefill covering the same
    /// text would select for that position. This is the property the
    /// cache and the `past_len` arithmetic exist for, and it is tested
    /// with a budget small enough that the selection actually binds —
    /// at 1 block of 4 from 9 positions, two blocks compete.
    #[test]
    fn a_decode_step_selects_what_the_prefill_would() {
        let (hidden_size, head_dim) = (8, 8);
        let rope = rope_for(head_dim);
        let h = hidden(1, 9, hidden_size, 4.4);

        let mut whole = indexer(hidden_size, 4, head_dim, 4, 4);
        let want = whole.visible_positions(&h, &rope, 0).unwrap()[0][8].clone();

        let mut split = indexer(hidden_size, 4, head_dim, 4, 4);
        let prefill = h.narrow(1, 0, 8).unwrap().contiguous().unwrap();
        split.visible_positions(&prefill, &rope, 0).unwrap();
        let step = h.narrow(1, 8, 1).unwrap().contiguous().unwrap();
        let got = split.visible_positions(&step, &rope, 8).unwrap()[0][0].clone();

        assert_eq!(got, want);
        // And the budget really did bind, or this proves nothing.
        assert!(
            got.len() < 9,
            "expected a bound selection, got all {} positions",
            got.len()
        );
        // The tail is 9 mod 4 = 1 position: the query's own.
        assert!(got.contains(&8));
    }

    /// A cache that has drifted from the main KV selects blocks of some
    /// other request's text. Fail loudly instead.
    #[test]
    fn a_cache_out_of_step_with_the_main_kv_is_an_error() {
        let (hidden_size, head_dim) = (8, 8);
        let mut ix = indexer(hidden_size, 4, head_dim, 4, 2048);
        let h = hidden(1, 2, hidden_size, 5.5);
        let err = ix
            .visible_positions(&h, &rope_for(head_dim), 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("diverged"), "got: {err}");
    }
}
