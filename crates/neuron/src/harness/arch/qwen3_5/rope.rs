//! Rotary position embedding for Qwen3-Next's full-attention layers.
//!
//! Qwen3.6 declares **interleaved M-RoPE** (multimodal RoPE): the
//! rotary half-dimension is split across three position axes —
//! `[text, height, width]` per `mrope_section` (`[11,11,10]` for
//! Qwen3.6) — interleaved per-frequency. For **text** every token's
//! three axes carry the same position id, so the interleave is a no-op
//! and this reduces exactly to plain RoPE. For **image** tokens the
//! height/width axes carry the patch's 2D grid coordinates, which is
//! how the model reads the 14×14 patch layout (without it, all patches
//! share a height position and the image reads as vertical repetition).
//!
//! Two cos/sin builders feed a shared [`RotaryEmbedding::apply`]:
//! - [`RotaryEmbedding::plain_cos_sin`] narrows the precomputed tables
//!   at a scalar position — the text / decode fast path.
//! - [`RotaryEmbedding::mrope_cos_sin`] builds per-token cos/sin from a
//!   `(3, seq)` position-id tensor, blending the three axes' frequencies
//!   at the interleave index sets — the vision-prefill path.
//!
//! Rotation flavour: **GLM-style** rotate-half (candle's `rope_slow`),
//! matching the reference Python's `apply_rotary_pos_emb` + `rotate_half`.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

use super::TextConfig;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Inverse frequencies, shape `(1, rotary_dim/2)`. Retained (beyond
    /// the precomputed `sin`/`cos` tables) so [`Self::mrope_cos_sin`] can
    /// build cos/sin from arbitrary per-axis position ids.
    inv_freq: Tensor,
    /// Per-axis column masks over the rotary half-dim, shape `(1, half)`,
    /// f32 0/1. `mask_t + mask_h + mask_w` partitions the columns; a
    /// column belongs to exactly one axis. For a non-MRoPE config
    /// `mask_t` is all-ones and the others all-zero (→ plain RoPE).
    mask_t: Tensor,
    mask_h: Tensor,
    mask_w: Tensor,
    dtype: DType,
    /// Number of dims at the head's leading edge that the rotation
    /// covers. The remaining `head_dim - rotary_dim` dims pass through
    /// unchanged. Qwen3-Next uses `partial_rotary_factor = 0.25`, so
    /// for `head_dim = 256` only 64 dims rotate.
    rotary_dim: usize,
    head_dim: usize,
}

/// Build the per-axis 0/1 column masks over the rotary half-dim from
/// `mrope_section`. Returns `(temporal, height, width)` each length
/// `half`. Temporal is the complement of height ∪ width, so the three
/// masks always partition `0..half` and reduce to all-temporal (plain
/// RoPE) when no usable section is given.
fn mrope_masks(
    half: usize,
    section: &[usize],
    interleaved: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut mh = vec![0f32; half];
    let mut mw = vec![0f32; half];
    if section.len() == 3 {
        if interleaved {
            // Qwen3-VL: height at columns 1,4,7,… ; width at 2,5,8,… ;
            // temporal keeps 0,3,6,… — each `take`n from `mrope_section`.
            for i in (1..half).step_by(3).take(section[1]) {
                mh[i] = 1.0;
            }
            for i in (2..half).step_by(3).take(section[2]) {
                mw[i] = 1.0;
            }
        } else {
            // Qwen2-VL: contiguous blocks [text | height | width].
            let h_start = section[0].min(half);
            let h_end = (section[0] + section[1]).min(half);
            for m in mh.iter_mut().take(h_end).skip(h_start) {
                *m = 1.0;
            }
            for m in mw.iter_mut().take(half).skip(h_end) {
                *m = 1.0;
            }
        }
    }
    let mt: Vec<f32> = (0..half)
        .map(|i| {
            if mh[i] == 0.0 && mw[i] == 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (mt, mh, mw)
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &TextConfig, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let rope = &cfg.rope_parameters;
        let rotary_dim = (head_dim as f32 * rope.partial_rotary_factor) as usize;
        if !rotary_dim.is_multiple_of(2) {
            anyhow::bail!(
                "rotary_dim = head_dim * partial_rotary_factor = {head_dim} * {} = {rotary_dim} \
                 must be even (cos/sin are paired)",
                rope.partial_rotary_factor
            );
        }
        if rotary_dim == 0 {
            anyhow::bail!(
                "rotary_dim = 0 (partial_rotary_factor = {} too small)",
                rope.partial_rotary_factor
            );
        }
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let half = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        // MRoPE axis masks. `sum(mrope_section)` should equal `half`;
        // warn-tolerant: any shortfall just stays on the temporal axis.
        let (mt, mh, mw) = mrope_masks(half, &rope.mrope_section, rope.mrope_interleaved);
        let mask_t = Tensor::from_vec(mt, (1, half), dev)?;
        let mask_h = Tensor::from_vec(mh, (1, half), dev)?;
        let mask_w = Tensor::from_vec(mw, (1, half), dev)?;

        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            inv_freq,
            mask_t,
            mask_h,
            mask_w,
            dtype,
            rotary_dim,
            head_dim,
        })
    }

    /// cos/sin for a contiguous run of `seq_len` positions starting at
    /// `pos`, by narrowing the precomputed tables. The text / decode
    /// path (all three MRoPE axes equal → plain RoPE). Shape
    /// `(seq_len, rotary_dim/2)`.
    pub fn plain_cos_sin(
        &self,
        pos: usize,
        seq_len: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        Ok((cos, sin))
    }

    /// cos/sin from explicit per-token 3D position ids, shape
    /// `(3, seq_len)` (axes: text, height, width). Builds each axis's
    /// frequencies and blends them at the interleave index sets, so
    /// every rotary frequency slot is driven by exactly one axis.
    /// Reduces exactly to [`Self::plain_cos_sin`] when the three axes are
    /// equal. Returns cos/sin of shape `(seq_len, rotary_dim/2)`.
    pub fn mrope_cos_sin(&self, position_ids: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let pos = position_ids.to_dtype(DType::F32)?;
        let (axes, seq_len) = pos.dims2()?;
        debug_assert_eq!(axes, 3, "mrope position_ids must have 3 axes");
        // Per-axis freqs: pos[a] (seq,1) @ inv_freq (1,half) → (seq,half).
        let ft = pos.i(0)?.reshape((seq_len, 1))?.matmul(&self.inv_freq)?;
        let fh = pos.i(1)?.reshape((seq_len, 1))?.matmul(&self.inv_freq)?;
        let fw = pos.i(2)?.reshape((seq_len, 1))?.matmul(&self.inv_freq)?;
        // Blend: each column belongs to exactly one axis (masks partition
        // the half-dim), so this picks the right axis per frequency slot.
        let blended = ft
            .broadcast_mul(&self.mask_t)?
            .add(&fh.broadcast_mul(&self.mask_h)?)?
            .add(&fw.broadcast_mul(&self.mask_w)?)?;
        let cos = blended.cos()?.to_dtype(self.dtype)?;
        let sin = blended.sin()?.to_dtype(self.dtype)?;
        Ok((cos, sin))
    }

    /// Apply rotary to `q`, `k` (shape `(B, H, L, head_dim)`) using
    /// precomputed `cos`/`sin` of shape `(L, rotary_dim/2)`. Partial
    /// rotary: only the first `rotary_dim` dims rotate; the tail passes
    /// through unchanged.
    pub fn apply_cos_sin(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, _seq_len, head_dim_in) = q.dims4()?;
        debug_assert_eq!(head_dim_in, self.head_dim, "q head_dim mismatch");
        if self.rotary_dim == self.head_dim {
            let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, cos, sin)?;
            let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, cos, sin)?;
            Ok((q_embed, k_embed))
        } else {
            // Partial rotation: narrow → rotate → cat the untouched tail.
            let tail = self.head_dim - self.rotary_dim;
            let q_rot = q
                .narrow(candle_core::D::Minus1, 0, self.rotary_dim)?
                .contiguous()?;
            let q_pass = q.narrow(candle_core::D::Minus1, self.rotary_dim, tail)?;
            let k_rot = k
                .narrow(candle_core::D::Minus1, 0, self.rotary_dim)?
                .contiguous()?;
            let k_pass = k.narrow(candle_core::D::Minus1, self.rotary_dim, tail)?;
            let q_rotated = candle_nn::rotary_emb::rope_slow(&q_rot, cos, sin)?;
            let k_rotated = candle_nn::rotary_emb::rope_slow(&k_rot, cos, sin)?;
            let q_embed =
                Tensor::cat(&[&q_rotated, &q_pass.contiguous()?], candle_core::D::Minus1)?;
            let k_embed =
                Tensor::cat(&[&k_rotated, &k_pass.contiguous()?], candle_core::D::Minus1)?;
            Ok((q_embed, k_embed))
        }
    }

    /// Text/decode convenience: build plain cos/sin for a scalar offset
    /// and apply in one call. The current call sites use this; Stages 3–4
    /// move cos/sin construction up into the model forward (computed once
    /// per forward) and call [`Self::apply_cos_sin`] directly.
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let (cos, sin) = self.plain_cos_sin(offset, seq_len)?;
        self.apply_cos_sin(q, k, &cos, &sin)
    }
}

/// Compute interleaved-M-RoPE 3D position ids for a full prompt that may
/// contain image-placeholder runs, plus the decode `rope_delta`.
///
/// Mirrors the reference `get_rope_index`:
/// - text tokens advance a single running counter `c`, all three axes
///   equal (`[c, c, c]`);
/// - each contiguous run of `image_token_id` is one image; its tokens get
///   `[base + t, base + h, base + w]` in row-major (t outer, h, w inner),
///   where `base` is the counter at the run's start; after the run the
///   counter resumes from `base + max(grid_t, grid_h, grid_w)`.
///
/// Returns `(text_pos, height_pos, width_pos, rope_delta)`, each pos `Vec`
/// length `input_ids.len()`. `rope_delta = final_counter - seq_len`: add it
/// to a plain decode offset so text resumes from the counter after the
/// (position-compressed) image blocks.
///
/// Fixed-resolution assumption (Stage C): each image run is a perfect
/// square with `grid_t = 1` (still image) and `grid_h = grid_w =
/// isqrt(run_len)` — 196 → 14×14. Dynamic resolution (#14) would thread
/// real per-image grids instead.
pub(crate) fn get_rope_index(input_ids: &[u32], image_token_id: u32) -> Result<MRopeIndex> {
    let n = input_ids.len();
    let mut text = Vec::with_capacity(n);
    let mut height = Vec::with_capacity(n);
    let mut width = Vec::with_capacity(n);
    let mut counter: i64 = 0;
    let mut i = 0;
    while i < n {
        if input_ids[i] == image_token_id {
            let start = i;
            while i < n && input_ids[i] == image_token_id {
                i += 1;
            }
            let run = i - start;
            let g = run.isqrt();
            if g * g != run {
                anyhow::bail!(
                    "get_rope_index: image run length {run} is not a perfect square \
                     (fixed-resolution Stage C assumes a square grid; dynamic resolution is #14)"
                );
            }
            let (grid_t, grid_h, grid_w) = (1usize, g, g);
            let base = counter;
            for tt in 0..grid_t {
                for hh in 0..grid_h {
                    for ww in 0..grid_w {
                        text.push(base + tt as i64);
                        height.push(base + hh as i64);
                        width.push(base + ww as i64);
                    }
                }
            }
            counter = base + grid_t.max(grid_h).max(grid_w) as i64;
        } else {
            text.push(counter);
            height.push(counter);
            width.push(counter);
            counter += 1;
            i += 1;
        }
    }
    let delta = counter - n as i64;
    Ok((text, height, width, delta))
}

/// `(text_pos, height_pos, width_pos, rope_delta)` returned by
/// [`get_rope_index`]; the three vectors combine into the `(3, seq)`
/// MRoPE position-id tensor.
pub(crate) type MRopeIndex = (Vec<i64>, Vec<i64>, Vec<i64>, i64);

/// Build the `(3, seq)` position-id tensor consumed by
/// [`RotaryEmbedding::mrope_cos_sin`] from the three axis vectors.
pub(crate) fn mrope_position_tensor(
    text: &[i64],
    height: &[i64],
    width: &[i64],
    dev: &Device,
) -> candle_core::Result<Tensor> {
    let seq = text.len();
    let mut flat = Vec::with_capacity(3 * seq);
    flat.extend_from_slice(text);
    flat.extend_from_slice(height);
    flat.extend_from_slice(width);
    Tensor::from_vec(flat, (3, seq), dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    /// A TextConfig stub with Qwen3.6's rope params (head_dim 256,
    /// partial 0.25 → rotary_dim 64 → half 32; section [11,11,10]).
    fn qwen36_cfg() -> TextConfig {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 5120,
            "num_hidden_layers": 1,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "intermediate_size": 1,
            "vocab_size": 10,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64,
            "layer_types": ["full_attention"],
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": true
            }
        }))
        .expect("cfg")
    }

    #[test]
    fn mrope_masks_partition_the_half_dim() {
        let (mt, mh, mw) = mrope_masks(32, &[11, 11, 10], true);
        // Each column belongs to exactly one axis.
        for i in 0..32 {
            let s = mt[i] + mh[i] + mw[i];
            assert_eq!(s, 1.0, "column {i} covered {s} times");
        }
        assert_eq!(mt.iter().sum::<f32>(), 11.0);
        assert_eq!(mh.iter().sum::<f32>(), 11.0);
        assert_eq!(mw.iter().sum::<f32>(), 10.0);
        // Interleave: temporal 0,3,…; height 1,4,…; width 2,5,…
        assert_eq!(mt[0], 1.0);
        assert_eq!(mh[1], 1.0);
        assert_eq!(mw[2], 1.0);
        assert_eq!(mt[3], 1.0);
    }

    /// The load-bearing invariant: when all three position axes are
    /// equal (text), `mrope_cos_sin` must reproduce `plain_cos_sin`
    /// bit-for-bit — i.e. M-RoPE is a no-op for text, so text inference
    /// is unchanged.
    #[test]
    fn mrope_reduces_to_plain_for_equal_axes() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(DType::F32, &qwen36_cfg(), &dev).unwrap();

        // positions 5,6,7 on all three axes.
        let base: Vec<i64> = vec![5, 6, 7];
        let pos =
            Tensor::from_vec([base.clone(), base.clone(), base].concat(), (3, 3), &dev).unwrap();

        let (mc, ms) = rope.mrope_cos_sin(&pos).unwrap();
        let (pc, ps) = rope.plain_cos_sin(5, 3).unwrap();

        let dcos = (mc - pc).unwrap().abs().unwrap().max_all().unwrap();
        let dsin = (ms - ps).unwrap().abs().unwrap().max_all().unwrap();
        assert!(
            dcos.to_scalar::<f32>().unwrap() < 1e-6,
            "cos mismatch {dcos:?}"
        );
        assert!(
            dsin.to_scalar::<f32>().unwrap() < 1e-6,
            "sin mismatch {dsin:?}"
        );
    }

    /// Hand-checked interleave: a width-axis column (index 2) must track
    /// the WIDTH position, while a temporal column (index 0) tracks the
    /// TEXT position, even when the axes differ.
    #[test]
    fn mrope_blends_axes_at_interleave_columns() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(DType::F32, &qwen36_cfg(), &dev).unwrap();
        let half = rope.inv_freq.dim(1).unwrap();
        let inv: Vec<f32> = rope.inv_freq.i(0).unwrap().to_vec1().unwrap();

        // One token: text=10, height=3, width=7 — all distinct.
        let pos = Tensor::from_vec(vec![10i64, 3, 7], (3, 1), &dev).unwrap();
        let (cos, _sin) = rope.mrope_cos_sin(&pos).unwrap();
        let cos_row: Vec<f32> = cos.i(0).unwrap().to_vec1().unwrap();
        assert_eq!(cos_row.len(), half);

        // Column 0 (temporal) → text pos 10. Column 1 (height) → 3.
        // Column 2 (width) → 7.
        assert!((cos_row[0] - (10.0 * inv[0]).cos()).abs() < 1e-5);
        assert!((cos_row[1] - (3.0 * inv[1]).cos()).abs() < 1e-5);
        assert!((cos_row[2] - (7.0 * inv[2]).cos()).abs() < 1e-5);
        assert!((cos_row[3] - (10.0 * inv[3]).cos()).abs() < 1e-5);
    }

    #[test]
    fn get_rope_index_text_only_is_sequential() {
        let (t, h, w, delta) = get_rope_index(&[1, 2, 3, 4], 99).unwrap();
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 1, 2, 3]);
        assert_eq!(w, vec![0, 1, 2, 3]);
        assert_eq!(delta, 0, "no image → delta 0 → plain decode positions");
    }

    #[test]
    fn get_rope_index_text_image_text() {
        // [text, image(2x2 run of 4), text]. image_token = 99.
        let ids = [1u32, 99, 99, 99, 99, 2];
        let (t, h, w, delta) = get_rope_index(&ids, 99).unwrap();
        // token 0: text → 0. image base=1, grid 1x2x2:
        //   t all = 1; h = base+row = [1,1,2,2]; w = base+col = [1,2,1,2].
        // resume from base + max(1,2,2) = 3. trailing text → 3.
        assert_eq!(t, vec![0, 1, 1, 1, 1, 3]);
        assert_eq!(h, vec![0, 1, 1, 2, 2, 3]);
        assert_eq!(w, vec![0, 1, 2, 1, 2, 3]);
        // final counter = 4, seq_len = 6 → delta = -2 (the 4 image tokens
        // advanced the counter by only 2).
        assert_eq!(delta, -2);
        // Decode after the prompt (offset = 6) → text position 6 + (-2) = 4.
        assert_eq!(6 + delta, 4);
    }

    #[test]
    fn get_rope_index_rejects_non_square_image_run() {
        // 196 is square (14x14) — ok. 195 is not.
        assert!(get_rope_index(&[99u32; 196], 99).is_ok());
        assert!(get_rope_index(&[99u32; 195], 99).is_err());
    }

    #[test]
    fn position_tensor_round_trips_through_mrope_cos_sin() {
        // get_rope_index → (3,seq) tensor → mrope_cos_sin, and confirm an
        // image token's height column tracks its grid row (not the text
        // counter), i.e. the end-to-end position plumbing is wired right.
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(DType::F32, &qwen36_cfg(), &dev).unwrap();
        let ids = [1u32, 99, 99, 99, 99]; // text + 2x2 image
        let (t, h, w, _d) = get_rope_index(&ids, 99).unwrap();
        let pos = mrope_position_tensor(&t, &h, &w, &dev).unwrap();
        assert_eq!(pos.dims(), &[3, 5]);
        let (cos, _sin) = rope.mrope_cos_sin(&pos).unwrap();
        assert_eq!(cos.dims(), &[5, rope.inv_freq.dim(1).unwrap()]);

        let inv: Vec<f32> = rope.inv_freq.i(0).unwrap().to_vec1().unwrap();
        // Last image token (index 4): grid (h=1, w=1) → base 1 → h=2, w=2.
        // Height column (index 1) must track h-position 2, not text.
        let last: Vec<f32> = cos.i(4).unwrap().to_vec1().unwrap();
        assert!((last[1] - (2.0 * inv[1]).cos()).abs() < 1e-5);
    }

    #[test]
    fn get_rope_index_196_is_14x14() {
        let mut ids = vec![1u32]; // one text token
        ids.extend(std::iter::repeat_n(99u32, 196));
        let (t, h, w, _delta) = get_rope_index(&ids, 99).unwrap();
        // image base = 1. Last image token (index 196) is grid (h=13,w=13).
        assert_eq!(*t.last().unwrap(), 1, "grid_t=1 → temporal const at base");
        assert_eq!(h[1], 1, "first image row at base");
        assert_eq!(w[1], 1, "first image col at base");
        assert_eq!(h[196], 1 + 13, "last image row = base + 13");
        assert_eq!(w[196], 1 + 13, "last image col = base + 13");
    }
}
