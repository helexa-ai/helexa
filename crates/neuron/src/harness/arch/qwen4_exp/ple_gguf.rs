//! The PLE n-gram table read from a GGUF (#427).
//!
//! Unsloth's GGUFs carry the table as `per_layer_token_embd.weight`:
//! `[160, 320001536]` IQ4_NL, 27 GB against the checkpoint's 102 GB of
//! BF16. Its rows are addressed exactly as the checkpoint's (the same
//! hash, the same flat row index), so it is a drop-in [`NGramTable`].
//!
//! Each token reads sixteen 90-byte rows at hashed, effectively random
//! positions, so the mapping is advised random: read-ahead around a hit
//! would only evict rows another token needs. Rows are dequantised on
//! the host, as llama.cpp's IQ4_NL does it: per 32 values, an f16 scale
//! times a fixed 16-entry codebook.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use memmap2::{Advice, Mmap};

use super::gguf_index::GgufIndex;
use super::ple::NGramTable;

/// ggml's type id for IQ4_NL.
const GGML_IQ4_NL: u32 = 20;
/// Values per IQ4_NL block, and its bytes: an f16 scale, 16 code bytes.
const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 18;
/// llama.cpp's `kvalues_iq4nl`.
const CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// The tensor's name in the GGUF.
pub const TABLE_NAME: &str = "per_layer_token_embd.weight";

pub struct GgufNGramTable {
    map: Mmap,
    /// Byte position of row 0 in `map`.
    offset: usize,
    rows: usize,
    head_dim: usize,
    row_bytes: usize,
    device: Device,
}

impl GgufNGramTable {
    /// Maps the table out of the GGUF whose first shard is `first`.
    /// `device` is where [`NGramTable::gather`] puts its rows.
    pub fn open(first: &Path, head_dim: usize, device: &Device) -> Result<Self> {
        let index = GgufIndex::open(first)?;
        let t = index
            .tensors
            .get(TABLE_NAME)
            .with_context(|| format!("{} has no {TABLE_NAME}", first.display()))?;
        ensure!(
            t.ggml_type == GGML_IQ4_NL,
            "{TABLE_NAME} is ggml type {}, only IQ4_NL ({GGML_IQ4_NL}) is read",
            t.ggml_type
        );
        ensure!(
            t.dims.len() == 2 && t.dims[0] as usize == head_dim,
            "{TABLE_NAME} is {:?}, expected rows of {head_dim}",
            t.dims
        );
        ensure!(
            head_dim.is_multiple_of(BLOCK),
            "a {head_dim}-wide row is not whole IQ4_NL blocks"
        );
        let rows = t.dims[1] as usize;
        let row_bytes = head_dim / BLOCK * BLOCK_BYTES;
        let path = &index.shards[t.shard];
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: model files are not modified while serving.
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?;
        map.advise(Advice::Random)
            .context("advise the n-gram table random")?;
        let offset = t.offset as usize;
        ensure!(
            offset + rows * row_bytes <= map.len(),
            "{TABLE_NAME} runs past its shard"
        );
        tracing::info!(
            rows,
            gib = (rows * row_bytes) as f64 / (1u64 << 30) as f64,
            "PLE n-gram table from GGUF (IQ4_NL)"
        );
        Ok(Self {
            map,
            offset,
            rows,
            head_dim,
            row_bytes,
            device: device.clone(),
        })
    }

    /// How many rows the table has.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Dequantises `rows`, in order, into `out` (`rows.len() * head_dim`).
    pub fn gather_into(&self, rows: &[i64], out: &mut [f32]) -> Result<()> {
        ensure!(
            out.len() == rows.len() * self.head_dim,
            "{} values for {} rows of {}",
            out.len(),
            rows.len(),
            self.head_dim
        );
        for (&row, dst) in rows.iter().zip(out.chunks_exact_mut(self.head_dim)) {
            ensure!(
                row >= 0 && (row as usize) < self.rows,
                "n-gram row {row} outside the table's {}",
                self.rows
            );
            let start = self.offset + row as usize * self.row_bytes;
            dequant_row(&self.map[start..start + self.row_bytes], dst);
        }
        Ok(())
    }
}

/// One IQ4_NL row into f32: per block, value `j` is the low nibble of
/// code byte `j` for `j < 16`, else the high nibble of byte `j - 16`.
fn dequant_row(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<BLOCK>().0)
    {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let (lo, hi) = out.split_at_mut(BLOCK / 2);
        for ((q, l), h) in block[2..].iter().zip(lo).zip(hi) {
            *l = d * f32::from(CODEBOOK[usize::from(q & 0xf)]);
            *h = d * f32::from(CODEBOOK[usize::from(q >> 4)]);
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: renormalise.
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << shift) & 0x3ff) << 13
        }
        (0x1f, _) => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

impl NGramTable for GgufNGramTable {
    fn gather(&self, rows: &[i64]) -> Result<Tensor> {
        let mut out = vec![0f32; rows.len() * self.head_dim];
        self.gather_into(rows, &mut out)?;
        Ok(Tensor::from_vec(
            out,
            (rows.len(), self.head_dim),
            &self.device,
        )?)
    }

    fn gather_into(&self, rows: &[i64], out: &mut [f32]) -> Result<()> {
        GgufNGramTable::gather_into(self, rows, out)
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_matches_known_values() {
        for (bits, want) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x3555, 0.333_251_95),
            (0x0001, 5.960_464_5e-8),
            (0x7bff, 65504.0),
            (0x0000, 0.0),
        ] {
            assert_eq!(f16_to_f32(bits), want, "{bits:#06x}");
        }
    }

    /// A block whose code bytes put a different nibble in each half: the
    /// low nibbles fill the first 16 values, the high ones the last.
    #[test]
    fn a_row_splits_nibbles_into_halves() {
        let mut block = vec![0x00u8, 0x40]; // f16 2.0
        block.extend((0..16u8).map(|j| (15 - j) << 4 | j));
        let mut out = [0f32; 32];
        dequant_row(&block, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], 2.0 * f32::from(CODEBOOK[j]), "low {j}");
            assert_eq!(out[16 + j], 2.0 * f32::from(CODEBOOK[15 - j]), "high {j}");
        }
    }
}
