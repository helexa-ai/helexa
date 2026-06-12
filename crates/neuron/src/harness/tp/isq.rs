//! Parallel in-situ quantization (#1).
//!
//! `candle_core::quantized::QTensor::quantize` processes a tensor's
//! quantization blocks strictly sequentially on one CPU core (its
//! CUDA storage round-trips through the same CPU path), which made
//! Q6K ISQ the dominant phase of the Qwen3.6-27B TP cold load —
//! minutes of single-threaded block math per rank while 31 cores
//! idled.
//!
//! Each block is independent, so this module re-implements the same
//! quantization through candle's public per-block API
//! (`k_quants::GgmlType::from_float`) with rayon fanning the blocks
//! across the CPU pool, producing **byte-identical** output to
//! candle's sequential path (pinned by the parity tests below).
//!
//! Threading discipline: the device-to-host read and the final
//! device upload (`QStorage::from_data`) run on the *calling* thread
//! — the device worker / subprocess main thread that owns the CUDA
//! context. The rayon workers only ever touch host memory.

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_core::quantized::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8K, GgmlType,
};
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use rayon::prelude::*;
use std::borrow::Cow;

/// Quantization blocks per rayon task. Blocks are 32–256 elements; 32
/// of them per task keeps scheduling overhead negligible while a 27B
/// shard's largest tensors still split into thousands of tasks.
const BLOCKS_PER_TASK: usize = 32;

/// Drop-in replacement for `QTensor::quantize` that parallelises the
/// per-block work. Dtypes without a plain block encoding (the f32 /
/// f16 / bf16 casts, Q8_1) fall through to candle's implementation.
pub(crate) fn quantize_parallel(weight: &Tensor, dtype: GgmlDType) -> Result<QTensor> {
    match dtype {
        GgmlDType::Q2K => quantize_blocks::<BlockQ2K>(weight),
        GgmlDType::Q3K => quantize_blocks::<BlockQ3K>(weight),
        GgmlDType::Q4K => quantize_blocks::<BlockQ4K>(weight),
        GgmlDType::Q5K => quantize_blocks::<BlockQ5K>(weight),
        GgmlDType::Q6K => quantize_blocks::<BlockQ6K>(weight),
        GgmlDType::Q8K => quantize_blocks::<BlockQ8K>(weight),
        GgmlDType::Q4_0 => quantize_blocks::<BlockQ4_0>(weight),
        GgmlDType::Q4_1 => quantize_blocks::<BlockQ4_1>(weight),
        GgmlDType::Q5_0 => quantize_blocks::<BlockQ5_0>(weight),
        GgmlDType::Q5_1 => quantize_blocks::<BlockQ5_1>(weight),
        GgmlDType::Q8_0 => quantize_blocks::<BlockQ8_0>(weight),
        _ => QTensor::quantize(weight, dtype)
            .with_context(|| format!("QTensor::quantize fallback for {dtype:?}")),
    }
}

fn quantize_blocks<T: GgmlType + Send + Sync>(weight: &Tensor) -> Result<QTensor> {
    let shape = weight.shape().clone();
    let block_size = T::BLCK_SIZE;
    // Same constraint QTensor::quantize enforces: the last dim must
    // tile into whole blocks so a block never spans two rows.
    let last_dim = shape.dims().last().copied().unwrap_or(0);
    if last_dim == 0 || !last_dim.is_multiple_of(block_size) {
        anyhow::bail!(
            "quantize_parallel: last dim of {shape:?} is not divisible by the {:?} block size {block_size}",
            T::DTYPE
        );
    }

    // Device→host read + f32 cast on the calling thread (the one
    // that owns the CUDA context, when there is one).
    let host: Vec<f32> = weight
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()
        .context("copy weight to host for quantization")?;
    let n_blocks = host.len() / block_size;

    // Zero-initialised block buffer. The block structs have no public
    // constructor, but every dispatch above is a plain `repr(C)`
    // bundle of integers and (half-)floats, for which the all-zero
    // bit pattern is a valid value — and `from_float` overwrites
    // every block in full.
    let mut blocks: Vec<T> = Vec::with_capacity(n_blocks);
    // SAFETY: the buffer was allocated with capacity `n_blocks`;
    // `write_bytes` zero-initialises exactly that many elements
    // before `set_len` exposes them, and all-zero is a valid bit
    // pattern for these POD block types (no references, no enums,
    // no padding-sensitive invariants).
    unsafe {
        std::ptr::write_bytes(blocks.as_mut_ptr(), 0, n_blocks);
        blocks.set_len(n_blocks);
    }

    blocks
        .par_chunks_mut(BLOCKS_PER_TASK)
        .zip(host.par_chunks(BLOCKS_PER_TASK * block_size))
        .for_each(|(bs, xs)| T::from_float(xs, bs));

    // SAFETY: a `repr(C)` slice viewed as its raw bytes; the length
    // is exactly the allocation's initialised extent. `from_data`
    // copies the bytes (host-side for CPU, `memcpy_htod` for CUDA)
    // before this view is dropped.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            n_blocks * std::mem::size_of::<T>(),
        )
    };
    let storage = QStorage::from_data(Cow::Borrowed(bytes), weight.device(), T::DTYPE)
        .context("upload quantized blocks")?;
    Ok(QTensor::new(storage, shape)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The parity gate: parallel quantization must be byte-identical
    /// to candle's sequential path — same per-block math, different
    /// scheduling only.
    fn assert_byte_parity(dtype: GgmlDType) {
        let dev = Device::Cpu;
        let weight = Tensor::randn(0f32, 1.0, (8, 512), &dev).unwrap();
        let seq = QTensor::quantize(&weight, dtype).unwrap();
        let par = quantize_parallel(&weight, dtype).unwrap();
        assert_eq!(par.dtype(), seq.dtype());
        assert_eq!(par.shape(), seq.shape());
        let a = seq.data().unwrap();
        let b = par.data().unwrap();
        assert_eq!(a.as_ref(), b.as_ref(), "byte mismatch for {dtype:?}");
    }

    #[test]
    fn parity_q6k() {
        assert_byte_parity(GgmlDType::Q6K);
    }

    #[test]
    fn parity_q4k() {
        assert_byte_parity(GgmlDType::Q4K);
    }

    #[test]
    fn parity_q5k() {
        assert_byte_parity(GgmlDType::Q5K);
    }

    #[test]
    fn parity_q8_0() {
        assert_byte_parity(GgmlDType::Q8_0);
    }

    #[test]
    fn rejects_non_divisible_last_dim() {
        let dev = Device::Cpu;
        // 100 is not a multiple of the 256-element k-quant block.
        let weight = Tensor::randn(0f32, 1.0, (4, 100), &dev).unwrap();
        assert!(quantize_parallel(&weight, GgmlDType::Q6K).is_err());
    }

    /// Fallback dtypes still produce a usable QTensor.
    #[test]
    fn fallback_f16_roundtrips() {
        let dev = Device::Cpu;
        let weight = Tensor::randn(0f32, 1.0, (4, 64), &dev).unwrap();
        let qt = quantize_parallel(&weight, GgmlDType::F16).unwrap();
        assert_eq!(qt.dtype(), GgmlDType::F16);
    }
}
