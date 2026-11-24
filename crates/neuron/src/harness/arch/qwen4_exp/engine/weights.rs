//! GGUF tensors as the native decode engine (#427) holds them.
//!
//! The engine reads the GGUF as it is stored: llama.cpp's layout and
//! head order (the linear-attention value heads tiled by key head), the
//! norm weights already `1 + w`, `ssm_a` already `-exp(A_log)`. That is
//! the form Strata's kernels take. Quantized projections stay quantized;
//! the tensors Strata's pack converts to BF16 (the hyper-connection mixes
//! and injections, the GDN alpha/beta, the routers) are converted the
//! same way here, rounding to nearest even.

/// ggml type ids the engine handles.
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q6_K: u32 = 14;
pub const GGML_BF16: u32 = 30;

const Q8_0_BLOCK_VALUES: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

/// IEEE half to single, subnormals included.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, _) => {
            // Subnormal: renormalise.
            let shift = man.leading_zeros() - 21;
            let man = (man << shift) & 0x3ff;
            (sign << 31) | ((113 - shift) << 23) | (man << 13)
        }
        (0x1f, _) => (sign << 31) | (0xff << 23) | (man << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Single to BF16 bits, round to nearest even (NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Q8_0 blocks (an f16 scale, then 32 signed bytes) to f32.
pub fn q8_0_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_VALUES);
    for block in bytes.as_chunks::<Q8_0_BLOCK_BYTES>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        out.extend(block[2..].iter().map(|&q| d * f32::from(q as i8)));
    }
    out
}

const Q6_K_BLOCK_VALUES: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;

/// Q6_K blocks to f32, as ggml's `dequantize_row_q6_K`: per 256 values,
/// 128 bytes of low nibbles, 64 of high bit pairs, 16 signed scales and
/// an f16 scale.
pub fn q6_k_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / Q6_K_BLOCK_BYTES * Q6_K_BLOCK_VALUES);
    for block in bytes.as_chunks::<Q6_K_BLOCK_BYTES>().0 {
        let (ql, rest) = block.split_at(128);
        let (qh, rest) = rest.split_at(64);
        let (sc, d) = rest.split_at(16);
        let d = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
        let mut y = [0f32; Q6_K_BLOCK_VALUES];
        for half in 0..2 {
            let (ql, qh, sc) = (&ql[64 * half..], &qh[32 * half..], &sc[8 * half..]);
            let y = &mut y[128 * half..];
            for l in 0..32 {
                let is = l / 16;
                let q = |low: u8, shift: u32| -> f32 {
                    f32::from(((low | (((qh[l] >> shift) & 3) << 4)) as i8) - 32)
                };
                let s = |i: usize| f32::from(sc[i] as i8);
                y[l] = d * s(is) * q(ql[l] & 0xf, 0);
                y[l + 32] = d * s(is + 2) * q(ql[l + 32] & 0xf, 2);
                y[l + 64] = d * s(is + 4) * q(ql[l] >> 4, 4);
                y[l + 96] = d * s(is + 6) * q(ql[l + 32] >> 4, 6);
            }
        }
        out.extend_from_slice(&y);
    }
    out
}

/// A tensor's values as f32, from the GGUF types the engine converts.
pub fn to_f32(bytes: &[u8], ggml_type: u32) -> Option<Vec<f32>> {
    Some(match ggml_type {
        GGML_F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect(),
        GGML_F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| f16_to_f32(u16::from_le_bytes(b)))
            .collect(),
        GGML_BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
            .collect(),
        GGML_Q8_0 => q8_0_to_f32(bytes),
        GGML_Q6_K => q6_k_to_f32(bytes),
        _ => return None,
    })
}

/// Bytes a tensor of `ggml_type` with `values` elements occupies, for
/// the types the engine reads whole.
pub fn tensor_bytes(ggml_type: u32, values: usize) -> Option<usize> {
    Some(match ggml_type {
        GGML_F32 => values * 4,
        GGML_F16 | GGML_BF16 => values * 2,
        GGML_Q8_0 if values.is_multiple_of(Q8_0_BLOCK_VALUES) => {
            values / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES
        }
        GGML_Q6_K if values.is_multiple_of(Q6_K_BLOCK_VALUES) => {
            values / Q6_K_BLOCK_VALUES * Q6_K_BLOCK_BYTES
        }
        _ => return None,
    })
}

#[cfg(feature = "strata-kernels")]
pub use device::{DeviceTensor, Gguf};

#[cfg(feature = "strata-kernels")]
mod device {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use anyhow::{Context, Result, bail, ensure};
    use candle_core::CudaDevice;
    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
    use memmap2::Mmap;

    use super::{GGML_BF16, GGML_F32, f32_to_bf16, tensor_bytes, to_f32};
    use crate::harness::arch::qwen4_exp::gguf_dense::GgufDenseSource;
    use crate::harness::arch::qwen4_exp::gguf_index::{GgufIndex, GgufTensor};

    /// Where a device tensor's bytes live.
    enum Backing {
        /// Uploaded by the engine.
        Owned(CudaSlice<u8>),
        /// Uploaded by candle's GGUF dense source, in the same layout.
        Shared(Arc<CudaSlice<u8>>),
        /// One of candle's own tensors (the embedding, the LM head).
        Tensor(candle_core::Tensor),
    }

    /// A tensor on the device, in whatever form it was uploaded.
    pub struct DeviceTensor {
        buf: Backing,
        pub ggml_type: u32,
        /// ggml order: `dims[0]` is the fastest-varying (the input width
        /// of a projection).
        pub dims: Vec<u64>,
    }

    impl DeviceTensor {
        pub fn address(&self) -> u64 {
            match &self.buf {
                Backing::Owned(s) => s.device_ptr(s.stream()).0,
                Backing::Shared(s) => s.device_ptr(s.stream()).0,
                Backing::Tensor(t) => {
                    let (storage, layout) = t.storage_and_layout();
                    let candle_core::Storage::Cuda(storage) = &*storage else {
                        unreachable!("checked CUDA in from_tensor");
                    };
                    let (base, size) = match t.dtype() {
                        candle_core::DType::BF16 => {
                            let s = storage.as_cuda_slice::<half::bf16>().expect("checked bf16");
                            (s.device_ptr(s.stream()).0, 2)
                        }
                        _ => {
                            let s = storage.as_cuda_slice::<f32>().expect("checked f32");
                            (s.device_ptr(s.stream()).0, 4)
                        }
                    };
                    base + (layout.start_offset() * size) as u64
                }
            }
        }

        /// Wraps a contiguous BF16 or f32 candle tensor on the GPU, which
        /// the engine then reads in place. `dims` in ggml order.
        pub fn from_tensor(t: candle_core::Tensor, dims: Vec<u64>) -> Result<Self> {
            ensure!(t.device().is_cuda(), "a shared tensor must be on the GPU");
            ensure!(t.is_contiguous(), "a shared tensor must be contiguous");
            let ggml_type = match t.dtype() {
                candle_core::DType::BF16 => GGML_BF16,
                candle_core::DType::F32 => GGML_F32,
                other => bail!("cannot share a {other:?} tensor"),
            };
            ensure!(
                t.elem_count() as u64 == dims.iter().product::<u64>(),
                "a {:?} tensor is not {dims:?}",
                t.dims()
            );
            Ok(Self {
                buf: Backing::Tensor(t),
                ggml_type,
                dims,
            })
        }

        pub fn values(&self) -> usize {
            self.dims.iter().product::<u64>() as usize
        }
    }

    /// The GGUF's shards, mapped, and the device tensors go to.
    pub struct Gguf {
        pub index: GgufIndex,
        maps: Vec<Mmap>,
        device: CudaDevice,
        /// Candle's dense source, whose uploads in the GGUF's own layout
        /// the engine shares instead of uploading again.
        shared: Option<Arc<GgufDenseSource>>,
    }

    impl Gguf {
        pub fn open(first: &Path, device: &CudaDevice) -> Result<Self> {
            let index = GgufIndex::open(first)?;
            let mut maps = Vec::with_capacity(index.shards.len());
            for path in &index.shards {
                let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
                // SAFETY: model files are not modified while serving.
                maps.push(
                    unsafe { Mmap::map(&file) }
                        .with_context(|| format!("map {}", path.display()))?,
                );
            }
            Ok(Self {
                index,
                maps,
                device: device.clone(),
                shared: None,
            })
        }

        /// Shares `source`'s uploads wherever it has the same tensor in
        /// the same layout.
        pub fn sharing(mut self, source: Arc<GgufDenseSource>) -> Self {
            self.shared = Some(source);
            self
        }

        fn tensor(&self, name: &str) -> Result<&GgufTensor> {
            self.index
                .tensors
                .get(name)
                .with_context(|| format!("the GGUF has no {name}"))
        }

        /// The tensor's stored bytes. Only for the types whose size the
        /// engine knows; quantized projections use `upload_raw`.
        fn bytes(&self, name: &str, size: usize) -> Result<&[u8]> {
            let t = self.tensor(name)?;
            let map = &self.maps[t.shard];
            let start = t.offset as usize;
            ensure!(start + size <= map.len(), "{name} runs past its shard");
            Ok(&map[start..start + size])
        }

        fn upload(&self, host: &[u8], ggml_type: u32, dims: Vec<u64>) -> Result<DeviceTensor> {
            // SAFETY: written in full by the copy below.
            let mut buf = unsafe { self.device.alloc::<u8>(host.len().max(1)) }?;
            self.device.memcpy_htod(host, &mut buf)?;
            Ok(DeviceTensor {
                buf: Backing::Owned(buf),
                ggml_type,
                dims,
            })
        }

        /// The tensor as stored, for the kernels that read its format.
        /// `row_bytes` gives the size of one `dims[0]`-wide row.
        pub fn upload_raw(
            &self,
            name: &str,
            row_bytes: impl Fn(u32, u64) -> usize,
        ) -> Result<DeviceTensor> {
            let t = self.tensor(name)?;
            let rows: u64 = t.dims.iter().skip(1).product();
            let size = row_bytes(t.ggml_type, t.dims[0]) * rows as usize;
            ensure!(size > 0, "{name}: no row size for type {}", t.ggml_type);
            let host = self.bytes(name, size)?;
            if let Some(buf) = self.shared.as_ref().and_then(|s| s.shared(name)) {
                ensure!(
                    buf.len() == size,
                    "{name}: the shared copy is {} bytes, not {size}",
                    buf.len()
                );
                return Ok(DeviceTensor {
                    buf: Backing::Shared(buf),
                    ggml_type: t.ggml_type,
                    dims: t.dims.clone(),
                });
            }
            self.upload(host, t.ggml_type, t.dims.clone())
        }

        fn values_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<u64>)> {
            let t = self.tensor(name)?;
            let n: u64 = t.dims.iter().product();
            let Some(size) = tensor_bytes(t.ggml_type, n as usize) else {
                bail!("{name}: the engine cannot convert type {}", t.ggml_type);
            };
            let values = to_f32(self.bytes(name, size)?, t.ggml_type)
                .with_context(|| format!("{name}: unconvertible type {}", t.ggml_type))?;
            Ok((values, t.dims.clone()))
        }

        /// The tensor as f32.
        pub fn upload_f32(&self, name: &str) -> Result<DeviceTensor> {
            let (values, dims) = self.values_f32(name)?;
            let host: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.upload(&host, GGML_F32, dims)
        }

        /// The tensor as BF16, rounded to nearest even.
        pub fn upload_bf16(&self, name: &str) -> Result<DeviceTensor> {
            let (values, dims) = self.values_f32(name)?;
            let host: Vec<u8> = values
                .iter()
                .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
                .collect();
            self.upload(&host, GGML_BF16, dims)
        }

        pub fn device(&self) -> &CudaDevice {
            &self.device
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_converts_exactly() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        // Smallest subnormal, 2^-24, and a larger one, 2^-15.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    /// Ties go to the even mantissa, and anything past half rounds up.
    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        // 1 + 2^-8 is exactly halfway between 1 and 1 + 2^-7: to even.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3f80);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 (odd) and 1 + 2^-6.
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3f82);
        // Just past halfway rounds up.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8) + 2f32.powi(-20)), 0x3f81);
        assert_eq!(f32_to_bf16(-2.5), 0xc020);
    }

    /// One Q6_K block built by hand: every 6-bit value is assembled from
    /// its low nibble and high bit pair, offset by 32, and scaled by its
    /// group's scale and the block scale.
    #[test]
    fn q6_k_assembles_six_bit_values() {
        let mut block = vec![0u8; 210];
        // Value 0: low nibble 5, high bits 0b10 -> 0b10_0101 = 37, -32 = 5.
        block[0] = 0x05;
        block[128] = 0b10;
        // Value 96 (fourth quarter: ql[32]'s high nibble, qh[0] bits 6-7):
        // nibble 0xf, high bits 0b01 -> 0b01_1111 = 31, -32 = -1.
        block[32] = 0xf0;
        block[128] |= 0b01 << 6;
        // Scales: group 0 = 3, group 6 (value 96's) = -2.
        block[192] = 3;
        block[198] = (-2i8) as u8;
        block[208..210].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        let y = q6_k_to_f32(&block);
        assert_eq!(y.len(), 256);
        assert_eq!(y[0], 0.5 * 3.0 * 5.0);
        // scale 0.5, group scale -2, value -1.
        assert_eq!(y[96], 1.0);
        // An all-zero position reads 0 - 32 = -32 times its scale.
        assert_eq!(y[1], 0.5 * 3.0 * -32.0);
        assert_eq!(tensor_bytes(GGML_Q6_K, 512), Some(420));
    }

    #[test]
    fn q8_0_blocks_scale_their_bytes() {
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        for (i, b) in block[2..].iter_mut().enumerate() {
            *b = (i as i8 - 16) as u8;
        }
        let values = q8_0_to_f32(&block);
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], -8.0);
        assert_eq!(values[16], 0.0);
        assert_eq!(values[31], 7.5);
        assert_eq!(tensor_bytes(GGML_Q8_0, 64), Some(68));
        assert_eq!(tensor_bytes(GGML_Q8_0, 33), None);
    }
}
