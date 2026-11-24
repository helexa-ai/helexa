//! A full-attention layer's mixer half, native, for contexts within the
//! QSA budget.
//!
//! One decode token: the attention hyper-connection's read; the Q8_0
//! q (query and output gate interleaved per head), k and v projections;
//! RMS norm and rotary on q and k with the position read from device
//! memory; the K/V append into a cache allocated once; dense decode
//! attention over every cached position, with the per-head sigmoid
//! output gate; the Q8_0 output projection; the hyper-connection's write.
//!
//! Dense is exact while the context fits QSA's budget (2,048 tokens),
//! where the selection keeps every position. The indexer's raw key is
//! still projected and cached each step, so selection can take over
//! beyond the budget without rebuilding that cache.

use std::ffi::c_void;

use anyhow::{Result, ensure};
use candle_core::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};

use super::gdn::{GdnShapes, check};
use super::weights::{DeviceTensor, Gguf};
use crate::cuda::strata as ffi;

const Q8_1_BLOCK_VALUES: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;

fn address<T>(slice: &CudaSlice<T>) -> u64 {
    slice.device_ptr(slice.stream()).0
}

#[derive(Debug, Clone, Copy)]
pub struct AttnShapes {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub index_dim: usize,
    /// Positions the cache holds.
    pub max_len: usize,
}

impl AttnShapes {
    fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }
    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

pub struct AttnWeights {
    hc_norm: DeviceTensor,
    hc_down: DeviceTensor,
    hc_up: DeviceTensor,
    hc_inject: DeviceTensor,
    q: DeviceTensor,
    k: DeviceTensor,
    v: DeviceTensor,
    o: DeviceTensor,
    q_norm: DeviceTensor,
    k_norm: DeviceTensor,
    index_k: DeviceTensor,
}

fn row_bytes(ggml_type: u32, n: u64) -> usize {
    // SAFETY: a pure function of its arguments.
    unsafe { ffi::neuron_strata_row_bytes(ggml_type as i32, n as i64) }
}

impl AttnWeights {
    pub fn load(gguf: &Gguf, layer: usize, s: &GdnShapes, a: &AttnShapes) -> Result<Self> {
        let name = |t: &str| format!("blk.{layer}.{t}");
        let projection = |t: &str, n_in: usize, n_out: usize| -> Result<DeviceTensor> {
            let w = gguf.upload_raw(&name(t), row_bytes)?;
            ensure!(
                w.dims == [n_in as u64, n_out as u64],
                "{} is {:?}, expected [{n_in}, {n_out}]",
                name(t),
                w.dims
            );
            // SAFETY: a pure function of its argument.
            ensure!(
                unsafe { ffi::neuron_strata_mmvq_supported(w.ggml_type as i32) } != 0,
                "{}: no mmvq for type {}",
                name(t),
                w.ggml_type
            );
            Ok(w)
        };
        let wide = s.hc * s.n_embd;
        let w = Self {
            hc_norm: gguf.upload_f32(&name("hc_attn_norm.weight"))?,
            hc_down: gguf.upload_bf16(&name("hc_attn_down.weight"))?,
            hc_up: gguf.upload_bf16(&name("hc_attn_up.weight"))?,
            hc_inject: gguf.upload_bf16(&name("hc_attn_inject.weight"))?,
            q: projection("attn_q.weight", s.n_embd, 2 * a.q_width())?,
            k: projection("attn_k.weight", s.n_embd, a.kv_width())?,
            v: projection("attn_v.weight", s.n_embd, a.kv_width())?,
            o: projection("attn_output.weight", a.q_width(), s.n_embd)?,
            q_norm: gguf.upload_f32(&name("attn_q_norm.weight"))?,
            k_norm: gguf.upload_f32(&name("attn_k_norm.weight"))?,
            index_k: gguf.upload_bf16(&name("indexer.k_proj.weight"))?,
        };
        for (t, want) in [
            (&w.hc_norm, wide),
            (&w.hc_down, s.hc_lr * wide),
            (&w.hc_up, wide * s.hc_lr),
            (&w.hc_inject, s.hc * wide),
            (&w.q_norm, a.head_dim),
            (&w.k_norm, a.head_dim),
            (&w.index_k, a.index_dim * s.n_embd),
        ] {
            ensure!(
                t.values() == want,
                "layer {layer}: a tensor has {} values, expected {want}",
                t.values()
            );
        }
        Ok(w)
    }
}

/// A sequence's caches for one full-attention layer.
pub struct AttnCache {
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// The indexer's raw (unnormalised, unrotated) keys.
    index_keys: CudaSlice<f32>,
}

impl AttnCache {
    /// K, V and the indexer keys, for export.
    pub fn slices(&self) -> (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>) {
        (&self.k, &self.v, &self.index_keys)
    }

    /// Fills the front of the caches from rows already in the engine's
    /// layout: K and V `(len, kv_heads, head_dim)`, keys `(len, index_dim)`.
    pub fn import(
        &mut self,
        device: &CudaDevice,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        index_keys: &candle_core::Tensor,
    ) -> Result<()> {
        use super::decoder::copy_into;
        copy_into(device, k, &mut self.k)?;
        copy_into(device, v, &mut self.v)?;
        copy_into(device, index_keys, &mut self.index_keys)
    }

    /// Zeroes the caches in place: a captured step keeps its addresses.
    pub fn clear(&mut self, device: &CudaDevice) -> Result<()> {
        let stream = device.cuda_stream();
        stream.memset_zeros(&mut self.k)?;
        stream.memset_zeros(&mut self.v)?;
        stream.memset_zeros(&mut self.index_keys)?;
        Ok(())
    }

    pub fn new(device: &CudaDevice, a: &AttnShapes) -> Result<Self> {
        Ok(Self {
            k: device.alloc_zeros::<f32>(a.max_len * a.kv_width())?,
            v: device.alloc_zeros::<f32>(a.max_len * a.kv_width())?,
            index_keys: device.alloc_zeros::<f32>(a.max_len * a.index_dim)?,
        })
    }
}

pub struct AttnScratch {
    gr: CudaSlice<u8>,
    mixed: CudaSlice<f32>,
    inject: CudaSlice<f32>,
    x_q8: CudaSlice<u8>,
    q_raw: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k_raw: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    index_key: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    ctx_q8: CudaSlice<u8>,
    out: CudaSlice<f32>,
}

impl AttnScratch {
    pub fn new(device: &CudaDevice, s: &GdnShapes, a: &AttnShapes) -> Result<Self> {
        // SAFETY: a pure function of its arguments.
        let gr_bytes =
            unsafe { ffi::neuron_gr_workspace_bytes(s.n_embd as i64, s.hc as i64, s.hc_lr as i64) };
        let q8 = |n: usize| n / Q8_1_BLOCK_VALUES * Q8_1_BLOCK_BYTES;
        Ok(Self {
            gr: device.alloc_zeros::<u8>(gr_bytes)?,
            mixed: device.alloc_zeros::<f32>(s.n_embd)?,
            inject: device.alloc_zeros::<f32>(s.hc)?,
            x_q8: device.alloc_zeros::<u8>(q8(s.n_embd))?,
            q_raw: device.alloc_zeros::<f32>(2 * a.q_width())?,
            q: device.alloc_zeros::<f32>(a.q_width())?,
            k_raw: device.alloc_zeros::<f32>(a.kv_width())?,
            k: device.alloc_zeros::<f32>(a.kv_width())?,
            v: device.alloc_zeros::<f32>(a.kv_width())?,
            index_key: device.alloc_zeros::<f32>(a.index_dim)?,
            ctx: device.alloc_zeros::<f32>(a.q_width())?,
            ctx_q8: device.alloc_zeros::<u8>(q8(a.q_width()))?,
            out: device.alloc_zeros::<f32>(s.n_embd)?,
        })
    }
}

pub(super) fn mmvq(w: &DeviceTensor, x_q8: u64, y: u64, stream: *mut c_void) {
    // SAFETY: `w` is `dims[1]` rows of `dims[0]`, `x_q8` holds `dims[0]`
    // values in q8_1 and `y` has room for `dims[1]` floats.
    unsafe {
        ffi::neuron_strata_mmvq(
            w.ggml_type as i32,
            w.address() as *const c_void,
            x_q8 as *const c_void,
            y as *mut f32,
            w.dims[0] as i32,
            w.dims[1] as i32,
            1,
            stream,
        )
    }
}

pub(super) fn quantize(x: u64, n: usize, q8: u64, stream: *mut c_void) {
    // SAFETY: `x` holds `n` floats and `q8` room for their q8_1 blocks.
    unsafe {
        ffi::neuron_strata_quantize_q8_1(x as *const f32, 1, n as i64, q8 as *mut c_void, stream)
    }
}

/// The mixer half of one full-attention layer, for one token at the
/// position in `positions[0]` (a device array of at least `heads` equal
/// entries: the rotary reads one per row).
#[allow(clippy::too_many_arguments)]
pub fn forward(
    w: &AttnWeights,
    cache: &mut AttnCache,
    sc: &mut AttnScratch,
    s: &GdnShapes,
    a: &AttnShapes,
    positions: u64,
    residual: u64,
    residual_out: u64,
    stream: *mut c_void,
) -> Result<()> {
    let (n_embd, hc, hc_lr) = (s.n_embd as i64, s.hc as i64, s.hc_lr as i64);
    let pos = positions as *const i32;
    // SAFETY (all calls below): every pointer is a live device buffer of
    // the size its kernel reads or writes, as allocated in `AttnScratch`,
    // `AttnCache` and `AttnWeights`; `positions` holds at least `heads`
    // entries; every launch is on `stream`.
    unsafe {
        check(
            ffi::neuron_gr_read(
                residual as *const f32,
                w.hc_norm.address() as *const f32,
                w.hc_down.address() as *const u16,
                w.hc_up.address() as *const u16,
                w.hc_inject.address() as *const u16,
                s.eps,
                n_embd,
                hc,
                hc_lr,
                address(&sc.gr) as *mut c_void,
                address(&sc.mixed) as *mut f32,
                address(&sc.inject) as *mut f32,
                stream,
            ),
            "attn gr_read",
        )?;
    }
    quantize(address(&sc.mixed), s.n_embd, address(&sc.x_q8), stream);
    mmvq(&w.q, address(&sc.x_q8), address(&sc.q_raw), stream);
    mmvq(&w.k, address(&sc.x_q8), address(&sc.k_raw), stream);
    mmvq(&w.v, address(&sc.x_q8), address(&sc.v), stream);
    // SAFETY: as above.
    unsafe {
        // q rows are [query | gate] per head: read the query half.
        check(
            ffi::neuron_norm_rope(
                address(&sc.q_raw) as *const f32,
                (2 * a.head_dim) as i32,
                w.q_norm.address() as *const f32,
                address(&sc.q) as *mut f32,
                a.heads as i32,
                a.head_dim as i32,
                a.n_rot as i32,
                s.eps,
                pos,
                stream,
            ),
            "q norm + rope",
        )?;
        check(
            ffi::neuron_norm_rope(
                address(&sc.k_raw) as *const f32,
                a.head_dim as i32,
                w.k_norm.address() as *const f32,
                address(&sc.k) as *mut f32,
                a.kv_heads as i32,
                a.head_dim as i32,
                a.n_rot as i32,
                s.eps,
                pos,
                stream,
            ),
            "k norm + rope",
        )?;
        check(
            ffi::neuron_bf16_gemv(
                address(&sc.mixed) as *const f32,
                w.index_k.address() as *const u16,
                address(&sc.index_key) as *mut f32,
                n_embd,
                a.index_dim as i64,
                stream,
            ),
            "indexer key",
        )?;
        check(
            ffi::neuron_append_rows(
                address(&sc.k) as *const f32,
                address(&sc.v) as *const f32,
                address(&cache.k) as *mut f32,
                address(&cache.v) as *mut f32,
                pos,
                a.kv_width() as i32,
                stream,
            ),
            "kv append",
        )?;
        check(
            ffi::neuron_append_rows(
                address(&sc.index_key) as *const f32,
                std::ptr::null(),
                address(&cache.index_keys) as *mut f32,
                std::ptr::null_mut(),
                pos,
                a.index_dim as i32,
                stream,
            ),
            "indexer key append",
        )?;
        check(
            ffi::neuron_decode_attn(
                address(&sc.q) as *const f32,
                address(&cache.k) as *const f32,
                address(&cache.v) as *const f32,
                pos,
                a.heads as i32,
                a.kv_heads as i32,
                a.head_dim as i32,
                1.0 / (a.head_dim as f32).sqrt(),
                (address(&sc.q_raw) + (a.head_dim * 4) as u64) as *const f32,
                (2 * a.head_dim) as i32,
                address(&sc.ctx) as *mut f32,
                stream,
            ),
            "decode attention",
        )?;
    }
    quantize(address(&sc.ctx), a.q_width(), address(&sc.ctx_q8), stream);
    mmvq(&w.o, address(&sc.ctx_q8), address(&sc.out), stream);
    // SAFETY: as above.
    unsafe {
        check(
            ffi::neuron_gr_write(
                residual as *const f32,
                address(&sc.out) as *const f32,
                address(&sc.inject) as *const f32,
                n_embd,
                hc,
                hc_lr,
                residual_out as *mut f32,
                stream,
            ),
            "attn gr_write",
        )?;
    }
    Ok(())
}

/// The attention shapes of a qwen4_exp config, with a cache of
/// `max_len` positions.
pub fn shapes(cfg: &super::super::config::TextConfig, max_len: usize) -> Result<AttnShapes> {
    let n_rot =
        (cfg.head_dim as f64 * f64::from(cfg.rope_parameters.partial_rotary_factor)) as usize;
    let a = AttnShapes {
        heads: cfg.num_attention_heads,
        kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        n_rot,
        index_dim: cfg.indexer_head_dim,
        max_len,
    };
    ensure!(
        a.head_dim == 256 && n_rot == 64,
        "native attention is built for 256-wide heads, 64 rotated"
    );
    ensure!(
        a.heads.is_multiple_of(a.kv_heads),
        "query heads must be a multiple of kv heads"
    );
    Ok(a)
}
