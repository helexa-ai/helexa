//! The native decode step (#427): every layer of the model for one token,
//! over buffers allocated once.
//!
//! A step is the embedding, the layers before PLE, the PLE block, the
//! layers from PLE on, the final mixer and the LM head: fixed work over
//! fixed addresses, with the token, its position and its n-gram rows
//! read from pinned host memory at the start of the step, so the whole
//! step is captured as one CUDA graph and replayed. Only the n-gram hash
//! and the table read stay on the host, between steps.

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use candle_core::cuda_backend::cudarc::driver::{CudaGraph, CudaSlice, DevicePtr, sys};
use candle_core::{CudaDevice, Storage, Tensor};

use super::attn::{self, AttnCache, AttnScratch, AttnShapes, AttnWeights};
use super::gdn::{self, GdnScratch, GdnShapes, GdnState, GdnWeights, check};
use super::moe::{self, MoeScratch, MoeShapes, MoeWeights};
use super::ple::PleNative;
use super::weights::{DeviceTensor, Gguf};
use crate::cuda::strata as ffi;
use crate::harness::arch::qwen4_exp::config::TextConfig;
use crate::harness::arch::qwen4_exp::native_experts::GgufExpertSource;

/// Positions are written for every query-head row the rotary reads.
const POSITION_SLOTS: usize = 32;
/// QSA's step: pos, n_kv, completed blocks, selection width.
const QSA_STEP_SLOTS: usize = 4;
/// The step buffer: `POSITION_SLOTS` positions, the token id, QSA's step.
const STEP_SLOTS: usize = QSA_STEP_AT + QSA_STEP_SLOTS;
/// Where QSA's step starts in the step buffer, in i32s. Kept 16-byte
/// aligned for the kernels that read it as a vector.
const QSA_STEP_AT: usize = POSITION_SLOTS + 4;

fn address<T>(slice: &CudaSlice<T>) -> u64 {
    slice.device_ptr(slice.stream()).0
}

enum Mixer {
    Linear(GdnWeights, GdnState),
    Full(AttnWeights, AttnCache),
}

struct Layer {
    mixer: Mixer,
    moe: MoeWeights,
}

/// Weights the engine reads from candle's model instead of loading its
/// own: the embedding and the LM head, both `(vocab, hidden)` BF16.
pub struct SharedWeights {
    pub embed: Tensor,
    pub head: Tensor,
}

/// Pinned host memory the step inputs are copied from.
struct HostStep(*mut i32);

// SAFETY: plain host memory, written by the device worker between steps.
unsafe impl Send for HostStep {}

impl Drop for HostStep {
    fn drop(&mut self) {
        // SAFETY: allocated by `neuron_strata_pinned_alloc`, freed once.
        unsafe { ffi::neuron_strata_pinned_free(self.0.cast()) };
    }
}

/// A captured segment.
struct Graph(CudaGraph);

// SAFETY: a graph is launched only from the device worker that owns the
// decoder, on the stream it was captured on.
unsafe impl Send for Graph {}

pub struct NativeDecoder {
    /// The captured step, once a warm step has run.
    graph: Option<Graph>,
    warmed: bool,
    ple: PleNative,
    device: CudaDevice,
    s: GdnShapes,
    a: AttnShapes,
    m: MoeShapes,
    layers: Vec<Layer>,
    ple_layer: usize,
    experts: Arc<GgufExpertSource>,
    gdn_sc: GdnScratch,
    attn_sc: AttnScratch,
    moe_sc: MoeScratch,
    embed: DeviceTensor,
    embed_row: CudaSlice<f32>,
    mixer_norm: DeviceTensor,
    mixer_down: DeviceTensor,
    mixer_up: DeviceTensor,
    mixer_ws: CudaSlice<u8>,
    mixed: CudaSlice<f32>,
    unused_inject: CudaSlice<f32>,
    head: DeviceTensor,
    logits: CudaSlice<f32>,
    /// The residual and its other half: every half-layer reads one and
    /// writes the other, so each layer starts and ends in `residual[0]`.
    residual: [CudaSlice<f32>; 2],
    step: CudaSlice<i32>,
    host_step: HostStep,
    vocab: usize,
}

fn row_bytes(ggml_type: u32, n: u64) -> usize {
    // SAFETY: a pure function of its arguments.
    unsafe { ffi::neuron_strata_row_bytes(ggml_type as i32, n as i64) }
}

impl NativeDecoder {
    /// Loads every layer from the GGUF. `experts` must route on the
    /// device; `max_len` sizes the attention caches.
    pub fn load(
        gguf: &Gguf,
        cfg: &TextConfig,
        experts: Arc<GgufExpertSource>,
        max_len: usize,
        shared: Option<SharedWeights>,
    ) -> Result<Self> {
        gdn::configure();
        let device = gguf.device().clone();
        let s = gdn::shapes(cfg)?;
        let a = attn::shapes(cfg, max_len)?;
        let m = moe::shapes(cfg)?;
        ensure!(
            a.heads <= POSITION_SLOTS,
            "more query heads than position slots"
        );
        let ple_layers = cfg.ple_layers();
        ensure!(
            ple_layers.len() == 1,
            "the engine expects exactly one PLE layer"
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, kind) in cfg
            .layer_types
            .iter()
            .enumerate()
            .take(cfg.num_hidden_layers)
        {
            let mixer = match kind.as_str() {
                "linear_attention" => Mixer::Linear(
                    GdnWeights::load(gguf, i, &s)?,
                    GdnState::zeroed(&device, &s)?,
                ),
                "full_attention" => Mixer::Full(
                    AttnWeights::load(gguf, i, &s, &a)?,
                    AttnCache::new(&device, &a)?,
                ),
                other => bail!("layer {i}: unknown layer type {other}"),
            };
            layers.push(Layer {
                mixer,
                moe: MoeWeights::load(gguf, i, &s, &m)?,
            });
        }
        let (embed, head) = match shared {
            // Candle's own BF16 embedding and LM head, read in place: the
            // checkpoint's values, and no second copy on the card.
            Some(w) => {
                let dims = vec![s.n_embd as u64, cfg.vocab_size as u64];
                (
                    DeviceTensor::from_tensor(w.embed, dims.clone())?,
                    DeviceTensor::from_tensor(w.head, dims)?,
                )
            }
            None => (
                gguf.upload_raw("token_embd.weight", row_bytes)?,
                gguf.upload_bf16("output.weight")?,
            ),
        };
        ensure!(
            embed.dims == [s.n_embd as u64, cfg.vocab_size as u64],
            "token_embd is {:?}",
            embed.dims
        );
        ensure!(
            head.dims == [s.n_embd as u64, cfg.vocab_size as u64],
            "output is {:?}",
            head.dims
        );
        // SAFETY: a pure function of its arguments.
        let gr_bytes =
            unsafe { ffi::neuron_gr_workspace_bytes(s.n_embd as i64, s.hc as i64, s.hc_lr as i64) };
        // SAFETY: returns null or STEP_SLOTS i32 of pinned host memory.
        let host = unsafe { ffi::neuron_strata_pinned_alloc(STEP_SLOTS * 4) }.cast::<i32>();
        ensure!(!host.is_null(), "cannot pin the step buffer");
        let wide = s.hc * s.n_embd;
        Ok(Self {
            graph: None,
            warmed: false,
            gdn_sc: GdnScratch::new(&device, &s)?,
            attn_sc: AttnScratch::new(&device, &s, &a)?,
            moe_sc: MoeScratch::new(&device, &s, &m)?,
            embed,
            embed_row: device.alloc_zeros::<f32>(s.n_embd)?,
            mixer_norm: gguf.upload_f32("output_hc_norm.weight")?,
            mixer_down: gguf.upload_bf16("output_hc_down.weight")?,
            mixer_up: gguf.upload_bf16("output_hc_up.weight")?,
            mixer_ws: device.alloc_zeros::<u8>(gr_bytes)?,
            mixed: device.alloc_zeros::<f32>(s.n_embd)?,
            unused_inject: device.alloc_zeros::<f32>(s.hc)?,
            head,
            logits: device.alloc_zeros::<f32>(cfg.vocab_size)?,
            residual: [
                device.alloc_zeros::<f32>(wide)?,
                device.alloc_zeros::<f32>(wide)?,
            ],
            step: device.alloc_zeros::<i32>(STEP_SLOTS)?,
            host_step: HostStep(host),
            vocab: cfg.vocab_size,
            ple: PleNative::load(gguf, ple_layers[0], &s)?,
            ple_layer: ple_layers[0],
            device,
            s,
            a,
            m,
            layers,
            experts,
        })
    }

    pub fn stream(&self) -> *mut c_void {
        self.device.cuda_stream().cu_stream() as *mut c_void
    }

    /// Where the step's n-gram rows go (`n_embd` f32): written by the
    /// caller between steps, with [`Self::set_step`].
    pub fn ple_input(&mut self) -> &mut [f32] {
        self.ple.input()
    }

    pub fn shapes(&self) -> (GdnShapes, AttnShapes, MoeShapes) {
        (self.s, self.a, self.m)
    }

    /// Writes the step's inputs to the pinned buffer. Must come before
    /// `segment_head` and after the previous step's logits were read.
    pub fn set_step(&mut self, token: u32, pos: usize) -> Result<()> {
        ensure!(
            pos < self.a.max_len,
            "position {pos} is past the {}-position cache",
            self.a.max_len
        );
        // SAFETY: the buffer holds STEP_SLOTS i32, and no copy from it is
        // in flight (the previous step's logits were read).
        let host = unsafe { std::slice::from_raw_parts_mut(self.host_step.0, STEP_SLOTS) };
        host[..POSITION_SLOTS].fill(pos as i32);
        host[POSITION_SLOTS] = token as i32;
        // Strata's `qsa_step_write`: every count derived from `pos` here,
        // in one place.
        let n_kv = pos + 1;
        host[QSA_STEP_AT..STEP_SLOTS].copy_from_slice(&[
            pos as i32,
            n_kv as i32,
            (n_kv / attn::BLOCK) as i32,
            n_kv.min(self.a.cap) as i32,
        ]);
        Ok(())
    }

    /// The host-side tier work between steps: what the experts learned
    /// from the last step, and admissions.
    pub fn adapt(&self) -> Result<()> {
        for l in 0..self.layers.len() {
            self.experts.adapt(l)?;
        }
        Ok(())
    }

    /// After a replay: the usage copies it made are this step's.
    pub fn usage_replayed(&self) {
        for l in 0..self.layers.len() {
            self.experts.usage_replayed(l);
        }
    }

    fn run_layer(&mut self, i: usize) -> Result<()> {
        let stream = self.stream();
        let (r0, r1) = (address(&self.residual[0]), address(&self.residual[1]));
        let positions = address(&self.step);
        let qsa_step = positions + (QSA_STEP_AT * 4) as u64;
        let layer = &mut self.layers[i];
        match &mut layer.mixer {
            Mixer::Linear(w, state) => {
                gdn::forward(w, state, &mut self.gdn_sc, &self.s, r0, r1, stream)?
            }
            Mixer::Full(w, cache) => attn::forward(
                w,
                cache,
                &mut self.attn_sc,
                &self.s,
                &self.a,
                positions,
                qsa_step,
                r0,
                r1,
                stream,
            )?,
        }
        moe::forward(
            &layer.moe,
            &self.experts,
            i,
            &mut self.moe_sc,
            &self.s,
            &self.m,
            r1,
            r0,
            stream,
        )
    }

    /// The step, eagerly: [`Self::set_step`] and [`Self::ple_input`]
    /// must have been written.
    pub fn segment(&mut self) -> Result<()> {
        self.segment_head()?;
        self.ple
            .forward(address(&self.residual[0]), self.stream())
            .with_context(|| format!("PLE (layer {})", self.ple_layer))?;
        self.segment_tail()
    }

    /// The step inputs, the embedding into every stream, and the layers
    /// before the PLE layer.
    fn segment_head(&mut self) -> Result<()> {
        let stream = self.stream();
        // SAFETY: the pinned source holds STEP_SLOTS i32 and the device
        // buffer as many; both live as long as `self`.
        let status = unsafe {
            ffi::neuron_strata_copy_to_device(
                address(&self.step) as *mut c_void,
                self.host_step.0.cast(),
                STEP_SLOTS * 4,
                stream,
            )
        };
        ensure!(
            status == 0,
            "step input copy failed with CUDA status {status}"
        );
        let (n_embd, hc) = (self.s.n_embd, self.s.hc);
        let token = address(&self.step) + (POSITION_SLOTS * 4) as u64;
        // SAFETY: the table holds `vocab` rows; `token` is one device id;
        // the outputs are sized for one row and `hc` copies of it.
        unsafe {
            check(
                ffi::neuron_embed_rows(
                    self.embed.ggml_type as i32,
                    self.embed.address() as *const c_void,
                    if self.embed.ggml_type == super::weights::GGML_BF16 {
                        n_embd * 2
                    } else {
                        row_bytes(self.embed.ggml_type, n_embd as u64)
                    },
                    token as *const i32,
                    1,
                    n_embd as i64,
                    address(&self.embed_row) as *mut f32,
                    stream,
                ),
                "embedding",
            )?;
            check(
                ffi::neuron_repeat_row(
                    address(&self.embed_row) as *const f32,
                    address(&self.residual[0]) as *mut f32,
                    n_embd as i32,
                    hc as i32,
                    stream,
                ),
                "embedding into the streams",
            )?;
        }
        for i in 0..self.ple_layer {
            self.run_layer(i).with_context(|| format!("layer {i}"))?;
        }
        Ok(())
    }

    /// The step, from its graph: captured on first use after a warm
    /// eager step (which sizes every buffer the step touches), then
    /// replayed. The first call of all runs it eagerly.
    pub fn run_step(&mut self) -> Result<()> {
        if !self.warmed {
            self.segment()?;
            self.warmed = true;
            return Ok(());
        }
        if self.graph.is_none() {
            let stream = self.device.cuda_stream();
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let ran = self.segment();
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            ran.context("while capturing")?;
            self.graph = Some(Graph(graph?.context("capture produced no graph")?));
        }
        self.graph.as_ref().expect("captured above").0.launch()?;
        self.usage_replayed();
        Ok(())
    }

    /// The layers from the PLE layer on, the final mixer and the LM head,
    /// into the logits buffer.
    fn segment_tail(&mut self) -> Result<()> {
        for i in self.ple_layer..self.layers.len() {
            self.run_layer(i).with_context(|| format!("layer {i}"))?;
        }
        let stream = self.stream();
        let s = self.s;
        // SAFETY: as in the layers; a null injection is the final
        // mixer's, which collapses the streams without a write-back.
        unsafe {
            check(
                ffi::neuron_gr_read(
                    address(&self.residual[0]) as *const f32,
                    self.mixer_norm.address() as *const f32,
                    self.mixer_down.address() as *const u16,
                    self.mixer_up.address() as *const u16,
                    std::ptr::null(),
                    s.eps,
                    s.n_embd as i64,
                    s.hc as i64,
                    s.hc_lr as i64,
                    address(&self.mixer_ws) as *mut c_void,
                    address(&self.mixed) as *mut f32,
                    address(&self.unused_inject) as *mut f32,
                    stream,
                ),
                "final mixer",
            )?;
            check(
                ffi::neuron_bf16_gemv(
                    address(&self.mixed) as *const f32,
                    self.head.address() as *const u16,
                    address(&self.logits) as *mut f32,
                    s.n_embd as i64,
                    self.vocab as i64,
                    stream,
                ),
                "lm head",
            )?;
        }
        Ok(())
    }

    /// The final mixer's output of the last step, `n_embd` f32: what the
    /// LM head reads.
    pub fn mixed(&self) -> &CudaSlice<f32> {
        &self.mixed
    }

    /// The logits of the last step (waits for it).
    pub fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.device.clone_dtoh(&self.logits)?)
    }

    /// The layer that carries PLE.
    pub fn ple_layer(&self) -> usize {
        self.ple_layer
    }

    /// How many layers there are, and whether layer `i` is linear.
    pub fn layer_is_linear(&self, i: usize) -> Option<bool> {
        self.layers
            .get(i)
            .map(|l| matches!(l.mixer, Mixer::Linear(..)))
    }

    /// Takes layer `i`'s linear-attention state from candle after a
    /// prefill (#427): `conv` is candle's `(1, channels, 4)` window and
    /// `recurrent` its `(1, v_heads, 128, 128)` state, both in the
    /// checkpoint's head order. The engine keeps the GGUF's order (value
    /// heads tiled by key head) and Strata's layouts: a `[channel][3]`
    /// history of the last three inputs, and `[row][head][col]` state.
    pub fn import_linear(&mut self, i: usize, conv: &Tensor, recurrent: &Tensor) -> Result<()> {
        let s = self.s;
        let (hk, hv) = (s.k_heads, s.v_heads);
        let r = hv / hk;
        // GGUF value head p = g * hk + k holds checkpoint head r * k + g.
        let head_src: Vec<u32> = (0..hv).map(|p| ((p % hk) * r + p / hk) as u32).collect();
        let qk = 2 * hk * 128;
        let mut channel_src: Vec<u32> = (0..qk as u32).collect();
        for &h in &head_src {
            channel_src.extend((0..128u32).map(|d| qk as u32 + h * 128 + d));
        }
        let device = candle_core::Device::Cuda(self.device.clone());
        let channels = Tensor::new(channel_src.as_slice(), &device)?;
        let heads = Tensor::new(head_src.as_slice(), &device)?;
        let width = conv.dim(2)?;
        ensure!(width >= 3, "conv state is {width} wide");
        let conv = conv
            .squeeze(0)?
            .narrow(1, width - 3, 3)?
            .contiguous()?
            .index_select(&channels, 0)?
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?;
        let rec = recurrent
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?
            .index_select(&heads, 0)?
            .permute((1, 0, 2))?
            .contiguous()?;
        let Mixer::Linear(_, state) = &mut self.layers[i].mixer else {
            bail!("layer {i} is not linear attention");
        };
        copy_into(&self.device, &conv, &mut state.conv)?;
        copy_into(&self.device, &rec, &mut state.recurrent)
    }

    /// The engine's linear-attention state of layer `i` back in candle's
    /// form (the inverse of [`Self::import_linear`]): the conv window
    /// `(1, channels, 4)` in `dtype` (its first column, which candle's
    /// next step shifts out, zero) and the recurrent state
    /// `(1, v_heads, 128, 128)` f32, both in the checkpoint's head order.
    pub fn export_linear(&self, i: usize, dtype: candle_core::DType) -> Result<(Tensor, Tensor)> {
        let s = self.s;
        let (hk, hv) = (s.k_heads, s.v_heads);
        let r = hv / hk;
        // Checkpoint head h = r * k + g sits at GGUF position g * hk + k.
        let head_at: Vec<u32> = (0..hv).map(|h| ((h % r) * hk + h / r) as u32).collect();
        let qk = 2 * hk * 128;
        let mut channel_at: Vec<u32> = (0..qk as u32).collect();
        for &p in &head_at {
            channel_at.extend((0..128u32).map(|d| qk as u32 + p * 128 + d));
        }
        let device = candle_core::Device::Cuda(self.device.clone());
        let Mixer::Linear(_, state) = &self.layers[i].mixer else {
            bail!("layer {i} is not linear attention");
        };
        let channels = s.channels_total();
        let hist = tensor_of(&self.device, &state.conv, (channels, 3))?;
        let conv = Tensor::cat(
            &[
                &Tensor::zeros((channels, 1), candle_core::DType::F32, &device)?,
                &hist,
            ],
            1,
        )?
        .index_select(&Tensor::new(channel_at.as_slice(), &device)?, 0)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;
        let rec = tensor_of(&self.device, &state.recurrent, (128, hv, 128))?
            .permute((1, 0, 2))?
            .contiguous()?
            .index_select(&Tensor::new(head_at.as_slice(), &device)?, 0)?
            .unsqueeze(0)?;
        Ok((conv, rec))
    }

    /// The engine's attention caches of layer `i` for the first `len`
    /// positions, in candle's form: K and V `(1, kv_heads, len, head_dim)`
    /// and the indexer keys `(1, len, index_dim)`, in `dtype`.
    pub fn export_full(
        &self,
        i: usize,
        len: usize,
        dtype: candle_core::DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let Mixer::Full(_, cache) = &self.layers[i].mixer else {
            bail!("layer {i} is not full attention");
        };
        cache.export(&self.device, &self.a, len, dtype)
    }

    /// Takes layer `i`'s attention caches from candle after a prefill:
    /// K and V `(1, kv_heads, len, head_dim)` (normed and rotated) and the
    /// indexer's raw keys `(1, len, index_dim)`.
    pub fn import_full(
        &mut self,
        i: usize,
        k: &Tensor,
        v: &Tensor,
        index_keys: &Tensor,
    ) -> Result<()> {
        let Mixer::Full(w, cache) = &mut self.layers[i].mixer else {
            bail!("layer {i} is not full attention");
        };
        cache.import(&self.device, &self.a, k, v, index_keys, w, self.s.eps)
    }

    /// How many positions the attention caches hold.
    pub fn max_len(&self) -> usize {
        self.a.max_len
    }

    /// Takes PLE's rolling conv context from candle after a prefill
    /// (`(1, channels, 9)`, or `None` before any).
    pub fn import_ple(&mut self, state: Option<&Tensor>) -> Result<()> {
        self.ple.import(&self.device, state)
    }

    /// PLE's conv context in candle's form, for a hand-back.
    pub fn export_ple(&self, dtype: candle_core::DType) -> Result<Tensor> {
        self.ple.export(&self.device, dtype)
    }

    /// Clears every sequence state, in place, for a new request.
    pub fn reset(&mut self) -> Result<()> {
        self.ple.clear(&self.device)?;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Linear(_, state) => state.clear(&self.device)?,
                Mixer::Full(_, cache) => cache.clear(&self.device)?,
            }
        }
        Ok(())
    }
}

/// Copies a contiguous tensor's values into the front of `dst` (the
/// tensor's dtype must be `T`).
pub(crate) fn copy_into<T>(device: &CudaDevice, src: &Tensor, dst: &mut CudaSlice<T>) -> Result<()>
where
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
{
    let n = src.elem_count();
    ensure!(n <= dst.len(), "{n} values into a buffer of {}", dst.len());
    let (storage, layout) = src.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        bail!("expected a CUDA tensor");
    };
    let slice = storage.as_cuda_slice::<T>()?;
    let start = layout.start_offset();
    device
        .cuda_stream()
        .memcpy_dtod(&slice.slice(start..start + n), &mut dst.slice_mut(0..n))?;
    Ok(())
}

/// A copy of a device buffer's front as a candle tensor of `shape`.
pub(super) fn tensor_of<T, S: Into<candle_core::Shape>>(
    device: &CudaDevice,
    src: &CudaSlice<T>,
    shape: S,
) -> Result<Tensor>
where
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
{
    let shape = shape.into();
    let n = shape.elem_count();
    ensure!(n <= src.len(), "{n} values from a buffer of {}", src.len());
    // SAFETY: written in full by the copy below.
    let mut copy = unsafe { device.alloc::<T>(n)? };
    device
        .cuda_stream()
        .memcpy_dtod(&src.slice(0..n), &mut copy)?;
    Ok(Tensor::from((
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            copy,
            device.clone(),
        )),
        shape,
    )))
}
