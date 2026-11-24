//! Routed experts on Strata's grouped kernels, from a GGUF's own blocks.
//!
//! Neuron keeps routing. For each layer this module groups the routed
//! tokens by expert, makes each active expert's `[gate | up | down]`
//! blob addressable on the device, and runs them all in one
//! `native_expert_grouped` launch.
//!
//! An expert comes from the VRAM tier when it is held. Decode-sized
//! forwards admit what they miss. A prompt chunk must not (see
//! [`ExpertTier::peek`]), so its misses are copied into a reusable arena
//! for the duration of that layer.

use std::ffi::c_void;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::result::mem_pool;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, sys};
use candle_core::{CudaDevice, CudaStorage, DType, Device, Storage, Tensor};
use memmap2::{Advice, Mmap};
use rayon::prelude::*;

use super::gguf_index::{GgufIndex, GgufTensor, read_expert_profile};
use crate::cuda::strata as ffi;
use crate::harness::arch::qwen3_5::expert_tier::ExpertTier;
use crate::harness::arch::qwen3_5::moe::{
    ExternalExpertSource, ExternalExperts, TIER_ADMIT_MAX_ROWS,
};

const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_BLOCK_VALUES: usize = 32;
/// Pinned host memory expert blobs are staged through. Pinned because
/// an unpinned source makes the driver stage it again, at a fraction of
/// the link's speed.
const STAGING_BYTES: usize = 2 << 30;

/// Pinned host memory. Freed on the device worker with the model.
struct PinnedBuffer {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is plain host memory, and every access is made
// while holding the workspace lock.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    fn new(len: usize) -> candle_core::Result<Self> {
        // SAFETY: returns null or an allocation of `len` bytes.
        let ptr = unsafe { ffi::neuron_strata_pinned_alloc(len) }.cast::<u8>();
        if ptr.is_null() {
            candle_core::bail!("cannot pin {len} bytes of staging memory");
        }
        Ok(Self { ptr, len })
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated by `neuron_strata_pinned_alloc`, freed once.
        unsafe { ffi::neuron_strata_pinned_free(self.ptr.cast()) };
    }
}

/// An expert's blob held in VRAM.
pub struct ExpertSlot {
    blob: CudaSlice<u8>,
    /// Where the device table points for this expert while the slot
    /// lives; on drop it is pointed back at the host copy (#425).
    repoint: Option<Repoint>,
}

struct Repoint {
    /// Device address of the expert's table entry.
    entry: u64,
    /// Device address of the expert's host copy.
    host: u64,
    device: CudaDevice,
}

impl Drop for ExpertSlot {
    fn drop(&mut self) {
        if let Some(r) = &self.repoint {
            let stream = r.device.cuda_stream().cu_stream() as *mut c_void;
            // SAFETY: the entry is a live table slot. Queued before the
            // blob's stream-ordered free (fields drop after this body),
            // so no later kernel reads the freed slot.
            let status =
                unsafe { ffi::neuron_strata_table_set(r.entry as *mut u64, r.host, stream) };
            if status != 0 {
                tracing::error!(status, "could not repoint an evicted expert");
            }
        }
    }
}

/// Every expert, in page-locked host memory the device reads directly:
/// a miss is read over PCIe by the expert kernel, with no copy and no
/// host involvement (#425). One allocation per layer.
struct HostStore {
    layers: Vec<(*mut u8, u64)>,
}

// SAFETY: plain host memory, written once at load and only read after.
unsafe impl Send for HostStore {}
// SAFETY: as above.
unsafe impl Sync for HostStore {}

impl Drop for HostStore {
    fn drop(&mut self) {
        for &(ptr, _) in &self.layers {
            // SAFETY: from `neuron_strata_host_alloc_mapped`, freed once.
            unsafe { ffi::neuron_strata_pinned_free(ptr.cast()) };
        }
    }
}

/// One layer's last device-routed ids, on their way to the host so the
/// tier can learn from them between steps.
///
/// No event guards the read: every caller adapts only after it has read
/// the previous step's logits, which waits for the whole step, and an
/// event recorded inside a captured step cannot be waited on from the
/// host.
struct Pending {
    ids: MappedIds,
    entries: usize,
    recorded: bool,
}

/// Mapped pinned host memory for a layer's ids: written by a kernel
/// through `device`, read by the host at `ptr` (#427).
struct MappedIds {
    ptr: *mut u32,
    device: u64,
    len: usize,
}

// SAFETY: plain host memory, read by the host only after the step whose
// kernel writes it has been waited on.
unsafe impl Send for MappedIds {}

impl MappedIds {
    fn new(words: usize) -> candle_core::Result<Self> {
        // SAFETY: returns null or `words * 4` bytes of mapped host memory.
        let ptr = unsafe { ffi::neuron_strata_host_alloc_mapped(words * 4) }.cast::<u32>();
        if ptr.is_null() {
            candle_core::bail!("cannot pin {words} words for expert usage");
        }
        // SAFETY: as allocated just above.
        let device = unsafe { ffi::neuron_strata_host_device_ptr(ptr.cast()) };
        if device == 0 {
            // SAFETY: allocated above, freed once.
            unsafe { ffi::neuron_strata_pinned_free(ptr.cast()) };
            candle_core::bail!("the device cannot address the expert usage buffer");
        }
        Ok(Self {
            ptr,
            device,
            len: words,
        })
    }
}

impl Drop for MappedIds {
    fn drop(&mut self) {
        // SAFETY: allocated by `neuron_strata_host_alloc_mapped`, freed once.
        unsafe { ffi::neuron_strata_pinned_free(self.ptr.cast()) };
    }
}

/// Decode-sized MoE layers routed on the device (#425).
struct DeviceRouting {
    store: HostStore,
    /// `[layer][expert]` device addresses: a tier slot, or the host copy.
    table: CudaSlice<u64>,
    /// `[layer][expert]` host-copy addresses: where `table` equals this,
    /// the expert is a miss.
    host_table: CudaSlice<u64>,
    pending: Vec<Mutex<Option<Pending>>>,
    /// Misses a layer may admit to the tier per step (LRU admission).
    admit_per_layer: usize,
    /// Decayed-count admission instead (`NEURON_EXPERT_ADMIT=decayed`).
    decayed: Option<Mutex<Decayed>>,
    experts: usize,
    /// Buffers for device-routed layers only, sized once for the largest
    /// decode batch so they never move: a captured decode step holds
    /// their addresses (#427). Prompt chunks use the source's workspace.
    workspace: Mutex<Workspace>,
    /// The largest blob of any layer, which sizes the decode arena.
    max_blob_bytes: usize,
}

/// Routing counts for [`super::tier_policy`], and the steps since load.
struct Decayed {
    policy: super::tier_policy::DecayedPolicy,
    usage: Vec<f32>,
    steps: usize,
}

/// A CUDA memory pool for the tier's experts alone.
///
/// Thousands of long-lived expert blobs allocated from the default pool
/// sit among the forward pass's transient tensors and pin the blocks
/// they share, so the pool cannot hand those blocks back: measured on
/// one 5090, 29 GB reserved against a 21 GB peak of live data. With
/// their own pool the experts only ever share blocks with each other,
/// and the default pool's trims return the transients' memory.
struct TierPool(sys::CUmemoryPool);

// SAFETY: a pool handle may be used from any thread of its context, and
// the source is only used from its device worker.
unsafe impl Send for TierPool {}
// SAFETY: as above; allocation from a pool is thread-safe in the driver.
unsafe impl Sync for TierPool {}

impl TierPool {
    fn new(ordinal: usize) -> candle_core::Result<Self> {
        // SAFETY: an all-zero props struct is valid; the fields that
        // matter are set below, and zero handle types means none.
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = ordinal as i32;
        // SAFETY: `props` describes this device.
        let pool = unsafe { mem_pool::create(&props) }.map_err(candle_err)?;
        // Keep what eviction frees: the next admission reuses it.
        let mut keep = u64::MAX;
        // SAFETY: the release threshold is a cuuint64_t.
        unsafe {
            mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&raw mut keep).cast(),
            )
        }
        .map_err(candle_err)?;
        Ok(Self(pool))
    }

    fn alloc(&self, device: &CudaDevice, bytes: usize) -> candle_core::Result<CudaSlice<u8>> {
        let stream = device.cuda_stream();
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: the pool and stream belong to this device's context.
        unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, bytes, self.0, stream.cu_stream()) }
            .result()
            .map_err(candle_err)?;
        // SAFETY: `ptr` is a fresh allocation of `bytes` bytes. Its drop
        // frees it stream-ordered, which returns it to this pool.
        Ok(unsafe { stream.upgrade_device_ptr(ptr, bytes) })
    }
}

impl Drop for TierPool {
    fn drop(&mut self) {
        // SAFETY: created by `new`; the driver defers the destruction
        // until every allocation from it has been freed.
        let _ = unsafe { mem_pool::destroy(self.0) };
    }
}

fn device_address<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    slice: &CudaSlice<T>,
) -> u64 {
    slice.device_ptr(slice.stream()).0
}

/// `NEURON_EXPERT_DEVICE_ROUTING=1`: pin every expert in host memory
/// and route decode on the device (#425).
fn device_routing_enabled() -> bool {
    std::env::var("NEURON_EXPERT_DEVICE_ROUTING").is_ok_and(|v| v == "1")
}

/// Per-layer timing, from `NEURON_EXPERT_TIMING`.
fn timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NEURON_EXPERT_TIMING").is_some())
}

fn candle_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::msg(e.to_string())
}

/// Where one projection of a layer's experts lies.
#[derive(Clone, Copy)]
struct Role {
    shard: usize,
    offset: usize,
    bytes: usize,
}

struct Layer {
    gu_type: i32,
    d_type: i32,
    gate: Role,
    up: Role,
    down: Role,
    blob_bytes: usize,
}

/// One expert blob to copy to the device.
#[derive(Clone, Copy)]
struct Upload {
    expert: usize,
    address: u64,
    in_arena: bool,
}

/// Device buffers reused from layer to layer.
///
/// Reuse is safe because each layer synchronizes the stream before
/// returning, so no queued kernel still reads them.
#[derive(Default)]
struct Workspace {
    q8: Option<CudaSlice<u8>>,
    scratch: Option<CudaSlice<u8>>,
    grp_ptr: Option<CudaSlice<u64>>,
    grp_start: Option<CudaSlice<i32>>,
    n_groups: Option<CudaSlice<i32>>,
    ent_dst: Option<CudaSlice<i32>>,
    ent_tok: Option<CudaSlice<i32>>,
    arena: Option<CudaSlice<u8>>,
    staging: Option<PinnedBuffer>,
    miss_src: Option<CudaSlice<u64>>,
    n_miss: Option<CudaSlice<i32>>,
}

/// The device address of a contiguous tensor's first element.
fn tensor_address<
    T: candle_core::cuda_backend::CudaDType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr
        + 'static,
>(
    t: &Tensor,
) -> candle_core::Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("expected a tensor on the GPU");
    };
    let slice = storage.as_cuda_slice::<T>()?;
    Ok(device_address(slice) + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
}

fn ensure_capacity<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    device: &CudaDevice,
    slot: &mut Option<CudaSlice<T>>,
    len: usize,
) -> candle_core::Result<()> {
    if slot.as_ref().is_none_or(|s| s.len() < len) {
        *slot = None;
        // SAFETY: every user writes the range it later reads.
        *slot = Some(unsafe { device.alloc::<T>(len.max(1))? });
    }
    Ok(())
}

fn upload<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    device: &CudaDevice,
    slot: &mut Option<CudaSlice<T>>,
    data: &[T],
) -> candle_core::Result<u64> {
    ensure_capacity(device, slot, data.len())?;
    let buffer = slot.as_mut().expect("capacity was just ensured");
    device.memcpy_htod(data, &mut buffer.slice_mut(..data.len()))?;
    Ok(device_address(buffer))
}

/// The model's GGUF experts, shared by its MoE blocks.
pub struct GgufExpertSource {
    device: CudaDevice,
    n_embd: usize,
    n_ff: usize,
    maps: Vec<Mmap>,
    layers: Vec<Layer>,
    tier: Option<Arc<ExpertTier<ExpertSlot>>>,
    workspace: Mutex<Workspace>,
    /// Declared after `tier`, so its slots repoint the table before the
    /// table goes.
    routing: Option<DeviceRouting>,
    /// Declared after `tier`, so the experts are freed first.
    tier_pool: TierPool,
}

fn role(
    index: &GgufIndex,
    layer: usize,
    name: &str,
    want_dims: [u64; 3],
) -> Result<(GgufTensor, usize)> {
    let key = format!("blk.{layer}.ffn_{name}_exps.weight");
    let tensor = index
        .tensors
        .get(&key)
        .with_context(|| format!("GGUF has no {key}"))?;
    ensure!(
        tensor.dims == want_dims,
        "{key} is {:?}, expected {want_dims:?}",
        tensor.dims
    );
    // SAFETY: a pure function of its integer arguments.
    let row = unsafe { ffi::neuron_strata_row_bytes(tensor.ggml_type as i32, want_dims[0] as i64) };
    ensure!(row > 0, "{key}: unsupported type {}", tensor.ggml_type);
    Ok((tensor.clone(), row * want_dims[1] as usize))
}

impl GgufExpertSource {
    /// Indexes the GGUF whose first shard is `first` and maps its shards.
    pub fn open(
        first: &Path,
        device: &Device,
        layers: usize,
        experts: usize,
        n_embd: usize,
        n_ff: usize,
        tier_budget_mb: Option<u64>,
    ) -> Result<Self> {
        let device = device.as_cuda_device()?.clone();
        let index = GgufIndex::open(first)?;
        let mut maps = Vec::with_capacity(index.shards.len());
        for path in &index.shards {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: model files are not modified while serving.
            let map =
                unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?;
            map.advise(Advice::Random).context("advise expert map")?;
            maps.push(map);
        }
        let (e, h, f) = (experts as u64, n_embd as u64, n_ff as u64);
        let mut parsed = Vec::with_capacity(layers);
        for layer in 0..layers {
            let (gate, gate_bytes) = role(&index, layer, "gate", [h, f, e])?;
            let (up, up_bytes) = role(&index, layer, "up", [h, f, e])?;
            let (down, down_bytes) = role(&index, layer, "down", [f, h, e])?;
            ensure!(
                gate.ggml_type == up.ggml_type,
                "layer {layer}: gate and up differ in type"
            );
            let (gu_type, d_type) = (gate.ggml_type as i32, down.ggml_type as i32);
            // SAFETY: pure functions of their integer arguments.
            let (supported, blob_bytes) = unsafe {
                (
                    ffi::neuron_strata_expert_supported(
                        gu_type,
                        d_type,
                        n_embd as i64,
                        n_ff as i64,
                    ),
                    ffi::neuron_strata_expert_bytes(gu_type, d_type, n_embd as i64, n_ff as i64),
                )
            };
            ensure!(
                supported != 0,
                "layer {layer}: no kernels for types {gu_type}/{d_type}"
            );
            ensure!(
                blob_bytes == gate_bytes + up_bytes + down_bytes,
                "layer {layer}: blob is {blob_bytes} bytes, roles sum to {}",
                gate_bytes + up_bytes + down_bytes
            );
            for (tensor, bytes) in [(&gate, gate_bytes), (&up, up_bytes), (&down, down_bytes)] {
                let end = tensor.offset + (bytes * experts) as u64;
                ensure!(
                    end <= maps[tensor.shard].len() as u64,
                    "layer {layer}: expert tensor runs past the end of its shard"
                );
            }
            let as_role = |t: &GgufTensor, bytes| Role {
                shard: t.shard,
                offset: t.offset as usize,
                bytes,
            };
            parsed.push(Layer {
                gu_type,
                d_type,
                gate: as_role(&gate, gate_bytes),
                up: as_role(&up, up_bytes),
                down: as_role(&down, down_bytes),
                blob_bytes,
            });
        }
        let tier = tier_budget_mb.map(|mb| ExpertTier::new(mb << 20));
        let tier_pool = TierPool::new(device.cuda_stream().context().ordinal())?;
        let profile = std::env::var_os("NEURON_EXPERT_PROFILE")
            .map(|path| read_expert_profile(Path::new(&path), layers, experts))
            .transpose()?;
        tracing::info!(
            shards = maps.len(),
            layers,
            tier_budget_mb,
            "native expert kernels enabled, reading experts from GGUF"
        );
        let mut source = Self {
            device,
            n_embd,
            n_ff,
            maps,
            layers: parsed,
            tier,
            workspace: Mutex::new(Workspace::default()),
            routing: None,
            tier_pool,
        };
        if device_routing_enabled() {
            source.routing = Some(source.build_routing(experts)?);
        }
        if let Some(ranked) = profile {
            let started = std::time::Instant::now();
            let filled = source.warm_tier(&ranked)?;
            tracing::info!(
                filled,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "VRAM expert tier filled from profile"
            );
        }
        Ok(source)
    }

    fn copy_blob(&self, layer: &Layer, expert: usize, dst: &mut [u8]) {
        let (gate, rest) = dst.split_at_mut(layer.gate.bytes);
        let (up, down) = rest.split_at_mut(layer.up.bytes);
        for (out, role) in [(gate, layer.gate), (up, layer.up), (down, layer.down)] {
            let start = role.offset + expert * role.bytes;
            out.copy_from_slice(&self.maps[role.shard][start..start + role.bytes]);
        }
    }

    /// Queues blob copies on the stream.
    ///
    /// The last batch's staging memory stays in use until the caller
    /// synchronizes. Neighbouring arena blobs are copied as one; slots
    /// are separate allocations even when their addresses adjoin, and a
    /// copy may not span two allocations.
    fn stage_blobs(
        &self,
        staging: &mut Option<PinnedBuffer>,
        layer: &Layer,
        items: &[Upload],
    ) -> candle_core::Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        if let Some(routing) = &self.routing {
            // Every expert is already in page-locked memory: copy each
            // straight from there, with no staging pass and no file read.
            let index = self
                .layers
                .iter()
                .position(|l| std::ptr::eq(l, layer))
                .expect("a layer of this source");
            let base = routing.store.layers[index].0;
            let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
            for item in items {
                // SAFETY: the source is the expert's pinned copy; the
                // destination is a live slot or arena range of its size.
                let status = unsafe {
                    ffi::neuron_strata_copy_to_device(
                        item.address as *mut c_void,
                        base.add(item.expert * layer.blob_bytes).cast(),
                        layer.blob_bytes,
                        stream,
                    )
                };
                if status != 0 {
                    candle_core::bail!("expert upload failed with CUDA status {status}");
                }
            }
            return Ok(());
        }
        if staging.is_none() {
            *staging = Some(PinnedBuffer::new(STAGING_BYTES)?);
        }
        let pinned = staging.as_ref().expect("allocated above");
        let stream = self.device.cuda_stream();
        let per_batch = pinned.len / layer.blob_bytes;
        for (b, batch) in items.chunks(per_batch).enumerate() {
            if b > 0 {
                stream.synchronize().map_err(candle_err)?;
            }
            // SAFETY: the stream has passed any earlier use of this
            // range, and the lock gives this call the only reference.
            let host = unsafe {
                std::slice::from_raw_parts_mut(pinned.ptr, batch.len() * layer.blob_bytes)
            };
            host.par_chunks_mut(layer.blob_bytes)
                .zip(batch.par_iter())
                .for_each(|(dst, item)| self.copy_blob(layer, item.expert, dst));
            let mut i = 0;
            while i < batch.len() {
                let mut run = 1;
                while batch[i].in_arena
                    && i + run < batch.len()
                    && batch[i + run].in_arena
                    && batch[i + run].address == batch[i].address + (run * layer.blob_bytes) as u64
                {
                    run += 1;
                }
                // SAFETY: both ranges are live for `run` blobs, and the
                // staging range is not rewritten before a synchronize.
                let status = unsafe {
                    ffi::neuron_strata_copy_to_device(
                        batch[i].address as *mut c_void,
                        host.as_ptr().add(i * layer.blob_bytes).cast(),
                        run * layer.blob_bytes,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if status != 0 {
                    candle_core::bail!("expert upload failed with CUDA status {status}");
                }
                i += run;
            }
        }
        Ok(())
    }

    /// Fills the tier with the profile's hottest experts, so the first
    /// requests do not pay for its warm-up.
    fn warm_tier(&self, ranked: &[(u16, u16)]) -> candle_core::Result<usize> {
        let Some(tier) = &self.tier else {
            return Ok(0);
        };
        let mut workspace = self.workspace.lock().map_err(candle_err)?;
        let capacity = tier.capacity_bytes();
        let mut used = 0_u64;
        let mut by_layer: Vec<Vec<Upload>> = vec![Vec::new(); self.layers.len()];
        let mut filled = 0;
        for &(l, e) in ranked {
            let (l, e) = (l as usize, e as usize);
            let bytes = self.layers[l].blob_bytes as u64;
            if used + bytes > capacity {
                break;
            }
            used += bytes;
            let pending = &mut by_layer[l];
            tier.get_or_load(l as u32, e as u32, bytes, || -> candle_core::Result<_> {
                // Filled by `stage_blobs` before first use.
                let blob = self.tier_pool.alloc(&self.device, bytes as usize)?;
                pending.push(Upload {
                    expert: e,
                    address: device_address(&blob),
                    in_arena: false,
                });
                let repoint = self.point_table_at(l, e, device_address(&blob))?;
                Ok(ExpertSlot { blob, repoint })
            })?;
            filled += 1;
        }
        for (l, pending) in by_layer.iter().enumerate() {
            self.stage_blobs(&mut workspace.staging, &self.layers[l], pending)?;
            self.device
                .cuda_stream()
                .synchronize()
                .map_err(candle_err)?;
        }
        Ok(filled)
    }

    /// Copies every expert into page-locked host memory the device can
    /// read, and points the table at those copies.
    fn build_routing(&self, experts: usize) -> candle_core::Result<DeviceRouting> {
        let started = std::time::Instant::now();
        let mut layers = Vec::with_capacity(self.layers.len());
        let mut host_table = Vec::with_capacity(self.layers.len() * experts);
        let mut bytes_total = 0_usize;
        for layer in &self.layers {
            let bytes = layer.blob_bytes * experts;
            // SAFETY: returns null or `bytes` of mapped host memory.
            let ptr = unsafe { ffi::neuron_strata_host_alloc_mapped(bytes) }.cast::<u8>();
            if ptr.is_null() {
                candle_core::bail!(
                    "cannot pin {bytes} bytes of host memory for the expert store \
                     (raise LimitMEMLOCK, or unset NEURON_EXPERT_DEVICE_ROUTING)"
                );
            }
            // SAFETY: as allocated just above.
            let device = unsafe { ffi::neuron_strata_host_device_ptr(ptr.cast()) };
            layers.push((ptr, device));
            if device == 0 {
                candle_core::bail!("the device cannot address the pinned expert store");
            }
            // SAFETY: `ptr` is `bytes` long and only this call holds it.
            let host = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
            host.par_chunks_mut(layer.blob_bytes)
                .enumerate()
                .for_each(|(e, dst)| self.copy_blob(layer, e, dst));
            host_table.extend((0..experts).map(|e| device + (e * layer.blob_bytes) as u64));
            bytes_total += bytes;
        }
        let store = HostStore { layers };
        // SAFETY: written in full by the copy below.
        let mut table = unsafe { self.device.alloc::<u64>(host_table.len())? };
        self.device.memcpy_htod(&host_table, &mut table)?;
        // SAFETY: as above.
        let mut host_table_d = unsafe { self.device.alloc::<u64>(host_table.len())? };
        self.device.memcpy_htod(&host_table, &mut host_table_d)?;
        let mut pending = Vec::with_capacity(self.layers.len());
        pending.resize_with(self.layers.len(), || Mutex::new(None));
        let admit_per_layer = std::env::var("NEURON_EXPERT_ADMIT_PER_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let decayed =
            (std::env::var("NEURON_EXPERT_ADMIT").as_deref() == Ok("decayed")).then(|| {
                let mut policy = super::tier_policy::DecayedPolicy::default();
                if let Some(n) = std::env::var("NEURON_EXPERT_SWAPS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                {
                    policy.max_swaps = n;
                }
                Mutex::new(Decayed {
                    policy,
                    usage: vec![0.0; self.layers.len() * experts],
                    steps: 0,
                })
            });
        tracing::info!(
            gib = format!("{:.1}", bytes_total as f64 / (1u64 << 30) as f64),
            elapsed_ms = started.elapsed().as_millis() as u64,
            admit_per_layer,
            admission = if decayed.is_some() { "decayed" } else { "lru" },
            "expert host store pinned; decode routes on the device"
        );
        Ok(DeviceRouting {
            store,
            table,
            host_table: host_table_d,
            pending,
            admit_per_layer,
            decayed,
            experts,
            workspace: Mutex::new(Workspace::default()),
            max_blob_bytes: self.layers.iter().map(|l| l.blob_bytes).max().unwrap_or(0),
        })
    }

    /// Points `layer`/`expert`'s table entry at `address` (queued), and
    /// returns how to point it back. `None` without device routing.
    fn point_table_at(
        &self,
        layer: usize,
        expert: usize,
        address: u64,
    ) -> candle_core::Result<Option<Repoint>> {
        let Some(routing) = &self.routing else {
            return Ok(None);
        };
        let experts = routing.table.len() / self.layers.len();
        let entry = device_address(&routing.table) + ((layer * experts + expert) * 8) as u64;
        let host = routing.store.layers[layer].1 + (expert * self.layers[layer].blob_bytes) as u64;
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        // SAFETY: `entry` is inside the live table.
        let status = unsafe { ffi::neuron_strata_table_set(entry as *mut u64, address, stream) };
        if status != 0 {
            candle_core::bail!("table write failed with CUDA status {status}");
        }
        Ok(Some(Repoint {
            entry,
            host,
            device: self.device.clone(),
        }))
    }

    /// Lets the tier learn from `layer`'s previous device-routed step:
    /// counts its uses and admits up to `admit_per_layer` misses, each a
    /// queued copy from the host store and a table write.
    pub fn adapt(&self, layer_index: usize) -> candle_core::Result<()> {
        let (Some(routing), Some(tier)) = (&self.routing, &self.tier) else {
            return Ok(());
        };
        let mut guard = routing.pending[layer_index].lock().map_err(candle_err)?;
        let Some(pending) = guard.as_mut().filter(|p| p.recorded) else {
            return Ok(());
        };
        pending.recorded = false;
        // SAFETY: the copy above wrote `entries` ids, and the next copy
        // into this buffer is only queued after this read.
        let ids =
            unsafe { std::slice::from_raw_parts(pending.ids.ptr.cast::<u32>(), pending.entries) };
        if let Some(decayed) = &routing.decayed {
            let mut d = decayed.lock().map_err(candle_err)?;
            let base = layer_index * routing.experts;
            for &e in ids {
                if let Some(c) = d.usage.get_mut(base + e as usize) {
                    *c += 1.0;
                }
            }
            drop(guard);
            if layer_index + 1 == self.layers.len() {
                d.steps += 1;
                if d.steps.is_multiple_of(d.policy.every.max(1)) {
                    self.swap_round(&mut d, routing.experts, tier)?;
                }
            }
            return Ok(());
        }
        let mut used: Vec<u32> = ids.to_vec();
        used.sort_unstable();
        used.dedup();
        drop(guard);

        let layer = &self.layers[layer_index];
        let host_base = routing.store.layers[layer_index].0;
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        let mut admitted = 0;
        for e in used {
            let e = e as usize;
            if admitted >= routing.admit_per_layer {
                tier.touch(layer_index as u32, e as u32);
                continue;
            }
            tier.get_or_load(
                layer_index as u32,
                e as u32,
                layer.blob_bytes as u64,
                || -> candle_core::Result<_> {
                    admitted += 1;
                    self.admit_slot(layer_index, e, host_base, stream)
                },
            )?;
        }
        Ok(())
    }

    /// A tier slot for `layer`/`expert`, copied from the pinned store and
    /// with the table pointed at it (both queued on `stream`).
    fn admit_slot(
        &self,
        layer_index: usize,
        e: usize,
        host_base: *mut u8,
        stream: *mut c_void,
    ) -> candle_core::Result<ExpertSlot> {
        let layer = &self.layers[layer_index];
        let blob = self.tier_pool.alloc(&self.device, layer.blob_bytes)?;
        // SAFETY: the source is this expert's pinned copy and the slot was
        // just allocated at its size.
        let status = unsafe {
            ffi::neuron_strata_copy_to_device(
                device_address(&blob) as *mut c_void,
                host_base.add(e * layer.blob_bytes).cast(),
                layer.blob_bytes,
                stream,
            )
        };
        if status != 0 {
            candle_core::bail!("expert admission copy failed with CUDA status {status}");
        }
        let repoint = self.point_table_at(layer_index, e, device_address(&blob))?;
        Ok(ExpertSlot { blob, repoint })
    }

    /// One round of decayed-count admission: plan from the counts so far,
    /// decay them, and carry out the swaps (each victim's table entry goes
    /// back to its host copy before its slot's bytes are reused).
    fn swap_round(
        &self,
        d: &mut Decayed,
        experts: usize,
        tier: &ExpertTier<ExpertSlot>,
    ) -> candle_core::Result<()> {
        let Some(routing) = &self.routing else {
            return Ok(());
        };
        let swaps =
            super::tier_policy::plan_swaps(&d.policy, &d.usage, experts, |l| tier.held(l as u32));
        let decay = d.policy.decay;
        d.usage.iter_mut().for_each(|c| *c *= decay);
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        for swap in swaps {
            let layer = &self.layers[swap.layer];
            if !tier.remove(swap.layer as u32, swap.evict) {
                continue;
            }
            let host_base = routing.store.layers[swap.layer].0;
            tier.get_or_load(
                swap.layer as u32,
                swap.admit,
                layer.blob_bytes as u64,
                || self.admit_slot(swap.layer, swap.admit as usize, host_base, stream),
            )?;
        }
        Ok(())
    }

    /// A decode-sized layer from routing left on the device: the group
    /// arrays are built there from the table, the grouped kernel reads
    /// resident experts from VRAM and the rest from the host store, and
    /// nothing waits on the device (#425).
    fn forward_layer_device(
        &self,
        layer_index: usize,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        k: usize,
    ) -> candle_core::Result<Tensor> {
        let (rows, hidden) = xs.dims2()?;
        let entries = rows * k;
        self.adapt(layer_index)?;
        let ids = ids.to_dtype(DType::U32)?.contiguous()?;
        let weights = weights.to_dtype(DType::F32)?.contiguous()?;
        let x = xs.to_dtype(DType::F32)?.contiguous()?;
        // SAFETY: written in full by the route kernel.
        let w_route = unsafe { self.device.alloc::<f32>(entries)? };
        // SAFETY: written in full by the expert kernel.
        let out = unsafe { self.device.alloc::<f32>(entries * hidden)? };
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        self.launch_routed(
            layer_index,
            tensor_address::<f32>(&x)?,
            tensor_address::<u32>(&ids)?,
            tensor_address::<f32>(&weights)?,
            rows,
            k,
            device_address(&out),
            device_address(&w_route),
            stream,
        )?;
        self.record_usage(layer_index, tensor_address::<u32>(&ids)?, entries)?;

        let out = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, self.device.clone())),
            (rows, k, hidden),
        ));
        let w_route = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(w_route, self.device.clone())),
            (rows, k, 1),
        ));
        out.broadcast_mul(&w_route)?.sum(1)
    }

    /// The routed experts of one device-routed layer, over device
    /// pointers: `x` is `(rows, n_embd)` f32, `ids` `(rows, k)` 32-bit
    /// expert ids and `weights` `(rows, k)` f32. Writes `parts`
    /// `(rows, k, n_embd)` f32, each token's outputs ordered by ascending
    /// expert id, and `w_route` `(rows, k)`, the weights in that order.
    ///
    /// Once its buffers exist (first call) this allocates nothing and
    /// does not wait, so it can be captured. It does not adapt the tier:
    /// the caller does that between steps.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_routed(
        &self,
        layer_index: usize,
        x: u64,
        ids: u64,
        weights: u64,
        rows: usize,
        k: usize,
        parts: u64,
        w_route: u64,
        stream: *mut c_void,
    ) -> candle_core::Result<()> {
        let routing = self
            .routing
            .as_ref()
            .ok_or_else(|| candle_err("device routing is not enabled"))?;
        if rows > TIER_ADMIT_MAX_ROWS {
            candle_core::bail!(
                "device routing takes at most {TIER_ADMIT_MAX_ROWS} rows, got {rows}"
            );
        }
        let layer = &self.layers[layer_index];
        let hidden = self.n_embd;
        let entries = rows * k;
        let experts = routing.table.len() / self.layers.len();

        let mut workspace = routing.workspace.lock().map_err(candle_err)?;
        let ws = &mut *workspace;
        let cap_groups = entries.min(experts);
        // Sized for the largest batch this path takes, once, so nothing
        // here is reallocated under a captured step.
        let max_entries = TIER_ADMIT_MAX_ROWS * k;
        let max_groups = max_entries.min(experts);
        ensure_capacity(&self.device, &mut ws.grp_ptr, max_groups)?;
        ensure_capacity(&self.device, &mut ws.grp_start, max_groups + 1)?;
        ensure_capacity(&self.device, &mut ws.n_groups, 1)?;
        ensure_capacity(&self.device, &mut ws.ent_dst, max_entries)?;
        ensure_capacity(&self.device, &mut ws.ent_tok, max_entries)?;
        ensure_capacity(&self.device, &mut ws.miss_src, max_groups)?;
        ensure_capacity(&self.device, &mut ws.n_miss, 1)?;
        // A miss is copied into the arena by the gather below, so the
        // expert kernel reads VRAM only: reading host memory from the
        // kernel itself is latency-bound at decode's few warps.
        ensure_capacity(
            &self.device,
            &mut ws.arena,
            max_groups * routing.max_blob_bytes,
        )?;
        ensure_capacity(
            &self.device,
            &mut ws.q8,
            TIER_ADMIT_MAX_ROWS * (hidden / Q8_1_BLOCK_VALUES) * Q8_1_BLOCK_BYTES,
        )?;
        // SAFETY: a pure function of its integer arguments.
        let scratch_bytes = unsafe {
            ffi::neuron_strata_expert_scratch_bytes(max_entries as i64, self.n_ff as i64)
        };
        ensure_capacity(&self.device, &mut ws.scratch, scratch_bytes)?;

        let table_row = device_address(&routing.table) + (layer_index * experts * 8) as u64;
        let host_row = device_address(&routing.host_table) + (layer_index * experts * 8) as u64;
        let arena = device_address(ws.arena.as_ref().expect("allocated"));
        let miss_src = device_address(ws.miss_src.as_ref().expect("allocated"));
        let n_miss = device_address(ws.n_miss.as_ref().expect("allocated"));
        let q8 = device_address(ws.q8.as_ref().expect("allocated"));
        // SAFETY: every pointer is a live device allocation sized above or
        // by the caller, and every kernel is ordered on the one stream.
        unsafe {
            let status = ffi::neuron_strata_route_groups(
                ids as *const u32,
                weights as *const f32,
                rows as i32,
                k as i32,
                table_row as *const u64,
                host_row as *const u64,
                arena,
                layer.blob_bytes as u64,
                experts as i32,
                device_address(ws.grp_ptr.as_ref().expect("allocated")) as *mut u64,
                device_address(ws.grp_start.as_ref().expect("allocated")) as *mut i32,
                device_address(ws.n_groups.as_ref().expect("allocated")) as *mut i32,
                device_address(ws.ent_dst.as_ref().expect("allocated")) as *mut i32,
                device_address(ws.ent_tok.as_ref().expect("allocated")) as *mut i32,
                w_route as *mut f32,
                miss_src as *mut u64,
                n_miss as *mut i32,
                stream,
            );
            if status != 0 {
                candle_core::bail!("route kernel failed with status {status}");
            }
            let status = ffi::neuron_strata_gather_blobs(
                miss_src as *const u64,
                n_miss as *const i32,
                cap_groups as i32,
                arena,
                layer.blob_bytes as u64,
                stream,
            );
            if status != 0 {
                candle_core::bail!("miss gather failed with status {status}");
            }
            ffi::neuron_strata_quantize_q8_1(
                x as *const f32,
                rows as i64,
                hidden as i64,
                q8 as *mut c_void,
                stream,
            );
            let status = ffi::neuron_strata_last_error();
            if status != 0 {
                candle_core::bail!("expert activation quantize failed with CUDA status {status}");
            }
            ffi::neuron_strata_expert_grouped(
                layer.gu_type,
                layer.d_type,
                self.n_embd as i64,
                self.n_ff as i64,
                device_address(ws.grp_ptr.as_ref().expect("allocated")) as *const u64,
                device_address(ws.grp_start.as_ref().expect("allocated")) as *const i32,
                device_address(ws.n_groups.as_ref().expect("allocated")) as *const i32,
                device_address(ws.ent_dst.as_ref().expect("allocated")) as *const i32,
                device_address(ws.ent_tok.as_ref().expect("allocated")) as *const i32,
                cap_groups as i64,
                entries as i64,
                q8 as *const c_void,
                device_address(ws.scratch.as_ref().expect("allocated")) as *mut c_void,
                parts as *mut f32,
                stream,
            );
            let status = ffi::neuron_strata_last_error();
            if status != 0 {
                candle_core::bail!("grouped expert kernel failed with CUDA status {status}");
            }
        }
        Ok(())
    }

    /// Queues this step's ids for the host, for `adapt` to read next
    /// time the layer runs. `ids` is `entries` 32-bit ids on the device.
    pub fn record_usage(
        &self,
        layer_index: usize,
        ids: u64,
        entries: usize,
    ) -> candle_core::Result<()> {
        let (Some(routing), Some(_)) = (&self.routing, &self.tier) else {
            return Ok(());
        };
        let mut guard = routing.pending[layer_index].lock().map_err(candle_err)?;
        if guard.as_ref().is_none_or(|p| p.ids.len < entries) {
            // Sized once for the largest batch: a captured step writes
            // this address on every replay.
            *guard = Some(Pending {
                ids: MappedIds::new((TIER_ADMIT_MAX_ROWS * 64).max(entries))?,
                entries: 0,
                recorded: false,
            });
        }
        let pending = guard.as_mut().expect("set above");
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        // SAFETY: the buffer holds `entries` ids and is not read until the
        // step has been waited on.
        unsafe {
            let status = ffi::neuron_strata_copy_words(
                pending.ids.device as *mut u32,
                ids as *const u32,
                entries as i32,
                stream,
            );
            if status != 0 {
                candle_core::bail!("expert usage copy failed with CUDA status {status}");
            }
        }
        pending.entries = entries;
        pending.recorded = true;
        Ok(())
    }

    /// After a captured step's replay: the usage copies it queued are
    /// the ones its capture recorded, so mark them for `adapt`.
    pub fn usage_replayed(&self, layer_index: usize) {
        if let Some(routing) = &self.routing
            && let Ok(mut guard) = routing.pending[layer_index].lock()
            && let Some(pending) = guard.as_mut()
        {
            pending.recorded = true;
        }
    }

    fn forward_layer(
        &self,
        layer_index: usize,
        xs: &Tensor,
        tokens_for: &[Vec<u32>],
        weights_for: &[Vec<f32>],
        k: usize,
    ) -> candle_core::Result<Tensor> {
        let started = std::time::Instant::now();
        let layer = &self.layers[layer_index];
        let (rows, hidden) = xs.dims2()?;
        if hidden != self.n_embd {
            candle_core::bail!(
                "activations are {hidden} wide, experts take {}",
                self.n_embd
            );
        }
        let entries = rows * k;
        let active: Vec<usize> = (0..tokens_for.len())
            .filter(|&e| !tokens_for[e].is_empty())
            .collect();
        let admit = rows <= TIER_ADMIT_MAX_ROWS;

        let mut workspace = self.workspace.lock().map_err(candle_err)?;
        let ws = &mut *workspace;

        // Resolve every active expert to a device address.
        let mut held = Vec::with_capacity(active.len());
        let mut addresses = vec![0_u64; active.len()];
        let mut misses = Vec::new();
        let mut uploads: Vec<Upload> = Vec::new();
        for (g, &e) in active.iter().enumerate() {
            let slot = match &self.tier {
                Some(tier) if admit => Some(tier.get_or_load(
                    layer_index as u32,
                    e as u32,
                    layer.blob_bytes as u64,
                    || -> candle_core::Result<_> {
                        // Filled by `stage_blobs` before the launch.
                        let blob = self.tier_pool.alloc(&self.device, layer.blob_bytes)?;
                        uploads.push(Upload {
                            expert: e,
                            address: device_address(&blob),
                            in_arena: false,
                        });
                        let repoint = self.point_table_at(layer_index, e, device_address(&blob))?;
                        Ok(ExpertSlot { blob, repoint })
                    },
                )?),
                Some(tier) => tier.peek(layer_index as u32, e as u32),
                None => None,
            };
            match slot {
                Some(slot) => {
                    addresses[g] = device_address(&slot.blob);
                    held.push(slot);
                }
                None => misses.push((g, e)),
            }
        }
        let resolved = started.elapsed();
        if !misses.is_empty() {
            ensure_capacity(&self.device, &mut ws.arena, misses.len() * layer.blob_bytes)?;
            let arena_address = device_address(ws.arena.as_ref().expect("allocated above"));
            for (i, &(g, e)) in misses.iter().enumerate() {
                let address = arena_address + (i * layer.blob_bytes) as u64;
                addresses[g] = address;
                uploads.push(Upload {
                    expert: e,
                    address,
                    in_arena: true,
                });
            }
        }
        self.stage_blobs(&mut ws.staging, layer, &uploads)?;
        let streamed = started.elapsed();

        // Each token writes its k outputs side by side, so the mix below
        // is one multiply and one sum rather than a scatter.
        let mut grp_start = Vec::with_capacity(active.len() + 1);
        let mut ent_dst = Vec::with_capacity(entries);
        let mut ent_tok = Vec::with_capacity(entries);
        let mut weights = vec![0_f32; entries];
        let mut rank = vec![0_usize; rows];
        for &e in &active {
            grp_start.push(ent_dst.len() as i32);
            for (&token, &weight) in tokens_for[e].iter().zip(&weights_for[e]) {
                let token = token as usize;
                let dst = token * k + rank[token];
                rank[token] += 1;
                ent_dst.push(dst as i32);
                ent_tok.push(token as i32);
                weights[dst] = weight;
            }
        }
        grp_start.push(ent_dst.len() as i32);
        if ent_dst.len() != entries || rank.iter().any(|&r| r != k) {
            candle_core::bail!("routing did not give every token {k} experts");
        }

        let grp_ptr = upload(&self.device, &mut ws.grp_ptr, &addresses)?;
        let grp_start_ptr = upload(&self.device, &mut ws.grp_start, &grp_start)?;
        let n_groups = upload(&self.device, &mut ws.n_groups, &[active.len() as i32])?;
        let ent_dst_ptr = upload(&self.device, &mut ws.ent_dst, &ent_dst)?;
        let ent_tok_ptr = upload(&self.device, &mut ws.ent_tok, &ent_tok)?;
        ensure_capacity(
            &self.device,
            &mut ws.q8,
            rows * (hidden / Q8_1_BLOCK_VALUES) * Q8_1_BLOCK_BYTES,
        )?;
        // SAFETY: a pure function of its integer arguments.
        let scratch_bytes =
            unsafe { ffi::neuron_strata_expert_scratch_bytes(entries as i64, self.n_ff as i64) };
        ensure_capacity(&self.device, &mut ws.scratch, scratch_bytes)?;

        let prepared = started.elapsed();
        let x = xs.to_dtype(DType::F32)?.contiguous()?;
        let (storage, x_layout) = x.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("native experts need activations on the GPU");
        };
        let x_slice = storage.as_cuda_slice::<f32>()?;
        let x_ptr = x_slice
            .slice(x_layout.start_offset()..)
            .device_ptr(x_slice.stream())
            .0 as *const f32;

        // SAFETY: the kernel writes `entries * hidden` values.
        let out = unsafe { self.device.alloc::<f32>(entries * hidden)? };
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        // SAFETY: every pointer is a live device allocation sized above
        // and owned for the call. `held` keeps tier slots alive until
        // the synchronize below, including any evicted while resolving.
        unsafe {
            let q8 = device_address(ws.q8.as_ref().expect("allocated")) as *mut c_void;
            ffi::neuron_strata_quantize_q8_1(x_ptr, rows as i64, hidden as i64, q8, stream);
            ffi::neuron_strata_expert_grouped(
                layer.gu_type,
                layer.d_type,
                self.n_embd as i64,
                self.n_ff as i64,
                grp_ptr as *const u64,
                grp_start_ptr as *const i32,
                n_groups as *const i32,
                ent_dst_ptr as *const i32,
                ent_tok_ptr as *const i32,
                active.len() as i64,
                entries as i64,
                q8 as *const c_void,
                device_address(ws.scratch.as_ref().expect("allocated")) as *mut c_void,
                device_address(&out) as *mut f32,
                stream,
            );
        }
        self.device
            .cuda_stream()
            .synchronize()
            .map_err(candle_err)?;
        drop(held);
        if timing_enabled() {
            let finished = started.elapsed();
            tracing::info!(
                layer = layer_index,
                rows,
                active = active.len(),
                misses = misses.len(),
                resolve_us = resolved.as_micros() as u64,
                arena_us = (streamed - resolved).as_micros() as u64,
                arrays_us = (prepared - streamed).as_micros() as u64,
                kernel_us = (finished - prepared).as_micros() as u64,
                "native experts"
            );
        }

        let out = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, self.device.clone())),
            (rows, k, hidden),
        ));
        let weights = Tensor::from_vec(weights, (rows, k, 1), xs.device())?;
        out.broadcast_mul(&weights)?.sum(1)
    }
}

struct NativeBlock {
    source: Arc<GgufExpertSource>,
    layer: usize,
}

impl ExternalExperts for NativeBlock {
    fn routes_on_device(&self, rows: usize) -> bool {
        self.source.routing.is_some() && rows <= TIER_ADMIT_MAX_ROWS
    }

    fn forward_device(
        &self,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        k: usize,
    ) -> candle_core::Result<Tensor> {
        self.source
            .forward_layer_device(self.layer, xs, ids, weights, k)
    }

    fn forward(
        &self,
        xs: &Tensor,
        tokens_for: &[Vec<u32>],
        weights_for: &[Vec<f32>],
        k: usize,
    ) -> candle_core::Result<Tensor> {
        self.source
            .forward_layer(self.layer, xs, tokens_for, weights_for, k)
    }
}

/// Hands each MoE block its view of one [`GgufExpertSource`].
pub struct NativeExpertFactory(pub Arc<GgufExpertSource>);

impl ExternalExpertSource for NativeExpertFactory {
    fn block(&self, layer: usize) -> Result<Box<dyn ExternalExperts>> {
        ensure!(
            layer < self.0.layers.len(),
            "no GGUF experts for layer {layer}"
        );
        Ok(Box::new(NativeBlock {
            source: self.0.clone(),
            layer,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host-side builder `forward_layer` uses: groups by ascending
    /// expert, tokens ascending within a group, each token's outputs
    /// ranked by ascending expert.
    #[allow(clippy::type_complexity)]
    fn host_groups(
        ids: &[u32],
        w: &[f32],
        rows: usize,
        k: usize,
        experts: usize,
    ) -> (Vec<u64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>) {
        let mut tokens_for = vec![Vec::new(); experts];
        let mut weights_for = vec![Vec::new(); experts];
        for t in 0..rows {
            for j in 0..k {
                tokens_for[ids[t * k + j] as usize].push(t);
                weights_for[ids[t * k + j] as usize].push(w[t * k + j]);
            }
        }
        let (mut ptr, mut start, mut dst, mut tok) = (vec![], vec![], vec![], vec![]);
        let mut w_out = vec![0f32; rows * k];
        let mut rank = vec![0usize; rows];
        for e in 0..experts {
            if tokens_for[e].is_empty() {
                continue;
            }
            ptr.push(1000 + e as u64);
            start.push(dst.len() as i32);
            for (&t, &wt) in tokens_for[e].iter().zip(&weights_for[e]) {
                let d = t * k + rank[t];
                rank[t] += 1;
                dst.push(d as i32);
                tok.push(t as i32);
                w_out[d] = wt;
            }
        }
        start.push(dst.len() as i32);
        (ptr, start, dst, tok, w_out)
    }

    /// The device builder produces exactly the host builder's arrays,
    /// for one token, a few, and a full decode-sized batch. Skips
    /// without a CUDA device.
    #[test]
    fn device_groups_match_the_host_builder() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("no CUDA device; skipped");
            return;
        };
        let cuda = device.as_cuda_device().unwrap();
        let (experts, k) = (512usize, 10usize);
        let table: Vec<u64> = (0..experts as u64).map(|e| 1000 + e).collect();
        let mut table_d = unsafe { cuda.alloc::<u64>(experts).unwrap() };
        cuda.memcpy_htod(&table, &mut table_d).unwrap();
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for rows in [1usize, 3, 16] {
            let mut ids = Vec::new();
            for _ in 0..rows {
                let mut row: Vec<u32> = Vec::new();
                while row.len() < k {
                    // Few distinct experts, so tokens share groups.
                    let e = next() % 40 * 7;
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                ids.extend(row);
            }
            let w: Vec<f32> = (0..rows * k).map(|i| i as f32 * 0.25 + 1.0).collect();
            let entries = rows * k;
            let ids_t = Tensor::from_vec(ids.clone(), (rows, k), &device).unwrap();
            let w_t = Tensor::from_vec(w.clone(), (rows, k), &device).unwrap();
            let grp_ptr = unsafe { cuda.alloc::<u64>(entries).unwrap() };
            let grp_start = unsafe { cuda.alloc::<i32>(entries + 1).unwrap() };
            let n_groups = unsafe { cuda.alloc::<i32>(1).unwrap() };
            let ent_dst = unsafe { cuda.alloc::<i32>(entries).unwrap() };
            let ent_tok = unsafe { cuda.alloc::<i32>(entries).unwrap() };
            let w_out = unsafe { cuda.alloc::<f32>(entries).unwrap() };
            let stream = cuda.cuda_stream().cu_stream() as *mut c_void;
            let status = unsafe {
                ffi::neuron_strata_route_groups(
                    tensor_address::<u32>(&ids_t).unwrap() as *const u32,
                    tensor_address::<f32>(&w_t).unwrap() as *const f32,
                    rows as i32,
                    k as i32,
                    device_address(&table_d) as *const u64,
                    std::ptr::null(),
                    0,
                    0,
                    experts as i32,
                    device_address(&grp_ptr) as *mut u64,
                    device_address(&grp_start) as *mut i32,
                    device_address(&n_groups) as *mut i32,
                    device_address(&ent_dst) as *mut i32,
                    device_address(&ent_tok) as *mut i32,
                    device_address(&w_out) as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    stream,
                )
            };
            assert_eq!(status, 0);
            let (ptr, start, dst, tok, w_want) = host_groups(&ids, &w, rows, k, experts);
            let n = cuda.clone_dtoh(&n_groups).unwrap()[0] as usize;
            assert_eq!(n, ptr.len(), "rows {rows}: group count");
            assert_eq!(&cuda.clone_dtoh(&grp_ptr).unwrap()[..n], &ptr[..]);
            assert_eq!(&cuda.clone_dtoh(&grp_start).unwrap()[..=n], &start[..]);
            assert_eq!(
                cuda.clone_dtoh(&ent_dst).unwrap(),
                dst,
                "rows {rows}: ent_dst"
            );
            assert_eq!(
                cuda.clone_dtoh(&ent_tok).unwrap(),
                tok,
                "rows {rows}: ent_tok"
            );
            assert_eq!(
                cuda.clone_dtoh(&w_out).unwrap(),
                w_want,
                "rows {rows}: weights"
            );
        }
    }

    /// Experts whose table entry is their host copy get arena slots in
    /// group order and are listed for the gather, which copies them; the
    /// rest keep their table address. Blob size is odd so the gather's
    /// byte tail runs too. Skips without a CUDA device.
    #[test]
    fn misses_get_arena_slots_and_the_gather_copies_them() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("no CUDA device; skipped");
            return;
        };
        let cuda = device.as_cuda_device().unwrap();
        let (experts, k, rows, bytes) = (8usize, 2usize, 2usize, 1001usize);
        // Host copies of every expert, patterned by expert.
        let host = unsafe { ffi::neuron_strata_host_alloc_mapped(experts * bytes) }.cast::<u8>();
        assert!(!host.is_null());
        let host_dev = unsafe { ffi::neuron_strata_host_device_ptr(host.cast()) };
        for e in 0..experts {
            for b in 0..bytes {
                unsafe { *host.add(e * bytes + b) = (e * 31 + b) as u8 };
            }
        }
        let host_table: Vec<u64> = (0..experts as u64)
            .map(|e| host_dev + e * bytes as u64)
            .collect();
        // Experts 2 and 5 resident (fake VRAM addresses), the rest misses.
        let table: Vec<u64> = (0..experts)
            .map(|e| {
                if e == 2 || e == 5 {
                    7000 + e as u64
                } else {
                    host_table[e]
                }
            })
            .collect();
        let up = |v: &[u64]| {
            let mut d = unsafe { cuda.alloc::<u64>(v.len()).unwrap() };
            cuda.memcpy_htod(v, &mut d).unwrap();
            d
        };
        let (table_d, host_d) = (up(&table), up(&host_table));
        // Token 0 routes to 5 and 1, token 1 to 6 and 2: misses are 1, 6.
        let ids_t = Tensor::from_vec(vec![5u32, 1, 6, 2], (rows, k), &device).unwrap();
        let w_t = Tensor::from_vec(vec![0.5f32, 0.5, 0.5, 0.5], (rows, k), &device).unwrap();
        let entries = rows * k;
        let arena = unsafe { cuda.alloc::<u8>(entries * bytes).unwrap() };
        let arena_at = device_address(&arena);
        let grp_ptr = unsafe { cuda.alloc::<u64>(entries).unwrap() };
        let grp_start = unsafe { cuda.alloc::<i32>(entries + 1).unwrap() };
        let n_groups = unsafe { cuda.alloc::<i32>(1).unwrap() };
        let ent_dst = unsafe { cuda.alloc::<i32>(entries).unwrap() };
        let ent_tok = unsafe { cuda.alloc::<i32>(entries).unwrap() };
        let w_out = unsafe { cuda.alloc::<f32>(entries).unwrap() };
        let miss_src = unsafe { cuda.alloc::<u64>(entries).unwrap() };
        let n_miss = unsafe { cuda.alloc::<i32>(1).unwrap() };
        let stream = cuda.cuda_stream().cu_stream() as *mut c_void;
        unsafe {
            assert_eq!(
                ffi::neuron_strata_route_groups(
                    tensor_address::<u32>(&ids_t).unwrap() as *const u32,
                    tensor_address::<f32>(&w_t).unwrap() as *const f32,
                    rows as i32,
                    k as i32,
                    device_address(&table_d) as *const u64,
                    device_address(&host_d) as *const u64,
                    arena_at,
                    bytes as u64,
                    experts as i32,
                    device_address(&grp_ptr) as *mut u64,
                    device_address(&grp_start) as *mut i32,
                    device_address(&n_groups) as *mut i32,
                    device_address(&ent_dst) as *mut i32,
                    device_address(&ent_tok) as *mut i32,
                    device_address(&w_out) as *mut f32,
                    device_address(&miss_src) as *mut u64,
                    device_address(&n_miss) as *mut i32,
                    stream,
                ),
                0
            );
            assert_eq!(
                ffi::neuron_strata_gather_blobs(
                    device_address(&miss_src) as *const u64,
                    device_address(&n_miss) as *const i32,
                    entries as i32,
                    arena_at,
                    bytes as u64,
                    stream,
                ),
                0
            );
        }
        // Groups in ascending expert: 1 (miss 0), 2, 5, 6 (miss 1).
        assert_eq!(cuda.clone_dtoh(&n_miss).unwrap()[0], 2);
        assert_eq!(
            cuda.clone_dtoh(&grp_ptr).unwrap(),
            [arena_at, 7002, 7005, arena_at + bytes as u64]
        );
        assert_eq!(
            &cuda.clone_dtoh(&miss_src).unwrap()[..2],
            &[host_table[1], host_table[6]]
        );
        let copied = cuda.clone_dtoh(&arena).unwrap();
        for (slot, e) in [(0usize, 1usize), (1, 6)] {
            for b in 0..bytes {
                assert_eq!(
                    copied[slot * bytes + b],
                    (e * 31 + b) as u8,
                    "slot {slot} byte {b}"
                );
            }
        }
        unsafe { ffi::neuron_strata_pinned_free(host.cast()) };
    }
}
