//! Direct safetensors readers for fused-region weight tensors.
//!
//! Qwen3-Next's `in_proj_qkv` and `conv1d` weights are *fused* —
//! three regions stored sequentially along dim 0 (`[key_q, key_k,
//! value]`). The per-rank shard for each region has unequal size
//! (`key_dim/ws` vs `value_dim/ws`), so candle's `ShardedSafeTensors`
//! built-in `Shard { dim, rank, world_size }` (uniform split) doesn't
//! map to the right slices.
//!
//! The previous approach loaded the full fused tensor onto the device,
//! `narrow`ed the three regions, and `Tensor::cat(...).contiguous()`'d
//! the per-rank slice. That left ~100 MB of transient device memory
//! per linear-attention layer — 48 layers × 100 MB = ~4.8 GB of
//! allocator pressure during load, enough to trigger fragmentation
//! OOM on tight-VRAM consumer GPUs.
//!
//! This module reads the three per-rank byte ranges *directly from
//! the safetensors mmap* (host-side), concatenates them into a single
//! contiguous byte buffer, and uploads as one device allocation. No
//! full-tensor device materialisation.

use anyhow::{Context, Result, bail};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};

/// Read a 2D fused-QKV tensor `[conv_dim, hidden_size]` and return
/// this rank's per-region slice as a `[per_rank_conv_dim, hidden_size]`
/// device tensor.
///
/// `tensor_name` must be the fully-qualified safetensors key (e.g.
/// `"model.language_model.layers.5.linear_attn.in_proj_qkv.weight"`).
#[allow(clippy::too_many_arguments)]
pub fn load_fused_qkv_2d(
    mmap: &MmapedSafetensors,
    tensor_name: &str,
    hidden_size: usize,
    key_dim: usize,
    value_dim: usize,
    rank: u32,
    world_size: u32,
    target_dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let ws = world_size as usize;
    let r = rank as usize;
    if !key_dim.is_multiple_of(ws) || !value_dim.is_multiple_of(ws) {
        bail!(
            "fused qkv shard: key_dim ({key_dim}) and value_dim ({value_dim}) \
             must each be divisible by world_size ({ws})"
        );
    }
    let per_rank_key = key_dim / ws;
    let per_rank_value = value_dim / ws;
    let per_rank_conv_dim = per_rank_key * 2 + per_rank_value;

    let view = mmap
        .get(tensor_name)
        .with_context(|| format!("mmap.get('{tensor_name}') for fused qkv 2D"))?;
    let view_dtype: DType = view
        .dtype()
        .try_into()
        .with_context(|| format!("safetensors dtype unsupported for '{tensor_name}'"))?;

    let shape = view.shape();
    if shape.len() != 2 {
        bail!(
            "fused qkv tensor '{tensor_name}' has shape {shape:?}, expected 2D \
             [conv_dim, hidden_size]"
        );
    }
    let conv_dim = key_dim * 2 + value_dim;
    if shape[0] != conv_dim || shape[1] != hidden_size {
        bail!(
            "fused qkv tensor '{tensor_name}' shape {shape:?} \
             doesn't match expected [{conv_dim}, {hidden_size}]"
        );
    }

    let q_bytes = slice_dim0_bytes(&view, r * per_rank_key, per_rank_key, tensor_name, "q")?;
    let k_bytes = slice_dim0_bytes(
        &view,
        key_dim + r * per_rank_key,
        per_rank_key,
        tensor_name,
        "k",
    )?;
    let v_bytes = slice_dim0_bytes(
        &view,
        2 * key_dim + r * per_rank_value,
        per_rank_value,
        tensor_name,
        "v",
    )?;

    let mut bytes = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
    bytes.extend_from_slice(&q_bytes);
    bytes.extend_from_slice(&k_bytes);
    bytes.extend_from_slice(&v_bytes);

    let tensor = Tensor::from_raw_buffer(
        &bytes,
        view_dtype,
        &[per_rank_conv_dim, hidden_size],
        device,
    )
    .with_context(|| format!("Tensor::from_raw_buffer for per-rank fused qkv '{tensor_name}'"))?;
    tensor
        .to_dtype(target_dtype)
        .with_context(|| format!("cast '{tensor_name}' to {target_dtype:?}"))
}

/// Read a 3D fused-QKV tensor `[conv_dim, 1, kernel_size]` (the
/// depthwise conv1d weight) and return this rank's per-region slice
/// as a `[per_rank_conv_dim, 1, kernel_size]` device tensor.
#[allow(clippy::too_many_arguments)]
pub fn load_fused_qkv_3d(
    mmap: &MmapedSafetensors,
    tensor_name: &str,
    mid: usize,
    kernel_size: usize,
    key_dim: usize,
    value_dim: usize,
    rank: u32,
    world_size: u32,
    target_dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let ws = world_size as usize;
    let r = rank as usize;
    if !key_dim.is_multiple_of(ws) || !value_dim.is_multiple_of(ws) {
        bail!(
            "fused conv shard: key_dim ({key_dim}) and value_dim ({value_dim}) \
             must each be divisible by world_size ({ws})"
        );
    }
    let per_rank_key = key_dim / ws;
    let per_rank_value = value_dim / ws;
    let per_rank_conv_dim = per_rank_key * 2 + per_rank_value;

    let view = mmap
        .get(tensor_name)
        .with_context(|| format!("mmap.get('{tensor_name}') for fused qkv 3D"))?;
    let view_dtype: DType = view
        .dtype()
        .try_into()
        .with_context(|| format!("safetensors dtype unsupported for '{tensor_name}'"))?;

    let shape = view.shape();
    if shape.len() != 3 {
        bail!(
            "fused conv tensor '{tensor_name}' has shape {shape:?}, expected 3D \
             [conv_dim, mid, kernel_size]"
        );
    }
    let conv_dim = key_dim * 2 + value_dim;
    if shape[0] != conv_dim || shape[1] != mid || shape[2] != kernel_size {
        bail!(
            "fused conv tensor '{tensor_name}' shape {shape:?} \
             doesn't match expected [{conv_dim}, {mid}, {kernel_size}]"
        );
    }

    let q_bytes = slice_dim0_bytes(&view, r * per_rank_key, per_rank_key, tensor_name, "q")?;
    let k_bytes = slice_dim0_bytes(
        &view,
        key_dim + r * per_rank_key,
        per_rank_key,
        tensor_name,
        "k",
    )?;
    let v_bytes = slice_dim0_bytes(
        &view,
        2 * key_dim + r * per_rank_value,
        per_rank_value,
        tensor_name,
        "v",
    )?;

    let mut bytes = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
    bytes.extend_from_slice(&q_bytes);
    bytes.extend_from_slice(&k_bytes);
    bytes.extend_from_slice(&v_bytes);

    let tensor = Tensor::from_raw_buffer(
        &bytes,
        view_dtype,
        &[per_rank_conv_dim, mid, kernel_size],
        device,
    )
    .with_context(|| format!("Tensor::from_raw_buffer for per-rank fused conv '{tensor_name}'"))?;
    tensor
        .to_dtype(target_dtype)
        .with_context(|| format!("cast '{tensor_name}' to {target_dtype:?}"))
}

/// Read `len` consecutive rows along dim 0 starting at `start` from
/// the safetensors view, returning the raw bytes. Wraps the same
/// `view.slice(start..stop)` machinery that candle's
/// `ShardedSafeTensors::get` uses internally.
fn slice_dim0_bytes(
    view: &safetensors::tensor::TensorView<'_>,
    start: usize,
    len: usize,
    tensor_name: &str,
    region: &str,
) -> Result<Vec<u8>> {
    use safetensors::slice::IndexOp;
    let stop = start + len;
    let iter = view.slice(start..stop).map_err(|e| {
        anyhow::anyhow!("slice '{tensor_name}' region {region} ({start}..{stop}): {e:?}")
    })?;
    Ok(iter.into_iter().flatten().copied().collect())
}
