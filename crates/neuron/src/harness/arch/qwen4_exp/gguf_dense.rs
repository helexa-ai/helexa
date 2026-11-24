//! Dense projections from a GGUF's quantized blocks, run on Strata's
//! kernels: a quantized matrix-vector product (`mmvq`) for decode-sized
//! inputs, and dequantize + cuBLAS for prompt chunks.
//!
//! Most of the model's dense tensors are stored as Q8_0 in the UD packs,
//! half of BF16. They map to the checkpoint by name, each in its own
//! layout, except the linear-attention value heads: llama.cpp's
//! converter stores them tiled by key head where the checkpoint groups
//! them. With `r = value_heads / key_heads`, GGUF head `g * key_heads +
//! k` holds checkpoint head `r * k + g`. That is undone at load by
//! moving whole quantized rows (or whole blocks within each row, where
//! the heads are the input dimension), so no value is requantized.

use anyhow::{Result, ensure};

/// How a GGUF tensor's value heads must move to match the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reorder {
    None,
    /// Output rows, after the first `skip_rows` (the q and k of a fused
    /// qkv projection, which are stored in order).
    Rows {
        skip_rows: usize,
    },
    /// Input columns: within every row, per head of `head_dim` values.
    Cols,
}

/// The GGUF tensor holding checkpoint tensor `name`, and its reorder.
pub(crate) fn gguf_name(name: &str) -> Option<(String, Reorder, bool)> {
    // The bool says whether `skip_rows` must be filled in from the
    // config: only the fused qkv knows its q and k widths.
    if name == "lm_head.weight" {
        return Some(("output.weight".into(), Reorder::None, false));
    }
    if let Some(rest) = name.strip_prefix("model.language_model.hyper_connection_mixer.") {
        let gguf = match rest {
            "input_mix_weight_up.weight" => "output_hc_up.weight",
            "input_mix_weight_down.weight" => "output_hc_down.weight",
            _ => return None,
        };
        return Some((gguf.into(), Reorder::None, false));
    }
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (gguf, reorder, needs_skip) = match rest {
        "linear_attn.in_proj_qkv.weight" => ("attn_qkv", Reorder::Rows { skip_rows: 0 }, true),
        "linear_attn.in_proj_z.weight" => ("attn_gate", Reorder::Rows { skip_rows: 0 }, false),
        "linear_attn.out_proj.weight" => ("ssm_out", Reorder::Cols, false),
        "self_attn.q_proj.weight" => ("attn_q", Reorder::None, false),
        "self_attn.k_proj.weight" => ("attn_k", Reorder::None, false),
        "self_attn.v_proj.weight" => ("attn_v", Reorder::None, false),
        "self_attn.o_proj.weight" => ("attn_output", Reorder::None, false),
        "mlp.shared_expert.gate_proj.weight" => ("ffn_gate_shexp", Reorder::None, false),
        "mlp.shared_expert.up_proj.weight" => ("ffn_up_shexp", Reorder::None, false),
        "mlp.shared_expert.down_proj.weight" => ("ffn_down_shexp", Reorder::None, false),
        "attn_hyper_connection.input_mix_weight_up.weight" => ("hc_attn_up", Reorder::None, false),
        "attn_hyper_connection.input_mix_weight_down.weight" => {
            ("hc_attn_down", Reorder::None, false)
        }
        "mlp_hyper_connection.input_mix_weight_up.weight" => ("hc_ffn_up", Reorder::None, false),
        "mlp_hyper_connection.input_mix_weight_down.weight" => {
            ("hc_ffn_down", Reorder::None, false)
        }
        _ => return None,
    };
    Some((format!("blk.{layer}.{gguf}.weight"), reorder, needs_skip))
}

/// The linear-attention head geometry the reorder needs.
#[derive(Debug, Clone, Copy)]
pub struct Heads {
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
}

impl Heads {
    /// The GGUF head that holds checkpoint head `h`.
    fn source(&self, h: usize) -> usize {
        let r = self.value_heads / self.key_heads;
        (h % r) * self.key_heads + h / r
    }
}

/// Reorders a tensor of `n_out` rows of `row_bytes` into the
/// checkpoint's layout. `head_bytes` is one head's span within a row
/// (only read for [`Reorder::Cols`]).
pub(crate) fn reorder(
    src: &[u8],
    n_out: usize,
    row_bytes: usize,
    head_bytes: usize,
    how: Reorder,
    heads: Heads,
) -> Result<Vec<u8>> {
    ensure!(src.len() == n_out * row_bytes, "tensor is not n_out rows");
    match how {
        Reorder::None => Ok(src.to_vec()),
        Reorder::Rows { skip_rows } => {
            let per_head = heads.head_dim * row_bytes;
            ensure!(
                n_out == skip_rows + heads.value_heads * heads.head_dim,
                "{n_out} rows are not {skip_rows} + {} value heads of {}",
                heads.value_heads,
                heads.head_dim
            );
            let mut out = src[..skip_rows * row_bytes].to_vec();
            let base = skip_rows * row_bytes;
            for h in 0..heads.value_heads {
                let at = base + heads.source(h) * per_head;
                out.extend_from_slice(&src[at..at + per_head]);
            }
            Ok(out)
        }
        Reorder::Cols => {
            ensure!(
                row_bytes == heads.value_heads * head_bytes,
                "a {row_bytes}-byte row is not {} heads of {head_bytes}",
                heads.value_heads
            );
            let mut out = Vec::with_capacity(src.len());
            for row in src.chunks_exact(row_bytes) {
                for h in 0..heads.value_heads {
                    let at = heads.source(h) * head_bytes;
                    out.extend_from_slice(&row[at..at + head_bytes]);
                }
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "strata-kernels")]
pub use kernels::GgufDenseSource;

#[cfg(feature = "strata-kernels")]
mod kernels {
    use std::ffi::c_void;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use anyhow::{Context, Result, bail, ensure};
    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
    use candle_core::{CudaDevice, CudaStorage, DType, Device, Storage, Tensor};
    use memmap2::{Advice, Mmap};

    use super::{Heads, gguf_name, reorder};
    use crate::cuda::strata as ffi;
    use crate::harness::arch::qwen3_5::proj::{ExternalLinear, LinearSource};
    use crate::harness::arch::qwen4_exp::gguf_index::GgufIndex;

    const Q8_1_BLOCK_BYTES: usize = 36;
    /// Inputs of up to this many rows take the matrix-vector kernel;
    /// larger ones dequantize the weight and use a matmul.
    const MMVQ_MAX_ROWS: usize = 8;

    fn address<T>(slice: &CudaSlice<T>) -> u64 {
        slice.device_ptr(slice.stream()).0
    }

    pub struct GgufDenseSource {
        device: CudaDevice,
        index: GgufIndex,
        maps: Vec<Mmap>,
        heads: Heads,
        key_dim: usize,
    }

    impl GgufDenseSource {
        pub fn open(first: &Path, device: &Device, heads: Heads, key_dim: usize) -> Result<Self> {
            let device = device.as_cuda_device()?.clone();
            let index = GgufIndex::open(first)?;
            let mut maps = Vec::with_capacity(index.shards.len());
            for path in &index.shards {
                let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
                // SAFETY: model files are not modified while serving.
                let map = unsafe { Mmap::map(&file) }
                    .with_context(|| format!("map {}", path.display()))?;
                map.advise(Advice::Sequential).context("advise dense map")?;
                maps.push(map);
            }
            tracing::info!(
                shards = maps.len(),
                "dense projections from GGUF on native kernels"
            );
            Ok(Self {
                device,
                index,
                maps,
                heads,
                key_dim,
            })
        }
    }

    impl LinearSource for GgufDenseSource {
        fn linear(
            &self,
            name: &str,
            in_dim: usize,
            out_dim: usize,
        ) -> Result<Option<Arc<dyn ExternalLinear>>> {
            let Some((gguf, mut how, needs_skip)) = gguf_name(name) else {
                return Ok(None);
            };
            let Some(tensor) = self.index.tensors.get(&gguf) else {
                return Ok(None);
            };
            let ty = tensor.ggml_type as i32;
            // SAFETY: a pure function of its argument.
            if unsafe { ffi::neuron_strata_mmvq_supported(ty) } == 0 {
                tracing::info!(
                    name,
                    ggml_type = ty,
                    "no native kernel for this type; dense"
                );
                return Ok(None);
            }
            ensure!(
                tensor.dims == [in_dim as u64, out_dim as u64],
                "{gguf} is {:?}, the checkpoint's {name} is [{in_dim}, {out_dim}]",
                tensor.dims
            );
            if needs_skip {
                how = super::Reorder::Rows {
                    skip_rows: 2 * self.key_dim,
                };
            }
            // SAFETY: pure functions of their integer arguments.
            let (row_bytes, head_bytes) = unsafe {
                (
                    ffi::neuron_strata_row_bytes(ty, in_dim as i64),
                    ffi::neuron_strata_row_bytes(ty, self.heads.head_dim as i64),
                )
            };
            ensure!(row_bytes > 0, "{gguf}: no row size for type {ty}");
            if how == super::Reorder::Cols && head_bytes * self.heads.value_heads != row_bytes {
                bail!(
                    "{gguf}: type {ty} blocks do not divide a {}-wide head",
                    self.heads.head_dim
                );
            }
            let start = tensor.offset as usize;
            let bytes = out_dim * row_bytes;
            let map = &self.maps[tensor.shard];
            ensure!(start + bytes <= map.len(), "{gguf} runs past its shard");
            let host = reorder(
                &map[start..start + bytes],
                out_dim,
                row_bytes,
                head_bytes,
                how,
                self.heads,
            )
            .with_context(|| format!("reorder {gguf}"))?;
            // SAFETY: written in full by the copy below.
            let mut weight = unsafe { self.device.alloc::<u8>(bytes) }?;
            self.device.memcpy_htod(&host, &mut weight)?;
            Ok(Some(Arc::new(QuantLinear {
                device: self.device.clone(),
                weight,
                ty,
                n_in: in_dim,
                n_out: out_dim,
            })))
        }
    }

    struct QuantLinear {
        device: CudaDevice,
        weight: CudaSlice<u8>,
        ty: i32,
        n_in: usize,
        n_out: usize,
    }

    impl QuantLinear {
        fn stream(&self) -> *mut c_void {
            self.device.cuda_stream().cu_stream() as *mut c_void
        }

        /// `(rows, n_in)` f32 activations to `(rows, n_out)` f32.
        fn mmvq(&self, x: &Tensor, rows: usize) -> candle_core::Result<Tensor> {
            let (storage, layout) = x.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                candle_core::bail!("native dense projections need activations on the GPU");
            };
            let x_slice = storage.as_cuda_slice::<f32>()?;
            let x_ptr = (address(x_slice) as usize + layout.start_offset() * 4) as *const f32;
            // SAFETY: both are written in full before they are read.
            let q8 = unsafe {
                self.device
                    .alloc::<u8>(rows * (self.n_in / 32) * Q8_1_BLOCK_BYTES)?
            };
            // SAFETY: as above.
            let out = unsafe { self.device.alloc::<f32>(rows * self.n_out)? };
            // SAFETY: every pointer is a live device allocation of the
            // size the kernels read or write; the stream orders them.
            unsafe {
                ffi::neuron_strata_quantize_q8_1(
                    x_ptr,
                    rows as i64,
                    self.n_in as i64,
                    address(&q8) as *mut c_void,
                    self.stream(),
                );
                ffi::neuron_strata_mmvq(
                    self.ty,
                    address(&self.weight) as *const c_void,
                    address(&q8) as *const c_void,
                    address(&out) as *mut f32,
                    self.n_in as i32,
                    self.n_out as i32,
                    rows as i32,
                    self.stream(),
                );
            }
            // `q8` is freed stream-ordered, after the kernel has read it.
            drop(q8);
            Ok(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, self.device.clone())),
                (rows, self.n_out),
            )))
        }

        /// The weight as a `(n_out, n_in)` tensor of `dtype`.
        fn dense(&self, dtype: DType) -> candle_core::Result<Tensor> {
            let n = self.n_in * self.n_out;
            // SAFETY: written in full by the dequantize below.
            let out = unsafe { self.device.alloc::<f32>(n)? };
            // SAFETY: the weight holds `n` values of `ty`; `out` holds n.
            unsafe {
                ffi::neuron_strata_dequant_f32(
                    self.ty,
                    address(&self.weight) as *const c_void,
                    n as i64,
                    address(&out) as *mut f32,
                    self.stream(),
                );
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, self.device.clone())),
                (self.n_out, self.n_in),
            ))
            .to_dtype(dtype)
        }
    }

    impl ExternalLinear for QuantLinear {
        fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
            let dims = x.dims().to_vec();
            let Some((&n_in, lead)) = dims.split_last() else {
                candle_core::bail!("a projection needs at least one dimension");
            };
            if n_in != self.n_in {
                candle_core::bail!("input is {n_in} wide, the weight takes {}", self.n_in);
            }
            let rows: usize = lead.iter().product();
            let x2 = x.reshape((rows, n_in))?;
            let y = if rows <= MMVQ_MAX_ROWS {
                let x32 = x2.to_dtype(DType::F32)?.contiguous()?;
                self.mmvq(&x32, rows)?.to_dtype(x.dtype())?
            } else {
                let w = self.dense(x.dtype())?;
                x2.contiguous()?.matmul(&w.t()?)?
            };
            let mut out_dims = lead.to_vec();
            out_dims.push(self.n_out);
            y.reshape(out_dims)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADS: Heads = Heads {
        key_heads: 2,
        value_heads: 6,
        head_dim: 1,
    };

    /// The mapping recovered from the shipped GGUF (16 key heads, 48
    /// value heads): GGUF head `g * 16 + k` is checkpoint head `3k + g`.
    #[test]
    fn the_shipped_geometry_matches_the_measured_permutation() {
        let heads = Heads {
            key_heads: 16,
            value_heads: 48,
            head_dim: 128,
        };
        // Checkpoint heads 0, 1, 2 are GGUF 0, 16, 32; checkpoint 3 is 1.
        assert_eq!(
            (0..5).map(|h| heads.source(h)).collect::<Vec<_>>(),
            [0, 16, 32, 1, 17]
        );
        // And measured directly: GGUF position 1 holds checkpoint head 3.
        let gguf_holds: Vec<usize> = (0..48)
            .map(|p| (0..48).find(|&h| heads.source(h) == p).unwrap())
            .collect();
        assert_eq!(&gguf_holds[..4], [0, 3, 6, 9]);
        assert_eq!(gguf_holds[16], 1);
    }

    /// Rows after the skipped q/k move as whole heads; the skipped rows
    /// keep their place.
    #[test]
    fn rows_move_by_head_after_the_skip() {
        // Two skipped rows, then six value heads of one row each, two
        // bytes per row. GGUF value position p holds head `gguf_holds[p]`.
        let gguf_holds = [0u8, 3, 1, 4, 2, 5];
        let mut src = vec![100, 101, 102, 103];
        for h in gguf_holds {
            src.extend_from_slice(&[h, h]);
        }
        let got = reorder(&src, 8, 2, 0, Reorder::Rows { skip_rows: 2 }, HEADS).unwrap();
        assert_eq!(
            got,
            [100, 101, 102, 103, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        );
    }

    /// Columns move as whole head spans within every row.
    #[test]
    fn columns_move_by_head_within_each_row() {
        let gguf_holds = [0u8, 3, 1, 4, 2, 5];
        let mut src = Vec::new();
        for row in 0..2u8 {
            for h in gguf_holds {
                src.extend_from_slice(&[10 * row + h, 10 * row + h]);
            }
        }
        let got = reorder(&src, 2, 12, 2, Reorder::Cols, HEADS).unwrap();
        let want: Vec<u8> = (0..2u8)
            .flat_map(|row| (0..6u8).flat_map(move |h| [10 * row + h, 10 * row + h]))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn names_map_to_their_gguf_tensors() {
        assert_eq!(
            gguf_name("model.language_model.layers.7.linear_attn.out_proj.weight"),
            Some(("blk.7.ssm_out.weight".into(), Reorder::Cols, false))
        );
        assert_eq!(
            gguf_name("model.language_model.layers.3.self_attn.q_proj.weight"),
            Some(("blk.3.attn_q.weight".into(), Reorder::None, false))
        );
        assert_eq!(
            gguf_name("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            Some((
                "blk.0.attn_qkv.weight".into(),
                Reorder::Rows { skip_rows: 0 },
                true
            ))
        );
        assert_eq!(
            gguf_name("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"),
            Some(("output_hc_up.weight".into(), Reorder::None, false))
        );
        // Unmapped names — the indexer, the MTP head — stay dense.
        assert_eq!(
            gguf_name("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"),
            None
        );
        assert_eq!(gguf_name("mtp.layers.0.self_attn.q_proj.weight"), None);
    }

    #[test]
    fn a_tensor_of_the_wrong_size_is_refused() {
        assert!(reorder(&[0; 5], 2, 2, 0, Reorder::None, HEADS).is_err());
        assert!(reorder(&[0; 14], 7, 2, 0, Reorder::Rows { skip_rows: 2 }, HEADS).is_err());
    }
}
