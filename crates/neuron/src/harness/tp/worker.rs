//! Entry point for `neuron --worker`.
//!
//! The worker reads one newline-delimited JSON `WorkerRequest` from
//! stdin per loop iteration, dispatches synchronously, and writes
//! exactly one `WorkerResponse` JSON line to stdout. tracing goes to
//! stderr so it doesn't collide with the RPC stream.
//!
//! NCCL operations (`Init`, `NcclSanityCheck`) and model lifecycle ops
//! (`LoadDenseShard`, `GenerateStep`, `ClearKvCache`, `UnloadModel`)
//! are real when built with the `cuda` feature; without it they reply
//! with `Error{kind="cuda_feature_not_enabled"}` so the leader can tell
//! the difference between a misconfigured build and a genuine NCCL or
//! model failure.

use anyhow::Result;
use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::nccl_state::NcclState;
use super::rpc::{WorkerRequest, WorkerResponse};

#[cfg(feature = "cuda")]
use super::tp_qwen3::TpQwen3ForCausalLM;
#[cfg(feature = "cuda")]
use super::tp_qwen3_5::TpQwen3_5ForCausalLM;

/// Worker-side discriminator over the architectures we can load via
/// `LoadDenseShard`. Mirrors `super::TpLeaderModel` on the leader
/// side — the dispatch happens on the `model_type` extracted from the
/// config JSON.
#[cfg(feature = "cuda")]
enum WorkerModel {
    Qwen3(TpQwen3ForCausalLM),
    Qwen3_5(TpQwen3_5ForCausalLM),
}

#[cfg(feature = "cuda")]
impl WorkerModel {
    fn forward(
        &mut self,
        input: &candle_core::Tensor,
        offset: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            WorkerModel::Qwen3(m) => m.forward(input, offset),
            WorkerModel::Qwen3_5(m) => m.forward(input, offset),
        }
    }

    /// Chunked image prefill on this rank. Only the vision-capable
    /// `qwen3_5` arch has a replicated tower; the dense `qwen3` arch
    /// errors. The returned logits are discarded by the caller (the
    /// leader samples from its own rank-0 copy) — the value is the NCCL
    /// collectives the forward issues, chunk by chunk in lockstep with
    /// the leader.
    fn prefill_with_images_chunked(
        &mut self,
        tokens: &[u32],
        base_offset: usize,
        image_pixels: &[candle_core::Tensor],
        image_token_id: u32,
        chunk_size: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            WorkerModel::Qwen3_5(m) => m.prefill_with_images_chunked(
                tokens,
                base_offset,
                image_pixels,
                image_token_id,
                chunk_size,
            ),
            WorkerModel::Qwen3(_) => {
                candle_core::bail!("prefill_with_images_chunked: qwen3 (dense) has no vision tower")
            }
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            WorkerModel::Qwen3(m) => m.clear_kv_cache(),
            WorkerModel::Qwen3_5(m) => m.clear_kv_cache(),
        }
    }

    fn device(&self) -> &candle_core::Device {
        match self {
            WorkerModel::Qwen3(m) => m.device(),
            WorkerModel::Qwen3_5(m) => m.device(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkerConfig {
    pub rank: u32,
    pub world_size: u32,
    pub cuda_device: u32,
}

/// Drive the worker RPC loop until `Shutdown` or EOF on stdin.
pub async fn run(config: WorkerConfig) -> Result<()> {
    tracing::info!(
        rank = config.rank,
        world_size = config.world_size,
        cuda_device = config.cuda_device,
        "tp worker starting"
    );

    let mut state = WorkerState::new(config);
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let req: WorkerRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = WorkerResponse::Error {
                    kind: "bad_request".into(),
                    message: format!("parse {line:?}: {e}"),
                };
                write_response(&mut stdout, &resp).await?;
                continue;
            }
        };

        let resp = state.handle(req).await;
        let is_bye = matches!(resp, WorkerResponse::Bye);
        write_response(&mut stdout, &resp).await?;
        if is_bye {
            break;
        }
    }

    tracing::info!(rank = config.rank, "tp worker exiting");
    Ok(())
}

async fn write_response(stdout: &mut tokio::io::Stdout, resp: &WorkerResponse) -> Result<()> {
    let mut line = serde_json::to_string(resp)?;
    line.push('\n');
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

/// One rank's local state. Owns the rank's NCCL communicator (via
/// `NcclState`) and the rank's shard of every loaded model.
struct WorkerState {
    config: WorkerConfig,
    nccl: NcclState,
    /// Loaded model shards keyed by `model_id`. Each entry wraps the
    /// rank's TP architecture handle (Qwen3 or Qwen3-Next) — the
    /// column/row-parallel layers hold an `Arc<Comm>` cloned from
    /// `nccl`. Cuda-only: the underlying types reference cudarc types
    /// that don't exist without the cuda feature.
    #[cfg(feature = "cuda")]
    models: HashMap<String, WorkerModel>,
    /// Placeholder so the non-cuda build keeps the same field name set
    /// and `WorkerState::new` reads the same on both.
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    models: HashMap<String, ()>,
}

impl WorkerState {
    fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            nccl: NcclState::new(),
            models: HashMap::new(),
        }
    }

    async fn handle(&mut self, req: WorkerRequest) -> WorkerResponse {
        match req {
            WorkerRequest::Ping => WorkerResponse::Pong {
                rank: self.config.rank,
                world_size: self.config.world_size,
                cuda_device: self.config.cuda_device,
            },
            WorkerRequest::Init { comm_id } => self.nccl.init(self.config, &comm_id),
            WorkerRequest::NcclSanityCheck => self.nccl.sanity_check(),
            WorkerRequest::LoadDenseShard {
                model_id,
                config_json,
                safetensors_paths,
                quant,
            } => self.handle_load_dense_shard(model_id, config_json, safetensors_paths, quant),
            WorkerRequest::GenerateStep {
                model_id,
                tokens,
                offset,
            } => self.handle_generate_step(&model_id, tokens, offset),
            WorkerRequest::GenerateStepWithImages {
                model_id,
                tokens,
                offset,
                image_token_id,
                image_data_uris,
                chunk_size,
            } => self.handle_generate_step_with_images(
                &model_id,
                tokens,
                offset,
                image_token_id,
                image_data_uris,
                chunk_size,
            ),
            WorkerRequest::ClearKvCache { model_id } => self.handle_clear_kv_cache(&model_id),
            WorkerRequest::UnloadModel { model_id } => self.handle_unload_model(&model_id),
            WorkerRequest::Shutdown => WorkerResponse::Bye,
        }
    }

    #[cfg(feature = "cuda")]
    fn handle_load_dense_shard(
        &mut self,
        model_id: String,
        config_json: String,
        safetensors_paths: Vec<String>,
        quant: Option<String>,
    ) -> WorkerResponse {
        use crate::harness::arch::qwen3_5 as qwen3_5_arch;
        use candle_core::{DType, Device};
        use candle_nn::var_builder::ShardedSafeTensors;
        use candle_transformers::models::qwen3 as qwen3_dense;
        use std::path::PathBuf;

        let quant_dtype = match parse_quant_string(quant.as_deref()) {
            Ok(q) => q,
            Err(e) => {
                return WorkerResponse::Error {
                    kind: "bad_request".into(),
                    message: format!("parse quant: {e}"),
                };
            }
        };

        if self.models.contains_key(&model_id) {
            return WorkerResponse::Error {
                kind: "already_loaded".into(),
                message: format!("model '{model_id}' already loaded on this rank"),
            };
        }
        let comm = match self.nccl.comm() {
            Some(c) => c,
            None => {
                return WorkerResponse::Error {
                    kind: "nccl_not_initialised".into(),
                    message: "LoadDenseShard requires Init to have completed first".into(),
                };
            }
        };

        // Peek at model_type so we know which architecture to build.
        let model_type = serde_json::from_str::<serde_json::Value>(&config_json)
            .ok()
            .as_ref()
            .and_then(|v| v.get("model_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let device = match Device::new_cuda(self.config.cuda_device as usize) {
            Ok(d) => d,
            Err(e) => {
                return WorkerResponse::Error {
                    kind: "cuda_unavailable".into(),
                    message: format!("Device::new_cuda({}) failed: {e}", self.config.cuda_device),
                };
            }
        };

        let paths: Vec<PathBuf> = safetensors_paths.into_iter().map(PathBuf::from).collect();
        // SAFETY: same invariant as the single-GPU dense path — the HF
        // cache files are treated as immutable while the mmap is held.
        let vb = match unsafe { ShardedSafeTensors::var_builder(&paths, DType::BF16, &device) } {
            Ok(v) => v,
            Err(e) => {
                return WorkerResponse::Error {
                    kind: "load_failed".into(),
                    message: format!("ShardedSafeTensors::var_builder: {e}"),
                };
            }
        };
        // Separate mmap of the same paths for the direct fused-region
        // loader in `fused_load`. Linux's page cache shares the
        // underlying pages between the two mmaps; the cost is one
        // extra set of safetensors-header parses.
        let mmap = match unsafe { candle_core::safetensors::MmapedSafetensors::multi(&paths) } {
            Ok(m) => m,
            Err(e) => {
                return WorkerResponse::Error {
                    kind: "load_failed".into(),
                    message: format!("MmapedSafetensors::multi: {e}"),
                };
            }
        };

        let loaded = match model_type.as_str() {
            "qwen3" => {
                let cfg: qwen3_dense::Config = match serde_json::from_str(&config_json) {
                    Ok(c) => c,
                    Err(e) => {
                        return WorkerResponse::Error {
                            kind: "bad_request".into(),
                            message: format!("parse Qwen3 Config JSON: {e}"),
                        };
                    }
                };
                match TpQwen3ForCausalLM::load(
                    &cfg,
                    &vb,
                    self.config.rank,
                    self.config.world_size,
                    comm,
                ) {
                    Ok(m) => WorkerModel::Qwen3(m),
                    Err(e) => {
                        return WorkerResponse::Error {
                            kind: "load_failed".into(),
                            message: format!("TpQwen3ForCausalLM::load: {e:#}"),
                        };
                    }
                }
            }
            "qwen3_5" => {
                let cfg: qwen3_5_arch::Config = match serde_json::from_str(&config_json) {
                    Ok(c) => c,
                    Err(e) => {
                        return WorkerResponse::Error {
                            kind: "bad_request".into(),
                            message: format!("parse Qwen3-Next Config JSON: {e}"),
                        };
                    }
                };
                match TpQwen3_5ForCausalLM::load(
                    cfg,
                    &vb,
                    &mmap,
                    self.config.rank,
                    self.config.world_size,
                    comm,
                    quant_dtype,
                ) {
                    Ok(m) => WorkerModel::Qwen3_5(m),
                    Err(e) => {
                        return WorkerResponse::Error {
                            kind: "load_failed".into(),
                            message: format!("TpQwen3_5ForCausalLM::load: {e:#}"),
                        };
                    }
                }
            }
            other => {
                return WorkerResponse::Error {
                    kind: "unsupported_arch".into(),
                    message: format!(
                        "worker: unsupported model_type '{other}' (supported: qwen3, qwen3_5)"
                    ),
                };
            }
        };

        self.models.insert(model_id.clone(), loaded);
        tracing::info!(
            rank = self.config.rank,
            model = %model_id,
            model_type = %model_type,
            "loaded TP shard"
        );
        WorkerResponse::LoadDenseShardOk
    }

    #[cfg(not(feature = "cuda"))]
    fn handle_load_dense_shard(
        &mut self,
        _model_id: String,
        _config_json: String,
        _safetensors_paths: Vec<String>,
        _quant: Option<String>,
    ) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "LoadDenseShard requires --features cuda".into(),
        }
    }

    #[cfg(feature = "cuda")]
    fn handle_generate_step(
        &mut self,
        model_id: &str,
        tokens: Vec<u32>,
        offset: usize,
    ) -> WorkerResponse {
        use candle_core::Tensor;

        let Some(model) = self.models.get_mut(model_id) else {
            return WorkerResponse::Error {
                kind: "model_not_loaded".into(),
                message: format!("model '{model_id}' not loaded on rank {}", self.config.rank),
            };
        };
        let device = model.device().clone();
        let input = match Tensor::new(tokens.as_slice(), &device).and_then(|t| t.unsqueeze(0)) {
            Ok(t) => t,
            Err(e) => {
                return WorkerResponse::Error {
                    kind: "forward_failed".into(),
                    message: format!("build input tensor: {e}"),
                };
            }
        };
        let start = std::time::Instant::now();
        tracing::debug!(
            rank = self.config.rank,
            model = %model_id,
            tokens = tokens.len(),
            offset,
            "worker GenerateStep: forward starting"
        );
        // Drop the resulting logits — the leader uses its own copy from
        // rank 0. The forward's value here is the NCCL collectives it
        // issues, which let the leader's rank-0 forward make progress.
        if let Err(e) = model.forward(&input, offset) {
            tracing::warn!(
                rank = self.config.rank,
                model = %model_id,
                elapsed_ms = start.elapsed().as_millis(),
                error = %e,
                "worker GenerateStep: forward failed"
            );
            return WorkerResponse::Error {
                kind: "forward_failed".into(),
                message: format!("TP forward: {e}"),
            };
        }
        tracing::debug!(
            rank = self.config.rank,
            model = %model_id,
            elapsed_ms = start.elapsed().as_millis(),
            "worker GenerateStep: forward done"
        );
        WorkerResponse::GenerateStepOk
    }

    #[cfg(not(feature = "cuda"))]
    fn handle_generate_step(
        &mut self,
        _model_id: &str,
        _tokens: Vec<u32>,
        _offset: usize,
    ) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "GenerateStep requires --features cuda".into(),
        }
    }

    /// Image-bearing prefill on this rank. Preprocesses each source data
    /// URI through the same deterministic `preprocess_data_uri` the
    /// leader runs, encodes through this rank's replicated tower, and
    /// splices + forwards. The logits are discarded (the leader samples
    /// from rank 0); the row-parallel `AllReduce`s are the point.
    #[cfg(feature = "cuda")]
    fn handle_generate_step_with_images(
        &mut self,
        model_id: &str,
        tokens: Vec<u32>,
        offset: usize,
        image_token_id: u32,
        image_data_uris: Vec<String>,
        chunk_size: usize,
    ) -> WorkerResponse {
        use crate::harness::preprocess::{PreprocessProfile, preprocess_data_uri};
        use candle_core::Tensor;

        if image_data_uris.is_empty() {
            return WorkerResponse::Error {
                kind: "bad_request".into(),
                message: "GenerateStepWithImages with zero images".into(),
            };
        }
        let Some(model) = self.models.get_mut(model_id) else {
            return WorkerResponse::Error {
                kind: "model_not_loaded".into(),
                message: format!("model '{model_id}' not loaded on rank {}", self.config.rank),
            };
        };
        let device = model.device().clone();

        // Preprocess each image identically to the leader so the encoded
        // embeddings — and thus the spliced hidden state — match across
        // ranks. Fixed 448×448 profile.
        let profile = PreprocessProfile::qwen3_6();
        let (h, w) = (
            profile.target_height as usize,
            profile.target_width as usize,
        );
        let mut pixels: Vec<Tensor> = Vec::with_capacity(image_data_uris.len());
        for (idx, uri) in image_data_uris.iter().enumerate() {
            let px = match preprocess_data_uri(uri, &profile) {
                Ok(p) => p,
                Err(e) => {
                    return WorkerResponse::Error {
                        kind: "bad_request".into(),
                        message: format!("preprocess image[{idx}]: {e:#}"),
                    };
                }
            };
            match Tensor::from_vec(px, (3, h, w), &device) {
                Ok(t) => pixels.push(t),
                Err(e) => {
                    return WorkerResponse::Error {
                        kind: "forward_failed".into(),
                        message: format!("build image[{idx}] tensor: {e}"),
                    };
                }
            }
        }

        let start = std::time::Instant::now();
        tracing::debug!(
            rank = self.config.rank,
            model = %model_id,
            tokens = tokens.len(),
            offset,
            images = pixels.len(),
            chunk_size,
            "worker GenerateStepWithImages: chunked prefill starting"
        );
        // Drop the logits — the leader samples from its own rank-0 copy.
        // The chunked prefill builds its own per-chunk input tensors.
        if let Err(e) =
            model.prefill_with_images_chunked(&tokens, offset, &pixels, image_token_id, chunk_size)
        {
            tracing::warn!(
                rank = self.config.rank,
                model = %model_id,
                elapsed_ms = start.elapsed().as_millis(),
                error = %e,
                "worker GenerateStepWithImages: forward failed"
            );
            return WorkerResponse::Error {
                kind: "forward_failed".into(),
                message: format!("TP image forward: {e}"),
            };
        }
        tracing::debug!(
            rank = self.config.rank,
            model = %model_id,
            elapsed_ms = start.elapsed().as_millis(),
            "worker GenerateStepWithImages: forward done"
        );
        WorkerResponse::GenerateStepOk
    }

    #[cfg(not(feature = "cuda"))]
    fn handle_generate_step_with_images(
        &mut self,
        _model_id: &str,
        _tokens: Vec<u32>,
        _offset: usize,
        _image_token_id: u32,
        _image_data_uris: Vec<String>,
        _chunk_size: usize,
    ) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "GenerateStepWithImages requires --features cuda".into(),
        }
    }

    #[cfg(feature = "cuda")]
    fn handle_clear_kv_cache(&mut self, model_id: &str) -> WorkerResponse {
        let Some(model) = self.models.get_mut(model_id) else {
            return WorkerResponse::Error {
                kind: "model_not_loaded".into(),
                message: format!("model '{model_id}' not loaded on rank {}", self.config.rank),
            };
        };
        model.clear_kv_cache();
        WorkerResponse::KvCacheCleared
    }

    #[cfg(not(feature = "cuda"))]
    fn handle_clear_kv_cache(&mut self, _model_id: &str) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "ClearKvCache requires --features cuda".into(),
        }
    }

    #[cfg(feature = "cuda")]
    fn handle_unload_model(&mut self, model_id: &str) -> WorkerResponse {
        if self.models.remove(model_id).is_none() {
            return WorkerResponse::Error {
                kind: "model_not_loaded".into(),
                message: format!("model '{model_id}' not loaded on rank {}", self.config.rank),
            };
        }
        tracing::info!(rank = self.config.rank, model = %model_id, "unloaded TP shard");
        WorkerResponse::Unloaded
    }

    #[cfg(not(feature = "cuda"))]
    fn handle_unload_model(&mut self, _model_id: &str) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "UnloadModel requires --features cuda".into(),
        }
    }
}

/// Parse a `ModelSpec.quant` string into a `GgmlDType`. Accepts the
/// common ggml format names (case-insensitive). `None` and `Some("")`
/// both map to "no quantization".
///
/// Supported: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`,
/// `q2k`/`q2_k`, `q3k`/`q3_k`, `q4k`/`q4_k`, `q5k`/`q5_k`,
/// `q6k`/`q6_k`, `q8k`/`q8_k`, `f16`, `bf16`, `f32`. The underscore
/// is optional and the prefix is case-insensitive.
#[cfg(feature = "cuda")]
pub(crate) fn parse_quant_string(
    s: Option<&str>,
) -> anyhow::Result<Option<candle_core::quantized::GgmlDType>> {
    use candle_core::quantized::GgmlDType;
    let s = match s {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(None),
    };
    let normalised = s.to_ascii_lowercase().replace('_', "");
    let dtype = match normalised.as_str() {
        "q40" => GgmlDType::Q4_0,
        "q41" => GgmlDType::Q4_1,
        "q50" => GgmlDType::Q5_0,
        "q51" => GgmlDType::Q5_1,
        "q80" => GgmlDType::Q8_0,
        "q81" => GgmlDType::Q8_1,
        "q2k" => GgmlDType::Q2K,
        "q3k" => GgmlDType::Q3K,
        "q4k" | "q4km" => GgmlDType::Q4K,
        "q5k" | "q5km" => GgmlDType::Q5K,
        "q6k" => GgmlDType::Q6K,
        "q8k" => GgmlDType::Q8K,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "f32" => GgmlDType::F32,
        other => anyhow::bail!(
            "unknown quant '{other}' (expected one of: q4_0, q4_1, q5_0, q5_1, q8_0, \
             q8_1, q2k, q3k, q4k, q5k, q6k, q8k, f16, bf16, f32)"
        ),
    };
    Ok(Some(dtype))
}
