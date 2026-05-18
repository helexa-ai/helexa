//! Candle harness — in-process inference using huggingface/candle.
//!
//! This is the sole `Harness` implementation. Inference runs inside
//! the neuron process; there is no external subprocess. Stage 2 wires
//! up GGUF (currently Qwen3 only) model load/unload via
//! `candle-transformers::models::quantized_qwen3`. Stage 3 adds the
//! inference endpoint.

use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Weights;
use cortex_core::harness::{Harness, HarnessHealth, ModelInfo, ModelSpec};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, RwLock};

/// In-process candle harness. Owns the loaded model registry.
pub struct CandleHarness {
    models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    hf_cache: Option<PathBuf>,
    bind_url: String,
}

/// A loaded model with its tokenizer, device placement, and architecture-
/// specific weights. The `arch` field is mutexed because future inference
/// calls take `&mut self` on the underlying ModelWeights (KV cache state).
pub struct LoadedModel {
    pub model_id: String,
    pub arch: Mutex<ModelArch>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub quant: Option<String>,
    pub devices: Vec<u32>,
}

/// Architecture-specific weights. Stage 2 supports only Qwen3 quantized;
/// Stage 8 broadens this to additional families and non-quantized variants.
pub enum ModelArch {
    Qwen3Quantized(QuantizedQwen3Weights),
}

impl CandleHarness {
    pub fn new(bind_url: String, hf_cache: Option<PathBuf>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            hf_cache,
            bind_url,
        }
    }

    /// Pick a candle `Device` for the requested indices. Without the
    /// `cuda` feature, or if CUDA initialisation fails, falls back to CPU.
    fn pick_device(devices: &[u32]) -> Result<Device> {
        let _idx = devices.first().copied().unwrap_or(0) as usize;
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(_idx) {
                Ok(d) => return Ok(d),
                Err(e) => tracing::warn!(
                    device = _idx,
                    error = %e,
                    "CUDA device unavailable, falling back to CPU"
                ),
            }
        }
        Ok(Device::Cpu)
    }

    /// Resolve a model spec to local GGUF and tokenizer file paths via
    /// hf-hub. Downloads on first use; subsequent calls are cached.
    async fn resolve_files(&self, spec: &ModelSpec) -> Result<(PathBuf, PathBuf)> {
        let mut builder = hf_hub::api::tokio::ApiBuilder::new();
        if let Some(cache) = &self.hf_cache {
            builder = builder.with_cache_dir(cache.clone());
        }
        let api = builder.build().context("build hf-hub API")?;
        let repo = api.model(spec.model_id.clone());

        let info = repo
            .info()
            .await
            .with_context(|| format!("fetch HF repo info for {}", spec.model_id))?;

        let quant = spec.quant.as_deref().unwrap_or("");
        let quant_lc = quant.to_lowercase();
        let gguf_filename = info
            .siblings
            .iter()
            .map(|s| s.rfilename.as_str())
            .filter(|name| name.to_lowercase().ends_with(".gguf"))
            .find(|name| quant_lc.is_empty() || name.to_lowercase().contains(&quant_lc))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no GGUF file matching quant {:?} in repo {}",
                    spec.quant,
                    spec.model_id
                )
            })?
            .to_string();

        tracing::info!(
            model = %spec.model_id,
            file = %gguf_filename,
            "resolving GGUF (may be cached)"
        );
        let gguf_path = repo
            .get(&gguf_filename)
            .await
            .with_context(|| format!("fetch GGUF {gguf_filename}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .context("fetch tokenizer.json")?;
        Ok((gguf_path, tokenizer_path))
    }
}

#[async_trait]
impl Harness for CandleHarness {
    fn name(&self) -> &str {
        "candle"
    }

    async fn health(&self) -> HarnessHealth {
        HarnessHealth {
            name: "candle".into(),
            running: true,
            uptime_secs: None,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let models = self.models.read().await;
        Ok(models
            .values()
            .map(|m| ModelInfo {
                id: m.model_id.clone(),
                harness: "candle".into(),
                status: "loaded".into(),
                devices: m.devices.clone(),
                vram_used_mb: None,
            })
            .collect())
    }

    async fn load_model(&self, spec: &ModelSpec) -> Result<()> {
        if spec.harness != "candle" {
            anyhow::bail!("expected harness=candle, got harness={}", spec.harness);
        }

        {
            let models = self.models.read().await;
            if models.contains_key(&spec.model_id) {
                anyhow::bail!("model '{}' already loaded", spec.model_id);
            }
        }

        let devices = spec.devices.clone().unwrap_or_else(|| vec![0]);
        let device = Self::pick_device(&devices)?;

        let (gguf_path, tokenizer_path) = self.resolve_files(spec).await?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        // File I/O + GGUF parsing + tensor materialisation are CPU-bound,
        // so run them on a blocking task to avoid stalling the runtime.
        let device_for_load = device.clone();
        let gguf_path_for_load = gguf_path.clone();
        let model_id_for_log = spec.model_id.clone();
        let arch = tokio::task::spawn_blocking(move || -> Result<ModelArch> {
            tracing::info!(model = %model_id_for_log, path = ?gguf_path_for_load, "loading GGUF");
            let mut file = std::fs::File::open(&gguf_path_for_load).context("open GGUF file")?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|e| anyhow::anyhow!("parse GGUF: {e}"))?;

            let architecture = content
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_default();
            tracing::info!(architecture = %architecture, "GGUF architecture");

            match architecture.as_str() {
                "qwen3" => {
                    let weights =
                        QuantizedQwen3Weights::from_gguf(content, &mut file, &device_for_load)
                            .map_err(|e| anyhow::anyhow!("from_gguf qwen3: {e}"))?;
                    Ok(ModelArch::Qwen3Quantized(weights))
                }
                other => anyhow::bail!(
                    "unsupported GGUF architecture '{other}'; Stage 2 only supports qwen3"
                ),
            }
        })
        .await
        .context("blocking load task panicked")??;

        let loaded = Arc::new(LoadedModel {
            model_id: spec.model_id.clone(),
            arch: Mutex::new(arch),
            tokenizer,
            device,
            quant: spec.quant.clone(),
            devices,
        });

        let mut models = self.models.write().await;
        models.insert(spec.model_id.clone(), loaded);
        tracing::info!(model = %spec.model_id, "model loaded");
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        if models.remove(model_id).is_none() {
            anyhow::bail!("model '{model_id}' not loaded");
        }
        tracing::info!(model = %model_id, "model unloaded");
        Ok(())
    }

    async fn inference_endpoint(&self, model_id: &str) -> Option<String> {
        let models = self.models.read().await;
        models.contains_key(model_id).then(|| self.bind_url.clone())
    }
}
