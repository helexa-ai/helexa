//! A loaded Laya decision model, as the rest of neuron sees it (#336).
//!
//! The weights live in a device worker's decision slab (CUDA) or in an
//! `Arc` scored on the blocking pool (CPU); either way the caller holds
//! only this struct and asks it to [`DecisionCheckpoint::decide`] a
//! batch of already-assembled questions. Turning a `/v1/systemone`
//! request into that batch — rendering options, budgets, tokenisation —
//! and turning the scores back into calibrated answers happen outside
//! this module.
//!
//! One Hugging Face repo can carry several checkpoints (the released
//! `convaiinnovations/laya` has the English one at its root and
//! `multilingual/` and `typed-decisions/` beside it), so a loaded model
//! is a list of checkpoints, each named by its subfolder.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use candle_core::DType;
use tokenizers::Tokenizer;

use super::arch::laya::{CheckpointConfig, DecisionBatch, DecisionRow, LayaModel};
use super::device_worker::{DecisionHandle, DeviceWorkerHandle};
use cortex_core::harness::ModelSpec;

/// Name of the checkpoint at the repo root.
pub const ROOT_CHECKPOINT: &str = "root";

/// Where one checkpoint's weights live and how to reach them.
enum Backend {
    /// Owned by a CUDA device worker; scored through its channel.
    Worker {
        worker: Arc<DeviceWorkerHandle>,
        handle: DecisionHandle,
    },
    /// CPU fallback. `LayaModel::forward` takes `&self`, so concurrent
    /// requests share it without a lock.
    Cpu(Arc<LayaModel>),
}

/// The files one checkpoint needs, already on local disk.
#[derive(Debug, Clone)]
pub struct CheckpointFiles {
    /// The checkpoint directory: holds `rl_agent_config.json`,
    /// `encoder/config.json` and `model.safetensors`.
    pub dir: PathBuf,
    pub tokenizer: PathBuf,
}

pub struct DecisionCheckpoint {
    /// [`ROOT_CHECKPOINT`] or the subfolder name.
    pub name: String,
    pub config: CheckpointConfig,
    pub tokenizer: Tokenizer,
    /// Precision the weights were loaded in.
    pub dtype: DType,
    backend: Backend,
}

impl DecisionCheckpoint {
    /// Load one checkpoint onto `worker`'s device, or onto the CPU when
    /// `worker` is `None`.
    pub async fn load(
        name: String,
        files: CheckpointFiles,
        model_id: &str,
        worker: Option<Arc<DeviceWorkerHandle>>,
    ) -> Result<Self> {
        let config = CheckpointConfig::from_dir(&files.dir)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", files.tokenizer.display()))?;
        let (dtype, backend) = match worker {
            Some(worker) => {
                let wanted = device_dtype(config.laya.amp_dtype.as_deref());
                let (handle, dtype) = worker
                    .load_decision(files.dir.clone(), model_id.to_string(), wanted)
                    .await
                    .map_err(|e| anyhow::anyhow!("worker load_decision: {e:#}"))?;
                (dtype, Backend::Worker { worker, handle })
            }
            None => {
                let dir = files.dir.clone();
                let model = tokio::task::spawn_blocking(move || {
                    LayaModel::load(&dir, &candle_core::Device::Cpu, DType::F32)
                })
                .await
                .context("decision load task")??;
                (DType::F32, Backend::Cpu(Arc::new(model)))
            }
        };
        Ok(Self {
            name,
            config,
            tokenizer,
            dtype,
            backend,
        })
    }

    /// Score one batch: one [`DecisionRow`] per sequence, in order.
    pub async fn decide(&self, batch: DecisionBatch) -> Result<Vec<DecisionRow>> {
        match &self.backend {
            Backend::Worker { worker, handle } => worker
                .decide(*handle, batch)
                .await
                .map_err(|e| anyhow::anyhow!("{e:#}")),
            Backend::Cpu(model) => {
                let model = Arc::clone(model);
                tokio::task::spawn_blocking(move || model.forward(&batch))
                    .await
                    .context("decision forward task")?
            }
        }
    }

    /// Release the weights. On a worker this drops them on the thread
    /// that owns the device context.
    pub async fn release(&self) {
        if let Backend::Worker { worker, handle } = &self.backend
            && let Err(e) = worker.drop_decision(*handle).await
        {
            tracing::warn!(
                checkpoint = %self.name,
                error = %e,
                "decision unload: DropDecision RPC failed (weights may leak in worker slab)"
            );
        }
    }
}

/// The precision to serve in on a GPU: the autocast dtype the reference
/// itself runs there (`amp_dtype` in `rl_agent_config.json`), else bf16.
fn device_dtype(amp_dtype: Option<&str>) -> DType {
    match amp_dtype {
        Some("fp16") | Some("float16") => DType::F16,
        Some("fp32") | Some("float32") => DType::F32,
        _ => DType::BF16,
    }
}

pub struct LoadedDecisionModel {
    pub model_id: String,
    pub spec: ModelSpec,
    pub devices: Vec<u32>,
    /// At least one; the first is the default.
    pub checkpoints: Vec<DecisionCheckpoint>,
    /// Set when a forward returns a driver-level error, mirroring the
    /// other model kinds: `/models` reports it and requests fast-reject.
    pub poisoned: Arc<AtomicBool>,
    pub admission: super::admission::AdmissionController,
}

impl LoadedDecisionModel {
    /// The checkpoint called `name`, or the default one for `None`.
    pub fn checkpoint(&self, name: Option<&str>) -> Option<&DecisionCheckpoint> {
        match name {
            None => self.checkpoints.first(),
            Some(n) => self.checkpoints.iter().find(|c| c.name == n),
        }
    }

    pub async fn release(&self) {
        for c in &self.checkpoints {
            c.release().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> CheckpointFiles {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/harness/testdata/laya/tiny");
        CheckpointFiles {
            tokenizer: dir.join("tokenizer.json"),
            dir,
        }
    }

    #[test]
    fn gpu_dtype_follows_the_reference_autocast() {
        assert_eq!(device_dtype(Some("bf16")), DType::BF16);
        assert_eq!(device_dtype(Some("fp16")), DType::F16);
        assert_eq!(device_dtype(None), DType::BF16);
    }

    /// The whole async path on the CPU backend, and the device-worker
    /// path on a CPU build (the worker runs, with a CPU device), must
    /// give the same scores as the model called directly.
    #[tokio::test]
    async fn both_backends_score_like_the_model() {
        let files = tiny();
        let direct = LayaModel::load(&files.dir, &candle_core::Device::Cpu, DType::F32).unwrap();
        let batch = DecisionBatch {
            seqs: vec![super::super::arch::laya::DecisionSeq {
                input_ids: vec![5, 9, 11, 3, 7, 2],
                markers: vec![1, 3],
                qtype: 0,
            }],
        };
        let want = direct.forward(&batch).unwrap();

        let cpu = DecisionCheckpoint::load("root".into(), files.clone(), "tiny", None)
            .await
            .unwrap();
        assert_eq!(cpu.decide(batch.clone()).await.unwrap(), want);

        // A CPU build's worker holds a CPU device, so this exercises the
        // job round-trip (load, decide, drop) without a GPU — and the
        // worker must fall back to f32 there, reporting that it did.
        let worker = DeviceWorkerHandle::spawn(0).unwrap();
        let on_worker =
            DecisionCheckpoint::load("root".into(), files, "tiny", Some(worker.clone()))
                .await
                .unwrap();
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(on_worker.dtype, DType::F32);
            assert_eq!(on_worker.decide(batch).await.unwrap(), want);
        }
        #[cfg(feature = "cuda")]
        {
            let got = on_worker.decide(batch).await.unwrap();
            for (g, w) in got[0].logits.iter().zip(&want[0].logits) {
                assert!((g - w).abs() < 5e-2, "{g} vs {w}");
            }
        }
        on_worker.release().await;
        assert!(on_worker.decide(DecisionBatch::default()).await.is_err());
        worker.shutdown().unwrap();
    }
}
