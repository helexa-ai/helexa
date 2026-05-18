//! Candle harness — in-process inference using huggingface/candle.
//!
//! This is the sole `Harness` implementation. Unlike the previous
//! mistralrs/llamacpp harnesses, candle inference runs inside the neuron
//! process itself — no external subprocess, no systemd indirection.
//!
//! Stage 1 ships this as an inert skeleton; Stage 2 wires up actual
//! model load/unload via `candle-transformers`.

use anyhow::Result;
use async_trait::async_trait;
use cortex_core::harness::{Harness, HarnessHealth, ModelInfo, ModelSpec};

pub struct CandleHarness {
    /// URL where this neuron serves inference (its own bind address).
    bind_url: String,
}

impl CandleHarness {
    pub fn new(bind_url: String) -> Self {
        Self { bind_url }
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
        Ok(Vec::new())
    }

    async fn load_model(&self, _spec: &ModelSpec) -> Result<()> {
        anyhow::bail!("candle harness load_model not implemented yet (Stage 2)")
    }

    async fn unload_model(&self, _model_id: &str) -> Result<()> {
        anyhow::bail!("candle harness unload_model not implemented yet (Stage 2)")
    }

    async fn inference_endpoint(&self, _model_id: &str) -> Option<String> {
        Some(self.bind_url.clone())
    }
}
