//! mistral.rs harness implementation.
//!
//! Wraps the mistral.rs HTTP API for model lifecycle management
//! and optionally manages the process via systemd.

use anyhow::Result;
use async_trait::async_trait;
use cortex_core::harness::{Harness, HarnessConfig, HarnessHealth, ModelInfo, ModelSpec};
use reqwest::Client;
use serde::Deserialize;

pub struct MistralRsHarness {
    endpoint: String,
    systemd_unit: Option<String>,
    client: Client,
}

impl MistralRsHarness {
    pub fn new(endpoint: String, systemd_unit: Option<String>) -> Self {
        Self {
            endpoint,
            systemd_unit,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }
}

/// Response from mistral.rs `GET /v1/models`.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    status: Option<String>,
}

#[async_trait]
impl Harness for MistralRsHarness {
    fn name(&self) -> &str {
        "mistralrs"
    }

    async fn start(&self, _config: &HarnessConfig) -> Result<()> {
        let Some(unit) = &self.systemd_unit else {
            anyhow::bail!("no systemd unit configured for mistralrs harness");
        };

        let output = tokio::process::Command::new("systemctl")
            .args(["start", unit])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("systemctl start {unit} failed: {stderr}");
        }

        // Wait for the health endpoint to respond (up to 30s).
        let url = format!("{}/health", self.endpoint);
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if self.client.get(&url).send().await.is_ok() {
                tracing::info!(unit, "mistralrs started and healthy");
                return Ok(());
            }
        }
        anyhow::bail!("mistralrs started but health endpoint did not respond within 30s");
    }

    async fn stop(&self) -> Result<()> {
        let Some(unit) = &self.systemd_unit else {
            anyhow::bail!("no systemd unit configured for mistralrs harness");
        };

        let output = tokio::process::Command::new("systemctl")
            .args(["stop", unit])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("systemctl stop {unit} failed: {stderr}");
        }
        Ok(())
    }

    async fn health(&self) -> HarnessHealth {
        let url = format!("{}/health", self.endpoint);
        let running = self.client.get(&url).send().await.is_ok();
        HarnessHealth {
            name: "mistralrs".into(),
            running,
            uptime_secs: None,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.endpoint);
        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("GET /v1/models returned {}", resp.status());
        }

        let models_resp: ModelsResponse = resp.json().await?;
        Ok(models_resp
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                harness: "mistralrs".into(),
                status: m.status.unwrap_or_else(|| "loaded".into()),
                devices: vec![],
                vram_used_mb: None,
            })
            .collect())
    }

    async fn load_model(&self, spec: &ModelSpec) -> Result<()> {
        let url = format!("{}/v1/models/reload", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model_id": spec.model_id }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("POST /v1/models/reload failed: {body}");
        }
        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let url = format!("{}/v1/models/unload", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model_id": model_id }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("POST /v1/models/unload failed: {body}");
        }
        Ok(())
    }

    async fn inference_endpoint(&self, _model_id: &str) -> Option<String> {
        // mistral.rs routes internally by model name in the request body,
        // so the inference endpoint is always the base URL.
        Some(self.endpoint.clone())
    }
}
