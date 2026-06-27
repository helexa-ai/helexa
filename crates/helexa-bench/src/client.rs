//! Outbound calls to a benchmark target: build identity, host discovery,
//! and warm-model enumeration. Neuron targets use the native neuron API;
//! `openai` targets use the OpenAI-compatible surface (preliminary).

use crate::config::{TargetConfig, TargetKind};
use anyhow::{Context, Result};
use cortex_core::build_info::BuildInfo;
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::harness::ModelInfo;
use cortex_core::openai::ModelsResponse;
use std::time::Duration;

/// How long to wait on the cheap metadata polls (version/discovery/models).
const META_TIMEOUT: Duration = Duration::from_secs(10);

pub struct TargetClient {
    http: reqwest::Client,
}

impl TargetClient {
    pub fn new(request_timeout: Duration) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(request_timeout)
            .build()
            .context("building HTTP client")?;
        Ok(TargetClient { http })
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Chat-completions URL for the target.
    pub fn chat_url(&self, target: &TargetConfig) -> String {
        let base = target.endpoint.trim_end_matches('/');
        match target.kind {
            // neuron exposes OpenAI routes under /v1.
            TargetKind::Neuron => format!("{base}/v1/chat/completions"),
            // openai endpoint is the /v1 base already (bench.py convention).
            TargetKind::Openai => format!("{base}/chat/completions"),
        }
    }

    /// Build identity. Neuron: `GET /version`. Openai: a synthetic
    /// placeholder keyed by `"external"` so the version-aware skip logic
    /// treats it as one stable build (comparison runs are manual anyway).
    pub async fn fetch_version(&self, target: &TargetConfig) -> Result<BuildInfo> {
        match target.kind {
            TargetKind::Neuron => {
                let base = target.endpoint.trim_end_matches('/');
                let info = self
                    .http
                    .get(format!("{base}/version"))
                    .timeout(META_TIMEOUT)
                    .send()
                    .await
                    .context("GET /version")?
                    .error_for_status()
                    .context("GET /version status")?
                    .json::<BuildInfo>()
                    .await
                    .context("decoding /version")?;
                Ok(info)
            }
            TargetKind::Openai => {
                let mut info = BuildInfo::unknown();
                info.git_sha = "external".to_string();
                Ok(info)
            }
        }
    }

    /// Host discovery (neuron only).
    pub async fn fetch_discovery(
        &self,
        target: &TargetConfig,
    ) -> Result<Option<DiscoveryResponse>> {
        if target.kind != TargetKind::Neuron {
            return Ok(None);
        }
        let base = target.endpoint.trim_end_matches('/');
        let disco = self
            .http
            .get(format!("{base}/discovery"))
            .timeout(META_TIMEOUT)
            .send()
            .await
            .context("GET /discovery")?
            .error_for_status()
            .context("GET /discovery status")?
            .json::<DiscoveryResponse>()
            .await
            .context("decoding /discovery")?;
        Ok(Some(disco))
    }

    /// Runtime device health (neuron only): per-GPU VRAM used/free,
    /// utilization, and temperature from `GET /health`. Bench samples this
    /// around each measured run to record VRAM high-water + GPU telemetry
    /// (#87). Returns `Ok(None)` for non-neuron targets; a soft `Ok(None)`
    /// (not an error) on transport failure so a flaky `/health` never fails
    /// a measurement.
    pub async fn fetch_health(&self, target: &TargetConfig) -> Result<Option<HealthResponse>> {
        if target.kind != TargetKind::Neuron {
            return Ok(None);
        }
        let base = target.endpoint.trim_end_matches('/');
        let health = self
            .http
            .get(format!("{base}/health"))
            .timeout(META_TIMEOUT)
            .send()
            .await
            .context("GET /health")?
            .error_for_status()
            .context("GET /health status")?
            .json::<HealthResponse>()
            .await
            .context("decoding /health")?;
        Ok(Some(health))
    }

    /// Warm models — those ready to serve without a cold load.
    ///
    /// Neuron: `GET /models` filtered to `status == "loaded"` (skips
    /// `recovering`/`poisoned`). Openai: `GET /models`, honouring the
    /// helexa `loaded` extension when present, else treating all listed
    /// models as warm.
    pub async fn warm_models(&self, target: &TargetConfig) -> Result<Vec<ModelInfo>> {
        let base = target.endpoint.trim_end_matches('/');
        match target.kind {
            TargetKind::Neuron => {
                let models = self
                    .http
                    .get(format!("{base}/models"))
                    .timeout(META_TIMEOUT)
                    .send()
                    .await
                    .context("GET /models")?
                    .error_for_status()
                    .context("GET /models status")?
                    .json::<Vec<ModelInfo>>()
                    .await
                    .context("decoding /models")?;
                Ok(models
                    .into_iter()
                    .filter(|m| m.status == "loaded")
                    .collect())
            }
            TargetKind::Openai => {
                let resp = self
                    .http
                    .get(format!("{base}/models"))
                    .timeout(META_TIMEOUT)
                    .send()
                    .await
                    .context("GET /models")?
                    .error_for_status()
                    .context("GET /models status")?
                    .json::<ModelsResponse>()
                    .await
                    .context("decoding /models")?;
                Ok(resp
                    .data
                    .into_iter()
                    .filter(|m| {
                        // honour the helexa `loaded` extension if present
                        m.extra
                            .get("loaded")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true)
                    })
                    .map(|m| ModelInfo {
                        id: m.id,
                        harness: "openai".to_string(),
                        status: "loaded".to_string(),
                        devices: Vec::new(),
                        vram_used_mb: None,
                        capabilities: Vec::new(),
                        limit: None,
                        cost: None,
                        tool_call: false,
                        reasoning: false,
                    })
                    .collect())
            }
        }
    }
}
