//! Outbound calls to a benchmark target: build identity, host discovery,
//! and warm-model enumeration. Neuron targets use the native neuron API;
//! `openai` targets use the OpenAI-compatible surface (preliminary).

use crate::config::{TargetConfig, TargetKind};
use anyhow::{Context, Result, anyhow};
use cortex_core::build_info::BuildInfo;
use cortex_core::discovery::{DiscoveryResponse, HealthResponse};
use cortex_core::entitlements::{HEADER_ACCOUNT_ID, HEADER_KEY_ID};
use cortex_core::harness::{ModelInfo, ModelSpec};
use cortex_core::openai::ModelsResponse;
use std::time::Duration;

/// How long to wait on the cheap metadata polls (version/discovery/models).
const META_TIMEOUT: Duration = Duration::from_secs(10);

pub struct TargetClient {
    http: reqwest::Client,
}

impl TargetClient {
    pub fn new(request_timeout: Duration) -> Result<Self> {
        Self::with_principal(request_timeout, None)
    }

    /// Build a client that stamps `principal`'s identity headers on every
    /// request (#288).
    ///
    /// Set on the client rather than at each call site so no scenario can
    /// be added later that silently measures as anonymous — which is the
    /// failure this fixes, and it was invisible for a week.
    ///
    /// `None` keeps the historical anonymous behaviour, which is correct
    /// for `openai` targets (the headers mean nothing to a foreign
    /// engine) and for a fresh install with no principal configured.
    pub fn with_principal(
        request_timeout: Duration,
        principal: Option<&crate::config::PrincipalSettings>,
    ) -> Result<Self> {
        let mut builder = reqwest::Client::builder().timeout(request_timeout);
        if let Some(p) = principal {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                HEADER_ACCOUNT_ID,
                reqwest::header::HeaderValue::from_str(&p.account_id)
                    .context("principal.account_id is not a valid header value")?,
            );
            headers.insert(
                HEADER_KEY_ID,
                reqwest::header::HeaderValue::from_str(&p.key_id)
                    .context("principal.key_id is not a valid header value")?,
            );
            tracing::info!(
                account_id = %p.account_id,
                key_id = %p.key_id,
                "bench authenticates: measurements reflect identified-caller capacity (#288)"
            );
            builder = builder.default_headers(headers);
        } else {
            tracing::warn!(
                "bench has no [bench.principal]: requests are anonymous, so since #262 \
                 they measure the anonymous-yield policy rather than serving capacity (#288)"
            );
        }
        let http = builder.build().context("building HTTP client")?;
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

    /// Unload a model (neuron only): `POST /models/unload {model_id}`.
    /// Used by the deliberate swap-cost measurement (#90), never the sweep.
    pub async fn unload_model(&self, target: &TargetConfig, model_id: &str) -> Result<()> {
        let base = target.endpoint.trim_end_matches('/');
        self.http
            .post(format!("{base}/models/unload"))
            .json(&serde_json::json!({ "model_id": model_id }))
            .send()
            .await
            .context("POST /models/unload")?
            .error_for_status()
            .context("POST /models/unload status")?;
        Ok(())
    }

    /// Load a model from a spec (neuron only): `POST /models/load`. neuron
    /// returns synchronously once loaded, so the call duration is the reload
    /// cost the swap-cost measurement records (#90).
    pub async fn load_model(&self, target: &TargetConfig, spec: &ModelSpec) -> Result<()> {
        let base = target.endpoint.trim_end_matches('/');
        self.http
            .post(format!("{base}/models/load"))
            .json(spec)
            // A cold load can take tens of seconds; use the full request
            // timeout rather than the short metadata one.
            .send()
            .await
            .context("POST /models/load")?
            .error_for_status()
            .context("POST /models/load status")?;
        Ok(())
    }

    /// Reconstruct a reload [`ModelSpec`] from a model's `/models` entry.
    /// Tensor-parallel is inferred from the device count; `quant` is left
    /// `None` for neuron to resolve from the catalogue / its prior load.
    pub fn spec_from_info(info: &ModelInfo) -> Result<ModelSpec> {
        if info.devices.is_empty() {
            return Err(anyhow!(
                "model '{}' reports no devices; cannot reconstruct a load spec",
                info.id
            ));
        }
        Ok(ModelSpec {
            model_id: info.id.clone(),
            harness: info.harness.clone(),
            quant: None,
            tensor_parallel: (info.devices.len() > 1).then_some(info.devices.len() as u32),
            devices: Some(info.devices.clone()),
        })
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
                        // A generic OpenAI endpoint says nothing about
                        // serviceability, and absent means "no opinion".
                        servable: None,
                        reasoning_budget: Vec::new(),
                    })
                    .collect())
            }
        }
    }
}
