//! Model-to-node routing logic.
//!
//! Given a model ID from an inbound request, determine which node should
//! handle it. Priority:
//!   1. Node where the model is currently `Loaded` → use it.
//!   2. Node where the model is `Unloaded` → use it; neuron's existing
//!      lazy-load behaviour will reload before serving the request.
//!   3. Model is in the catalogue → pick a feasible neuron, call
//!      `POST /models/load`, wait for the load to complete, then
//!      proxy. First-request cold-load latency is acceptable per the
//!      unified-endpoint contract.
//!   4. Not in catalogue, not loaded anywhere → 404.

use crate::state::CortexState;
use cortex_core::catalogue::ModelProfile;
use cortex_core::harness::ModelSpec;
use cortex_core::node::ModelStatus;
use std::sync::Arc;
use std::time::Duration;

/// The routing decision: which node endpoint to proxy the request to.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub node_name: String,
    /// The inference endpoint to proxy to (from neuron's /models/{id}/endpoint).
    pub endpoint: String,
    /// Whether the model will need to load (cold start). Set to true
    /// when we proxied to an `Unloaded` node (lazy load on neuron) or
    /// when we just triggered an explicit cold-load via the catalogue
    /// path.
    pub cold_start: bool,
    /// The concrete model id we actually routed to. Equal to the
    /// caller's requested id unless an alias was resolved (e.g. caller
    /// asked for `helexa/small`, this carries `Qwen/Qwen3-1.7B`). The
    /// handler uses this to rewrite the request body's `model` field
    /// before proxying — neurons reject requests where the body's
    /// model name doesn't match a loaded model.
    pub resolved_model_id: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("model '{0}' not found on any node and not in catalogue")]
    ModelNotFound(String),
    #[error("no healthy nodes available")]
    NoHealthyNodes,
    #[error("failed to resolve inference endpoint for model '{0}' on node '{1}'")]
    EndpointResolveFailed(String, String),
    #[error(
        "model '{model_id}' is in the catalogue but no healthy neuron's topology satisfies its constraints"
    )]
    NoFeasibleNeuron { model_id: String },
    #[error("cold-load of '{model_id}' on '{node}' failed: {message}")]
    ColdLoadFailed {
        model_id: String,
        node: String,
        message: String,
    },
    #[error(
        "model '{model_id}' is recovering on node '{node}' (device context rebuild in progress) — retry shortly"
    )]
    ModelRecovering { model_id: String, node: String },
}

impl RouteError {
    /// HTTP status the gateway should answer with. `NoHealthyNodes` and
    /// `ModelRecovering` are the transient cases (503 service_unavailable,
    /// safe to retry the same request); everything else is 404.
    pub fn http_status(&self) -> u16 {
        match self {
            RouteError::NoHealthyNodes | RouteError::ModelRecovering { .. } => 503,
            _ => 404,
        }
    }

    /// Broad OpenAI error category for the JSON envelope.
    pub fn broad_type(&self) -> &'static str {
        match self {
            RouteError::ModelNotFound(_) => "invalid_request_error",
            RouteError::NoHealthyNodes
            | RouteError::EndpointResolveFailed(_, _)
            | RouteError::NoFeasibleNeuron { .. }
            | RouteError::ColdLoadFailed { .. }
            | RouteError::ModelRecovering { .. } => "api_error",
        }
    }

    /// Specific machine-readable error code.
    pub fn code(&self) -> &'static str {
        match self {
            RouteError::ModelNotFound(_) => "model_not_found",
            RouteError::NoHealthyNodes => "service_unavailable",
            RouteError::EndpointResolveFailed(_, _) => "service_unavailable",
            RouteError::NoFeasibleNeuron { .. } => "service_unavailable",
            RouteError::ColdLoadFailed { .. } => "service_unavailable",
            RouteError::ModelRecovering { .. } => "service_unavailable",
        }
    }
}

/// Resolve which node should serve a request for the given model.
/// Asks the neuron for the inference endpoint after selecting a node.
pub async fn resolve(
    fleet: &Arc<CortexState>,
    requested_model_id: &str,
) -> Result<RouteDecision, RouteError> {
    // Alias resolution first — swap `helexa/small` (etc.) for the
    // concrete id before any node lookups so the rest of routing,
    // loading, and metrics deal in concrete ids only. `resolve_alias`
    // returns the input verbatim when it isn't an alias.
    let model_id = fleet.catalogue.resolve_alias(requested_model_id);
    if model_id != requested_model_id {
        tracing::debug!(
            requested = requested_model_id,
            resolved = model_id,
            "alias resolved"
        );
    }
    // Snapshot loaded / unloaded / recovering state from the poller cache.
    let (loaded_route, unloaded_route, recovering_node, any_healthy) = {
        let nodes = fleet.nodes.read().await;
        let mut loaded_route = None;
        let mut unloaded_route = None;
        let mut recovering_node = None;
        let mut any_healthy = false;
        for node in nodes.values() {
            if !node.healthy {
                continue;
            }
            any_healthy = true;
            if let Some(entry) = node.models.get(model_id) {
                match entry.status {
                    ModelStatus::Loaded | ModelStatus::Reloading => {
                        loaded_route = Some((node.name.clone(), node.endpoint.clone(), false));
                        break;
                    }
                    ModelStatus::Unloaded => {
                        if unloaded_route.is_none() {
                            unloaded_route = Some((node.name.clone(), node.endpoint.clone(), true));
                        }
                    }
                    // Auto-recovering (#17/#20): the model is rebuilding
                    // its device context on this node. Hold the route —
                    // answer "retry shortly" rather than 404, and do NOT
                    // fall through to the catalogue cold-load, which
                    // would race a second placement (and a second copy's
                    // worth of VRAM) against the in-flight recovery.
                    ModelStatus::Recovering => {
                        if recovering_node.is_none() {
                            recovering_node = Some(node.name.clone());
                        }
                    }
                    // Loading is gateway-synthesised from neuron's
                    // activation snapshot; it never appears on the
                    // wire from neuron's `/models`. Skip — the model
                    // isn't actually servable yet. The pre-existing
                    // race (catalogue cold_load fires a parallel
                    // /models/load against the in-flight load) is no
                    // worse than before; fixing it needs neuron-side
                    // in-flight tracking on /models/load itself.
                    ModelStatus::Loading => {}
                }
            }
        }
        (loaded_route, unloaded_route, recovering_node, any_healthy)
    };

    if !any_healthy {
        return Err(RouteError::NoHealthyNodes);
    }

    // Priority 1: already loaded.
    if let Some((node_name, neuron_endpoint, cold_start)) = loaded_route {
        return finish(fleet, &node_name, &neuron_endpoint, model_id, cold_start).await;
    }

    // Priority 2: recovering somewhere — transient hold, not a reroute.
    if let Some(node) = recovering_node {
        return Err(RouteError::ModelRecovering {
            model_id: model_id.to_string(),
            node,
        });
    }

    // Priority 3: known to neuron but unloaded (neuron's lazy load).
    if let Some((node_name, neuron_endpoint, cold_start)) = unloaded_route {
        return finish(fleet, &node_name, &neuron_endpoint, model_id, cold_start).await;
    }

    // Priority 4: catalogue × topology cold-load.
    if let Some(profile) = fleet.catalogue.get(model_id) {
        let (node_name, neuron_endpoint) = pick_feasible_neuron(fleet, profile).await?;
        cold_load(fleet, &node_name, &neuron_endpoint, profile).await?;
        return finish(fleet, &node_name, &neuron_endpoint, model_id, true).await;
    }

    Err(RouteError::ModelNotFound(model_id.to_string()))
}

/// Pick a healthy neuron whose discovered topology satisfies the
/// profile. Preference order:
///   1. A neuron from `profile.pinned_on` that is healthy + feasible.
///   2. Otherwise, any healthy + feasible neuron, stable by name.
async fn pick_feasible_neuron(
    fleet: &Arc<CortexState>,
    profile: &ModelProfile,
) -> Result<(String, String), RouteError> {
    let nodes = fleet.nodes.read().await;
    let mut candidates: Vec<(String, String, bool)> = Vec::new();
    for node in nodes.values() {
        if !node.healthy {
            continue;
        }
        let Some(disc) = node.discovery.as_ref() else {
            continue;
        };
        if !profile.is_feasible_on(&node.name, &disc.devices) {
            continue;
        }
        let pinned = profile.pinned_on.iter().any(|n| n == &node.name);
        candidates.push((node.name.clone(), node.endpoint.clone(), pinned));
    }
    candidates.sort_by(|a, b| {
        b.2.cmp(&a.2) // pinned first (true > false)
            .then(a.0.cmp(&b.0))
    });
    let pick = candidates.into_iter().next();
    pick.map(|(n, e, _)| (n, e))
        .ok_or_else(|| RouteError::NoFeasibleNeuron {
            model_id: profile.id.clone(),
        })
}

/// Issue `POST {endpoint}/models/load` for this profile on this neuron,
/// blocking until the load completes (neuron's load endpoint is
/// synchronous — it returns 200 once VRAM is materialised). On success
/// also inserts a `Loaded` entry into the local NodeState cache so the
/// caller's subsequent endpoint lookup sees the new model without
/// waiting for the next poll cycle.
async fn cold_load(
    fleet: &Arc<CortexState>,
    node_name: &str,
    neuron_endpoint: &str,
    profile: &ModelProfile,
) -> Result<(), RouteError> {
    let spec = profile_to_spec(fleet, node_name, profile).await;
    let url = format!("{neuron_endpoint}/models/load");
    tracing::info!(model = %profile.id, node = node_name, "cold-loading via /models/load");

    // Generous timeout: a fresh download + safetensors mmap + device
    // copy for a 30B-class dense model can comfortably exceed 5 min on
    // a slow link. The HTTP client's own default already covers most
    // of this; pin a longer per-request bound just here.
    let resp = match fleet
        .http_client
        .post(&url)
        .timeout(Duration::from_secs(1800))
        .json(&spec)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return Err(RouteError::ColdLoadFailed {
                model_id: profile.id.clone(),
                node: node_name.to_string(),
                message: format!("HTTP request failed: {e}"),
            });
        }
    };

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        // Neuron returns 400 "already loaded" when two concurrent
        // requests race the same model. Treat that as success — both
        // requests effectively achieved the same end state.
        if body.contains("already loaded") {
            tracing::info!(
                model = %profile.id,
                node = node_name,
                "cold-load saw 'already loaded' — treating as success"
            );
        } else {
            return Err(RouteError::ColdLoadFailed {
                model_id: profile.id.clone(),
                node: node_name.to_string(),
                message: format!("HTTP {status}: {body}"),
            });
        }
    } else {
        tracing::info!(model = %profile.id, node = node_name, "cold-load returned 200");
    }

    // Warm the cache: insert a Loaded ModelEntry so the next
    // resolve() finds the model without waiting for the poll loop.
    {
        let mut nodes = fleet.nodes.write().await;
        if let Some(node) = nodes.get_mut(node_name) {
            node.models.insert(
                profile.id.clone(),
                cortex_core::node::ModelEntry {
                    id: profile.id.clone(),
                    status: ModelStatus::Loaded,
                    last_accessed: Some(chrono::Utc::now()),
                    vram_estimate_mb: profile.vram_mb,
                    capabilities: Vec::new(),
                    tool_call: false,
                    reasoning: false,
                },
            );
        }
    }
    Ok(())
}

/// Translate a `ModelProfile` to a `ModelSpec` neuron's /models/load
/// accepts. Devices are picked from the neuron's discovered topology —
/// the first `min_devices` indices that meet `min_device_vram_mb`.
async fn profile_to_spec(
    fleet: &Arc<CortexState>,
    node_name: &str,
    profile: &ModelProfile,
) -> ModelSpec {
    let devices = {
        let nodes = fleet.nodes.read().await;
        let mut picked: Vec<u32> = Vec::new();
        if let Some(node) = nodes.get(node_name)
            && let Some(disc) = &node.discovery
        {
            let min_vram = profile.min_device_vram_mb.unwrap_or(0);
            for d in &disc.devices {
                if d.vram_total_mb >= min_vram {
                    picked.push(d.index);
                    if picked.len() as u32 >= profile.min_devices {
                        break;
                    }
                }
            }
        }
        if picked.is_empty() {
            // Fall back to a 0..min_devices default; pick_feasible_neuron
            // already verified the topology satisfies the constraints,
            // so this only fires if discovery raced or was lost.
            (0..profile.min_devices).collect()
        } else {
            picked
        }
    };

    let tensor_parallel = if profile.min_devices > 1 {
        Some(profile.min_devices)
    } else {
        None
    };

    ModelSpec {
        model_id: qualified_model_id(profile),
        harness: profile.harness.clone(),
        quant: profile.quant.clone(),
        tensor_parallel,
        devices: Some(devices),
    }
}

/// Prefix the catalogue id with the scheme when one is declared, so
/// neuron resolves the load against the right registry. Without this,
/// a profile pointing at the helexa registry would resolve via
/// neuron's `default_source` (typically `huggingface`) and fetch
/// bytes from the wrong place. Profiles that omit `source` continue
/// to pass the bare id through, preserving the pre-Phase-3 contract.
///
/// Stays at module scope (not nested in `profile_to_spec`) so the unit
/// tests can exercise it without spinning up CortexState topology.
fn qualified_model_id(profile: &ModelProfile) -> String {
    match profile.source.as_deref() {
        Some(scheme) if !scheme.is_empty() => format!("{scheme}:{}", profile.id),
        _ => profile.id.clone(),
    }
}

/// Resolve neuron's `/models/{id}/endpoint` to its inference URL and
/// build the final `RouteDecision`. Shared by all three priority
/// branches above.
async fn finish(
    fleet: &Arc<CortexState>,
    node_name: &str,
    neuron_endpoint: &str,
    model_id: &str,
    cold_start: bool,
) -> Result<RouteDecision, RouteError> {
    let endpoint_url = format!(
        "{}/models/{}/endpoint",
        neuron_endpoint,
        urlencoding::encode(model_id)
    );

    let inference_endpoint = match fleet.http_client.get(&endpoint_url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(body) => body
                .get("url")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            Err(_) => None,
        },
        _ => None,
    };

    let raw = inference_endpoint.ok_or_else(|| {
        RouteError::EndpointResolveFailed(model_id.to_string(), node_name.to_string())
    })?;

    // Rewrite loopback inference URLs to use the configured neuron host.
    // Neuron's default bind_url is `http://localhost:13131` (it can't
    // reliably know its own externally-resolvable name). Cortex sees a
    // URL that's only meaningful from the neuron host's own perspective;
    // proxying directly to localhost from a different cortex host would
    // hit nothing. Keep neuron's port and path (a future harness could
    // serve inference on a different port than the management API), but
    // swap the host for the one in cortex.toml.
    let endpoint = rewrite_loopback_host(&raw, neuron_endpoint).unwrap_or(raw);

    Ok(RouteDecision {
        node_name: node_name.to_string(),
        endpoint,
        cold_start,
        resolved_model_id: model_id.to_string(),
    })
}

/// If `inference_url`'s host is a loopback name (localhost / 127.0.0.1 /
/// 0.0.0.0 / ::1), return a copy with the host replaced by
/// `neuron_endpoint`'s host. Otherwise return None and the caller falls
/// back to the inference URL as-is.
fn rewrite_loopback_host(inference_url: &str, neuron_endpoint: &str) -> Option<String> {
    let inf = url::Url::parse(inference_url).ok()?;
    let inf_host = inf.host_str()?;
    let is_loopback = matches!(inf_host, "localhost" | "127.0.0.1" | "0.0.0.0" | "::1");
    if !is_loopback {
        return None;
    }
    let neuron = url::Url::parse(neuron_endpoint).ok()?;
    let new_host = neuron.host_str()?;
    let mut out = inf.clone();
    out.set_host(Some(new_host)).ok()?;
    // url::Url::to_string normalises an empty path to "/", which then
    // breaks downstream callers that do format!("{endpoint}/v1/...")
    // and produce a double slash. The proxy URL is treated as a base
    // string that the caller appends paths to, so strip the trailing
    // slash here.
    let s = out.to_string();
    Some(s.trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::{ModelProfile, qualified_model_id, rewrite_loopback_host};

    fn bare_profile(id: &str, source: Option<&str>) -> ModelProfile {
        ModelProfile {
            id: id.into(),
            harness: "candle".into(),
            quant: None,
            vram_mb: None,
            min_devices: 1,
            min_device_vram_mb: None,
            pinned_on: vec![],
            source: source.map(String::from),
            limit: None,
            cost: None,
            capabilities: vec![],
        }
    }

    #[test]
    fn qualified_id_passes_through_when_source_absent() {
        let p = bare_profile("Qwen/Qwen3-30B", None);
        assert_eq!(qualified_model_id(&p), "Qwen/Qwen3-30B");
    }

    #[test]
    fn qualified_id_prefixes_when_source_set() {
        let p = bare_profile("Helexa/Qwen3.6-27B-Uncensored", Some("helexa"));
        assert_eq!(
            qualified_model_id(&p),
            "helexa:Helexa/Qwen3.6-27B-Uncensored"
        );
    }

    #[test]
    fn qualified_id_passes_through_when_source_is_empty_string() {
        // An empty scheme is treated as absent — neuron's default_source
        // substitution kicks in.
        let p = bare_profile("Qwen/Qwen3-30B", Some(""));
        assert_eq!(qualified_model_id(&p), "Qwen/Qwen3-30B");
    }

    #[test]
    fn rewrites_localhost_keeps_port_and_path() {
        let out = rewrite_loopback_host(
            "http://localhost:13131",
            "http://beast.hanzalova.internal:13131",
        );
        assert_eq!(
            out.as_deref(),
            Some("http://beast.hanzalova.internal:13131")
        );
    }

    #[test]
    fn rewrites_loopback_with_distinct_inference_port() {
        let out = rewrite_loopback_host("http://127.0.0.1:8080", "http://beast.lan:13131");
        assert_eq!(out.as_deref(), Some("http://beast.lan:8080"));
    }

    #[test]
    fn leaves_non_loopback_alone() {
        let out = rewrite_loopback_host("http://other.host:1234", "http://beast.lan:13131");
        assert_eq!(out, None);
    }

    #[test]
    fn malformed_inference_url_returns_none() {
        let out = rewrite_loopback_host("not a url", "http://beast.lan:13131");
        assert_eq!(out, None);
    }
}
