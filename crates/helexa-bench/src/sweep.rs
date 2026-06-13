//! The version-aware sweep loop.
//!
//! Each sweep visits every configured target, polls its build identity
//! and warm models, and tops up benchmark samples per
//! (target, build SHA, model, scenario) to `samples_per_version`. Cells
//! already at target are skipped — so once every neuron's current build
//! is fully sampled, sweeps cost only the cheap metadata polls until a
//! new SHA ships. Runs are recorded to SQLite with full provenance.

use crate::client::TargetClient;
use crate::config::{BenchConfig, TargetConfig, TargetKind};
use crate::scenario::{RunCtx, build_scenarios};
use crate::store::{RunRecord, Store};
use anyhow::Result;
use cortex_core::build_info::BuildInfo;
use cortex_core::discovery::DiscoveryResponse;
use cortex_core::harness::ModelInfo;

/// helexa-bench's own build version.
fn bench_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// helexa-bench's own build SHA, injected by CI via `HELEXA_BUILD_SHA`
/// at compile time; `"unknown"` for ad-hoc local builds.
fn bench_sha() -> String {
    option_env!("HELEXA_BUILD_SHA")
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

#[derive(Debug, Default, Clone)]
pub struct SweepSummary {
    pub measured: usize,
    pub skipped: usize,
    pub failed: usize,
    pub targets_unreachable: usize,
}

pub struct Sweeper {
    cfg: BenchConfig,
    client: TargetClient,
    store: Store,
}

impl Sweeper {
    pub fn new(cfg: BenchConfig) -> Result<Self> {
        let client = TargetClient::new(cfg.bench.request_timeout())?;
        let store = Store::open(&cfg.bench.db_path)?;
        Ok(Sweeper { cfg, client, store })
    }

    /// Run sweeps forever, pausing `sweep_interval` between them.
    pub async fn run_forever(&self) -> ! {
        loop {
            match self.run_once().await {
                Ok(s) => tracing::info!(
                    measured = s.measured,
                    skipped = s.skipped,
                    failed = s.failed,
                    unreachable = s.targets_unreachable,
                    "sweep complete"
                ),
                Err(e) => tracing::error!(error = %format!("{e:#}"), "sweep errored"),
            }
            tracing::debug!(
                secs = self.cfg.bench.sweep_interval_secs,
                "sleeping until next sweep"
            );
            tokio::time::sleep(self.cfg.bench.sweep_interval()).await;
        }
    }

    /// One full pass over all targets.
    pub async fn run_once(&self) -> Result<SweepSummary> {
        let mut summary = SweepSummary::default();
        for target in &self.cfg.targets {
            if let Err(e) = self.sweep_target(target, &mut summary).await {
                summary.targets_unreachable += 1;
                tracing::warn!(target = %target.name, error = %format!("{e:#}"), "target skipped");
            }
        }
        Ok(summary)
    }

    async fn sweep_target(&self, target: &TargetConfig, summary: &mut SweepSummary) -> Result<()> {
        let build = self.client.fetch_version(target).await?;
        let discovery = self.client.fetch_discovery(target).await.unwrap_or(None);
        let models = self.client.warm_models(target).await?;

        tracing::info!(
            target = %target.name,
            sha = %build.git_sha,
            warm_models = models.len(),
            "sweeping target"
        );

        let scenarios = build_scenarios(&self.cfg.scenarios);
        for model in &models {
            for scenario in scenarios.iter().filter(|s| s.applies_to(model)) {
                let have = self.store.count_samples(
                    &target.name,
                    &build.git_sha,
                    &model.id,
                    scenario.id(),
                )?;
                let need = self.cfg.bench.samples_per_version.saturating_sub(have);
                if need == 0 {
                    summary.skipped += 1;
                    tracing::debug!(
                        target = %target.name, model = %model.id, scenario = scenario.id(),
                        sha = %build.git_sha, "cell already satisfied, skipping"
                    );
                    continue;
                }

                let ctx = RunCtx {
                    client: self.client.http(),
                    chat_url: self.client.chat_url(target),
                    model_id: model.id.clone(),
                    max_tokens: self.cfg.scenarios.max_tokens,
                    timeout: self.cfg.bench.request_timeout(),
                };

                // One unmeasured warmup when the cell is empty (matches
                // bench.py — first run after a load hits cold caches).
                if have == 0 {
                    tracing::debug!(model = %model.id, scenario = scenario.id(), "warmup run");
                    let _ = scenario.run(&ctx).await;
                }

                for i in 0..need {
                    match scenario.run(&ctx).await {
                        Ok(m) => {
                            let rec = self.build_record(
                                target,
                                &build,
                                discovery.as_ref(),
                                model,
                                scenario.id(),
                                scenario.prompt_size(),
                                Ok(&m),
                            );
                            self.store.insert_run(&rec)?;
                            summary.measured += 1;
                            tracing::info!(
                                target = %target.name, model = %model.id, scenario = scenario.id(),
                                ttft_s = m.ttft_s, decode_tps = ?m.decode_tps, total_s = m.total_s,
                                "{}/{} recorded", have + i + 1, self.cfg.bench.samples_per_version
                            );
                        }
                        Err(e) => {
                            let msg = format!("{e:#}");
                            let rec = self.build_record(
                                target,
                                &build,
                                discovery.as_ref(),
                                model,
                                scenario.id(),
                                scenario.prompt_size(),
                                Err(&msg),
                            );
                            self.store.insert_run(&rec)?;
                            summary.failed += 1;
                            tracing::warn!(
                                target = %target.name, model = %model.id, scenario = scenario.id(),
                                error = %msg, "iteration failed"
                            );
                        }
                    }
                    tokio::time::sleep(self.cfg.bench.iteration_pause()).await;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_record(
        &self,
        target: &TargetConfig,
        build: &BuildInfo,
        discovery: Option<&DiscoveryResponse>,
        model: &ModelInfo,
        scenario_id: &str,
        prompt_size: u32,
        result: Result<&crate::scenario::ScenarioMetrics, &str>,
    ) -> RunRecord {
        let (ok, error, ttft, decode, total, prompt_tokens, completion) = match result {
            Ok(m) => (
                true,
                None,
                Some(m.ttft_s),
                m.decode_tps,
                Some(m.total_s),
                m.prompt_tokens,
                Some(m.completion_tokens),
            ),
            Err(e) => (false, Some(e.to_string()), None, None, None, None, None),
        };

        RunRecord {
            ts: chrono::Utc::now().to_rfc3339(),
            target_name: target.name.clone(),
            target_kind: kind_str(target.kind).to_string(),
            endpoint: target.endpoint.clone(),
            hostname: discovery.map(|d| d.hostname.clone()),
            driver_version: discovery.and_then(|d| d.driver_version.clone()),
            cuda_version: discovery.and_then(|d| d.cuda_version.clone()),
            gpus_json: discovery
                .map(|d| serde_json::to_string(&d.devices).unwrap_or_else(|_| "[]".to_string())),
            git_sha: build.git_sha.clone(),
            git_sha_long: build.git_sha_long.clone(),
            package_version: build.package_version.clone(),
            git_dirty: build.git_dirty,
            build_timestamp: build.build_timestamp.clone(),
            rustc_version: build.rustc_version.clone(),
            profile: build.profile.clone(),
            features_json: serde_json::to_string(&build.features)
                .unwrap_or_else(|_| "[]".to_string()),
            candle_version: build.candle_version.clone(),
            bench_version: bench_version(),
            bench_sha: bench_sha(),
            model_id: model.id.clone(),
            harness: model.harness.clone(),
            capabilities_json: serde_json::to_string(&model.capabilities)
                .unwrap_or_else(|_| "[]".to_string()),
            devices_json: serde_json::to_string(&model.devices)
                .unwrap_or_else(|_| "[]".to_string()),
            scenario_id: scenario_id.to_string(),
            prompt_size_approx: prompt_size,
            prompt_tokens_actual: prompt_tokens,
            max_tokens: self.cfg.scenarios.max_tokens,
            ttft_s: ttft,
            decode_tps: decode,
            total_s: total,
            completion_tokens: completion,
            ok,
            error,
        }
    }
}

fn kind_str(kind: TargetKind) -> &'static str {
    match kind {
        TargetKind::Neuron => "neuron",
        TargetKind::Openai => "openai",
    }
}
