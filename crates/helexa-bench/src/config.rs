//! Bench configuration: loaded from `helexa-bench.toml` with figment,
//! `BENCH_`-prefixed env overrides (mirrors `NeuronConfig::load`).

use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Top-level bench config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    #[serde(default)]
    pub bench: BenchSettings,
    #[serde(default)]
    pub scenarios: ScenarioConfig,
    /// Endpoints to benchmark. At least one is required for `run`/`once`.
    #[serde(default)]
    pub targets: Vec<TargetConfig>,
}

/// Loop/timing knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchSettings {
    /// Pause between full sweeps of all targets.
    #[serde(default = "default_sweep_interval")]
    pub sweep_interval_secs: u64,
    /// Target number of measured samples to record for a given
    /// (target, build SHA, model, scenario). Once met, later sweeps skip
    /// that cell — so a fully-sampled build costs only cheap version
    /// polls until a new SHA ships.
    #[serde(default = "default_samples")]
    pub samples_per_version: u32,
    /// Pause between successive measured iterations against one model.
    #[serde(default = "default_iter_pause")]
    pub iteration_pause_secs: u64,
    /// Per-request timeout (cold lazy-loads can be slow; generous like
    /// bench.py's 600s default).
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
    /// SQLite system-of-record path.
    #[serde(default = "default_db_path")]
    pub db_path: String,
}

impl Default for BenchSettings {
    fn default() -> Self {
        BenchSettings {
            sweep_interval_secs: default_sweep_interval(),
            samples_per_version: default_samples(),
            iteration_pause_secs: default_iter_pause(),
            request_timeout_secs: default_timeout(),
            db_path: default_db_path(),
        }
    }
}

impl BenchSettings {
    pub fn iteration_pause(&self) -> Duration {
        Duration::from_secs(self.iteration_pause_secs)
    }
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }
    pub fn sweep_interval(&self) -> Duration {
        Duration::from_secs(self.sweep_interval_secs)
    }
}

/// Which scenarios to run and their shared parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioConfig {
    /// Approximate prompt sizes (in tokens) — one chat-latency scenario
    /// is generated per size, e.g. `chat:128`, `chat:4096`. This is the
    /// per-cell dimension that the version-aware skip logic keys on.
    #[serde(default = "default_prompt_sizes")]
    pub prompt_sizes: Vec<u32>,
    /// Max generated tokens per request.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        ScenarioConfig {
            prompt_sizes: default_prompt_sizes(),
            max_tokens: default_max_tokens(),
        }
    }
}

/// One endpoint to benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Stable label used as the engine column and in the DB.
    pub name: String,
    /// Which protocol/metadata surface the target exposes.
    #[serde(default)]
    pub kind: TargetKind,
    /// Base URL. For `neuron`: the daemon root (e.g.
    /// `http://beast.internal:13131`). For `openai`: the OpenAI `/v1`
    /// base (e.g. `http://host:8080/v1`).
    pub endpoint: String,
    /// Optional display label override for reports (defaults to `name`).
    #[serde(default)]
    pub label: Option<String>,
}

impl TargetConfig {
    pub fn display_label(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.name)
    }
}

/// The two target surfaces. `neuron` gets rich build metadata and warm
/// model discovery via the native neuron API; `openai` is the seam for
/// later comparison against mistral.rs / llama.cpp / vLLM (phase 1
/// implements `neuron` fully; `openai` is preliminary plumbing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    #[default]
    Neuron,
    Openai,
}

impl BenchConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("BENCH_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

fn default_sweep_interval() -> u64 {
    1800
}
fn default_samples() -> u32 {
    5
}
fn default_iter_pause() -> u64 {
    2
}
fn default_timeout() -> u64 {
    600
}
fn default_db_path() -> String {
    "/var/lib/helexa-bench/bench.sqlite".to_string()
}
fn default_prompt_sizes() -> Vec<u32> {
    vec![128, 4096]
}
fn default_max_tokens() -> u64 {
    256
}

#[cfg(test)]
// Jail's closure must return figment::Result; the large-Err type is
// figment's, not ours, so suppress the lint here.
#[allow(clippy::result_large_err)]
mod tests {
    use super::*;
    use figment::Jail;

    #[test]
    fn loads_minimal_with_defaults() {
        Jail::expect_with(|jail| {
            jail.create_file(
                "helexa-bench.toml",
                r#"
                [[targets]]
                name = "beast"
                endpoint = "http://beast.internal:13131"
                "#,
            )?;
            let cfg = BenchConfig::load("helexa-bench.toml").unwrap();
            assert_eq!(cfg.targets.len(), 1);
            assert_eq!(cfg.targets[0].kind, TargetKind::Neuron);
            assert_eq!(cfg.bench.samples_per_version, 5);
            assert_eq!(cfg.scenarios.prompt_sizes, vec![128, 4096]);
            Ok(())
        });
    }

    #[test]
    fn env_overrides_apply() {
        Jail::expect_with(|jail| {
            jail.create_file(
                "helexa-bench.toml",
                r#"
                [bench]
                samples_per_version = 3
                [[targets]]
                name = "benjy"
                kind = "openai"
                endpoint = "http://benjy:8080/v1"
                "#,
            )?;
            jail.set_env("BENCH_BENCH__SAMPLES_PER_VERSION", "9");
            let cfg = BenchConfig::load("helexa-bench.toml").unwrap();
            assert_eq!(cfg.bench.samples_per_version, 9);
            assert_eq!(cfg.targets[0].kind, TargetKind::Openai);
            Ok(())
        });
    }
}
