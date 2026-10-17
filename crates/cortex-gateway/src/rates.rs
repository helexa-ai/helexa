//! Per-model rates for work metered in units other than tokens (#386).
//!
//! The budget ledger counts tokens. Audio jobs are metered in seconds of
//! generated audio, converted to ledger tokens at a per-model rate. The
//! rates are configuration held by the clearing house (helexa-upstream's
//! `model_rates` table), so pricing changes without a deploy: cortex
//! fetches the table from `/authz/v1/rates`, caches it, and refreshes it in
//! the background.
//!
//! Resolution: the model's own rate, else the unit's `"*"` default. A model
//! with neither is not priced, and cortex refuses its metered work rather
//! than serving it free. Until the first fetch succeeds nothing is priced,
//! for the same reason.
//!
//! A cortex running without upstream has no clearing house to ask. It
//! prices every audio-second at [`LOCAL_DEFAULT_TOKENS_PER_AUDIO_SECOND`],
//! which only matters for the operator's own local key caps.

use cortex_core::config::UpstreamClientConfig;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Unit name for one second of generated audio, as stored upstream.
pub const AUDIO_SECOND: &str = "audio_second";

/// The default-rate model id.
pub const ANY_MODEL: &str = "*";

/// Audio rate used when this cortex runs without an upstream authority.
pub const LOCAL_DEFAULT_TOKENS_PER_AUDIO_SECOND: u64 = 1000;

/// How often the rate table is refreshed from upstream.
const REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// Retry interval while no table has loaded yet.
const INITIAL_RETRY: Duration = Duration::from_secs(5);

/// A snapshot of the rate table: `(unit, model_id) → tokens per unit`.
#[derive(Debug, Clone, Default)]
pub struct RateTable {
    rates: HashMap<(String, String), u64>,
}

impl RateTable {
    /// The table a cortex without upstream uses.
    pub fn local_default() -> Self {
        let mut t = Self::default();
        t.set(
            AUDIO_SECOND,
            ANY_MODEL,
            LOCAL_DEFAULT_TOKENS_PER_AUDIO_SECOND,
        );
        t
    }

    pub fn set(&mut self, unit: &str, model_id: &str, tokens_per_unit: u64) {
        self.rates
            .insert((unit.to_string(), model_id.to_string()), tokens_per_unit);
    }

    /// The model's rate for `unit`, else the unit's default, else `None`.
    pub fn rate(&self, unit: &str, model_id: &str) -> Option<u64> {
        self.rates
            .get(&(unit.to_string(), model_id.to_string()))
            .or_else(|| self.rates.get(&(unit.to_string(), ANY_MODEL.to_string())))
            .copied()
    }

    /// Ledger tokens per second of audio for `model_id`.
    pub fn audio_tokens_per_second(&self, model_id: &str) -> Option<u64> {
        self.rate(AUDIO_SECOND, model_id)
    }
}

/// The live rate table, held in `CortexState`.
pub struct Rates {
    table: RwLock<RateTable>,
}

impl Rates {
    /// Rates for a cortex with no upstream: the local default.
    pub fn local() -> Self {
        Self {
            table: RwLock::new(RateTable::local_default()),
        }
    }

    /// Rates for a cortex with upstream: empty (nothing priced) until the
    /// first fetch lands.
    pub fn upstream() -> Self {
        Self {
            table: RwLock::new(RateTable::default()),
        }
    }

    pub async fn snapshot(&self) -> RateTable {
        self.table.read().await.clone()
    }

    pub async fn replace(&self, table: RateTable) {
        *self.table.write().await = table;
    }

    pub async fn audio_tokens_per_second(&self, model_id: &str) -> Option<u64> {
        self.table.read().await.audio_tokens_per_second(model_id)
    }
}

#[derive(Deserialize)]
struct RatesResp {
    data: Vec<RateRow>,
}

#[derive(Deserialize)]
struct RateRow {
    model_id: String,
    unit: String,
    tokens_per_unit: i64,
}

/// Fetch the full rate table from upstream's `/authz/v1/rates`.
pub async fn fetch(
    client: &reqwest::Client,
    base_url: &str,
    bearer: &str,
) -> anyhow::Result<RateTable> {
    let url = format!("{}/authz/v1/rates", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .bearer_auth(bearer)
        .send()
        .await?
        .error_for_status()?;
    let body: RatesResp = resp.json().await?;
    let mut table = RateTable::default();
    for row in body.data {
        // The database forbids non-positive rates; skip one rather than
        // trust it if that ever changes.
        if let Ok(tokens) = u64::try_from(row.tokens_per_unit)
            && tokens > 0
        {
            table.set(&row.unit, &row.model_id, tokens);
        }
    }
    Ok(table)
}

/// Keep `rates` in step with upstream. Retries quickly until the first
/// table loads, then refreshes every minute. A failed refresh keeps the
/// last good table.
pub fn spawn_refresh(rates: Arc<Rates>, cfg: &UpstreamClientConfig) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
        .build()
        .expect("failed to build rates HTTP client");
    let url = cfg.url.clone();
    let bearer = cfg.bearer.clone();
    tokio::spawn(async move {
        let mut loaded = false;
        loop {
            match fetch(&client, &url, &bearer).await {
                Ok(table) => {
                    if !loaded {
                        tracing::info!("model rates loaded from upstream");
                    }
                    loaded = true;
                    rates.replace(table).await;
                }
                Err(e) => tracing::warn!(error = %e, "model rate refresh failed"),
            }
            tokio::time::sleep(if loaded {
                REFRESH_INTERVAL
            } else {
                INITIAL_RETRY
            })
            .await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_rate_beats_default_and_absent_unit_is_unpriced() {
        let mut t = RateTable::default();
        assert_eq!(t.audio_tokens_per_second("m"), None);
        t.set(AUDIO_SECOND, ANY_MODEL, 1000);
        assert_eq!(t.audio_tokens_per_second("m"), Some(1000));
        t.set(AUDIO_SECOND, "m", 250);
        assert_eq!(t.audio_tokens_per_second("m"), Some(250));
        assert_eq!(t.audio_tokens_per_second("other"), Some(1000));
        assert_eq!(t.rate("some_other_unit", "m"), None);
    }

    #[test]
    fn local_default_prices_every_audio_model() {
        let t = RateTable::local_default();
        assert_eq!(
            t.audio_tokens_per_second("anything"),
            Some(LOCAL_DEFAULT_TOKENS_PER_AUDIO_SECOND)
        );
    }
}
