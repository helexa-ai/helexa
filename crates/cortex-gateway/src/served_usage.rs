//! Served-usage ledger (#58): cortex meters, per principal and per UTC day,
//! the tokens it has served on behalf of mesh accounts, and periodically
//! reports **absolute** cumulative counters to helexa-upstream for
//! reconciliation (operators are compensated for served tokens).
//!
//! Counters are cumulative-since-process-start for the current period;
//! upstream upserts them monotonically (GREATEST), so re-sending the same
//! value is idempotent and a flush that races another is harmless. (A
//! process restart resets the in-memory counter; the monotonic upsert keeps
//! upstream from regressing — at most it under-counts the restarted window,
//! acceptable for beta. One cortex per operator token is assumed.)

use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ServedRow {
    pub account_id: String,
    pub key_id: String,
    pub period: String, // YYYY-MM-DD (UTC)
    pub served_tokens: u64,
}

#[derive(Default)]
pub struct ServedUsage {
    inner: Mutex<HashMap<(String, String, String), u64>>,
}

impl ServedUsage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add served tokens for a principal in today's (UTC) period.
    pub fn add(&self, account_id: &str, key_id: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let period = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let mut m = self.inner.lock().expect("served-usage lock");
        *m.entry((account_id.to_string(), key_id.to_string(), period))
            .or_insert(0) += tokens;
    }

    /// Absolute cumulative counters, for a flush to upstream.
    pub fn snapshot(&self) -> Vec<ServedRow> {
        let m = self.inner.lock().expect("served-usage lock");
        m.iter()
            .map(|((account_id, key_id, period), &served_tokens)| ServedRow {
                account_id: account_id.clone(),
                key_id: key_id.clone(),
                period: period.clone(),
                served_tokens,
            })
            .collect()
    }
}

/// POST the absolute counters to upstream's `/authz/v1/served-usage`.
pub async fn report(
    client: &reqwest::Client,
    base_url: &str,
    bearer: &str,
    rows: &[ServedRow],
) -> Result<(), reqwest::Error> {
    if rows.is_empty() {
        return Ok(());
    }
    let url = format!("{}/authz/v1/served-usage", base_url.trim_end_matches('/'));
    client
        .post(url)
        .bearer_auth(bearer)
        .json(&serde_json::json!({ "rows": rows }))
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_per_principal_and_period() {
        let su = ServedUsage::new();
        su.add("acct", "key", 10);
        su.add("acct", "key", 5);
        su.add("acct", "other", 7);
        su.add("acct", "key", 0); // no-op
        let mut rows = su.snapshot();
        rows.sort_by(|a, b| a.key_id.cmp(&b.key_id));
        assert_eq!(rows.len(), 2);
        let key_row = rows.iter().find(|r| r.key_id == "key").unwrap();
        assert_eq!(key_row.served_tokens, 15);
        assert_eq!(
            rows.iter()
                .find(|r| r.key_id == "other")
                .unwrap()
                .served_tokens,
            7
        );
    }
}
