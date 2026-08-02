//! Database-backed application config (`app_config`).
//!
//! The split with `helexa-upstream.toml` is deliberate: the toml describes
//! how the *process* runs (listen address, database URL, SMTP credentials)
//! and needs a restart to change; this table describes how the *product*
//! behaves (grant sizes, thresholds, caps) and an operator should be able
//! to change it from an admin UI, live.
//!
//! Rows are self-describing — type, bounds, label, help text — so the
//! planned admin UI can render an editor for settings that did not exist
//! when it was written. Reads go through the typed accessors here, which
//! fall back to a caller-supplied default when a key is absent (a fresh
//! deployment, or a setting added by a newer binary than the database) and
//! clamp to the row's own bounds, so a bad write cannot take the service
//! outside a sane range.

use serde_json::Value;
use sqlx::Row;
use sqlx::postgres::PgPool;

/// One setting as the admin UI needs to see it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigEntry {
    pub key: String,
    pub value: Value,
    pub value_type: String,
    pub category: String,
    pub label: String,
    pub description: Option<String>,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
}

/// Every setting, ordered for display. The admin UI groups on `category`.
pub async fn list(pool: &PgPool) -> Result<Vec<ConfigEntry>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT key, value, value_type, category, label, description, min_value, max_value \
         FROM app_config ORDER BY category, key",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| ConfigEntry {
            key: r.get("key"),
            value: r.get("value"),
            value_type: r.get("value_type"),
            category: r.get("category"),
            label: r.get("label"),
            description: r.get("description"),
            min_value: r.get("min_value"),
            max_value: r.get("max_value"),
        })
        .collect())
}

/// Raw row fetch: the JSON value plus its declared bounds.
async fn fetch(
    pool: &PgPool,
    key: &str,
) -> Result<Option<(Value, Option<i64>, Option<i64>)>, sqlx::Error> {
    let row = sqlx::query("SELECT value, min_value, max_value FROM app_config WHERE key = $1")
        .bind(key)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|r| (r.get("value"), r.get("min_value"), r.get("max_value"))))
}

/// Integer setting, clamped to the row's declared bounds.
///
/// Falls back to `default` when the key is absent or holds a non-integer —
/// a malformed row degrades to shipped behaviour rather than breaking the
/// request path.
pub async fn get_i64(pool: &PgPool, key: &str, default: i64) -> i64 {
    match fetch(pool, key).await {
        Ok(Some((value, min, max))) => {
            let raw = value.as_i64().unwrap_or(default);
            clamp(raw, min, max)
        }
        Ok(None) => default,
        Err(e) => {
            tracing::warn!(key, error = %e, "app_config read failed; using default");
            default
        }
    }
}

/// Boolean setting, `default` when absent or malformed.
pub async fn get_bool(pool: &PgPool, key: &str, default: bool) -> bool {
    match fetch(pool, key).await {
        Ok(Some((value, _, _))) => value.as_bool().unwrap_or(default),
        Ok(None) => default,
        Err(e) => {
            tracing::warn!(key, error = %e, "app_config read failed; using default");
            default
        }
    }
}

/// Clamp `v` into whichever bounds the row declares.
pub fn clamp(v: i64, min: Option<i64>, max: Option<i64>) -> i64 {
    let v = match min {
        Some(m) => v.max(m),
        None => v,
    };
    match max {
        Some(m) => v.min(m),
        None => v,
    }
}

/// Keys used by the self-service top-up path. Named constants rather than
/// inline strings so a rename is a compile error, not a silent fallback to
/// the default.
pub mod keys {
    pub const TOPUP_AUTO_ENABLED: &str = "topup.auto.enabled";
    pub const TOPUP_AUTO_THRESHOLD_PCT: &str = "topup.auto.threshold_pct";
    pub const TOPUP_AUTO_GRANT_TOKENS: &str = "topup.auto.grant_tokens";
    pub const TOPUP_AUTO_MAX_PER_ACCOUNT: &str = "topup.auto.max_per_account";
    pub const TOPUP_AUTO_COOLDOWN_SECS: &str = "topup.auto.cooldown_secs";
}

/// Shipped defaults, used when the database has no row for a key. These
/// mirror the values seeded by migration 0002 so a binary running ahead of
/// its migrations behaves identically.
pub mod defaults {
    pub const TOPUP_AUTO_ENABLED: bool = true;
    pub const TOPUP_AUTO_THRESHOLD_PCT: i64 = 75;
    pub const TOPUP_AUTO_GRANT_TOKENS: i64 = 1_000_000;
    pub const TOPUP_AUTO_MAX_PER_ACCOUNT: i64 = 3;
    pub const TOPUP_AUTO_COOLDOWN_SECS: i64 = 86_400;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_respects_bounds() {
        assert_eq!(clamp(150, Some(1), Some(100)), 100);
        assert_eq!(clamp(0, Some(1), Some(100)), 1);
        assert_eq!(clamp(50, Some(1), Some(100)), 50);
    }

    #[test]
    fn clamp_tolerates_open_bounds() {
        assert_eq!(clamp(-5, None, None), -5);
        assert_eq!(clamp(-5, Some(0), None), 0);
        assert_eq!(clamp(9_999, None, Some(10)), 10);
    }
}
