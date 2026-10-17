//! Per-model rates for work metered in units other than tokens (#386).
//!
//! The ledger only counts tokens; a rate converts another unit (one second
//! of generated audio) into ledger tokens. Rates live in `model_rates` so
//! they can change without a deploy. cortex reads the whole table from
//! `/authz/v1/rates` and resolves a model's rate itself: the model's own
//! row, else the unit's `'*'` default, else the work is not priced.

use serde::Serialize;
use sqlx::postgres::PgPool;

/// One row of `model_rates`, as served to cortex.
#[derive(Debug, Clone, Serialize)]
pub struct ModelRate {
    pub model_id: String,
    pub unit: String,
    pub tokens_per_unit: i64,
}

/// Every configured rate.
pub async fn list(pool: &PgPool) -> Result<Vec<ModelRate>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT model_id, unit, tokens_per_unit FROM model_rates ORDER BY unit, model_id",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| ModelRate {
            model_id: sqlx::Row::get(&r, "model_id"),
            unit: sqlx::Row::get(&r, "unit"),
            tokens_per_unit: sqlx::Row::get(&r, "tokens_per_unit"),
        })
        .collect())
}
