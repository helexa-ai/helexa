//! Reconciliation rollup (#58): aggregate the served-usage ledger per
//! operator and period for operator compensation, stamping rows
//! `reconciled_at` so each window is settled once. The payout mechanism
//! itself is out of scope — this produces the authoritative per-operator
//! totals a settlement process consumes.

use sqlx::Row;
use sqlx::postgres::PgPool;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollupRow {
    pub operator_id: String,
    pub period: chrono::NaiveDate,
    pub total_served_tokens: i64,
}

/// Roll up all not-yet-reconciled served-usage into per-(operator, period)
/// totals, then stamp those rows `reconciled_at`. Returns the rollup.
/// Idempotent: a second run finds nothing unreconciled and returns empty.
pub async fn reconcile(pool: &PgPool) -> Result<Vec<RollupRow>, sqlx::Error> {
    let mut tx = pool.begin().await?;
    let rows = sqlx::query(
        // SUM(bigint) is numeric in Postgres — cast back to bigint for i64.
        "SELECT operator_id, period, SUM(served_tokens)::bigint AS total \
         FROM served_usage WHERE reconciled_at IS NULL \
         GROUP BY operator_id, period ORDER BY operator_id, period",
    )
    .fetch_all(&mut *tx)
    .await?;
    let rollup: Vec<RollupRow> = rows
        .iter()
        .map(|r| RollupRow {
            operator_id: r.get("operator_id"),
            period: r.get("period"),
            total_served_tokens: r.get::<i64, _>("total"),
        })
        .collect();
    sqlx::query("UPDATE served_usage SET reconciled_at = now() WHERE reconciled_at IS NULL")
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(rollup)
}
