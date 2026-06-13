//! SQLite system-of-record. One row per measured iteration, keyed so a
//! benchmark can be attributed to the exact neuron build that produced
//! it. Replaces hand edits to `doc/benchmarks.md`.
//!
//! Calls are synchronous (SQLite is local and the sweep is batch-1
//! sequential), so the connection is used inline between `await` points,
//! never held across one.

use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::Path;

/// A single measured (or failed) iteration, with full provenance.
#[derive(Debug, Clone)]
pub struct RunRecord {
    pub ts: String, // RFC3339
    // target
    pub target_name: String,
    pub target_kind: String,
    pub endpoint: String,
    // host (from /discovery)
    pub hostname: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub gpus_json: Option<String>,
    // neuron build (from /version)
    pub git_sha: String,
    pub git_sha_long: Option<String>,
    pub package_version: String,
    pub git_dirty: bool,
    pub build_timestamp: Option<String>,
    pub rustc_version: Option<String>,
    pub profile: Option<String>,
    pub features_json: String,
    pub candle_version: Option<String>,
    // bench's own build
    pub bench_version: String,
    pub bench_sha: String,
    // model
    pub model_id: String,
    pub harness: String,
    pub capabilities_json: String,
    pub devices_json: String,
    // scenario
    pub scenario_id: String,
    pub prompt_size_approx: u32,
    pub prompt_tokens_actual: Option<u64>,
    pub max_tokens: u64,
    // metrics
    pub ttft_s: Option<f64>,
    pub decode_tps: Option<f64>,
    pub total_s: Option<f64>,
    pub completion_tokens: Option<u64>,
    // outcome
    pub ok: bool,
    pub error: Option<String>,
}

pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (creating parent dirs + schema as needed).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating db dir {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening sqlite db {}", path.display()))?;
        Self::init(&conn)?;
        Ok(Store { conn })
    }

    /// In-memory store for tests.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(&conn)?;
        Ok(Store { conn })
    }

    fn init(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS runs (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                   TEXT NOT NULL,
                target_name          TEXT NOT NULL,
                target_kind          TEXT NOT NULL,
                endpoint             TEXT NOT NULL,
                hostname             TEXT,
                driver_version       TEXT,
                cuda_version         TEXT,
                gpus_json            TEXT,
                git_sha              TEXT NOT NULL,
                git_sha_long         TEXT,
                package_version      TEXT NOT NULL,
                git_dirty            INTEGER NOT NULL,
                build_timestamp      TEXT,
                rustc_version        TEXT,
                profile              TEXT,
                features_json        TEXT NOT NULL,
                candle_version       TEXT,
                bench_version        TEXT NOT NULL,
                bench_sha            TEXT NOT NULL,
                model_id             TEXT NOT NULL,
                harness              TEXT NOT NULL,
                capabilities_json    TEXT NOT NULL,
                devices_json         TEXT NOT NULL,
                scenario_id          TEXT NOT NULL,
                prompt_size_approx   INTEGER NOT NULL,
                prompt_tokens_actual INTEGER,
                max_tokens           INTEGER NOT NULL,
                ttft_s               REAL,
                decode_tps           REAL,
                total_s              REAL,
                completion_tokens    INTEGER,
                ok                   INTEGER NOT NULL,
                error                TEXT
            );
            -- The version-aware skip query keys on this tuple. scenario_id
            -- encodes the prompt size (chat:<n>), so it subsumes the cell.
            CREATE INDEX IF NOT EXISTS idx_runs_cell
                ON runs (target_name, git_sha, model_id, scenario_id, ok);
            "#,
        )
        .context("initialising sqlite schema")?;
        Ok(())
    }

    /// Count successful samples already recorded for a cell. Only `ok`
    /// rows count toward the per-version target so transient failures
    /// don't permanently starve a cell.
    pub fn count_samples(
        &self,
        target_name: &str,
        git_sha: &str,
        model_id: &str,
        scenario_id: &str,
    ) -> Result<u32> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM runs WHERE target_name=?1 AND git_sha=?2 \
             AND model_id=?3 AND scenario_id=?4 AND ok=1",
            params![target_name, git_sha, model_id, scenario_id],
            |row| row.get(0),
        )?;
        Ok(n as u32)
    }

    pub fn insert_run(&self, r: &RunRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO runs (
                ts, target_name, target_kind, endpoint,
                hostname, driver_version, cuda_version, gpus_json,
                git_sha, git_sha_long, package_version, git_dirty,
                build_timestamp, rustc_version, profile, features_json, candle_version,
                bench_version, bench_sha,
                model_id, harness, capabilities_json, devices_json,
                scenario_id, prompt_size_approx, prompt_tokens_actual, max_tokens,
                ttft_s, decode_tps, total_s, completion_tokens,
                ok, error
            ) VALUES (
                ?1, ?2, ?3, ?4,
                ?5, ?6, ?7, ?8,
                ?9, ?10, ?11, ?12,
                ?13, ?14, ?15, ?16, ?17,
                ?18, ?19,
                ?20, ?21, ?22, ?23,
                ?24, ?25, ?26, ?27,
                ?28, ?29, ?30, ?31,
                ?32, ?33
            )",
            params![
                r.ts,
                r.target_name,
                r.target_kind,
                r.endpoint,
                r.hostname,
                r.driver_version,
                r.cuda_version,
                r.gpus_json,
                r.git_sha,
                r.git_sha_long,
                r.package_version,
                r.git_dirty as i64,
                r.build_timestamp,
                r.rustc_version,
                r.profile,
                r.features_json,
                r.candle_version,
                r.bench_version,
                r.bench_sha,
                r.model_id,
                r.harness,
                r.capabilities_json,
                r.devices_json,
                r.scenario_id,
                r.prompt_size_approx,
                r.prompt_tokens_actual,
                r.max_tokens,
                r.ttft_s,
                r.decode_tps,
                r.total_s,
                r.completion_tokens,
                r.ok as i64,
                r.error,
            ],
        )?;
        Ok(())
    }

    /// One reportable cell: the median metrics over the most-recently-seen
    /// build SHA for each (target, model, scenario).
    pub fn report_rows(&self) -> Result<Vec<ReportRow>> {
        // For each (target, model, scenario), find the SHA of the latest
        // successful run, then median that SHA's samples.
        let mut stmt = self.conn.prepare(
            "SELECT target_name, model_id, scenario_id, prompt_size_approx, git_sha,
                    ttft_s, decode_tps, total_s, prompt_tokens_actual
             FROM runs
             WHERE ok=1
             ORDER BY target_name, model_id, scenario_id, id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RawRow {
                target_name: row.get(0)?,
                model_id: row.get(1)?,
                scenario_id: row.get(2)?,
                prompt_size_approx: row.get(3)?,
                git_sha: row.get(4)?,
                ttft_s: row.get(5)?,
                decode_tps: row.get(6)?,
                total_s: row.get(7)?,
                prompt_tokens_actual: row.get(8)?,
            })
        })?;
        let raws: Vec<RawRow> = rows.collect::<rusqlite::Result<_>>()?;
        Ok(aggregate(raws))
    }
}

struct RawRow {
    target_name: String,
    model_id: String,
    scenario_id: String,
    prompt_size_approx: u32,
    git_sha: String,
    ttft_s: Option<f64>,
    decode_tps: Option<f64>,
    total_s: Option<f64>,
    prompt_tokens_actual: Option<u64>,
}

/// An aggregated cell ready for the report table.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportRow {
    pub target_name: String,
    pub model_id: String,
    pub scenario_id: String,
    pub prompt_size_approx: u32,
    pub git_sha: String,
    pub prompt_tokens: Option<u64>,
    pub ttft_s_median: Option<f64>,
    pub decode_tps_median: Option<f64>,
    pub total_s_median: Option<f64>,
    pub samples: usize,
}

/// Group by (target, model, scenario), keep only the latest SHA's rows
/// (latest = the SHA of the last-inserted row, since input is id-ordered),
/// and median each metric.
fn aggregate(raws: Vec<RawRow>) -> Vec<ReportRow> {
    use std::collections::BTreeMap;
    // key -> (latest_sha, rows for that sha)
    let mut groups: BTreeMap<(String, String, String), Vec<RawRow>> = BTreeMap::new();
    for r in raws {
        groups
            .entry((
                r.target_name.clone(),
                r.model_id.clone(),
                r.scenario_id.clone(),
            ))
            .or_default()
            .push(r);
    }
    let mut out = Vec::new();
    for ((target_name, model_id, scenario_id), rows) in groups {
        // id-ordered, so the last row carries the latest SHA.
        let latest_sha = rows.last().map(|r| r.git_sha.clone()).unwrap_or_default();
        let cell: Vec<&RawRow> = rows.iter().filter(|r| r.git_sha == latest_sha).collect();
        let prompt_size_approx = cell.first().map(|r| r.prompt_size_approx).unwrap_or(0);
        out.push(ReportRow {
            target_name,
            model_id,
            scenario_id,
            prompt_size_approx,
            git_sha: latest_sha,
            prompt_tokens: cell.iter().find_map(|r| r.prompt_tokens_actual),
            ttft_s_median: median(cell.iter().filter_map(|r| r.ttft_s)),
            decode_tps_median: median(cell.iter().filter_map(|r| r.decode_tps)),
            total_s_median: median(cell.iter().filter_map(|r| r.total_s)),
            samples: cell.len(),
        });
    }
    out
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // lo == hi for odd lengths (the middle element); they straddle the
    // centre for even lengths. Avoids a `% 2` branch.
    let lo = (v.len() - 1) / 2;
    let hi = v.len() / 2;
    Some((v[lo] + v[hi]) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(target: &str, sha: &str, model: &str, scenario: &str, ok: bool) -> RunRecord {
        RunRecord {
            ts: "2026-06-13T00:00:00Z".into(),
            target_name: target.into(),
            target_kind: "neuron".into(),
            endpoint: "http://x:13131".into(),
            hostname: Some("x".into()),
            driver_version: None,
            cuda_version: None,
            gpus_json: None,
            git_sha: sha.into(),
            git_sha_long: None,
            package_version: "0.1.16".into(),
            git_dirty: false,
            build_timestamp: None,
            rustc_version: None,
            profile: None,
            features_json: "[]".into(),
            candle_version: None,
            bench_version: "0.1.16".into(),
            bench_sha: "deadbee".into(),
            model_id: model.into(),
            harness: "candle".into(),
            capabilities_json: "[]".into(),
            devices_json: "[]".into(),
            scenario_id: scenario.into(),
            prompt_size_approx: 128,
            prompt_tokens_actual: Some(130),
            max_tokens: 256,
            ttft_s: Some(0.1),
            decode_tps: Some(50.0),
            total_s: Some(1.0),
            completion_tokens: Some(50),
            ok,
            error: if ok { None } else { Some("boom".into()) },
        }
    }

    #[test]
    fn counts_only_successful_samples() {
        let s = Store::open_in_memory().unwrap();
        s.insert_run(&rec("beast", "abc", "m", "chat:128", true))
            .unwrap();
        s.insert_run(&rec("beast", "abc", "m", "chat:128", true))
            .unwrap();
        s.insert_run(&rec("beast", "abc", "m", "chat:128", false))
            .unwrap();
        assert_eq!(s.count_samples("beast", "abc", "m", "chat:128").unwrap(), 2);
        // Different SHA is a different cell.
        assert_eq!(s.count_samples("beast", "xyz", "m", "chat:128").unwrap(), 0);
    }

    #[test]
    fn report_uses_latest_sha_per_cell() {
        let s = Store::open_in_memory().unwrap();
        // old build
        s.insert_run(&rec("beast", "old", "m", "chat:128", true))
            .unwrap();
        // new build, two samples
        let mut r = rec("beast", "new", "m", "chat:128", true);
        r.ttft_s = Some(0.2);
        s.insert_run(&r).unwrap();
        r.ttft_s = Some(0.4);
        s.insert_run(&r).unwrap();
        let rows = s.report_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].git_sha, "new");
        assert_eq!(rows[0].samples, 2);
        assert!((rows[0].ttft_s_median.unwrap() - 0.3).abs() < 1e-9);
    }
}
