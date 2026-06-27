//! SQLite system-of-record. One row per measured iteration, keyed so a
//! benchmark can be attributed to the exact neuron build that produced
//! it. Replaces hand edits to `doc/benchmarks.md`.
//!
//! Calls are synchronous (SQLite is local and the sweep is batch-1
//! sequential), so the connection is used inline between `await` points,
//! never held across one.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
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
    // server-measured prefill/decode split (#85), null on engines/paths
    // that don't emit `usage.helexa_timing`.
    pub prefill_ms: Option<u64>,
    pub decode_ms: Option<u64>,
    pub prefill_tokens: Option<u64>,
    // GPU telemetry sampled from /health around the run (#87), null for
    // non-neuron targets or when /health was unreachable. vram_used_mb is
    // the node sum; util/temp are the hottest single device.
    pub vram_used_mb: Option<u64>,
    pub gpu_util_pct: Option<u32>,
    pub gpu_temp_c: Option<u32>,
    // concurrency / agentic-load burst metrics (#89), null for single-request
    // scenarios. For a burst, ttft_s/decode_tps/total_s carry the aggregate.
    pub concurrency: Option<u32>,
    pub ttft_p95_s: Option<f64>,
    pub queue_wait_ms: Option<f64>,
    pub rejected: Option<u32>,
    // cold-load / model-swap cost (#90), set only by the deliberate
    // `swap-cost` measurement (scenario_id = "swap"). The other metric
    // fields carry the cold first-request after reload.
    pub swap_unload_ms: Option<u64>,
    pub swap_load_ms: Option<u64>,
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
            -- WAL so the read-only API connection never blocks the
            -- sweep writer (and vice versa).
            PRAGMA journal_mode=WAL;
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
                prefill_ms           INTEGER,
                decode_ms            INTEGER,
                prefill_tokens       INTEGER,
                vram_used_mb         INTEGER,
                gpu_util_pct         INTEGER,
                gpu_temp_c           INTEGER,
                concurrency          INTEGER,
                ttft_p95_s           REAL,
                queue_wait_ms        REAL,
                rejected             INTEGER,
                swap_unload_ms       INTEGER,
                swap_load_ms         INTEGER,
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
        // Additive migrations for DBs created before a column existed.
        // `CREATE TABLE IF NOT EXISTS` above only seeds fresh DBs; existing
        // ones need the columns backfilled (as NULL) so older rows coexist
        // with new metrics. There is no migration framework — each entry is
        // an idempotent "add if missing".
        Self::ensure_columns(
            conn,
            "runs",
            &[
                ("prefill_ms", "INTEGER"),
                ("decode_ms", "INTEGER"),
                ("prefill_tokens", "INTEGER"),
                ("vram_used_mb", "INTEGER"),
                ("gpu_util_pct", "INTEGER"),
                ("gpu_temp_c", "INTEGER"),
                ("concurrency", "INTEGER"),
                ("ttft_p95_s", "REAL"),
                ("queue_wait_ms", "REAL"),
                ("rejected", "INTEGER"),
                ("swap_unload_ms", "INTEGER"),
                ("swap_load_ms", "INTEGER"),
            ],
        )?;
        Ok(())
    }

    /// Add any of `columns` that the table is missing (`ALTER TABLE ADD
    /// COLUMN`). Idempotent: existing columns are read from
    /// `PRAGMA table_info` and skipped, so this is safe to run on every open.
    fn ensure_columns(conn: &Connection, table: &str, columns: &[(&str, &str)]) -> Result<()> {
        let mut existing = std::collections::HashSet::new();
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let names = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for name in names {
            existing.insert(name?);
        }
        for (name, ty) in columns {
            if !existing.contains(*name) {
                conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {name} {ty};"))
                    .with_context(|| format!("adding column {table}.{name}"))?;
            }
        }
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
                prefill_ms, decode_ms, prefill_tokens,
                vram_used_mb, gpu_util_pct, gpu_temp_c,
                concurrency, ttft_p95_s, queue_wait_ms, rejected,
                swap_unload_ms, swap_load_ms,
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
                ?32, ?33, ?34,
                ?35, ?36, ?37,
                ?38, ?39, ?40, ?41,
                ?42, ?43,
                ?44, ?45
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
                r.prefill_ms,
                r.decode_ms,
                r.prefill_tokens,
                r.vram_used_mb,
                r.gpu_util_pct,
                r.gpu_temp_c,
                r.concurrency,
                r.ttft_p95_s,
                r.queue_wait_ms,
                r.rejected,
                r.swap_unload_ms,
                r.swap_load_ms,
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
                    ttft_s, decode_tps, total_s, prompt_tokens_actual, gpus_json,
                    prefill_ms, decode_ms, prefill_tokens,
                    vram_used_mb, gpu_util_pct, gpu_temp_c,
                    concurrency, ttft_p95_s, queue_wait_ms, rejected
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
                gpus_json: row.get(9)?,
                prefill_ms: row.get(10)?,
                decode_ms: row.get(11)?,
                prefill_tokens: row.get(12)?,
                vram_used_mb: row.get(13)?,
                gpu_util_pct: row.get(14)?,
                gpu_temp_c: row.get(15)?,
                concurrency: row.get(16)?,
                ttft_p95_s: row.get(17)?,
                queue_wait_ms: row.get(18)?,
                rejected: row.get(19)?,
            })
        })?;
        let raws: Vec<RawRow> = rows.collect::<rusqlite::Result<_>>()?;
        Ok(aggregate(raws))
    }

    /// Context-length scaling curves (#88): per (target, model), the latest
    /// build's `chat:<n>` cells pivoted by prompt size into prefill & decode
    /// tok/s vs context. The headline is `decode_flatness` — decode tok/s at
    /// the largest context divided by the smallest. Near 1.0 confirms the
    /// Gated-DeltaNet O(1)-in-sequence-length decode; a sharp drop locates
    /// where the model stops scaling for free.
    pub fn scaling(&self) -> Result<Vec<ScalingCurve>> {
        use std::collections::BTreeMap;
        // Reuse the already-aggregated report cells; the chat:<n> rows are
        // exactly the per-context measurement points.
        let mut by_model: BTreeMap<(String, String), Vec<ReportRow>> = BTreeMap::new();
        for r in self.report_rows()? {
            if r.scenario_id.starts_with("chat:") {
                by_model
                    .entry((r.target_name.clone(), r.model_id.clone()))
                    .or_default()
                    .push(r);
            }
        }
        let mut out = Vec::new();
        for ((target_name, model_id), mut rows) in by_model {
            rows.sort_by_key(|r| r.prompt_size_approx);
            let points: Vec<ScalingPoint> = rows
                .iter()
                .map(|r| ScalingPoint {
                    prompt_size: r.prompt_size_approx,
                    prompt_tokens: r.prompt_tokens,
                    prefill_tps: r.prefill_tps_median,
                    decode_tps: r.decode_tps_median,
                    samples: r.samples,
                })
                .collect();
            // Flatness across the smallest→largest points that both have a
            // decode rate (skips cells where the decode window was too short).
            let with_decode: Vec<&ScalingPoint> =
                points.iter().filter(|p| p.decode_tps.is_some()).collect();
            let decode_flatness = match (with_decode.first(), with_decode.last()) {
                (Some(lo), Some(hi)) if with_decode.len() >= 2 => {
                    match (lo.decode_tps, hi.decode_tps) {
                        (Some(a), Some(b)) if a > 0.0 => Some(b / a),
                        _ => None,
                    }
                }
                _ => None,
            };
            out.push(ScalingCurve {
                target_name,
                model_id,
                git_sha: rows.first().map(|r| r.git_sha.clone()).unwrap_or_default(),
                gpu: rows.iter().find_map(|r| r.gpu.clone()),
                points,
                decode_flatness,
            });
        }
        Ok(out)
    }

    /// Cold-load / model-swap costs (#90): per (target, model) at the latest
    /// build, the median unload→reload latency and the cold first-request
    /// latency after reload (the `scenario_id = "swap"` rows).
    pub fn swap_costs(&self) -> Result<Vec<SwapCost>> {
        use std::collections::BTreeMap;
        let mut stmt = self.conn.prepare(
            "SELECT target_name, model_id, git_sha, gpus_json,
                    swap_unload_ms, swap_load_ms, ttft_s, total_s
             FROM runs
             WHERE ok=1 AND scenario_id='swap'
             ORDER BY target_name, model_id, id",
        )?;
        struct Raw {
            target: String,
            model: String,
            sha: String,
            gpus_json: Option<String>,
            unload_ms: Option<f64>,
            load_ms: Option<f64>,
            ttft_s: Option<f64>,
            total_s: Option<f64>,
        }
        let raws: Vec<Raw> = stmt
            .query_map([], |r| {
                Ok(Raw {
                    target: r.get(0)?,
                    model: r.get(1)?,
                    sha: r.get(2)?,
                    gpus_json: r.get(3)?,
                    unload_ms: r.get::<_, Option<i64>>(4)?.map(|v| v as f64),
                    load_ms: r.get::<_, Option<i64>>(5)?.map(|v| v as f64),
                    ttft_s: r.get(6)?,
                    total_s: r.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut by: BTreeMap<(String, String), Vec<Raw>> = BTreeMap::new();
        for r in raws {
            by.entry((r.target.clone(), r.model.clone()))
                .or_default()
                .push(r);
        }
        let mut out = Vec::new();
        for ((target, model), rows) in by {
            let latest = rows.last().map(|r| r.sha.clone()).unwrap_or_default();
            let cell: Vec<&Raw> = rows.iter().filter(|r| r.sha == latest).collect();
            out.push(SwapCost {
                target_name: target,
                model_id: model,
                git_sha: latest,
                gpu: cell
                    .iter()
                    .find_map(|r| r.gpus_json.as_deref().and_then(gpu_label)),
                unload_ms_median: median(cell.iter().filter_map(|r| r.unload_ms)),
                load_ms_median: median(cell.iter().filter_map(|r| r.load_ms)),
                cold_ttft_s_median: median(cell.iter().filter_map(|r| r.ttft_s)),
                cold_total_s_median: median(cell.iter().filter_map(|r| r.total_s)),
                samples: cell.len(),
            });
        }
        Ok(out)
    }

    // ── Read API surface (consumed by api.rs) ─────────────────────────

    /// Total recorded runs (for `/api/health`).
    pub fn run_count(&self) -> Result<u64> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))?;
        Ok(n as u64)
    }

    /// Distinct hosts / models / scenarios / builds, for populating UI
    /// filters. Builds are ordered chronologically by build timestamp
    /// (falling back to first-seen wall-clock).
    pub fn dimensions(&self) -> Result<Dimensions> {
        let col = |sql: &str| -> Result<Vec<String>> {
            let mut stmt = self.conn.prepare(sql)?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<rusqlite::Result<_>>()?)
        };
        let hosts = col("SELECT DISTINCT target_name FROM runs ORDER BY target_name")?;
        let models = col("SELECT DISTINCT model_id FROM runs ORDER BY model_id")?;
        let scenarios = col("SELECT DISTINCT scenario_id FROM runs ORDER BY scenario_id")?;

        let mut stmt = self.conn.prepare(
            "SELECT git_sha, MAX(build_timestamp), MAX(package_version), MIN(COALESCE(build_timestamp, ts)) AS ord
             FROM runs GROUP BY git_sha ORDER BY ord",
        )?;
        let builds = stmt
            .query_map([], |r| {
                Ok(BuildRef {
                    git_sha: r.get(0)?,
                    build_timestamp: r.get(1)?,
                    package_version: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;

        // host/model → GPU label, taken from each one's most recent run.
        let gpu_map = |group_col: &str| -> Result<std::collections::HashMap<String, String>> {
            let sql = format!(
                "SELECT {group_col}, gpus_json FROM runs \
                 WHERE id IN (SELECT MAX(id) FROM runs GROUP BY {group_col})"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
            })?;
            let mut out = std::collections::HashMap::new();
            for row in rows {
                let (key, gpus) = row?;
                if let Some(label) = gpus.as_deref().and_then(gpu_label) {
                    out.insert(key, label);
                }
            }
            Ok(out)
        };
        let host_gpus = gpu_map("target_name")?;
        let model_gpus = gpu_map("model_id")?;

        Ok(Dimensions {
            hosts,
            models,
            scenarios,
            builds,
            host_gpus,
            model_gpus,
        })
    }

    /// Latest-SHA-per-cell medians (the report table as JSON).
    pub fn summary(&self) -> Result<Vec<ReportRow>> {
        self.report_rows()
    }

    /// Per-build median metrics for one (model, scenario) cell, ordered
    /// chronologically by build — the "over time" series. `host` is
    /// optional: when omitted it resolves to the host with the most recent
    /// run for this (model, scenario). Each model is served by a single
    /// host today, so this yields a coherent single-host series and lets
    /// callers (the public UI) select by model alone.
    pub fn series(
        &self,
        host: Option<&str>,
        model: &str,
        scenario: &str,
    ) -> Result<Vec<SeriesPoint>> {
        let host = match host {
            Some(h) => h.to_string(),
            None => {
                let resolved: Option<String> = self
                    .conn
                    .query_row(
                        "SELECT target_name FROM runs WHERE ok=1 AND model_id=?1 \
                         AND scenario_id=?2 ORDER BY id DESC LIMIT 1",
                        params![model, scenario],
                        |r| r.get(0),
                    )
                    .optional()?;
                match resolved {
                    Some(h) => h,
                    None => return Ok(Vec::new()),
                }
            }
        };
        let mut stmt = self.conn.prepare(
            "SELECT git_sha, build_timestamp, package_version, ttft_s, decode_tps, total_s, ts
             FROM runs
             WHERE ok=1 AND target_name=?1 AND model_id=?2 AND scenario_id=?3
             ORDER BY id",
        )?;
        let raws: Vec<SeriesRaw> = stmt
            .query_map(params![host, model, scenario], |r| {
                Ok(SeriesRaw {
                    git_sha: r.get(0)?,
                    build_timestamp: r.get(1)?,
                    package_version: r.get(2)?,
                    ttft_s: r.get(3)?,
                    decode_tps: r.get(4)?,
                    total_s: r.get(5)?,
                    ts: r.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(aggregate_series(raws))
    }

    /// Raw rows, optionally filtered. For drill-down + programmatic access.
    pub fn runs(&self, f: &RunFilter) -> Result<Vec<RunRow>> {
        let mut sql = String::from(
            "SELECT id, ts, target_name, hostname, git_sha, build_timestamp, package_version,
                    model_id, harness, scenario_id, prompt_size_approx, prompt_tokens_actual,
                    max_tokens, ttft_s, decode_tps, total_s, completion_tokens, ok, error,
                    gpus_json, prefill_ms, decode_ms, prefill_tokens,
                    vram_used_mb, gpu_util_pct, gpu_temp_c,
                    concurrency, ttft_p95_s, queue_wait_ms, rejected,
                    swap_unload_ms, swap_load_ms
             FROM runs",
        );
        let mut conds: Vec<String> = Vec::new();
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        let bind = |col: &str,
                    val: Option<&str>,
                    conds: &mut Vec<String>,
                    args: &mut Vec<Box<dyn rusqlite::ToSql>>| {
            if let Some(v) = val {
                args.push(Box::new(v.to_string()));
                conds.push(format!("{col}=?{}", args.len()));
            }
        };
        bind("target_name", f.host.as_deref(), &mut conds, &mut args);
        bind("model_id", f.model.as_deref(), &mut conds, &mut args);
        bind("scenario_id", f.scenario.as_deref(), &mut conds, &mut args);
        bind("git_sha", f.sha.as_deref(), &mut conds, &mut args);
        if let Some(ok) = f.ok {
            args.push(Box::new(ok as i64));
            conds.push(format!("ok=?{}", args.len()));
        }
        if !conds.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conds.join(" AND "));
        }
        sql.push_str(" ORDER BY id DESC");
        let limit = f.limit.unwrap_or(500).min(5000);
        args.push(Box::new(limit as i64));
        sql.push_str(&format!(" LIMIT ?{}", args.len()));

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(args.iter()), |r| {
                let gpus_json: Option<String> = r.get(19)?;
                Ok(RunRow {
                    id: r.get(0)?,
                    ts: r.get(1)?,
                    host: r.get(2)?,
                    gpu: gpus_json.as_deref().and_then(gpu_label),
                    hostname: r.get(3)?,
                    git_sha: r.get(4)?,
                    build_timestamp: r.get(5)?,
                    package_version: r.get(6)?,
                    model_id: r.get(7)?,
                    harness: r.get(8)?,
                    scenario_id: r.get(9)?,
                    prompt_size_approx: r.get(10)?,
                    prompt_tokens_actual: r.get(11)?,
                    max_tokens: r.get(12)?,
                    ttft_s: r.get(13)?,
                    decode_tps: r.get(14)?,
                    total_s: r.get(15)?,
                    completion_tokens: r.get(16)?,
                    ok: r.get::<_, i64>(17)? != 0,
                    error: r.get(18)?,
                    prefill_ms: r.get(20)?,
                    decode_ms: r.get(21)?,
                    prefill_tokens: r.get(22)?,
                    vram_used_mb: r.get(23)?,
                    gpu_util_pct: r.get(24)?,
                    gpu_temp_c: r.get(25)?,
                    concurrency: r.get(26)?,
                    ttft_p95_s: r.get(27)?,
                    queue_wait_ms: r.get(28)?,
                    rejected: r.get(29)?,
                    swap_unload_ms: r.get(30)?,
                    swap_load_ms: r.get(31)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }
}

// ── Read-API serde types ──────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct Dimensions {
    pub hosts: Vec<String>,
    pub models: Vec<String>,
    pub scenarios: Vec<String>,
    pub builds: Vec<BuildRef>,
    /// host → GPU label (latest run), so the UI can show the GPU as the
    /// resource name instead of the internal hostname.
    pub host_gpus: std::collections::HashMap<String, String>,
    /// model → GPU label (latest run); model maps to one host today.
    pub model_gpus: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BuildRef {
    pub git_sha: String,
    pub build_timestamp: Option<String>,
    pub package_version: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SeriesPoint {
    pub git_sha: String,
    pub build_timestamp: Option<String>,
    pub package_version: Option<String>,
    pub ttft_s_median: Option<f64>,
    pub decode_tps_median: Option<f64>,
    pub total_s_median: Option<f64>,
    pub samples: usize,
}

struct SeriesRaw {
    git_sha: String,
    build_timestamp: Option<String>,
    package_version: Option<String>,
    ttft_s: Option<f64>,
    decode_tps: Option<f64>,
    total_s: Option<f64>,
    ts: String,
}

/// Group id-ordered rows by build SHA, median each metric, and order the
/// resulting points chronologically by build (timestamp, else first ts).
fn aggregate_series(raws: Vec<SeriesRaw>) -> Vec<SeriesPoint> {
    use std::collections::BTreeMap;
    // Preserve first-seen order per sha for the chronological sort key.
    let mut order: Vec<String> = Vec::new();
    let mut groups: BTreeMap<String, Vec<SeriesRaw>> = BTreeMap::new();
    for r in raws {
        if !groups.contains_key(&r.git_sha) {
            order.push(r.git_sha.clone());
        }
        groups.entry(r.git_sha.clone()).or_default().push(r);
    }
    let mut points: Vec<(String, SeriesPoint)> = order
        .into_iter()
        .map(|sha| {
            let rows = &groups[&sha];
            let sort_key = rows
                .iter()
                .map(|r| r.build_timestamp.clone().unwrap_or_else(|| r.ts.clone()))
                .min()
                .unwrap_or_default();
            let point = SeriesPoint {
                git_sha: sha,
                build_timestamp: rows.iter().find_map(|r| r.build_timestamp.clone()),
                package_version: rows.iter().find_map(|r| r.package_version.clone()),
                ttft_s_median: median(rows.iter().filter_map(|r| r.ttft_s)),
                decode_tps_median: median(rows.iter().filter_map(|r| r.decode_tps)),
                total_s_median: median(rows.iter().filter_map(|r| r.total_s)),
                samples: rows.len(),
            };
            (sort_key, point)
        })
        .collect();
    points.sort_by(|a, b| a.0.cmp(&b.0));
    points.into_iter().map(|(_, p)| p).collect()
}

#[derive(Debug, Clone, Default)]
pub struct RunFilter {
    pub host: Option<String>,
    pub model: Option<String>,
    pub scenario: Option<String>,
    pub sha: Option<String>,
    pub ok: Option<bool>,
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RunRow {
    pub id: i64,
    pub ts: String,
    pub host: String,
    /// Public-facing resource name (the host's GPU(s)), e.g. "RTX 4090".
    pub gpu: Option<String>,
    pub hostname: Option<String>,
    pub git_sha: String,
    pub build_timestamp: Option<String>,
    pub package_version: String,
    pub model_id: String,
    pub harness: String,
    pub scenario_id: String,
    pub prompt_size_approx: u32,
    pub prompt_tokens_actual: Option<u64>,
    pub max_tokens: u64,
    pub ttft_s: Option<f64>,
    pub decode_tps: Option<f64>,
    pub total_s: Option<f64>,
    pub completion_tokens: Option<u64>,
    pub prefill_ms: Option<u64>,
    pub decode_ms: Option<u64>,
    pub prefill_tokens: Option<u64>,
    pub vram_used_mb: Option<u64>,
    pub gpu_util_pct: Option<u64>,
    pub gpu_temp_c: Option<u64>,
    pub concurrency: Option<u64>,
    pub ttft_p95_s: Option<f64>,
    pub queue_wait_ms: Option<f64>,
    pub rejected: Option<u64>,
    pub swap_unload_ms: Option<u64>,
    pub swap_load_ms: Option<u64>,
    pub ok: bool,
    pub error: Option<String>,
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
    gpus_json: Option<String>,
    prefill_ms: Option<u64>,
    decode_ms: Option<u64>,
    prefill_tokens: Option<u64>,
    vram_used_mb: Option<u64>,
    gpu_util_pct: Option<u64>,
    gpu_temp_c: Option<u64>,
    concurrency: Option<u64>,
    ttft_p95_s: Option<f64>,
    queue_wait_ms: Option<f64>,
    rejected: Option<u64>,
}

/// An aggregated cell ready for the report table.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
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
    /// Latency tail percentiles — where batch-1 pain actually shows up, and
    /// invisible behind a bare median. p95/p99 nearest-rank; with few
    /// samples they collapse toward the max (honest, not interpolated).
    pub ttft_s_p95: Option<f64>,
    pub ttft_s_p99: Option<f64>,
    pub total_s_p95: Option<f64>,
    pub total_s_p99: Option<f64>,
    /// Server-measured prefill/decode split (#85). `prefill_tps_median` is
    /// the true prompt-encoding rate (prefill_tokens / prefill_ms),
    /// complementing `decode_tps_median` (the generation rate).
    pub prefill_ms_median: Option<f64>,
    pub decode_ms_median: Option<f64>,
    pub prefill_tps_median: Option<f64>,
    /// GPU telemetry sampled from /health around the run (#87). `vram_used_mb`
    /// is the node sum; `vram_total_mb` (from discovery) lets the report show
    /// real headroom — the "2/3 used" hunch as a number. util/temp are the
    /// hottest device. All `None` for non-neuron targets.
    pub vram_used_mb_median: Option<f64>,
    pub vram_total_mb: Option<u64>,
    pub gpu_util_pct_median: Option<f64>,
    pub gpu_temp_c_median: Option<f64>,
    /// Concurrency / agentic-load burst metrics (#89). `concurrency` is the
    /// burst width (constant per cell). `ttft_p95_load_s` is the within-burst
    /// TTFT tail; `queue_wait_ms_median` the admission wait; `rejected_median`
    /// the per-burst shed count. All `None` for non-concurrency scenarios.
    pub concurrency: Option<u64>,
    pub ttft_p95_load_s: Option<f64>,
    pub queue_wait_ms_median: Option<f64>,
    pub rejected_median: Option<f64>,
    pub samples: usize,
    /// Public-facing resource name (the host's GPU(s)), e.g. "2× RTX 5090".
    pub gpu: Option<String>,
}

/// One context-length scaling curve for a (target, model) at its latest
/// build — the points ordered by prompt size, plus the decode-flatness
/// summary (#88).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScalingCurve {
    pub target_name: String,
    pub model_id: String,
    pub git_sha: String,
    pub gpu: Option<String>,
    pub points: Vec<ScalingPoint>,
    /// decode tok/s at the largest context ÷ at the smallest. ~1.0 = flat
    /// (GDN O(1) decode); <1 quantifies the drop-off. `None` with <2 points.
    pub decode_flatness: Option<f64>,
}

/// Cold-load / model-swap cost for a (target, model) at its latest build
/// (#90): the reload latency and the cold first-request after it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SwapCost {
    pub target_name: String,
    pub model_id: String,
    pub git_sha: String,
    pub gpu: Option<String>,
    /// Time to free the model (`POST /models/unload`), ms.
    pub unload_ms_median: Option<f64>,
    /// Time to reload it (`POST /models/load`, synchronous), ms — the
    /// headline swap cost feeding the vision cold-swap policy (F4e).
    pub load_ms_median: Option<f64>,
    /// TTFT of the first request after reload (cold caches), seconds.
    pub cold_ttft_s_median: Option<f64>,
    /// Total wall-clock of that cold first request, seconds.
    pub cold_total_s_median: Option<f64>,
    pub samples: usize,
}

/// One point on a [`ScalingCurve`]: the throughput at a given context size.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScalingPoint {
    pub prompt_size: u32,
    pub prompt_tokens: Option<u64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
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
        // Per-row prefill tok/s, derived from the server-measured split.
        let prefill_tps = |r: &&RawRow| match (r.prefill_tokens, r.prefill_ms) {
            (Some(tok), Some(ms)) if ms > 0 => Some(tok as f64 * 1000.0 / ms as f64),
            _ => None,
        };
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
            ttft_s_p95: percentile(cell.iter().filter_map(|r| r.ttft_s), 95.0),
            ttft_s_p99: percentile(cell.iter().filter_map(|r| r.ttft_s), 99.0),
            total_s_p95: percentile(cell.iter().filter_map(|r| r.total_s), 95.0),
            total_s_p99: percentile(cell.iter().filter_map(|r| r.total_s), 99.0),
            prefill_ms_median: median(cell.iter().filter_map(|r| r.prefill_ms.map(|m| m as f64))),
            decode_ms_median: median(cell.iter().filter_map(|r| r.decode_ms.map(|m| m as f64))),
            prefill_tps_median: median(cell.iter().filter_map(prefill_tps)),
            vram_used_mb_median: median(
                cell.iter().filter_map(|r| r.vram_used_mb.map(|v| v as f64)),
            ),
            vram_total_mb: cell
                .iter()
                .find_map(|r| r.gpus_json.as_deref().and_then(gpu_total_vram_mb)),
            gpu_util_pct_median: median(
                cell.iter().filter_map(|r| r.gpu_util_pct.map(|v| v as f64)),
            ),
            gpu_temp_c_median: median(cell.iter().filter_map(|r| r.gpu_temp_c.map(|v| v as f64))),
            concurrency: cell.iter().find_map(|r| r.concurrency),
            ttft_p95_load_s: median(cell.iter().filter_map(|r| r.ttft_p95_s)),
            queue_wait_ms_median: median(cell.iter().filter_map(|r| r.queue_wait_ms)),
            rejected_median: median(cell.iter().filter_map(|r| r.rejected.map(|v| v as f64))),
            samples: cell.len(),
            gpu: cell
                .iter()
                .find_map(|r| r.gpus_json.as_deref().and_then(gpu_label)),
        });
    }
    out
}

/// Node total VRAM in MB, summed across the devices in a run's stored
/// `gpus_json` (the discovery `DeviceInfo` list, each with `vram_total_mb`).
/// Pairs with the sampled `vram_used_mb` to report real headroom (#87).
/// `None` when empty/absent or no device declares a total.
fn gpu_total_vram_mb(gpus_json: &str) -> Option<u64> {
    let devices: Vec<serde_json::Value> = serde_json::from_str(gpus_json).ok()?;
    let total: u64 = devices
        .iter()
        .filter_map(|d| d.get("vram_total_mb").and_then(|v| v.as_u64()))
        .sum();
    (total > 0).then_some(total)
}

/// Compact GPU label from a run's stored `gpus_json` (the discovery device
/// list) — e.g. "2× RTX 5090", "RTX 4090". `None` when empty/absent. Used
/// as the public-facing resource name in place of internal hostnames.
fn gpu_label(gpus_json: &str) -> Option<String> {
    let devices: Vec<serde_json::Value> = serde_json::from_str(gpus_json).ok()?;
    if devices.is_empty() {
        return None;
    }
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for d in &devices {
        let name = d.get("name").and_then(|v| v.as_str()).unwrap_or("GPU");
        let short = name
            .trim_start_matches("NVIDIA GeForce ")
            .trim_start_matches("NVIDIA ")
            .to_string();
        if !counts.contains_key(&short) {
            order.push(short.clone());
        }
        *counts.entry(short).or_insert(0) += 1;
    }
    Some(
        order
            .iter()
            .map(|n| {
                let c = counts[n];
                if c > 1 {
                    format!("{c}× {n}")
                } else {
                    n.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" + "),
    )
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

/// Nearest-rank percentile (`p` in 0..=100). Chosen over interpolation
/// because bench cells hold only a handful of samples: with n=5, p95/p99
/// resolve to the max, which honestly says "this is the worst we saw"
/// rather than inventing a value between samples we never observed.
fn percentile(values: impl Iterator<Item = f64>, p: f64) -> Option<f64> {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // rank = ceil(p/100 * n), clamped to [1, n]; index is rank-1.
    let rank = (p / 100.0 * v.len() as f64).ceil() as usize;
    let idx = rank.clamp(1, v.len()) - 1;
    Some(v[idx])
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
            prefill_ms: Some(200),
            decode_ms: Some(1000),
            prefill_tokens: Some(130),
            vram_used_mb: Some(42000),
            gpu_util_pct: Some(88),
            gpu_temp_c: Some(64),
            concurrency: None,
            ttft_p95_s: None,
            queue_wait_ms: None,
            rejected: None,
            swap_unload_ms: None,
            swap_load_ms: None,
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

    #[test]
    fn report_surfaces_percentiles_and_prefill_split() {
        let s = Store::open_in_memory().unwrap();
        // Five samples on one cell with spread TTFT so percentiles differ
        // from the median, plus a server-measured prefill/decode split.
        for (i, ttft) in [0.10, 0.12, 0.14, 0.16, 0.50].iter().enumerate() {
            let mut r = rec("beast", "sha", "m", "chat:128", true);
            r.ttft_s = Some(*ttft);
            r.total_s = Some(ttft + 1.0);
            r.prefill_ms = Some(200 + i as u64);
            r.prefill_tokens = Some(400);
            s.insert_run(&r).unwrap();
        }
        let rows = s.report_rows().unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.samples, 5);
        // p50 is the middle value; p95/p99 (nearest-rank, n=5) hit the max.
        assert!((row.ttft_s_median.unwrap() - 0.14).abs() < 1e-9);
        assert!((row.ttft_s_p95.unwrap() - 0.50).abs() < 1e-9);
        assert!((row.ttft_s_p99.unwrap() - 0.50).abs() < 1e-9);
        // prefill tok/s = 400 tok / ~0.2 s ≈ 2000 tok/s.
        assert!(row.prefill_tps_median.unwrap() > 1900.0);
        assert!(row.prefill_ms_median.is_some());
    }

    #[test]
    fn report_surfaces_vram_and_gpu_telemetry() {
        let s = Store::open_in_memory().unwrap();
        let mut r = rec("beast", "sha", "m", "chat:128", true);
        // Node total VRAM from discovery devices → headroom denominator.
        r.gpus_json =
            Some(r#"[{"name":"RTX 5090","vram_total_mb":32000},{"name":"RTX 5090","vram_total_mb":32000}]"#.into());
        r.vram_used_mb = Some(42000);
        r.gpu_util_pct = Some(90);
        r.gpu_temp_c = Some(66);
        s.insert_run(&r).unwrap();
        let rows = s.report_rows().unwrap();
        let row = &rows[0];
        assert_eq!(row.vram_used_mb_median, Some(42000.0));
        assert_eq!(row.vram_total_mb, Some(64000)); // 2× 32000
        assert_eq!(row.gpu_util_pct_median, Some(90.0));
        assert_eq!(row.gpu_temp_c_median, Some(66.0));
    }

    #[test]
    fn report_surfaces_concurrency_burst_metrics() {
        let s = Store::open_in_memory().unwrap();
        // Two concurrency:8 burst-runs with shed load and a queue-wait tail.
        for (qw, rej) in [(120.0, 1u32), (180.0, 3u32)] {
            let mut r = rec("beast", "sha", "m", "concurrency:8", true);
            r.concurrency = Some(8);
            r.ttft_p95_s = Some(0.9);
            r.queue_wait_ms = Some(qw);
            r.rejected = Some(rej);
            s.insert_run(&r).unwrap();
        }
        let row = s
            .report_rows()
            .unwrap()
            .into_iter()
            .find(|r| r.scenario_id == "concurrency:8")
            .unwrap();
        assert_eq!(row.concurrency, Some(8));
        assert_eq!(row.queue_wait_ms_median, Some(150.0)); // median(120,180)
        assert_eq!(row.rejected_median, Some(2.0)); // median(1,3)
        assert_eq!(row.ttft_p95_load_s, Some(0.9));
    }

    #[test]
    fn swap_costs_pivots_swap_rows() {
        let s = Store::open_in_memory().unwrap();
        for (unload, load) in [(300u64, 24000u64), (340, 26000)] {
            let mut r = rec("beast", "sha", "m", "swap", true);
            r.swap_unload_ms = Some(unload);
            r.swap_load_ms = Some(load);
            r.ttft_s = Some(2.5); // cold first-request
            r.total_s = Some(5.0);
            s.insert_run(&r).unwrap();
        }
        // A non-swap row must be ignored by swap_costs.
        s.insert_run(&rec("beast", "sha", "m", "chat:128", true))
            .unwrap();
        let costs = s.swap_costs().unwrap();
        assert_eq!(costs.len(), 1);
        let c = &costs[0];
        assert_eq!(c.unload_ms_median, Some(320.0)); // median(300,340)
        assert_eq!(c.load_ms_median, Some(25000.0)); // median(24000,26000)
        assert_eq!(c.cold_ttft_s_median, Some(2.5));
        assert_eq!(c.samples, 2);
    }

    #[test]
    fn scaling_pivots_chat_cells_and_computes_flatness() {
        let s = Store::open_in_memory().unwrap();
        // Two context points for one model: decode tok/s 50 @128, 45 @4096.
        let mut small = rec("beast", "sha", "m", "chat:128", true);
        small.prompt_size_approx = 128;
        small.decode_tps = Some(50.0);
        small.prefill_ms = Some(100);
        small.prefill_tokens = Some(128);
        s.insert_run(&small).unwrap();
        let mut big = rec("beast", "sha", "m", "chat:4096", true);
        big.prompt_size_approx = 4096;
        big.decode_tps = Some(45.0);
        big.prefill_ms = Some(1000);
        big.prefill_tokens = Some(4096);
        s.insert_run(&big).unwrap();
        // A concurrency cell must NOT leak into the scaling curve.
        let mut conc = rec("beast", "sha", "m", "concurrency:8", true);
        conc.concurrency = Some(8);
        s.insert_run(&conc).unwrap();

        let curves = s.scaling().unwrap();
        assert_eq!(curves.len(), 1);
        let c = &curves[0];
        assert_eq!(c.points.len(), 2); // only the two chat:<n> points
        assert_eq!(c.points[0].prompt_size, 128); // ordered ascending
        assert_eq!(c.points[1].prompt_size, 4096);
        // flatness = decode@largest / decode@smallest = 45/50 = 0.9
        assert!((c.decode_flatness.unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn gpu_total_vram_sums_devices() {
        let j = r#"[{"vram_total_mb":32000},{"vram_total_mb":32000}]"#;
        assert_eq!(gpu_total_vram_mb(j), Some(64000));
        assert_eq!(gpu_total_vram_mb("[]"), None);
        assert_eq!(gpu_total_vram_mb("not json"), None);
    }

    #[test]
    fn percentile_nearest_rank() {
        let vals = || [1.0, 2.0, 3.0, 4.0, 5.0].into_iter();
        assert_eq!(percentile(vals(), 50.0), Some(3.0));
        assert_eq!(percentile(vals(), 95.0), Some(5.0));
        assert_eq!(percentile(vals(), 99.0), Some(5.0));
        assert_eq!(percentile(std::iter::empty(), 95.0), None);
    }

    #[test]
    fn migration_is_idempotent_and_backfills() {
        // A DB whose `runs` table predates the prefill columns: create the
        // pre-#85 shape, insert a row, then run ensure_columns twice.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, ttft_s REAL);
             INSERT INTO runs (ttft_s) VALUES (0.1);",
        )
        .unwrap();
        for _ in 0..2 {
            Store::ensure_columns(
                &conn,
                "runs",
                &[("prefill_ms", "INTEGER"), ("decode_ms", "INTEGER")],
            )
            .unwrap();
        }
        // Columns now exist and the old row reads them back as NULL.
        let got: Option<i64> = conn
            .query_row("SELECT prefill_ms FROM runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn gpu_label_formats() {
        let two = r#"[{"name":"NVIDIA GeForce RTX 5090"},{"name":"NVIDIA GeForce RTX 5090"}]"#;
        assert_eq!(gpu_label(two).as_deref(), Some("2× RTX 5090"));
        let one = r#"[{"name":"NVIDIA GeForce RTX 4090"}]"#;
        assert_eq!(gpu_label(one).as_deref(), Some("RTX 4090"));
        let dc = r#"[{"name":"NVIDIA H100"}]"#;
        assert_eq!(gpu_label(dc).as_deref(), Some("H100"));
        assert_eq!(gpu_label("[]"), None);
    }
}
