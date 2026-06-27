//! Render the SQLite store as a results table — the automated
//! replacement for hand-editing `doc/benchmarks.md`. Columns match that
//! doc: engine, model, prompt tok, TTFT (s), decode tok/s, total (s),
//! plus the build SHA each cell was measured against.

use crate::store::ReportRow;
use anyhow::Result;

pub fn render_markdown(rows: &[ReportRow]) -> String {
    let mut out = String::new();
    out.push_str(
        "| engine | model | prompt tok | prefill tok/s | TTFT (s) | TTFT p95 | \
         decode tok/s | total (s) | total p95 | VRAM (GB) | conc | queue ms | rej | build | n |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n");
    for r in rows {
        let ptok = r
            .prompt_tokens
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("~{}", r.prompt_size_approx));
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` | {} |\n",
            r.target_name,
            r.model_id,
            ptok,
            fmt_opt(r.prefill_tps_median, 1),
            fmt_opt(r.ttft_s_median, 3),
            fmt_opt(r.ttft_s_p95, 3),
            fmt_opt(r.decode_tps_median, 1),
            fmt_opt(r.total_s_median, 3),
            fmt_opt(r.total_s_p95, 3),
            fmt_vram(r.vram_used_mb_median, r.vram_total_mb),
            fmt_u64(r.concurrency),
            fmt_opt(r.queue_wait_ms_median, 0),
            fmt_opt(r.rejected_median, 0),
            r.git_sha,
            r.samples,
        ));
    }
    out
}

pub fn render_json(rows: &[ReportRow]) -> Result<String> {
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "engine": r.target_name,
                "model": r.model_id,
                "scenario": r.scenario_id,
                "prompt_size_approx": r.prompt_size_approx,
                "prompt_tokens": r.prompt_tokens,
                "ttft_s_median": r.ttft_s_median,
                "ttft_s_p95": r.ttft_s_p95,
                "ttft_s_p99": r.ttft_s_p99,
                "decode_tps_median": r.decode_tps_median,
                "total_s_median": r.total_s_median,
                "total_s_p95": r.total_s_p95,
                "total_s_p99": r.total_s_p99,
                "prefill_ms_median": r.prefill_ms_median,
                "decode_ms_median": r.decode_ms_median,
                "prefill_tps_median": r.prefill_tps_median,
                "vram_used_mb_median": r.vram_used_mb_median,
                "vram_total_mb": r.vram_total_mb,
                "gpu_util_pct_median": r.gpu_util_pct_median,
                "gpu_temp_c_median": r.gpu_temp_c_median,
                "concurrency": r.concurrency,
                "ttft_p95_load_s": r.ttft_p95_load_s,
                "queue_wait_ms_median": r.queue_wait_ms_median,
                "rejected_median": r.rejected_median,
                "git_sha": r.git_sha,
                "samples": r.samples,
                "gpu": r.gpu,
            })
        })
        .collect();
    Ok(serde_json::to_string_pretty(&arr)?)
}

fn fmt_opt(v: Option<f64>, places: usize) -> String {
    match v {
        Some(x) => format!("{x:.places$}"),
        None => "—".to_string(),
    }
}

/// Integer cell (concurrency width); `—` when unset (non-concurrency rows).
fn fmt_u64(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "—".to_string(),
    }
}

/// `used/total` in GB (e.g. `42.0/64.0`) — the headroom-at-a-glance cell.
/// `used` alone if the node total is unknown; `—` if no telemetry.
fn fmt_vram(used_mb: Option<f64>, total_mb: Option<u64>) -> String {
    match (used_mb, total_mb) {
        (Some(u), Some(t)) => format!("{:.1}/{:.1}", u / 1024.0, t as f64 / 1024.0),
        (Some(u), None) => format!("{:.1}", u / 1024.0),
        _ => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_has_header_and_row() {
        let rows = vec![ReportRow {
            target_name: "beast".into(),
            model_id: "Qwen/Qwen3.6-27B".into(),
            scenario_id: "chat:128".into(),
            prompt_size_approx: 128,
            git_sha: "30d50d6".into(),
            prompt_tokens: Some(130),
            ttft_s_median: Some(0.123),
            decode_tps_median: Some(45.6),
            total_s_median: Some(1.234),
            ttft_s_p95: Some(0.222),
            ttft_s_p99: Some(0.250),
            total_s_p95: Some(1.5),
            total_s_p99: Some(1.6),
            prefill_ms_median: Some(120.0),
            decode_ms_median: Some(1100.0),
            prefill_tps_median: Some(1066.7),
            vram_used_mb_median: Some(43008.0),
            vram_total_mb: Some(65536),
            gpu_util_pct_median: Some(89.0),
            gpu_temp_c_median: Some(64.0),
            concurrency: None,
            ttft_p95_load_s: None,
            queue_wait_ms_median: None,
            rejected_median: None,
            samples: 5,
            gpu: Some("2× RTX 5090".into()),
        }];
        let md = render_markdown(&rows);
        assert!(md.contains("| engine |"));
        assert!(md.contains("prefill tok/s"));
        assert!(md.contains("VRAM (GB)"));
        assert!(md.contains("conc"));
        assert!(md.contains("beast"));
        assert!(md.contains("`30d50d6`"));
        assert!(md.contains("0.123"));
        // p95 column rendered.
        assert!(md.contains("0.222"));
        // VRAM used/total in GB (43008/65536 MiB → 42.0/64.0).
        assert!(md.contains("42.0/64.0"));
    }

    #[test]
    fn missing_decode_renders_dash() {
        let rows = vec![ReportRow {
            target_name: "benjy".into(),
            model_id: "m".into(),
            scenario_id: "chat:128".into(),
            prompt_size_approx: 128,
            git_sha: "abc".into(),
            prompt_tokens: None,
            ttft_s_median: Some(0.1),
            decode_tps_median: None,
            total_s_median: Some(0.5),
            ttft_s_p95: Some(0.1),
            ttft_s_p99: Some(0.1),
            total_s_p95: Some(0.5),
            total_s_p99: Some(0.5),
            prefill_ms_median: None,
            decode_ms_median: None,
            prefill_tps_median: None,
            vram_used_mb_median: None,
            vram_total_mb: None,
            gpu_util_pct_median: None,
            gpu_temp_c_median: None,
            concurrency: None,
            ttft_p95_load_s: None,
            queue_wait_ms_median: None,
            rejected_median: None,
            samples: 1,
            gpu: None,
        }];
        let md = render_markdown(&rows);
        assert!(md.contains("~128"));
        assert!(md.contains("—"));
    }
}
