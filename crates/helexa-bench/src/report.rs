//! Render the SQLite store as a results table — the automated
//! replacement for hand-editing `doc/benchmarks.md`. Columns match that
//! doc: engine, model, prompt tok, TTFT (s), decode tok/s, total (s),
//! plus the build SHA each cell was measured against.

use crate::store::{CapabilityRun, ConcurrencyCurve, ReportRow, ScalingCurve, SwapCost};
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

/// Context-length scaling view (#88): one block per (target, model) with
/// prefill & decode tok/s vs context, then the decode-flatness verdict.
pub fn render_scaling_markdown(curves: &[ScalingCurve]) -> String {
    let mut out = String::new();
    for c in curves {
        let gpu = c.gpu.as_deref().unwrap_or("");
        out.push_str(&format!(
            "### {} · {}  (`{}`{})\n\n",
            c.target_name,
            c.model_id,
            c.git_sha,
            if gpu.is_empty() {
                String::new()
            } else {
                format!(", {gpu}")
            },
        ));
        out.push_str("| ctx tok | prefill tok/s | decode tok/s | n |\n");
        out.push_str("|---:|---:|---:|---:|\n");
        for p in &c.points {
            let ctx = p
                .prompt_tokens
                .map(|t| t.to_string())
                .unwrap_or_else(|| format!("~{}", p.prompt_size));
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                ctx,
                fmt_opt(p.prefill_tps, 1),
                fmt_opt(p.decode_tps, 1),
                p.samples,
            ));
        }
        match c.decode_flatness {
            Some(f) => out.push_str(&format!(
                "\ndecode flatness: {f:.2} — decode tok/s {} across the context range \
                 ({})\n\n",
                if f >= 0.9 {
                    "holds"
                } else if f >= 0.7 {
                    "softens"
                } else {
                    "drops sharply"
                },
                if f >= 0.9 {
                    "Gated-DeltaNet O(1) decode confirmed"
                } else {
                    "investigate where it breaks"
                },
            )),
            None => out.push_str("\ndecode flatness: — (need ≥2 context points)\n\n"),
        }
    }
    out
}

pub fn render_scaling_json(curves: &[ScalingCurve]) -> Result<String> {
    Ok(serde_json::to_string_pretty(curves)?)
}

/// Concurrency-sweep view (#137): one block per (target, model) with the
/// throughput / latency-tail / shedding curve across burst widths, then the
/// knee — the max sustainable concurrency, the data-backed `max_in_flight`.
pub fn render_concurrency_markdown(curves: &[ConcurrencyCurve]) -> String {
    let mut out = String::new();
    for c in curves {
        let gpu = c.gpu.as_deref().unwrap_or("");
        out.push_str(&format!(
            "### {} · {}  (`{}`{})\n\n",
            c.target_name,
            c.model_id,
            c.git_sha,
            if gpu.is_empty() {
                String::new()
            } else {
                format!(", {gpu}")
            },
        ));
        out.push_str("| N | decode tok/s | p95 TTFT (s) | queue wait (ms) | reject % | n |\n");
        out.push_str("|---:|---:|---:|---:|---:|---:|\n");
        for p in &c.points {
            let reject = p
                .reject_rate
                .map(|r| format!("{:.0}%", r * 100.0))
                .unwrap_or_else(|| "—".into());
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                p.concurrency,
                fmt_opt(p.decode_tps, 1),
                fmt_opt(p.ttft_p95_s, 2),
                fmt_opt(p.queue_wait_ms, 0),
                reject,
                p.samples,
            ));
        }
        match c.knee_concurrency {
            Some(k) => out.push_str(&format!(
                "\nmax sustainable concurrency: **{k}** \
                 (no shedding, p95 TTFT within 2× of the lightest-load baseline)\n\n",
            )),
            None => out.push_str(
                "\nmax sustainable concurrency: — (sheds or breaks even at the lightest level)\n\n",
            ),
        }
    }
    out
}

pub fn render_concurrency_json(curves: &[ConcurrencyCurve]) -> Result<String> {
    Ok(serde_json::to_string_pretty(curves)?)
}

/// Cold-load / model-swap cost view (#90): reload latency + cold
/// first-request per model.
pub fn render_swap_markdown(costs: &[SwapCost]) -> String {
    let mut out = String::new();
    out.push_str(
        "| engine | model | unload (s) | reload (s) | cold TTFT (s) | cold total (s) | build | n |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---|---:|\n");
    for c in costs {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | `{}` | {} |\n",
            c.target_name,
            c.model_id,
            fmt_ms_as_s(c.unload_ms_median),
            fmt_ms_as_s(c.load_ms_median),
            fmt_opt(c.cold_ttft_s_median, 3),
            fmt_opt(c.cold_total_s_median, 3),
            c.git_sha,
            c.samples,
        ));
    }
    out
}

pub fn render_swap_json(costs: &[SwapCost]) -> Result<String> {
    Ok(serde_json::to_string_pretty(costs)?)
}

/// Capability-probe view (#91): per (model, probe) the median quality score
/// (the A/B number), then each run's id, score, and an artifact snippet so
/// unscored runs can be located and scored (`helexa-bench score --id …`).
pub fn render_capability_markdown(runs: &[CapabilityRun]) -> String {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(String, String, String), Vec<&CapabilityRun>> = BTreeMap::new();
    for r in runs {
        groups
            .entry((
                r.target_name.clone(),
                r.model_id.clone(),
                r.scenario_id.clone(),
            ))
            .or_default()
            .push(r);
    }
    let mut out = String::new();
    for ((target, model, scenario), rs) in groups {
        let scores: Vec<f64> = rs.iter().filter_map(|r| r.quality_score).collect();
        let median = median_slice(&scores);
        out.push_str(&format!(
            "### {target} · {model} · {scenario} — median score {} ({}/{} scored)\n\n",
            median
                .map(|m| format!("{m:.1}"))
                .unwrap_or_else(|| "—".into()),
            scores.len(),
            rs.len(),
        ));
        out.push_str("| run | score | scorer | build | artifact (snippet) |\n");
        out.push_str("|---:|---:|---|---|---|\n");
        for r in rs {
            out.push_str(&format!(
                "| {} | {} | {} | `{}` | {} |\n",
                r.id,
                r.quality_score
                    .map(|s| format!("{s:.1}"))
                    .unwrap_or_else(|| "—".into()),
                r.scorer.as_deref().unwrap_or("—"),
                r.git_sha,
                snippet(r.artifact.as_deref()),
            ));
        }
        out.push('\n');
    }
    out
}

pub fn render_capability_json(runs: &[CapabilityRun]) -> Result<String> {
    Ok(serde_json::to_string_pretty(runs)?)
}

/// First ~80 chars of an artifact on one line, for the table cell.
fn snippet(artifact: Option<&str>) -> String {
    match artifact {
        Some(a) => {
            let one_line: String = a.split_whitespace().collect::<Vec<_>>().join(" ");
            let trimmed: String = one_line.chars().take(80).collect();
            if one_line.chars().count() > 80 {
                format!("{trimmed}…")
            } else {
                trimmed
            }
        }
        None => "—".to_string(),
    }
}

fn median_slice(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = (s.len() - 1) / 2;
    let hi = s.len() / 2;
    Some((s[lo] + s[hi]) / 2.0)
}

/// Milliseconds rendered as seconds (reload costs read naturally in s).
fn fmt_ms_as_s(ms: Option<f64>) -> String {
    match ms {
        Some(x) => format!("{:.2}", x / 1000.0),
        None => "—".to_string(),
    }
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
    use crate::store::{ScalingCurve, ScalingPoint};

    #[test]
    fn capability_markdown_groups_with_median_and_snippet() {
        let runs = vec![
            CapabilityRun {
                id: 7,
                ts: "t".into(),
                target_name: "beast".into(),
                model_id: "m".into(),
                scenario_id: "capability:plan".into(),
                git_sha: "abc".into(),
                quality_score: Some(8.0),
                scorer: Some("manual".into()),
                artifact: Some("A detailed plan with trade-offs and sequencing.".into()),
            },
            CapabilityRun {
                id: 8,
                ts: "t".into(),
                target_name: "beast".into(),
                model_id: "m".into(),
                scenario_id: "capability:plan".into(),
                git_sha: "abc".into(),
                quality_score: Some(6.0),
                scorer: Some("manual".into()),
                artifact: Some("Shorter plan.".into()),
            },
        ];
        let md = render_capability_markdown(&runs);
        assert!(md.contains("capability:plan"));
        assert!(md.contains("median score 7.0")); // median(8,6)
        assert!(md.contains("trade-offs"));
        assert!(md.contains("| 7 |"));
    }

    #[test]
    fn swap_markdown_renders_reload_and_cold_costs() {
        let costs = vec![SwapCost {
            target_name: "beast".into(),
            model_id: "Qwen/Qwen3.6-27B".into(),
            git_sha: "abc1234".into(),
            gpu: Some("2× RTX 5090".into()),
            unload_ms_median: Some(320.0),
            load_ms_median: Some(25000.0),
            cold_ttft_s_median: Some(2.5),
            cold_total_s_median: Some(5.0),
            samples: 3,
        }];
        let md = render_swap_markdown(&costs);
        assert!(md.contains("reload (s)"));
        assert!(md.contains("beast"));
        assert!(md.contains("25.00")); // 25000 ms → 25.00 s
        assert!(md.contains("2.500"));
    }

    #[test]
    fn scaling_markdown_renders_curve_and_flatness() {
        let curves = vec![ScalingCurve {
            target_name: "beast".into(),
            model_id: "Qwen/Qwen3.6-27B".into(),
            git_sha: "abc1234".into(),
            gpu: Some("2× RTX 5090".into()),
            points: vec![
                ScalingPoint {
                    prompt_size: 128,
                    prompt_tokens: Some(130),
                    prefill_tps: Some(900.0),
                    decode_tps: Some(50.0),
                    samples: 5,
                },
                ScalingPoint {
                    prompt_size: 4096,
                    prompt_tokens: Some(4100),
                    prefill_tps: Some(2800.0),
                    decode_tps: Some(48.0),
                    samples: 5,
                },
            ],
            decode_flatness: Some(0.96),
        }];
        let md = render_scaling_markdown(&curves);
        assert!(md.contains("### beast · Qwen/Qwen3.6-27B"));
        assert!(md.contains("ctx tok"));
        assert!(md.contains("decode flatness: 0.96"));
        assert!(md.contains("holds"));
    }

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
