//! Render the SQLite store as a results table — the automated
//! replacement for hand-editing `doc/benchmarks.md`. Columns match that
//! doc: engine, model, prompt tok, TTFT (s), decode tok/s, total (s),
//! plus the build SHA each cell was measured against.

use crate::store::ReportRow;
use anyhow::Result;

pub fn render_markdown(rows: &[ReportRow]) -> String {
    let mut out = String::new();
    out.push_str(
        "| engine | model | prompt tok | TTFT (s) | decode tok/s | total (s) | build | n |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---|---:|\n");
    for r in rows {
        let ptok = r
            .prompt_tokens
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("~{}", r.prompt_size_approx));
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | `{}` | {} |\n",
            r.target_name,
            r.model_id,
            ptok,
            fmt_opt(r.ttft_s_median, 3),
            fmt_opt(r.decode_tps_median, 1),
            fmt_opt(r.total_s_median, 3),
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
                "decode_tps_median": r.decode_tps_median,
                "total_s_median": r.total_s_median,
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
            samples: 5,
            gpu: Some("2× RTX 5090".into()),
        }];
        let md = render_markdown(&rows);
        assert!(md.contains("| engine |"));
        assert!(md.contains("beast"));
        assert!(md.contains("`30d50d6`"));
        assert!(md.contains("0.123"));
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
            samples: 1,
            gpu: None,
        }];
        let md = render_markdown(&rows);
        assert!(md.contains("~128"));
        assert!(md.contains("—"));
    }
}
