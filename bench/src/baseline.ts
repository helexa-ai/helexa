// Pre-helexa-bench baseline, transcribed verbatim from doc/benchmarks.md.
//
// IMPORTANT — different measurement regime. These were measured by
// script/bench.py *through the cortex gateway* (so TTFT/total include a
// proxy hop), reported as medians only, before helexa-bench existed.
// helexa-bench measures each neuron *directly*. So these points are an
// honest historical anchor, NOT apples-to-apples with the live series —
// the Trends view renders them dashed + labelled, never merged into the
// live line.
//
// Host is inferred from the model via the doc's Fleet table
// (beast=27B, benjy=8B, quadbrat=1.7B). Timestamps are the two 2026-06-12
// snapshots in the doc, ordered (08:00 = pre-#11, 16:00 = post-#11) so
// they sort before the bench era on the shared time axis.

export interface BaselinePoint {
  host: string;
  model: string;
  scenario: string;
  git_sha: string;
  build_timestamp: string;
  ttft_s: number;
  decode_tps: number;
  total_s: number;
}

/** Source: bench.py via cortex gateway — see doc/benchmarks.md. */
export const BASELINE_SOURCE = "bench.py · via cortex gateway";

export const BASELINE: BaselinePoint[] = [
  // ── 8f6f1d3 — baseline (2026-06-12) ────────────────────────────────
  { host: "beast", model: "Qwen/Qwen3.6-27B", scenario: "chat:128", git_sha: "8f6f1d3", build_timestamp: "2026-06-12T08:00:00Z", ttft_s: 1.658, decode_tps: 35.0, total_s: 8.981 },
  { host: "beast", model: "Qwen/Qwen3.6-27B", scenario: "chat:4096", git_sha: "8f6f1d3", build_timestamp: "2026-06-12T08:00:00Z", ttft_s: 7.067, decode_tps: 33.7, total_s: 14.63 },
  { host: "benjy", model: "Qwen/Qwen3-8B", scenario: "chat:128", git_sha: "8f6f1d3", build_timestamp: "2026-06-12T08:00:00Z", ttft_s: 0.884, decode_tps: 62.4, total_s: 4.938 },
  { host: "benjy", model: "Qwen/Qwen3-8B", scenario: "chat:4096", git_sha: "8f6f1d3", build_timestamp: "2026-06-12T08:00:00Z", ttft_s: 1.818, decode_tps: 46.5, total_s: 7.27 },
  { host: "quadbrat", model: "Qwen/Qwen3-1.7B", scenario: "chat:128", git_sha: "8f6f1d3", build_timestamp: "2026-06-12T08:00:00Z", ttft_s: 0.685, decode_tps: 81.3, total_s: 3.741 },
  { host: "quadbrat", model: "Qwen/Qwen3-1.7B", scenario: "chat:4096", git_sha: "8f6f1d3", build_timestamp: "2026-06-12T08:00:00Z", ttft_s: 2.743, decode_tps: 35.4, total_s: 9.884 },
  // ── a1952a4 — post prefix-KV-cache (#11, 2026-06-12) ───────────────
  { host: "beast", model: "Qwen/Qwen3.6-27B", scenario: "chat:128", git_sha: "a1952a4", build_timestamp: "2026-06-12T16:00:00Z", ttft_s: 1.355, decode_tps: 45.8, total_s: 4.147 },
  { host: "beast", model: "Qwen/Qwen3.6-27B", scenario: "chat:4096", git_sha: "a1952a4", build_timestamp: "2026-06-12T16:00:00Z", ttft_s: 1.431, decode_tps: 43.3, total_s: 4.387 },
  { host: "benjy", model: "Qwen/Qwen3-8B", scenario: "chat:128", git_sha: "a1952a4", build_timestamp: "2026-06-12T16:00:00Z", ttft_s: 0.886, decode_tps: 78.6, total_s: 2.478 },
  { host: "benjy", model: "Qwen/Qwen3-8B", scenario: "chat:4096", git_sha: "a1952a4", build_timestamp: "2026-06-12T16:00:00Z", ttft_s: 1.824, decode_tps: 58.3, total_s: 3.969 },
  { host: "quadbrat", model: "Qwen/Qwen3-1.7B", scenario: "chat:128", git_sha: "a1952a4", build_timestamp: "2026-06-12T16:00:00Z", ttft_s: 0.702, decode_tps: 104.8, total_s: 1.895 },
  { host: "quadbrat", model: "Qwen/Qwen3-1.7B", scenario: "chat:4096", git_sha: "a1952a4", build_timestamp: "2026-06-12T16:00:00Z", ttft_s: 2.749, decode_tps: 44.9, total_s: 5.534 },
];

/** Baseline points for one (model, scenario) cell, oldest first. */
export function baselineFor(model: string, scenario: string): BaselinePoint[] {
  return BASELINE.filter(
    (b) => b.model === model && b.scenario === scenario,
  ).sort((a, b) => a.build_timestamp.localeCompare(b.build_timestamp));
}
