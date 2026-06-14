// Mirrors the JSON served by helexa-bench's read API (crates/helexa-bench/src/api.rs).

export interface BuildRef {
  git_sha: string;
  build_timestamp: string | null;
  package_version: string | null;
}

export interface Dimensions {
  hosts: string[];
  models: string[];
  scenarios: string[];
  builds: BuildRef[];
}

/** Latest-SHA-per-cell medians (the report table). */
export interface ReportRow {
  target_name: string;
  model_id: string;
  scenario_id: string;
  prompt_size_approx: number;
  git_sha: string;
  prompt_tokens: number | null;
  ttft_s_median: number | null;
  decode_tps_median: number | null;
  total_s_median: number | null;
  samples: number;
}

/** One point in a per-build time-series for a (host, model, scenario) cell. */
export interface SeriesPoint {
  git_sha: string;
  build_timestamp: string | null;
  package_version: string | null;
  ttft_s_median: number | null;
  decode_tps_median: number | null;
  total_s_median: number | null;
  samples: number;
}

export interface RunRow {
  id: number;
  ts: string;
  host: string;
  hostname: string | null;
  git_sha: string;
  build_timestamp: string | null;
  package_version: string;
  model_id: string;
  harness: string;
  scenario_id: string;
  prompt_size_approx: number;
  prompt_tokens_actual: number | null;
  max_tokens: number;
  ttft_s: number | null;
  decode_tps: number | null;
  total_s: number | null;
  completion_tokens: number | null;
  ok: boolean;
  error: string | null;
}
