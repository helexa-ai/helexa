import type { Dimensions, ReportRow, RunRow, SeriesPoint } from "./types";

// Empty default → `fetch('/api/...')` hits the dev proxy (vite.config.ts)
// or the same origin. For a separately-hosted build, set VITE_API_BASE to
// the bob API origin (e.g. http://bob.hanzalova.internal:13132).
const BASE = import.meta.env.VITE_API_BASE ?? "";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export const getDimensions = () => getJson<Dimensions>("/api/dimensions");
export const getSummary = () => getJson<ReportRow[]>("/api/summary");

// host is resolved server-side (each model maps to one host today), so the
// public UI selects by model + scenario alone.
export const getSeries = (model: string, scenario: string) =>
  getJson<SeriesPoint[]>(
    `/api/series?model=${encodeURIComponent(model)}&scenario=${encodeURIComponent(scenario)}`,
  );

export interface RunsParams {
  host?: string;
  model?: string;
  scenario?: string;
  sha?: string;
  ok?: boolean;
  limit?: number;
}

export const getRuns = (p: RunsParams = {}) => {
  const q = new URLSearchParams();
  if (p.host) q.set("host", p.host);
  if (p.model) q.set("model", p.model);
  if (p.scenario) q.set("scenario", p.scenario);
  if (p.sha) q.set("sha", p.sha);
  if (p.ok !== undefined) q.set("ok", String(p.ok));
  if (p.limit) q.set("limit", String(p.limit));
  const qs = q.toString();
  return getJson<RunRow[]>(`/api/runs${qs ? `?${qs}` : ""}`);
};
