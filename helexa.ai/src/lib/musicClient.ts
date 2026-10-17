// Music generation client → the mesh router's async job API
// (/v1/audio/generations, #382). A song takes seconds to minutes, so the
// API is submit → poll → fetch rather than one long request: a job
// survives a closed tab, and the page picks it up again from the list.

const DEFAULT_BASE = import.meta.env.VITE_ROUTER_BASE_URL || "";

/** What a music model can do, from `/v1/models` (`audio`). */
export interface AudioLimits {
  min_duration: number;
  max_duration: number;
  default_duration: number;
  vocals: boolean;
  sample_rate: number;
  channels: number;
  max_steps?: number;
}

export interface MusicModel {
  id: string;
  limits: AudioLimits;
  /** Ledger tokens per second of audio; absent when not priced. */
  tokensPerSecond?: number;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface AudioJob {
  id: string;
  status: JobStatus;
  model: string;
  prompt?: string;
  lyrics?: string;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  duration?: number;
  result?: {
    duration: number;
    sample_rate: number;
    channels: number;
    seed: number;
    format: string;
    bytes: number;
  };
  usage?: { helexa_audio_units: number };
  error?: { code?: string; message?: string };
}

export interface MusicRequest {
  model: string;
  prompt: string;
  lyrics?: string;
  duration?: number;
  seed?: number;
  steps?: number;
}

export class MusicError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.name = "MusicError";
    this.code = code;
  }
}

export function isTerminal(status: JobStatus): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

/** Estimated ledger tokens for a request, or undefined when unpriced. */
export function estimateTokens(seconds: number, tokensPerSecond?: number): number | undefined {
  if (tokensPerSecond === undefined || !Number.isFinite(seconds) || seconds <= 0) return undefined;
  return Math.ceil(seconds * tokensPerSecond);
}

function base(baseUrl?: string): string {
  return (baseUrl ?? DEFAULT_BASE).replace(/\/$/, "");
}

function headers(apiKey?: string, json = false): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["content-type"] = "application/json";
  if (apiKey) h.authorization = `Bearer ${apiKey}`;
  return h;
}

/** Turn a non-2xx response into a MusicError from its #63 envelope. */
async function fail(resp: Response): Promise<never> {
  let code = `http_${resp.status}`;
  let message = `Request failed (${resp.status})`;
  try {
    const body = (await resp.json()) as { error?: { code?: string; message?: string } };
    if (body.error?.code) code = body.error.code;
    if (body.error?.message) message = body.error.message;
  } catch {
    /* keep the status-derived fallback */
  }
  throw new MusicError(code, message);
}

async function call(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (e) {
    throw new MusicError("network_error", (e as Error)?.message ?? "Network error");
  }
}

/** Music models the fleet advertises (those with `audio` limits). */
export async function listMusicModels(opts: { baseUrl?: string; apiKey?: string } = {}): Promise<MusicModel[]> {
  const resp = await call(`${base(opts.baseUrl)}/v1/models`, { headers: headers(opts.apiKey) });
  if (!resp.ok) await fail(resp);
  const body = (await resp.json()) as {
    data?: { id: string; audio?: AudioLimits; audio_tokens_per_second?: number }[];
  };
  return (body.data ?? [])
    .filter((m) => m.audio)
    .map((m) => ({ id: m.id, limits: m.audio!, tokensPerSecond: m.audio_tokens_per_second }))
    .sort((a, b) => a.id.localeCompare(b.id));
}

export async function submitJob(
  req: MusicRequest,
  opts: { baseUrl?: string; apiKey?: string } = {},
): Promise<AudioJob> {
  const resp = await call(`${base(opts.baseUrl)}/v1/audio/generations`, {
    method: "POST",
    headers: headers(opts.apiKey, true),
    body: JSON.stringify({
      model: req.model,
      prompt: req.prompt,
      ...(req.lyrics ? { lyrics: req.lyrics } : {}),
      ...(req.duration !== undefined ? { duration: req.duration } : {}),
      ...(req.seed !== undefined ? { seed: req.seed } : {}),
      ...(req.steps !== undefined ? { steps: req.steps } : {}),
    }),
  });
  if (!resp.ok) await fail(resp);
  return (await resp.json()) as AudioJob;
}

export async function listJobs(opts: { baseUrl?: string; apiKey?: string } = {}): Promise<AudioJob[]> {
  const resp = await call(`${base(opts.baseUrl)}/v1/audio/generations`, { headers: headers(opts.apiKey) });
  if (!resp.ok) await fail(resp);
  const body = (await resp.json()) as { data?: AudioJob[] };
  return body.data ?? [];
}

export async function cancelJob(id: string, opts: { baseUrl?: string; apiKey?: string } = {}): Promise<AudioJob> {
  const resp = await call(`${base(opts.baseUrl)}/v1/audio/generations/${encodeURIComponent(id)}`, {
    method: "DELETE",
    headers: headers(opts.apiKey),
  });
  if (!resp.ok) await fail(resp);
  return (await resp.json()) as AudioJob;
}

/** The rendered audio of a finished job. */
export async function fetchAudio(id: string, opts: { baseUrl?: string; apiKey?: string } = {}): Promise<Blob> {
  const resp = await call(
    `${base(opts.baseUrl)}/v1/audio/generations/${encodeURIComponent(id)}/content`,
    { headers: headers(opts.apiKey) },
  );
  if (!resp.ok) await fail(resp);
  return await resp.blob();
}
