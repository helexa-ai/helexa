// Decision client → the mesh router's `/v1/systemone` (TypeSafe Jev wire
// protocol). A decision model (Laya) answers typed questions about a
// "state" in one forward pass: no stream and nothing generated, so the
// whole response arrives at once.
//
// Responses are read with the order-preserving reader, so answers and
// options come back in the order the request wrote them even when a label
// or question id looks like an integer (see orderedJson.ts).

import { JsonNumber, isMap, parseOrdered, stringifyOrdered, type OJ } from "./orderedJson";

export type QuestionType = "choice" | "score" | "noul";

/** One option of an answered question, in the question's own order. */
export interface AnswerOption {
  /** The probability key as the server wrote it. */
  key: string;
  probability: number;
  /** Score levels: the level's description from the legend. */
  legend?: string;
}

export interface ParsedAnswer {
  qid: string;
  type: QuestionType;
  options: AnswerOption[];
  /** Key of the chosen option (choice), most likely level (score), or
   *  "true"/"false" (noul). */
  chosen: string;
  /** Choice: the winning label as the caller wrote it. */
  choice?: string;
  /** Score: expected level index, which may fall between levels. */
  score?: number;
  /** Noul: P(true). */
  noul?: number;
  confidence: number;
  answerConfidence: number;
  actProbability?: number;
}

export interface Detection {
  script?: string;
  scriptProfile: [string, number][];
  language: string | null;
  isEnglish?: boolean;
  languageUndecided?: boolean;
  diacriticRate?: number;
  nonLatinFraction?: number;
  mixedSegment: string | null;
}

export interface Routing {
  /** Checkpoint that answered: english, multilingual or typed-decisions. */
  model: string;
  repo: string;
  reason: string;
  detection: Detection | null;
  workflow: string | null;
}

export interface CollapsedOptions {
  qid: string;
  total: number;
  distinct: number;
  tokensPerOption: number | null;
}

export interface ParsedResponse {
  model: string;
  answers: ParsedAnswer[];
  inputTokens: number;
  outputTokens: number;
  collapsed: CollapsedOptions[];
  routing: Routing | null;
}

export interface DecisionTiming {
  /** Send to parsed response, measured in the browser. */
  clientMs: number;
  /** Server inference time (Server-Timing / X-Inference-Time-Ms). */
  serverMs?: number;
}

export interface DecisionResult {
  response: ParsedResponse;
  /** The response body as received, for the raw view. */
  raw: string;
  timing: DecisionTiming;
}

/** What went wrong, in the classes the UI explains differently. */
export type ErrorKind =
  | "validation" // 400 / 422: the request itself needs fixing
  | "too_large" // 413: over a server limit
  | "rate_limited" // 429
  | "unavailable" // 503, or 502/504 from a proxy
  | "not_found" // 404: model not served
  | "server" // any other 5xx
  | "network" // no HTTP response at all
  | "timeout"
  | "cancelled"
  | "bad_response"; // a 2xx we could not read

export class DecisionError extends Error {
  kind: ErrorKind;
  code: string;
  status?: number;
  /** Seconds to wait before retrying, from `Retry-After`. */
  retryAfter?: number;
  /** The question a validation message names (`question 'x': …`). */
  questionId?: string;
  constructor(
    kind: ErrorKind,
    message: string,
    opts: { code?: string; status?: number; retryAfter?: number; questionId?: string } = {},
  ) {
    super(message);
    this.name = "DecisionError";
    this.kind = kind;
    this.code = opts.code ?? kind;
    this.status = opts.status;
    this.retryAfter = opts.retryAfter;
    this.questionId = opts.questionId;
  }
}

// ── Reading values ─────────────────────────────────────────────────────

function num(v: OJ | undefined): number | undefined {
  return v instanceof JsonNumber ? v.value : undefined;
}
function text(v: OJ | undefined): string | undefined {
  return typeof v === "string" ? v : undefined;
}
function field(m: OJ | undefined, k: string): OJ | undefined {
  return isMap(m) ? m.get(k) : undefined;
}
/** A label as the server writes it as a dict key (`json.dumps`). */
function keyOf(v: OJ | undefined): string {
  if (typeof v === "string") return v;
  if (v === undefined) return "";
  return stringifyOrdered(v, 0);
}

function parseAnswer(qid: string, a: OJ): ParsedAnswer {
  const type = text(field(a, "type"));
  if (type !== "choice" && type !== "score" && type !== "noul") {
    throw new DecisionError("bad_response", `answer '${qid}' has an unknown type`);
  }
  const qtype: QuestionType = type;
  const base = {
    qid,
    type: qtype,
    confidence: num(field(a, "confidence")) ?? 0,
    answerConfidence: num(field(a, "answer_confidence")) ?? 0,
    actProbability: num(field(field(a, "action"), "act_probability")),
  };
  if (type === "noul") {
    const p = num(field(a, "noul")) ?? 0;
    return {
      ...base,
      noul: p,
      chosen: p >= 0.5 ? "true" : "false",
      options: [
        { key: "true", probability: p },
        { key: "false", probability: 1 - p },
      ],
    };
  }
  const probs = field(a, "probabilities");
  const options: AnswerOption[] = isMap(probs)
    ? [...probs].map(([key, v]) => ({ key, probability: num(v) ?? 0 }))
    : [];
  const argmax = options.reduce<AnswerOption | undefined>(
    (best, o) => (!best || o.probability > best.probability ? o : best),
    undefined,
  );
  if (type === "score") {
    const legend = field(a, "legend");
    for (const o of options) {
      const d = field(legend, o.key);
      if (d !== undefined) o.legend = typeof d === "string" ? d : stringifyOrdered(d, 0);
    }
    return { ...base, score: num(field(a, "score")), chosen: argmax?.key ?? "", options };
  }
  const choiceKey = keyOf(field(a, "choice"));
  const chosen = options.some((o) => o.key === choiceKey) ? choiceKey : (argmax?.key ?? "");
  return { ...base, choice: choiceKey, chosen, options };
}

function parseDetection(d: OJ | undefined): Detection | null {
  if (!isMap(d)) return null;
  const profile = d.get("script_profile");
  return {
    script: text(d.get("script")),
    scriptProfile: isMap(profile) ? [...profile].map(([k, v]) => [k, num(v) ?? 0]) : [],
    language: text(d.get("language")) ?? null,
    isEnglish: typeof d.get("is_english") === "boolean" ? (d.get("is_english") as boolean) : undefined,
    languageUndecided:
      typeof d.get("language_undecided") === "boolean"
        ? (d.get("language_undecided") as boolean)
        : undefined,
    diacriticRate: num(d.get("diacritic_rate")),
    nonLatinFraction: num(d.get("non_latin_fraction")),
    mixedSegment: text(d.get("mixed_segment")) ?? null,
  };
}

/** Read a `/v1/systemone` response body, keeping every order it wrote. */
export function parseDecisionResponse(body: string): ParsedResponse {
  let root: OJ;
  try {
    root = parseOrdered(body);
  } catch {
    throw new DecisionError("bad_response", "The response was not valid JSON");
  }
  const answers = field(root, "answers");
  if (!isMap(answers)) throw new DecisionError("bad_response", "The response had no answers");
  const usage = field(root, "usage");
  const opts = field(usage, "options");
  const routing = field(root, "routing");
  return {
    model: text(field(root, "model")) ?? "",
    answers: [...answers].map(([qid, a]) => parseAnswer(qid, a)),
    inputTokens: num(field(usage, "input_tokens")) ?? 0,
    outputTokens: num(field(usage, "output_tokens")) ?? 0,
    collapsed: isMap(opts)
      ? [...opts].map(([qid, o]) => ({
          qid,
          total: num(field(o, "total")) ?? 0,
          distinct: num(field(o, "distinct")) ?? 0,
          tokensPerOption: num(field(o, "tokens_per_option")) ?? null,
        }))
      : [],
    routing: isMap(routing)
      ? {
          model: text(routing.get("model")) ?? "",
          repo: text(routing.get("repo")) ?? "",
          reason: text(routing.get("reason")) ?? "",
          detection: parseDetection(routing.get("detection")),
          workflow: text(routing.get("workflow")) ?? null,
        }
      : null,
  };
}

// ── Headers and errors ─────────────────────────────────────────────────

/** Server inference time from `X-Inference-Time-Ms` or `Server-Timing`. */
export function serverTimingMs(headers: Headers): number | undefined {
  const direct = headers.get("x-inference-time-ms");
  if (direct !== null && direct !== "" && Number.isFinite(Number(direct))) return Number(direct);
  const m = headers.get("server-timing")?.match(/inference;\s*dur=([0-9.]+)/);
  return m ? Number(m[1]) : undefined;
}

/** `Retry-After` in whole seconds (the delta form; an HTTP date is ignored). */
export function retryAfterSecs(headers: Headers): number | undefined {
  const v = headers.get("retry-after");
  if (v === null || v.trim() === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) && n >= 0 ? Math.ceil(n) : undefined;
}

function kindForStatus(status: number): ErrorKind {
  if (status === 400 || status === 422) return "validation";
  if (status === 413) return "too_large";
  if (status === 429) return "rate_limited";
  if (status === 404) return "not_found";
  if (status === 502 || status === 503 || status === 504) return "unavailable";
  return status >= 500 ? "server" : "validation";
}

/**
 * A non-2xx response as a `DecisionError`.
 *
 * Our servers answer with the #63 envelope (`{error: {code, message}}`)
 * plus FastAPI's top-level `detail`, which carries the reference server's
 * exact validation text and names the question it is about. A proxy in
 * front (the edge rate limiter) may answer with an HTML page instead, so
 * every field falls back to what the status alone says.
 */
export async function errorFromResponse(resp: Response): Promise<DecisionError> {
  const kind = kindForStatus(resp.status);
  const retryAfter = retryAfterSecs(resp.headers);
  let code: string = kind;
  let message = `HTTP ${resp.status}`;
  try {
    const body = parseOrdered(await resp.text());
    const err = field(body, "error");
    const detail = text(field(body, "detail"));
    code = text(field(err, "code")) ?? code;
    message = detail || text(field(err, "message")) || message;
  } catch {
    /* not JSON: keep what the status says */
  }
  const q = /^question '((?:[^'\\]|\\.)*)'/.exec(message);
  return new DecisionError(kind, message, {
    code,
    status: resp.status,
    retryAfter,
    questionId: q ? q[1].replace(/\\(.)/g, "$1") : undefined,
  });
}

// ── Sending ────────────────────────────────────────────────────────────

const DEFAULT_BASE = import.meta.env.VITE_ROUTER_BASE_URL || "";

/** A decision takes tens of milliseconds and a cold load seconds; this only
 *  stops a dead origin hanging the page. */
const TIMEOUT_MS = 60_000;

/**
 * Send a request body — already serialised, so its key order is exactly
 * the order the user wrote (see decisionRequest.ts) — and read the answer.
 */
export async function runDecision(opts: {
  body: string;
  apiKey?: string;
  baseUrl?: string;
  signal: AbortSignal;
}): Promise<DecisionResult> {
  const base = (opts.baseUrl ?? DEFAULT_BASE).replace(/\/$/, "");
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`;

  let timedOut = false;
  const ctl = new AbortController();
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, TIMEOUT_MS);
  const onAbort = (): void => ctl.abort();
  if (opts.signal.aborted) ctl.abort();
  opts.signal.addEventListener("abort", onAbort);

  const started = performance.now();
  try {
    const resp = await fetch(`${base}/v1/systemone`, {
      method: "POST",
      headers,
      signal: ctl.signal,
      body: opts.body,
    });
    if (!resp.ok) throw await errorFromResponse(resp);
    const raw = await resp.text();
    return {
      response: parseDecisionResponse(raw),
      raw,
      timing: { clientMs: performance.now() - started, serverMs: serverTimingMs(resp.headers) },
    };
  } catch (e) {
    if (e instanceof DecisionError) throw e;
    if ((e as Error)?.name === "AbortError") {
      throw timedOut
        ? new DecisionError("timeout", "The request timed out")
        : new DecisionError("cancelled", "Cancelled");
    }
    throw new DecisionError("network", (e as Error)?.message ?? "Network error");
  } finally {
    clearTimeout(timer);
    opts.signal.removeEventListener("abort", onAbort);
  }
}

