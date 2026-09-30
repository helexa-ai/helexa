// Shared by the example recorder and the test that holds examples to their
// recordings (#360). Node-only: never imported by the page.
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import type { ParsedResponse } from "../lib/decisionClient";
import { buildRequest } from "../lib/decisionRequest";
import { exampleDraft, type DecisionExample } from "./decisionExamples";

/** Tests run from helexa.ai/. */
export const RECORDED_PATH = resolve(process.cwd(), "src/data/decisionExamples.recorded.json");

export interface RecordedAnswer {
  type: string;
  /** Choice: the chosen label. */
  answer?: string;
  /** Choice: its probability. Noul: P(true). */
  p?: number;
  /** Score: the expected level. */
  score?: number;
}

export interface Recording {
  /** SHA-256 of the exact request body the example sends. */
  hash: string;
  checkpoint: string;
  inputTokens: number;
  answers: Record<string, RecordedAnswer>;
}

/** The body an example sends, and its hash. */
export function requestFor(ex: DecisionExample): { body: string; hash: string } {
  const built = buildRequest(exampleDraft(ex));
  if (!built.ok) throw new Error(`example ${ex.id} is invalid: ${JSON.stringify(built.issues)}`);
  const body = built.request.body;
  return { body, hash: createHash("sha256").update(body).digest("hex").slice(0, 16) };
}

const r3 = (x: number): number => Math.round(x * 1000) / 1000;

export function recordingOf(resp: ParsedResponse, hash: string): Recording {
  const answers: Record<string, RecordedAnswer> = {};
  for (const a of resp.answers) {
    if (a.type === "choice") {
      const p = a.options.find((o) => o.key === a.chosen)?.probability ?? 0;
      answers[a.qid] = { type: a.type, answer: a.chosen, p: r3(p) };
    } else if (a.type === "score") {
      answers[a.qid] = { type: a.type, score: r3(a.score ?? 0) };
    } else {
      answers[a.qid] = { type: a.type, p: r3(a.noul ?? 0) };
    }
  }
  return { hash, checkpoint: resp.routing?.model ?? "", inputTokens: resp.inputTokens, answers };
}
