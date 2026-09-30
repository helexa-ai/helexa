// Building a `/v1/systemone` request from the playground's editors, and
// checking it before it is sent.
//
// Everything works on the order-preserving representation (orderedJson.ts)
// and the body is serialised from it, so question ids, option labels and
// object states reach the server in exactly the order they were typed.
// The checks mirror the server's (the laya reference, which neuron
// reproduces) so a request that cannot succeed is explained before it is
// sent; the server stays the authority and its own message is shown when
// it disagrees.

import type { DecisionError, QuestionType } from "./decisionClient";
import {
  JsonNumber,
  JsonSyntaxError,
  fromPlain,
  isMap,
  parseOrdered,
  stringifyOrdered,
  type OJ,
  type OMap,
} from "./orderedJson";

/** The server's request limits. */
export const LIMITS = {
  questions: 64,
  choiceOptions: 100,
  scoreLevels: 32,
  totalOptions: 512,
  stateChars: 50_000,
  /** Upper bound the server accepts for `max_len` / `head_max_len`. */
  tokenBudget: 8192,
} as const;

/** How the state editor's text is sent: verbatim, or parsed as JSON. */
export type StateMode = "text" | "json";

/**
 * Which checkpoint answers. `auto` sends the class alias and lets the
 * service route by the state's language; the others pin one checkpoint.
 */
export type Checkpoint = "auto" | "english" | "multilingual" | "typed-decisions";
export const CHECKPOINTS: Checkpoint[] = ["auto", "english", "multilingual", "typed-decisions"];

/** The decision-model class alias every hop resolves. */
export const DEFAULT_MODEL = "helexa/one";

export function modelFor(checkpoint: Checkpoint): string {
  return checkpoint === "auto" ? DEFAULT_MODEL : checkpoint;
}

export interface Draft {
  stateText: string;
  stateMode: StateMode;
  questionsText: string;
  checkpoint: Checkpoint;
  /** Optional token budgets; empty means the checkpoint's default. */
  maxLen?: string;
  headMaxLen?: string;
}

/** Where a problem is, so the UI can put it beside the right editor. */
export type IssueField = "state" | "questions" | "budget";

/** A problem with the draft: an i18n key plus its interpolation values. */
export interface Issue {
  field: IssueField;
  key: string;
  params?: Record<string, string | number>;
  questionId?: string;
}

export interface BuiltRequest {
  /** The body exactly as it will be sent. */
  body: string;
  model: string;
  state: OJ;
  questions: OMap;
}

export type BuildResult = { ok: true; request: BuiltRequest } | { ok: false; issues: Issue[] };

const TYPES: QuestionType[] = ["choice", "score", "noul"];

function syntaxIssue(field: IssueField, e: unknown, key: string): Issue {
  if (e instanceof JsonSyntaxError) {
    return {
      field,
      key,
      params: { detail: e.message, line: e.line, column: e.column },
    };
  }
  return { field, key, params: { detail: String(e), line: 1, column: 1 } };
}

/**
 * Python treats `1`, `1.0` and `True` as one dict key, so the reference
 * rejects them as duplicate labels. Mirror that.
 */
function labelIdentity(v: OJ): string | null {
  if (typeof v === "string") return `s:${v}`;
  if (v instanceof JsonNumber) return `n:${v.value}`;
  if (typeof v === "boolean") return `n:${v ? 1 : 0}`;
  return null;
}

function isBlank(v: OJ | undefined): boolean {
  return v === undefined || v === null || (typeof v === "string" && !v.trim());
}

/** Check one question; returns how many options it contributes. */
function checkQuestion(id: string, q: OJ, issues: Issue[]): number {
  const push = (key: string, params: Record<string, string | number> = {}): void => {
    issues.push({ field: "questions", key, params: { id, ...params }, questionId: id });
  };
  if (!isMap(q)) {
    push("issues.questionNotObject");
    return 0;
  }
  const type = q.get("type");
  if (typeof type !== "string" || !TYPES.includes(type as QuestionType)) {
    push("issues.badType");
    return 0;
  }
  if (isBlank(q.get("instructions"))) push("issues.noInstructions");
  const labels = q.get("labels");
  if (labels !== undefined && type !== "noul") push("issues.labelsOnlyNoul");
  const crit = q.get("criteria");

  if (type === "choice") {
    const opts: OJ[] | null = Array.isArray(crit) ? crit : isMap(crit) ? [...crit.keys()] : null;
    if (!opts || opts.length === 0) {
      push("issues.choiceNeedsCriteria");
      return 0;
    }
    if (opts.length > LIMITS.choiceOptions) {
      push("issues.tooManyOptions", { max: LIMITS.choiceOptions });
    }
    if (Array.isArray(crit)) {
      const seen = new Set<string>();
      for (const label of crit) {
        const idn = labelIdentity(label);
        if (idn === null) {
          push("issues.labelNotScalar");
          break;
        }
        if (seen.has(idn)) {
          push("issues.duplicateLabel", { label: stringifyOrdered(label, 0) });
          break;
        }
        seen.add(idn);
      }
    }
    return opts.length;
  }

  if (type === "score") {
    if (!Array.isArray(crit) || crit.length === 0) {
      push("issues.scoreNeedsLevels");
      return 0;
    }
    if (crit.length > LIMITS.scoreLevels) {
      push("issues.tooManyLevels", { max: LIMITS.scoreLevels });
    }
    if (crit.some((c) => c === null)) push("issues.nullLevel");
    return crit.length;
  }

  // noul: criteria, if given, keyed only true/false; labels, if given,
  // exactly true and false, distinct and non-blank.
  if (crit !== undefined && crit !== null) {
    if (!isMap(crit) || [...crit.keys()].some((k) => !["true", "false"].includes(k.toLowerCase()))) {
      push("issues.noulCriteria");
    }
  }
  if (labels !== undefined) {
    const t = isMap(labels) ? labels.get("true") : undefined;
    const f = isMap(labels) ? labels.get("false") : undefined;
    const ok =
      isMap(labels) &&
      labels.size === 2 &&
      typeof t === "string" &&
      typeof f === "string" &&
      t.trim() !== "" &&
      f.trim() !== "" &&
      t.trim() !== f.trim();
    if (!ok) push("issues.noulLabels");
  }
  return 2;
}

function budget(text: string | undefined, name: string, issues: Issue[]): number | undefined {
  if (text === undefined || text.trim() === "") return undefined;
  const n = Number(text);
  if (!Number.isInteger(n) || n <= 0 || n > LIMITS.tokenBudget) {
    issues.push({ field: "budget", key: "issues.badBudget", params: { name, max: LIMITS.tokenBudget } });
    return undefined;
  }
  return n;
}

/**
 * Parse and check both editors. Returns the request when it can be sent,
 * else every problem found (not just the first), so they can be fixed in
 * one pass.
 */
export function buildRequest(draft: Draft): BuildResult {
  const issues: Issue[] = [];

  let state: OJ = draft.stateText;
  if (draft.stateMode === "json") {
    try {
      state = parseOrdered(draft.stateText);
    } catch (e) {
      issues.push(syntaxIssue("state", e, "issues.stateJson"));
    }
    if (state === null) issues.push({ field: "state", key: "issues.stateNull" });
  } else if (!draft.stateText.trim()) {
    issues.push({ field: "state", key: "issues.stateEmpty" });
  }
  // The server measures a string in code points and anything else by the
  // length of its Python repr; the serialised JSON is a close stand-in.
  const stateLen =
    typeof state === "string" ? [...state].length : stringifyOrdered(state, 0).length;
  if (stateLen > LIMITS.stateChars) {
    issues.push({ field: "state", key: "issues.stateTooLong", params: { max: LIMITS.stateChars } });
  }

  let questions: OJ;
  try {
    questions = parseOrdered(draft.questionsText);
  } catch (e) {
    issues.push(syntaxIssue("questions", e, "issues.questionsJson"));
    return { ok: false, issues };
  }
  if (!isMap(questions)) {
    issues.push({ field: "questions", key: "issues.questionsNotObject" });
    return { ok: false, issues };
  }
  if (questions.size === 0) issues.push({ field: "questions", key: "issues.noQuestions" });
  if (questions.size > LIMITS.questions) {
    issues.push({ field: "questions", key: "issues.tooManyQuestions", params: { max: LIMITS.questions } });
  }
  let total = 0;
  for (const [id, q] of questions) total += checkQuestion(id, q, issues);
  if (total > LIMITS.totalOptions) {
    issues.push({ field: "questions", key: "issues.tooManyTotal", params: { max: LIMITS.totalOptions } });
  }

  const maxLen = budget(draft.maxLen, "max_len", issues);
  const headMaxLen = budget(draft.headMaxLen, "head_max_len", issues);

  if (issues.length) return { ok: false, issues };

  const model = modelFor(draft.checkpoint);
  const root: OMap = new Map<string, OJ>([
    ["model", model],
    ["state", state],
    ["questions", questions],
  ]);
  if (maxLen !== undefined) root.set("max_len", new JsonNumber(String(maxLen)));
  if (headMaxLen !== undefined) root.set("head_max_len", new JsonNumber(String(headMaxLen)));
  return {
    ok: true,
    request: { body: stringifyOrdered(root, 0), model, state, questions },
  };
}

/**
 * The template each primitive inserts. Each shows the primitive's whole
 * shape — including the optional parts — so editing a template teaches
 * the format.
 */
export function questionTemplate(type: QuestionType): OJ {
  switch (type) {
    case "noul":
      return fromPlain({
        type: "noul",
        instructions: "Does the message ask for money back?",
        criteria: {
          true: "a refund, chargeback or credit is requested",
          false: "anything else",
        },
        labels: { true: "refund", false: "no refund" },
      });
    case "score":
      return fromPlain({
        type: "score",
        instructions: "How urgent is the message?",
        criteria: [
          "low: can wait a week",
          "normal: answer within a day",
          "high: the sender is blocked",
          "critical: outage, security or legal",
        ],
      });
    case "choice":
      return fromPlain({
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: {
          billing: "payments, invoices and refunds",
          technical: "bugs, errors and sign-in problems",
          other: "anything else",
        },
      });
  }
}

/**
 * Append a templated question to the questions editor's text. The new id
 * is the type plus the first free number (`choice`, `choice_2`, …). Returns
 * null when the current text is not a JSON object, rather than discarding
 * what the user typed.
 */
export function addQuestion(questionsText: string, type: QuestionType): string | null {
  let parsed: OJ = new Map();
  if (questionsText.trim()) {
    try {
      parsed = parseOrdered(questionsText);
    } catch {
      return null;
    }
  }
  if (!isMap(parsed)) return null;
  let id: string = type;
  for (let n = 2; parsed.has(id); n++) id = `${type}_${n}`;
  const next: OMap = new Map(parsed);
  next.set(id, questionTemplate(type));
  return stringifyOrdered(next);
}

/**
 * A server rejection that names a question, as an issue the compose pane
 * shows under the questions editor — beside the question it is about,
 * where it will be fixed. Null for anything else.
 */
export function issueFromError(error: DecisionError): Issue | null {
  if (error.kind !== "validation" || error.questionId === undefined) return null;
  const detail = error.message.replace(/^question '(?:[^'\\]|\\.)*':\s*/, "");
  return {
    field: "questions",
    key: "issues.server",
    params: { id: error.questionId, detail },
    questionId: error.questionId,
  };
}

/** Re-indent JSON text without reordering it; null if it does not parse. */
export function formatJson(text: string): string | null {
  try {
    return stringifyOrdered(parseOrdered(text));
  } catch {
    return null;
  }
}
