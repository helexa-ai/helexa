import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DecisionError,
  errorFromResponse,
  parseDecisionResponse,
  retryAfterSecs,
  runDecision,
  serverTimingMs,
} from "./decisionClient";

/**
 * Real responses: the laya reference server's output over the fixture
 * corpus the Rust port is tested against (#334). Every answer type,
 * checkpoint pins and multilingual routing are in it.
 */
const reference = JSON.parse(
  readFileSync(
    new URL("../../../crates/neuron/src/harness/testdata/laya/reference.json", import.meta.url),
    "utf8",
  ),
) as {
  cases: {
    name: string;
    request: { questions: Record<string, { type: string; criteria?: unknown }>; model?: string };
    response: Record<string, unknown> & {
      answers: Record<string, Record<string, unknown>>;
      usage: { input_tokens: number };
      routing: { model: string; reason: string };
    };
  }[];
};

describe("parseDecisionResponse over reference responses", () => {
  it.each(reference.cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const parsed = parseDecisionResponse(JSON.stringify(c.response));
    expect(parsed.answers.map((a) => a.qid)).toEqual(Object.keys(c.request.questions));
    expect(parsed.inputTokens).toBe(c.response.usage.input_tokens);
    expect(parsed.routing?.model).toBe(c.response.routing.model);
    expect(parsed.routing?.reason).toBe(c.response.routing.reason);
    for (const a of parsed.answers) {
      const want = c.response.answers[a.qid];
      expect(a.type).toBe(want.type);
      expect(a.answerConfidence).toBe(want.answer_confidence);
      if (a.type === "choice") {
        expect(a.chosen).toBe(String(want.choice));
        const crit = c.request.questions[a.qid].criteria;
        // Object-form criteria: options come back in the criteria's order.
        if (crit && !Array.isArray(crit)) {
          expect(a.options.map((o) => o.key)).toEqual(Object.keys(crit as object));
        }
      } else if (a.type === "score") {
        expect(a.score).toBe(want.score);
        const best = Math.max(...a.options.map((o) => o.probability));
        expect(a.options.find((o) => o.key === a.chosen)?.probability).toBe(best);
        expect(a.options.every((o) => typeof o.legend === "string")).toBe(true);
      } else {
        expect(a.noul).toBe(want.noul);
        expect(a.options[0].probability + a.options[1].probability).toBeCloseTo(1, 10);
      }
    }
  });

  it("covers every answer type, a pinned checkpoint and multilingual routing", () => {
    const types = new Set<string>();
    const checkpoints = new Set<string>();
    for (const c of reference.cases) {
      for (const a of parseDecisionResponse(JSON.stringify(c.response)).answers) types.add(a.type);
      checkpoints.add(c.response.routing.model);
    }
    expect([...types].sort()).toEqual(["choice", "noul", "score"]);
    expect([...checkpoints].sort()).toEqual(["english", "multilingual", "typed-decisions"]);
  });

  it("keeps integer-like option labels in the order the server wrote them", () => {
    const body =
      '{"model":"laya-rl-agent","answers":{"q":{"type":"choice","choice":2,' +
      '"probabilities":{"10":0.1,"2":0.9},"confidence":0.5,"answer_confidence":0.9,' +
      '"action":{"act_probability":1.0}}},"usage":{"input_tokens":7,"output_tokens":0}}';
    const [a] = parseDecisionResponse(body).answers;
    expect(a.options.map((o) => o.key)).toEqual(["10", "2"]);
    expect(a.chosen).toBe("2");
  });

  it("reads the detection block, script profile in order", () => {
    const ml = reference.cases.find((c) => c.name === "ml_mixed_script")!;
    const d = parseDecisionResponse(JSON.stringify(ml.response)).routing!.detection!;
    expect(d.scriptProfile[0][0]).toBe("latin");
    expect(d.scriptProfile.length).toBeGreaterThan(1);
  });

  it("rejects a body with no answers", () => {
    expect(() => parseDecisionResponse('{"usage":{}}')).toThrow(DecisionError);
    expect(() => parseDecisionResponse("<html>")).toThrow(DecisionError);
  });
});

describe("headers", () => {
  it("reads server timing from either header", () => {
    expect(serverTimingMs(new Headers({ "x-inference-time-ms": "7.39" }))).toBe(7.39);
    expect(serverTimingMs(new Headers({ "server-timing": "inference;dur=12.50" }))).toBe(12.5);
    expect(serverTimingMs(new Headers())).toBeUndefined();
  });

  it("reads Retry-After seconds and ignores dates", () => {
    expect(retryAfterSecs(new Headers({ "retry-after": "5" }))).toBe(5);
    expect(retryAfterSecs(new Headers({ "retry-after": "1.2" }))).toBe(2);
    expect(retryAfterSecs(new Headers({ "retry-after": "Wed, 21 Oct 2026 07:28:00 GMT" }))).toBeUndefined();
  });
});

describe("errorFromResponse", () => {
  const json = (status: number, body: unknown, headers: Record<string, string> = {}): Response =>
    new Response(JSON.stringify(body), { status, headers });

  it("prefers FastAPI's detail and names the question it is about", async () => {
    const e = await errorFromResponse(
      json(422, {
        error: { code: "invalid_decision_request", message: "x" },
        detail: "question 'urgency': a score question needs at least one level",
      }),
    );
    expect(e.kind).toBe("validation");
    expect(e.code).toBe("invalid_decision_request");
    expect(e.message).toMatch(/needs at least one level/);
    expect(e.questionId).toBe("urgency");
  });

  it.each([
    [400, "validation"],
    [404, "not_found"],
    [413, "too_large"],
    [429, "rate_limited"],
    [500, "server"],
    [502, "unavailable"],
    [503, "unavailable"],
  ])("classifies %i as %s", async (status, kind) => {
    const e = await errorFromResponse(json(status, { error: { code: "c", message: "m" } }));
    expect(e.kind).toBe(kind);
    expect(e.status).toBe(status);
  });

  it("handles the edge's HTML rate-limit page", async () => {
    const e = await errorFromResponse(
      new Response("<html><title>429 Too Many Requests</title></html>", {
        status: 429,
        headers: { "retry-after": "12" },
      }),
    );
    expect(e.kind).toBe("rate_limited");
    expect(e.retryAfter).toBe(12);
  });
});

describe("runDecision", () => {
  afterEach(() => vi.unstubAllGlobals());

  const ok = reference.cases[0].response;

  it("sends the body verbatim and returns the parsed answer with timing", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(ok), { status: 200, headers: { "server-timing": "inference;dur=9.5" } }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const body = '{"model":"helexa/one","state":"x","questions":{"2":{},"10":{}}}';
    const r = await runDecision({ body, baseUrl: "https://h", apiKey: "k", signal: new AbortController().signal });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://h/v1/systemone");
    expect(init.body).toBe(body);
    expect((init.headers as Record<string, string>).authorization).toBe("Bearer k");
    expect(r.timing.serverMs).toBe(9.5);
    expect(r.response.answers.length).toBeGreaterThan(0);
  });

  it("tells a network failure from a server rejection", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    await expect(runDecision({ body: "{}", signal: new AbortController().signal })).rejects.toMatchObject({
      kind: "network",
    });
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 503 })));
    await expect(runDecision({ body: "{}", signal: new AbortController().signal })).rejects.toMatchObject({
      kind: "unavailable",
    });
  });

  it("reports a user cancel as cancelled", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_u: string, init: RequestInit) =>
          new Promise((_, reject) =>
            init.signal!.addEventListener("abort", () =>
              reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
            ),
          ),
      ),
    );
    const ctl = new AbortController();
    const p = runDecision({ body: "{}", signal: ctl.signal });
    ctl.abort();
    await expect(p).rejects.toMatchObject({ kind: "cancelled" });
  });
});
