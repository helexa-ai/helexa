import { describe, expect, it } from "vitest";
import { LIMITS, addQuestion, buildRequest, formatJson, type Draft } from "./decisionRequest";

const base: Draft = {
  stateText: "I was charged twice",
  stateMode: "text",
  questionsText: '{"refund": {"type": "noul", "instructions": "Refund?"}}',
  checkpoint: "auto",
};

function issuesOf(d: Partial<Draft>): string[] {
  const r = buildRequest({ ...base, ...d });
  return r.ok ? [] : r.issues.map((i) => i.key);
}

describe("buildRequest", () => {
  it("builds the body in the order the editors were written", () => {
    const r = buildRequest({
      ...base,
      stateMode: "json",
      stateText: '{"z": "first", "a": "second"}',
      questionsText:
        '{"10": {"type": "noul", "instructions": "x"}, "2": {"type": "choice", "instructions": "y", "criteria": {"b": "", "a": ""}}}',
    });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.request.body).toBe(
      '{"model":"helexa/one","state":{"z":"first","a":"second"},"questions":{"10":{"type":"noul","instructions":"x"},"2":{"type":"choice","instructions":"y","criteria":{"b":"","a":""}}}}',
    );
  });

  it("sends a text state verbatim, as a string", () => {
    const r = buildRequest({ ...base, stateText: '{"looks": "like json"}' });
    expect(r.ok && JSON.parse(r.request.body).state).toBe('{"looks": "like json"}');
  });

  it("maps the checkpoint picker onto model", () => {
    for (const [checkpoint, model] of [
      ["auto", "helexa/one"],
      ["multilingual", "multilingual"],
      ["typed-decisions", "typed-decisions"],
    ] as const) {
      const r = buildRequest({ ...base, checkpoint });
      expect(r.ok && r.request.model).toBe(model);
    }
  });

  it("adds token budgets only when given", () => {
    const r = buildRequest({ ...base, maxLen: "256", headMaxLen: "" });
    expect(r.ok && JSON.parse(r.request.body)).toMatchObject({ max_len: 256 });
    expect(r.ok && "head_max_len" in JSON.parse(r.request.body)).toBe(false);
    expect(issuesOf({ maxLen: "0" })).toEqual(["issues.badBudget"]);
    expect(issuesOf({ maxLen: String(LIMITS.tokenBudget + 1) })).toEqual(["issues.badBudget"]);
  });

  it("reports JSON errors with their position", () => {
    const r = buildRequest({ ...base, questionsText: '{\n  "a": {"type": "noul",\n}' });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.issues[0].key).toBe("issues.questionsJson");
    expect(r.issues[0].params?.line).toBe(3);
  });

  it("names the question each problem is about", () => {
    const r = buildRequest({ ...base, questionsText: '{"q1": {"type": "maybe", "instructions": "x"}}' });
    expect(!r.ok && r.issues[0]).toMatchObject({ key: "issues.badType", questionId: "q1" });
  });

  it.each([
    ["an empty text state", { stateText: "  " }, "issues.stateEmpty"],
    ["a JSON null state", { stateMode: "json" as const, stateText: "null" }, "issues.stateNull"],
    ["a bad JSON state", { stateMode: "json" as const, stateText: "{" }, "issues.stateJson"],
    ["a state over the limit", { stateText: "x".repeat(LIMITS.stateChars + 1) }, "issues.stateTooLong"],
    ["questions that are a list", { questionsText: "[]" }, "issues.questionsNotObject"],
    ["no questions", { questionsText: "{}" }, "issues.noQuestions"],
    ["a question that is not an object", { questionsText: '{"q": 1}' }, "issues.questionNotObject"],
    ["a missing type", { questionsText: '{"q": {"instructions": "x"}}' }, "issues.badType"],
    ["missing instructions", { questionsText: '{"q": {"type": "noul"}}' }, "issues.noInstructions"],
    ["blank instructions", { questionsText: '{"q": {"type": "noul", "instructions": " "}}' }, "issues.noInstructions"],
    ["a choice with no options", { questionsText: '{"q": {"type": "choice", "instructions": "x", "criteria": []}}' }, "issues.choiceNeedsCriteria"],
    ["a score with no levels", { questionsText: '{"q": {"type": "score", "instructions": "x", "criteria": {}}}' }, "issues.scoreNeedsLevels"],
    ["a null score level", { questionsText: '{"q": {"type": "score", "instructions": "x", "criteria": ["a", null]}}' }, "issues.nullLevel"],
    ["labels on a choice", { questionsText: '{"q": {"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {}}}' }, "issues.labelsOnlyNoul"],
    ["a non-scalar label", { questionsText: '{"q": {"type": "choice", "instructions": "x", "criteria": [["a"]]}}' }, "issues.labelNotScalar"],
    ["noul criteria with other keys", { questionsText: '{"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}}' }, "issues.noulCriteria"],
    ["noul labels that collide", { questionsText: '{"q": {"type": "noul", "instructions": "x", "labels": {"true": "a", "false": "a"}}}' }, "issues.noulLabels"],
  ])("rejects %s", (_what, d, key) => {
    expect(issuesOf(d)).toContain(key);
  });

  it("treats 1, 1.0 and true as one choice label, as the server does", () => {
    for (const labels of ["[1, 1.0]", "[true, 1]", '["a", "a"]']) {
      expect(issuesOf({ questionsText: `{"q": {"type": "choice", "instructions": "x", "criteria": ${labels}}}` })).toContain(
        "issues.duplicateLabel",
      );
    }
    expect(issuesOf({ questionsText: '{"q": {"type": "choice", "instructions": "x", "criteria": [1, "1"]}}' })).toEqual([]);
  });

  it("enforces the server's count limits exactly at the boundary", () => {
    const qs = (n: number): string =>
      JSON.stringify(Object.fromEntries(Array.from({ length: n }, (_, i) => [`q${i}`, { type: "noul", instructions: "x" }])));
    expect(issuesOf({ questionsText: qs(LIMITS.questions) })).toEqual([]);
    expect(issuesOf({ questionsText: qs(LIMITS.questions + 1) })).toContain("issues.tooManyQuestions");

    const choice = (n: number): string =>
      JSON.stringify({ q: { type: "choice", instructions: "x", criteria: Array.from({ length: n }, (_, i) => `o${i}`) } });
    expect(issuesOf({ questionsText: choice(LIMITS.choiceOptions) })).toEqual([]);
    expect(issuesOf({ questionsText: choice(LIMITS.choiceOptions + 1) })).toContain("issues.tooManyOptions");

    const score = (n: number): string =>
      JSON.stringify({ q: { type: "score", instructions: "x", criteria: Array.from({ length: n }, (_, i) => `l${i}`) } });
    expect(issuesOf({ questionsText: score(LIMITS.scoreLevels) })).toEqual([]);
    expect(issuesOf({ questionsText: score(LIMITS.scoreLevels + 1) })).toContain("issues.tooManyLevels");

    // 6 choices of 86 options = 516 > 512, each under its own limit.
    const total = JSON.stringify(
      Object.fromEntries(
        Array.from({ length: 6 }, (_, q) => [
          `q${q}`,
          { type: "choice", instructions: "x", criteria: Array.from({ length: 86 }, (_, i) => `o${i}`) },
        ]),
      ),
    );
    expect(issuesOf({ questionsText: total })).toEqual(["issues.tooManyTotal"]);
  });

  it("collects every problem, not just the first", () => {
    expect(
      issuesOf({ stateText: "", questionsText: '{"a": {"type": "x"}, "b": {"type": "noul"}}' }),
    ).toEqual(["issues.stateEmpty", "issues.badType", "issues.noInstructions"]);
  });
});

describe("addQuestion", () => {
  it("appends a template under the first free id, keeping existing order", () => {
    let text = '{"10": {"type": "noul", "instructions": "x"}}';
    text = addQuestion(text, "choice")!;
    text = addQuestion(text, "choice")!;
    text = addQuestion(text, "score")!;
    expect(Object.keys(JSON.parse(text)).sort()).toEqual(["10", "choice", "choice_2", "score"].sort());
    expect(text.indexOf('"10"')).toBeLessThan(text.indexOf('"choice"'));
    expect(buildRequest({ ...base, questionsText: text }).ok).toBe(true);
  });

  it("inserts templates that are themselves valid requests", () => {
    for (const t of ["noul", "score", "choice"] as const) {
      expect(buildRequest({ ...base, questionsText: addQuestion("", t)! }).ok).toBe(true);
    }
  });

  it("refuses to overwrite text that is not a JSON object", () => {
    expect(addQuestion("{ broken", "noul")).toBeNull();
    expect(addQuestion("[]", "noul")).toBeNull();
  });
});

describe("formatJson", () => {
  it("re-indents without reordering", () => {
    expect(formatJson('{"2":1,"1":2}')).toBe('{\n  "2": 1,\n  "1": 2\n}');
    expect(formatJson("{")).toBeNull();
  });
});
