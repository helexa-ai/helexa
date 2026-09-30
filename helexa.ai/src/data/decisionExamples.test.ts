import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { EXAMPLES, EXAMPLE_GROUPS } from "./decisionExamples";
import { RECORDED_PATH, requestFor, type Recording } from "./decisionExamplesRecording";
import en from "../i18n/resources/en/decisions.json";

const recorded = JSON.parse(readFileSync(RECORDED_PATH, "utf8")) as {
  recordedAt: string;
  examples: Record<string, Recording>;
};

/** Every object key in a value, at any depth. */
function keysOf(v: unknown): string[] {
  if (Array.isArray(v)) return v.flatMap(keysOf);
  if (v && typeof v === "object") return Object.entries(v).flatMap(([k, x]) => [k, ...keysOf(x)]);
  return [];
}

describe("decision examples", () => {
  it("have unique ids", () => {
    expect(new Set(EXAMPLES.map((e) => e.id)).size).toBe(EXAMPLES.length);
  });

  it("include one lesson per primitive", () => {
    const lessons = EXAMPLES.filter((e) => e.group === "lessons").map((e) => e.primitive);
    expect(lessons.sort()).toEqual(["choice", "noul", "score"]);
  });

  it("use no integer-like keys, which JSON.stringify would reorder", () => {
    for (const ex of EXAMPLES) {
      expect(keysOf([ex.state, ex.questions]).filter((k) => /^\d+$/.test(k)), ex.id).toEqual([]);
    }
  });

  it("are all valid requests", () => {
    for (const ex of EXAMPLES) expect(() => requestFor(ex), ex.id).not.toThrow();
  });

  it("each match their live recording (re-record after editing one)", () => {
    for (const ex of EXAMPLES) {
      const rec = recorded.examples[ex.id];
      expect(rec, `${ex.id} has no recording`).toBeDefined();
      expect(rec.hash, `${ex.id} changed since it was recorded`).toBe(requestFor(ex).hash);
      expect(Object.keys(rec.answers).sort(), ex.id).toEqual(Object.keys(ex.questions).sort());
    }
    expect(Object.keys(recorded.examples).sort()).toEqual(EXAMPLES.map((e) => e.id).sort());
  });

  it("route where their group says they should", () => {
    for (const ex of EXAMPLES) {
      const want = ex.group === "multilingual" ? "multilingual" : "english";
      expect(recorded.examples[ex.id].checkpoint, ex.id).toBe(want);
    }
  });

  it("recorded clear answers, not coin flips", () => {
    for (const ex of EXAMPLES) {
      for (const [qid, a] of Object.entries(recorded.examples[ex.id].answers)) {
        if (a.type === "noul") expect(Math.abs(a.p! - 0.5), `${ex.id}.${qid}`).toBeGreaterThanOrEqual(0.15);
        if (a.type === "choice") expect(a.p!, `${ex.id}.${qid}`).toBeGreaterThanOrEqual(0.5);
      }
    }
  });

  it("each have an English title and note", () => {
    const strings = en.examples as unknown as Record<string, { title?: string; blurb?: string }>;
    for (const ex of EXAMPLES) {
      expect(strings[ex.id]?.title, ex.id).toBeTruthy();
      expect(strings[ex.id]?.blurb, ex.id).toBeTruthy();
    }
    for (const g of EXAMPLE_GROUPS) expect((en.examples.groups as Record<string, string>)[g]).toBeTruthy();
  });
});
