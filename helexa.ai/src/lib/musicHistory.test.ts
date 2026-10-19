import { describe, expect, it } from "vitest";
import type { AudioJob } from "./musicClient";
import { groupHistory, periodOf } from "./musicHistory";

/** Unix seconds for a local wall-clock time. */
const at = (y: number, m: number, d: number, h = 12, min = 0): number =>
  new Date(y, m - 1, d, h, min).getTime() / 1000;

function job(id: string, createdAt: number): AudioJob {
  return {
    id,
    object: "audio.generation",
    status: "succeeded",
    model: "m",
    created_at: createdAt,
  } as AudioJob;
}

describe("periodOf", () => {
  const now = new Date(2026, 9, 19, 0, 10); // 00:10 on the 19th

  it("uses calendar days, not 24-hour windows", () => {
    expect(periodOf(at(2026, 10, 19, 0, 5), now)).toBe("today");
    // Twenty minutes ago, but before midnight.
    expect(periodOf(at(2026, 10, 18, 23, 50), now)).toBe("yesterday");
    expect(periodOf(at(2026, 10, 18, 0, 0), now)).toBe("yesterday");
    expect(periodOf(at(2026, 10, 17, 23, 59), now)).toBe("week");
  });

  it("buckets the week, the month and older", () => {
    expect(periodOf(at(2026, 10, 12), now)).toBe("week");
    expect(periodOf(at(2026, 10, 11, 23), now)).toBe("month");
    expect(periodOf(at(2026, 9, 19, 12), now)).toBe("month");
    expect(periodOf(at(2026, 9, 18, 12), now)).toBe("older");
  });
});

describe("groupHistory", () => {
  const now = new Date(2026, 9, 19, 15, 0);
  const jobs = [
    job("a", at(2026, 10, 19, 9)),
    job("b", at(2026, 10, 19, 14)),
    job("c", at(2026, 10, 18, 9)),
    job("d", at(2026, 10, 1)),
    job("e", at(2026, 10, 19, 10)),
  ];

  it("files projects first and buckets the rest by date, newest first", () => {
    const filing = new Map<string, string | null>([
      ["e", "p1"],
      ["c", "p1"],
      ["d", null],
    ]);
    const g = groupHistory(jobs, filing, new Set(["p1"]), now);
    expect(g.byProject.get("p1")!.map((j) => j.id)).toEqual(["e", "c"]);
    expect(g.byPeriod.map((s) => [s.period, s.jobs.map((j) => j.id)])).toEqual([
      ["today", ["b", "a"]],
      ["month", ["d"]],
    ]);
  });

  it("shows songs of a deleted project under their dates", () => {
    const filing = new Map<string, string | null>([["a", "gone"]]);
    const g = groupHistory(jobs, filing, new Set(), now);
    expect(g.byProject.size).toBe(0);
    expect(g.byPeriod[0].jobs.map((j) => j.id)).toEqual(["b", "e", "a"]);
  });
});
