import { describe, expect, it } from "vitest";
import { clock, coverArt, generateBlocker, hasTag, peaks, placeholderPeaks, toggleTag } from "./musicUi";

describe("coverArt", () => {
  it("is stable per id and differs between ids", () => {
    expect(coverArt("audiogen_a")).toBe(coverArt("audiogen_a"));
    expect(coverArt("audiogen_a")).not.toBe(coverArt("audiogen_b"));
    expect(coverArt("x")).toMatch(/^linear-gradient\(\d+deg, hsl\(/);
  });
});

describe("style tags", () => {
  it("adds a tag, keeping what was typed", () => {
    expect(toggleTag("Retrowave, neon arpeggios", "dreamy")).toBe("Retrowave, neon arpeggios, dreamy");
    expect(toggleTag("", "jazz")).toBe("jazz");
    expect(toggleTag("  ", "jazz")).toBe("jazz");
  });

  it("removes a tag already present, case-insensitively, and only that phrase", () => {
    expect(toggleTag("Jazz, smoky bar, jazz trio", "jazz")).toBe("smoky bar, jazz trio");
    expect(hasTag("smoky bar, jazz trio", "jazz")).toBe(false);
    expect(hasTag("Lo-Fi Hip Hop , rain", "lo-fi hip hop")).toBe(true);
  });
});

describe("generateBlocker", () => {
  const ok = {
    ready: true,
    hasModel: true,
    surprising: false,
    submitting: false,
    prompt: "pop",
    seconds: 30,
    min: 10,
    max: 240,
  };

  it("allows a complete form", () => {
    expect(generateBlocker(ok)).toBeNull();
  });

  it("names the first thing in the way", () => {
    expect(generateBlocker({ ...ok, ready: false, prompt: "" })).toEqual({ reason: "preparing" });
    expect(generateBlocker({ ...ok, hasModel: false })).toEqual({ reason: "noModel" });
    expect(generateBlocker({ ...ok, surprising: true, prompt: "" })).toEqual({ reason: "surprising" });
    expect(generateBlocker({ ...ok, submitting: true })).toEqual({ reason: "submitting" });
    expect(generateBlocker({ ...ok, prompt: "  " })).toEqual({ reason: "needStyle" });
  });

  it("rejects a length outside the model's range", () => {
    expect(generateBlocker({ ...ok, seconds: 5 })).toEqual({ reason: "needLength", min: 10, max: 240 });
    expect(generateBlocker({ ...ok, seconds: 241 })?.reason).toBe("needLength");
    expect(generateBlocker({ ...ok, seconds: Number.NaN })?.reason).toBe("needLength");
    expect(generateBlocker({ ...ok, seconds: 240 })).toBeNull();
  });
});

describe("clock", () => {
  it("formats m:ss", () => {
    expect(clock(0)).toBe("0:00");
    expect(clock(9.9)).toBe("0:09");
    expect(clock(125)).toBe("2:05");
    expect(clock(-3)).toBe("0:00");
    expect(clock(Number.NaN)).toBe("0:00");
  });
});

describe("peaks", () => {
  it("takes the loudest channel per slice and normalises to 1", () => {
    const l = new Float32Array([0.1, -0.2, 0, 0, 0.5, 0]);
    const r = new Float32Array([0, 0, 0.1, 0, -0.25, 0]);
    const p = peaks([l, r], 3);
    [0.4, 0.2, 1].forEach((want, i) => expect(p[i]).toBeCloseTo(want, 6));
  });

  it("keeps silence at zero and handles empty input", () => {
    expect(peaks([new Float32Array(8)], 4)).toEqual([0, 0, 0, 0]);
    expect(peaks([], 4)).toEqual([]);
    expect(peaks([new Float32Array(2)], 5)).toHaveLength(5);
  });
});

describe("placeholderPeaks", () => {
  it("is stable, sized, and within 0..1", () => {
    const a = placeholderPeaks("audiogen_a", 64);
    expect(a).toHaveLength(64);
    expect(a).toEqual(placeholderPeaks("audiogen_a", 64));
    expect(a).not.toEqual(placeholderPeaks("audiogen_b", 64));
    expect(Math.min(...a)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...a)).toBeLessThanOrEqual(1);
  });
});
