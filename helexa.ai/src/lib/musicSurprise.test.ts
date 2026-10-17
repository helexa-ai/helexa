import { describe, expect, it } from "vitest";
import {
  clampDuration,
  localSurprise,
  parseSurpriseReply,
  surpriseInstruction,
} from "./musicSurprise";

const song = { vocals: true, minDuration: 10, maxDuration: 600 };
const sfx = { vocals: false, minDuration: 1, maxDuration: 10 };

/** A deterministic generator cycling through the given values. */
function seq(...values: number[]): () => number {
  let i = 0;
  return () => values[i++ % values.length];
}

describe("clampDuration", () => {
  it("rounds and clamps into the model's range", () => {
    expect(clampDuration(42.6, song)).toBe(43);
    expect(clampDuration(3, song)).toBe(10);
    expect(clampDuration(9000, song)).toBe(600);
    expect(clampDuration(Number.NaN, song)).toBe(10);
  });
});

describe("localSurprise", () => {
  it("only writes lyrics for a model that sings", () => {
    for (let i = 0; i < 50; i++) {
      const idea = localSurprise(sfx);
      expect(idea.lyrics).toBe("");
      expect(idea.duration).toBeGreaterThanOrEqual(1);
      expect(idea.duration).toBeLessThanOrEqual(10);
    }
  });

  it("gives a singing idea lyrics and a vocal caption", () => {
    // rng() < 0.5 at the "sing" draw picks vocals.
    const idea = localSurprise(song, seq(0.1));
    expect(idea.lyrics).toMatch(/\[Verse\]/);
    expect(idea.prompt.length).toBeGreaterThan(10);
  });

  it("keeps durations inside the range", () => {
    for (let i = 0; i < 50; i++) {
      const d = localSurprise(song).duration;
      expect(d).toBeGreaterThanOrEqual(10);
      expect(d).toBeLessThanOrEqual(600);
    }
  });
});

describe("surpriseInstruction", () => {
  it("asks for lyrics only when singing", () => {
    expect(surpriseInstruction(song, true, seq(0))).toContain('"lyrics"');
    expect(surpriseInstruction(song, false, seq(0))).not.toContain('"lyrics"');
  });
});

describe("parseSurpriseReply", () => {
  it("reads a JSON object out of surrounding text and thinking", () => {
    const idea = parseSurpriseReply(
      '<think>hmm</think>Sure! {"prompt": "dusty lo-fi, 80 BPM", "duration": 45} enjoy',
      song,
      false,
    );
    expect(idea).toEqual({ prompt: "dusty lo-fi, 80 BPM", lyrics: "", duration: 45 });
  });

  it("joins array lyrics and clamps the duration", () => {
    const idea = parseSurpriseReply(
      JSON.stringify({ prompt: "p", lyrics: ["[Verse]", "la"], duration: "9999" }),
      song,
      true,
    );
    expect(idea.lyrics).toBe("[Verse]\nla");
    expect(idea.duration).toBe(600);
  });

  it("rejects replies it cannot use", () => {
    expect(() => parseSurpriseReply("no json here", song, false)).toThrow();
    expect(() => parseSurpriseReply('{"duration": 30}', song, false)).toThrow();
    expect(() => parseSurpriseReply('{"prompt": "p", "duration": 30}', song, true)).toThrow();
  });
});
