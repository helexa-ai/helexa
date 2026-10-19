import { describe, expect, it } from "vitest";
import {
  GENRES,
  clampDuration,
  localSurprise,
  parseSurpriseReply,
  pickBrief,
  surpriseInstruction,
} from "./musicSurprise";

const song = { vocals: true, minDuration: 10, maxDuration: 600 };
const sfx = { vocals: false, minDuration: 1, maxDuration: 10 };

/** A seeded pseudo-random generator (mulberry32), for draws over many ideas. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

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

describe("pickBrief", () => {
  it("spreads ideas across the whole genre list", () => {
    const rng = seeded(1);
    const counts = new Map<string, number>();
    const draws = 7000;
    for (let i = 0; i < draws; i++) {
      const g = pickBrief(song, rng).genre.g;
      counts.set(g, (counts.get(g) ?? 0) + 1);
    }
    expect(counts.size).toBe(GENRES.length);
    // No genre may dominate: under twice its fair share even counting the
    // instrumental-only genres that singing ideas skip.
    const fair = draws / GENRES.length;
    expect(Math.max(...counts.values())).toBeLessThan(2 * fair);
  });

  it("gives sung ideas a singing genre and a lyric theme, and instrumentals neither", () => {
    const rng = seeded(2);
    for (let i = 0; i < 500; i++) {
      const b = pickBrief(song, rng);
      if (b.sing) {
        expect(b.genre.sings).not.toBe(false);
        expect(b.theme).toBeTruthy();
      } else {
        expect(b.theme).toBeUndefined();
      }
      expect(pickBrief(sfx, rng).sing).toBe(false);
    }
  });

  it("only gives an era the genre belongs to", () => {
    const rng = seeded(3);
    for (let i = 0; i < 500; i++) {
      const b = pickBrief(song, rng);
      if (b.era) expect(b.genre.eras).toContain(b.era);
    }
  });
});

describe("surpriseInstruction", () => {
  const genre = GENRES.find((g) => g.g === "bluegrass")!;

  it("pins the genre, mood and tempo range", () => {
    const text = surpriseInstruction(song, { genre, mood: "playful", bpm: 142, sing: false });
    expect(text).toContain("bluegrass");
    expect(text).toContain("playful");
    expect(text).toContain('"142 BPM"');
    expect(text).toContain("banjo");
    expect(text).toContain("no vocals");
    expect(text).not.toContain('"lyrics"');
  });

  it("asks for lyrics about the theme only when singing", () => {
    const text = surpriseInstruction(song, {
      genre,
      mood: "bittersweet",
      era: "1970s",
      theme: "a coastal road trip",
      bpm: 130,
      sing: true,
    });
    expect(text).toContain("1970s bluegrass");
    expect(text).toContain("lyrics about a coastal road trip");
    expect(text).toContain('"lyrics"');
  });
});

describe("localSurprise (genres)", () => {
  it("names the brief's genre and keeps its tempo range", () => {
    const rng = seeded(4);
    for (let i = 0; i < 300; i++) {
      const brief = pickBrief(song, rng);
      const idea = localSurprise(song, rng, brief);
      expect(idea.prompt).toContain(brief.genre.g);
      expect(idea.prompt).toContain(`${brief.bpm} BPM`);
      expect(brief.bpm).toBeGreaterThanOrEqual(brief.genre.bpm[0]);
      expect(brief.bpm).toBeLessThanOrEqual(brief.genre.bpm[1]);
      expect(idea.lyrics === "").toBe(!brief.sing);
    }
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

  it("appends the picked tempo when the model leaves it out", () => {
    const missing = parseSurpriseReply('{"prompt": "Samba with surdo and agogô.", "duration": 30}', song, false, 98);
    expect(missing.prompt).toBe("Samba with surdo and agogô, 98 BPM");
    const present = parseSurpriseReply('{"prompt": "Samba at 98 BPM", "duration": 30}', song, false, 98);
    expect(present.prompt).toBe("Samba at 98 BPM");
  });

  it("rejects replies it cannot use", () => {
    expect(() => parseSurpriseReply("no json here", song, false)).toThrow();
    expect(() => parseSurpriseReply('{"duration": 30}', song, false)).toThrow();
    expect(() => parseSurpriseReply('{"prompt": "p", "duration": 30}', song, true)).toThrow();
  });
});
