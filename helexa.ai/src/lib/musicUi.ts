/**
 * Pure helpers behind the `/music` page: cover art, style tags, the
 * generate button's blocking reason, clock formatting and waveform peaks.
 * Kept out of the component so they can be tested without a DOM.
 */

/** FNV-1a: a small, stable string hash. Cover art must not change between
 * visits, so nothing here may depend on randomness or time. */
function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * A generated cover for a song: two hues and an angle derived from its
 * id, so every song is recognisable in the history and on its card
 * without an image model or a stored picture.
 */
export function coverArt(id: string): string {
  const h = hash(id);
  const a = h % 360;
  const b = (a + 40 + ((h >>> 9) % 140)) % 360;
  const angle = (h >>> 17) % 360;
  return `linear-gradient(${angle}deg, hsl(${a} 75% 55%), hsl(${b} 70% 35%))`;
}

/**
 * Suggested style words, offered as one-click chips under the style box.
 * They stay in English whatever the interface language: they are words
 * for the model, whose style vocabulary is English.
 */
export const STYLE_TAGS = [
  "pop",
  "synthwave",
  "lo-fi hip hop",
  "rock",
  "jazz",
  "orchestral",
  "acoustic folk",
  "house",
  "ambient",
  "uplifting",
  "melancholic",
  "dreamy",
  "energetic",
] as const;

/** The comma-separated phrases of a style prompt, trimmed and lowercased. */
function phrases(prompt: string): string[] {
  return prompt
    .split(",")
    .map((p) => p.trim().toLowerCase())
    .filter(Boolean);
}

/** Whether the prompt already holds `tag` as one of its phrases. */
export function hasTag(prompt: string, tag: string): boolean {
  return phrases(prompt).includes(tag.toLowerCase());
}

/**
 * Add `tag` to a style prompt, or take it out if it is already there.
 * Everything the visitor typed is kept as written; only the tag's own
 * phrase is added or removed.
 */
export function toggleTag(prompt: string, tag: string): string {
  const parts = prompt
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  const lower = tag.toLowerCase();
  const kept = parts.filter((p) => p.toLowerCase() !== lower);
  return (kept.length === parts.length ? [...parts, tag] : kept).join(", ");
}

export type Blocker =
  | { reason: "preparing" }
  | { reason: "noModel" }
  | { reason: "surprising" }
  | { reason: "submitting" }
  | { reason: "needStyle" }
  | { reason: "needLength"; min: number; max: number };

/**
 * Why the generate button cannot be pressed right now, or null when it
 * can. The first reason that applies wins, ordered by what the visitor
 * can least do anything about.
 */
export function generateBlocker(s: {
  ready: boolean;
  hasModel: boolean;
  surprising: boolean;
  submitting: boolean;
  prompt: string;
  seconds: number;
  min?: number;
  max?: number;
}): Blocker | null {
  if (!s.ready) return { reason: "preparing" };
  if (!s.hasModel) return { reason: "noModel" };
  if (s.surprising) return { reason: "surprising" };
  if (s.submitting) return { reason: "submitting" };
  if (!s.prompt.trim()) return { reason: "needStyle" };
  if (
    s.min !== undefined &&
    s.max !== undefined &&
    (!Number.isFinite(s.seconds) || s.seconds < s.min || s.seconds > s.max)
  ) {
    return { reason: "needLength", min: s.min, max: s.max };
  }
  return null;
}

/** `m:ss` for a duration in seconds; negative or non-finite reads 0:00. */
export function clock(seconds: number): string {
  const s = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * Peak level of each of `bars` equal slices of the audio, the loudest
 * channel winning, scaled so the loudest bar is 1. Silence stays 0.
 */
export function peaks(channels: Float32Array[], bars: number): number[] {
  const length = channels[0]?.length ?? 0;
  if (length === 0 || bars <= 0) return [];
  const out = new Array<number>(bars).fill(0);
  for (let b = 0; b < bars; b++) {
    const from = Math.floor((b * length) / bars);
    const to = Math.max(from + 1, Math.floor(((b + 1) * length) / bars));
    let peak = 0;
    for (const ch of channels) {
      for (let i = from; i < to && i < ch.length; i++) {
        const v = Math.abs(ch[i]);
        if (v > peak) peak = v;
      }
    }
    out[b] = peak;
  }
  const max = Math.max(...out);
  return max > 0 ? out.map((v) => v / max) : out;
}

/**
 * Stand-in bars drawn before a song's audio has been fetched: a stable,
 * music-like shape per song, so the card looks like the player it is.
 */
export function placeholderPeaks(id: string, bars: number): number[] {
  let h = hash(id) || 1;
  return Array.from({ length: bars }, (_, i) => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    const jitter = ((h >>> 0) % 1000) / 1000;
    const swell = 0.55 + 0.35 * Math.sin((i / bars) * Math.PI);
    return Math.min(1, swell * (0.45 + 0.55 * jitter));
  });
}
