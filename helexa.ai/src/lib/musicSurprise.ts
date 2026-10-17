// "Surprise me" for the music page: an idea for a song, written by one of
// helexa's language models when one answers in time, otherwise drawn from
// the built-in lists below. Either way the result only fills the form —
// nothing is generated until the user presses Generate.

export interface SurpriseIdea {
  prompt: string;
  lyrics: string;
  duration: number;
}

export interface SurpriseTarget {
  /** Whether the chosen model sings. */
  vocals: boolean;
  minDuration: number;
  maxDuration: number;
}

const STYLES: { g: string; bpm: [number, number]; inst: string[] }[] = [
  { g: "synthwave", bpm: [95, 118], inst: ["analog arpeggiated synths", "gated reverb drums", "warm Juno pads", "punchy synth bass", "soaring lead synth"] },
  { g: "lo-fi hip hop", bpm: [70, 88], inst: ["dusty vinyl crackle", "mellow Rhodes chords", "boom-bap drums", "muted jazz guitar", "soft upright bass"] },
  { g: "deep house", bpm: [118, 124], inst: ["four-on-the-floor kick", "rolling sub bass", "shimmering open hi-hats", "stab chords", "airy vocal chops"] },
  { g: "cinematic orchestral", bpm: [70, 100], inst: ["swelling strings", "french horns", "timpani rolls", "choir pads", "solo cello"] },
  { g: "jazz trio", bpm: [110, 180], inst: ["brushed snare", "walking upright bass", "grand piano comping", "ride cymbal"] },
  { g: "ambient", bpm: [60, 80], inst: ["evolving granular pads", "distant piano notes", "field recordings of rain", "bowed metal textures", "tape-saturated drones"] },
  { g: "drum and bass", bpm: [170, 176], inst: ["chopped breakbeats", "reese bass", "atmospheric pads", "rimshot fills"] },
  { g: "indie rock", bpm: [110, 140], inst: ["jangly electric guitars", "driving bass", "live drum kit", "tambourine"] },
  { g: "bossa nova", bpm: [120, 140], inst: ["nylon-string guitar", "soft shaker", "upright bass", "flute melody"] },
  { g: "techno", bpm: [128, 136], inst: ["pounding kick", "hypnotic acid bassline", "metallic percussion", "dark rumble"] },
  { g: "afrobeat", bpm: [100, 115], inst: ["interlocking guitars", "horn section riffs", "talking drum", "shekere", "funky bassline"] },
  { g: "celtic folk", bpm: [100, 130], inst: ["tin whistle", "fiddle", "bodhrán", "acoustic guitar strumming"] },
  { g: "trip hop", bpm: [80, 95], inst: ["slow broken beats", "vinyl-sampled strings", "deep sub bass", "scratches"] },
  { g: "chiptune", bpm: [130, 160], inst: ["square wave lead", "noise-channel drums", "triangle bass", "fast arpeggios"] },
  { g: "flamenco", bpm: [100, 140], inst: ["rasgueado guitar", "palmas handclaps", "cajón"] },
  { g: "funk", bpm: [100, 112], inst: ["slap bass", "clavinet", "tight horn stabs", "wah guitar", "crisp snare"] },
  { g: "post-rock", bpm: [80, 120], inst: ["delay-drenched guitars", "crescendoing drums", "glockenspiel", "bowed bass"] },
  { g: "reggae", bpm: [70, 85], inst: ["offbeat skank guitar", "deep one-drop drums", "organ bubble", "heavy bass"] },
  { g: "baroque", bpm: [90, 120], inst: ["harpsichord", "string quartet", "continuo cello"] },
  { g: "city pop", bpm: [105, 120], inst: ["glossy electric piano", "slap bass", "brass section", "clean chorus guitar"] },
];
const MOODS = ["melancholic", "euphoric", "dreamy", "tense", "nostalgic", "playful", "triumphant", "brooding", "serene", "driving", "bittersweet", "mysterious"];
const PRODUCTION = ["wide stereo mix", "warm analog production", "crisp modern master", "lo-fi tape texture", "live room sound", "spacious reverb"];

const SHOWCASE_INSTRUMENTAL = [
  "Warm lo-fi hip hop beat with dusty vinyl crackle, mellow Rhodes chords and a lazy boom-bap groove, 82 BPM",
  "Epic cinematic trailer music: thunderous taiko drums, staccato strings and a massive brass swell, 90 BPM",
  "Solo fingerstyle acoustic guitar, gentle and intimate, in D major, recorded close in a small wooden room",
  "Minimal techno at 130 BPM, hypnotic acid bassline, crisp hats, dark warehouse atmosphere",
  "Peaceful ambient soundscape with slowly evolving pads, distant piano and soft rain, 70 BPM",
  "Upbeat 1980s synth-pop instrumental, gated snare, bright arpeggiated synths and a catchy lead melody, 118 BPM",
  "New Orleans brass band second line groove, sousaphone bass, trumpets and trombones, joyful, 105 BPM",
  "Tense stealth game music: pulsing low synth, ticking percussion, muted strings, 100 BPM",
];
const SHOWCASE_VOCAL = [
  "Dreamy shoegaze with walls of distorted guitar, breathy female vocals, washed-out drums",
  "Soulful neo-soul ballad, warm male vocals, Rhodes piano, brushed drums, 75 BPM",
  "High-energy pop punk anthem, shouted gang vocals, fast palm-muted guitars, 170 BPM",
  "Country ballad with pedal steel, acoustic guitar and a heartfelt male baritone",
  "Bright dance-pop track, layered female vocals, punchy synths and claps, 124 BPM",
  "Smoky late-night jazz club, sultry female vocals, muted trumpet, upright bass",
  "Anthemic stadium rock, powerful male vocals, big drums, soaring guitar solo",
  "Gentle indie folk duet, male and female harmonies, fingerpicked guitar, banjo",
];
// Original lyrics, written for helexa.
const LYRICS = [
  `[Verse]
Streetlights hum a song I used to know
Every window paints the rain in gold
I keep your number but I never call
Some things are sweeter when they stay unsaid

[Chorus]
So I drive, drive into the morning
Radio low and the city yawning
If you're out there, leave a light on for me
I'm halfway home, halfway free`,
  `[Verse]
We built a kingdom out of cardboard boxes
Painted the stars on the ceiling of your room
Summer was endless, the river was singing
Nobody told us it would end so soon

[Chorus]
Oh, carry me back to the long afternoons
Where the sky was a promise and the world was brand new
Oh, carry me back, carry me through
Everything I am, I learned it with you`,
  `[Verse]
Engines roaring, hearts on fire
We were born to climb up higher
No more waiting, no more fear
The only time we have is here

[Chorus]
Hey! Raise your hands up to the sky
Hey! We were never meant to hide
Tonight we're louder than the thunder
Tonight we're never going under`,
  `[Verse]
Coffee's cold and the clock says three
Paper moons and a melody
You left your sweater on the chair
I still talk like you're standing there

[Bridge]
Maybe love is just the space between
The things we say and the things we mean

[Chorus]
Slow dance in the kitchen light
Hold me like it's the last night
Don't say a word, don't say goodbye
Just slow dance in the kitchen light`,
  `[Verse]
Down by the harbour where the old boats lie
Fishermen sing to a silver sky
Salt on my lips and wind in my hair
I can feel the sea calling me there

[Chorus]
Sail away, sail away
Over the waves at the break of day
Wherever the current may carry me
I'll find my home on the open sea`,
  `[Verse]
Neon signals flicker in the dark
Electric heartbeat, digital spark
Running through the circuits of the night
Chasing shadows into the light

[Chorus]
We are the future, we are the sound
Spinning the planet round and round
Turn it up and let it go
Feel the bassline, lose control`,
];
const SPARKS = [
  "a rooftop at dawn", "a rainy night bus", "an abandoned arcade", "a desert highway", "a lighthouse in a storm",
  "a crowded night market", "a winter cabin", "a space station", "a summer wedding", "a submarine",
  "a 1920s speakeasy", "a heist", "a carnival after closing", "a mountain monastery", "a coastal road trip",
  "a basement rave", "a tea ceremony", "a thunderstorm over the plains", "a video game boss fight",
  "a detective's office", "a coral reef", "a late-night diner", "a forest at first snow", "a cyberpunk alley",
  "a farewell at a train station", "a childhood memory", "a festival main stage", "a quiet library",
];
const ERAS = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "contemporary", "futuristic", "timeless"];

type Rng = () => number;

function pick<T>(xs: readonly T[], rng: Rng): T {
  return xs[Math.floor(rng() * xs.length)];
}

function pickN<T>(xs: readonly T[], n: number, rng: Rng): T[] {
  const pool = [...xs];
  const out: T[] = [];
  while (out.length < n && pool.length) out.push(pool.splice(Math.floor(rng() * pool.length), 1)[0]);
  return out;
}

function rint(lo: number, hi: number, rng: Rng): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

/** Clamp a duration into the model's range, rounded to whole seconds. */
export function clampDuration(d: number, t: SurpriseTarget): number {
  const v = Number.isFinite(d) ? Math.round(d) : t.minDuration;
  return Math.min(t.maxDuration, Math.max(t.minDuration, v));
}

/** An idea from the built-in lists. */
export function localSurprise(t: SurpriseTarget, rng: Rng = Math.random): SurpriseIdea {
  const s = pick(STYLES, rng);
  const mood = pick(MOODS, rng);
  const composed = `${mood[0].toUpperCase()}${mood.slice(1)} ${s.g}, ${pickN(s.inst, 2, rng).join(" and ")}, ${pick(PRODUCTION, rng)}, ${rint(s.bpm[0], s.bpm[1], rng)} BPM`;
  // Showcase captions that describe a singer are only used with lyrics.
  const sing = t.vocals && rng() < 0.5;
  const prompt = sing
    ? rng() < 0.5
      ? pick(SHOWCASE_VOCAL, rng)
      : `${composed}, ${pick(["female", "male"], rng)} vocals`
    : rng() < 0.5
      ? pick(SHOWCASE_INSTRUMENTAL, rng)
      : `${composed}, instrumental`;
  return {
    prompt,
    lyrics: sing ? pick(LYRICS, rng) : "",
    duration: clampDuration(pick([30, 45, 60, 90, 120, 180], rng), t),
  };
}

/** The instruction given to the language model. */
export function surpriseInstruction(t: SurpriseTarget, sing: boolean, rng: Rng = Math.random): string {
  const hi = Math.min(t.maxDuration, 180);
  const lo = Math.min(Math.max(t.minDuration, 20), hi);
  const task = sing
    ? "Write a style caption for a song generator: genre, instruments, mood, tempo in BPM and the " +
      "singer's voice. Also write original lyrics of 2 verses and a chorus with [Verse] and [Chorus] " +
      `section tags, 16-24 lines in total. Suggest a duration between ${lo} and ${hi} seconds.`
    : "Write a prompt for an instrumental music generator: genre, instruments, mood, production " +
      `and tempo in BPM, with no vocals. Suggest a duration between ${lo} and ${hi} seconds.`;
  return (
    `${task}\n\nFor inspiration (interpret freely, do not quote it): ${pick(SPARKS, rng)}, ${pick(ERAS, rng)}.\n` +
    "The prompt must be one line of at most 40 words and must not mention the duration. " +
    "Do not name real artists, bands or songs.\n" +
    `Reply with only a JSON object with the keys "prompt", "duration"${sing ? ' and "lyrics"' : ""}.`
  );
}

/** Read the model's reply; throws when it is unusable. */
export function parseSurpriseReply(text: string, t: SurpriseTarget, sing: boolean): SurpriseIdea {
  const clean = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const start = clean.indexOf("{");
  const end = clean.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("no JSON object in reply");
  const obj = JSON.parse(clean.slice(start, end + 1)) as Record<string, unknown>;
  const prompt = typeof obj.prompt === "string" ? obj.prompt.trim() : "";
  if (!prompt) throw new Error("reply has no prompt");
  let lyrics = "";
  if (sing) {
    lyrics = Array.isArray(obj.lyrics)
      ? obj.lyrics.map(String).join("\n")
      : typeof obj.lyrics === "string"
        ? obj.lyrics.trim()
        : "";
    if (!lyrics) throw new Error("reply has no lyrics");
  }
  const d = typeof obj.duration === "number" ? obj.duration : Number.parseFloat(String(obj.duration));
  return { prompt, lyrics, duration: clampDuration(d, t) };
}

const SURPRISE_MODELS = (import.meta.env.VITE_SURPRISE_MODELS || "helexa/balanced,helexa/small")
  .split(",")
  .map((m: string) => m.trim())
  .filter(Boolean);

/**
 * An idea for the form, and where it came from. Tries helexa's language
 * models in order within a total deadline, then falls back to the
 * built-in lists — so the button always does something.
 */
export async function surprise(
  t: SurpriseTarget,
  opts: { apiKey?: string; baseUrl?: string; deadlineMs?: number } = {},
): Promise<{ idea: SurpriseIdea; source: "model" | "builtin" }> {
  const sing = t.vocals && Math.random() < 0.6;
  const base = (opts.baseUrl ?? import.meta.env.VITE_ROUTER_BASE_URL ?? "").replace(/\/$/, "");
  const deadline = Date.now() + (opts.deadlineMs ?? 30_000);
  const instruction = surpriseInstruction(t, sing);
  for (const model of SURPRISE_MODELS) {
    const remaining = deadline - Date.now();
    if (remaining < 3_000) break;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), remaining);
    try {
      const resp = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        signal: ctl.signal,
        headers: {
          "content-type": "application/json",
          ...(opts.apiKey ? { authorization: `Bearer ${opts.apiKey}` } : {}),
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: instruction }],
          temperature: 1.0,
          max_tokens: 700,
          chat_template_kwargs: { enable_thinking: false },
        }),
      });
      if (!resp.ok) continue;
      const body = (await resp.json()) as { choices?: { message?: { content?: string } }[] };
      const text = body.choices?.[0]?.message?.content ?? "";
      return { idea: parseSurpriseReply(text, t, sing), source: "model" };
    } catch {
      /* try the next model, then the built-in lists */
    } finally {
      clearTimeout(timer);
    }
  }
  return { idea: localSurprise(t), source: "builtin" };
}
