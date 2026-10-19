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

/**
 * The genres "Surprise me" draws from, picked here rather than left to the
 * language model: asked for music for a scene, a small model answers with
 * ambient pads and synthwave nearly every time. Picking the genre first
 * spreads ideas evenly across the list by construction; the model only
 * fills in instruments, mood and lyrics inside it.
 *
 * `bpm` bounds the tempo it may choose. `eras` lists the decades the genre
 * belongs to (none: the era is left out). `sings: false` keeps a genre
 * that is almost always instrumental out of songs with lyrics.
 */
export interface Genre {
  g: string;
  bpm: [number, number];
  inst: string[];
  eras?: string[];
  sings?: boolean;
}

export const GENRES: Genre[] = [
  { g: "pop", bpm: [95, 125], inst: ["punchy drums", "bright synths", "clean electric guitar", "layered claps"], eras: ["1980s", "1990s", "2000s", "contemporary"] },
  { g: "dance-pop", bpm: [118, 128], inst: ["four-on-the-floor kick", "sidechained synths", "plucky bass", "vocal chops"], eras: ["2000s", "contemporary"] },
  { g: "synth-pop", bpm: [105, 125], inst: ["analog synth arpeggios", "gated reverb snare", "warm pads", "synth bass"], eras: ["1980s", "contemporary"] },
  { g: "K-pop", bpm: [100, 130], inst: ["hard-hitting trap drums", "glossy synths", "funky bass", "dramatic build-ups"] },
  { g: "J-pop", bpm: [120, 150], inst: ["bright piano", "busy bass lines", "energetic drums", "string stabs"] },
  { g: "city pop", bpm: [105, 120], inst: ["glossy electric piano", "slap bass", "brass section", "clean chorus guitar"], eras: ["1980s"] },
  { g: "indie pop", bpm: [100, 125], inst: ["jangly guitars", "simple drum machine", "warm bass", "glockenspiel"] },
  { g: "classic rock", bpm: [100, 140], inst: ["overdriven electric guitar", "hammond organ", "live drum kit", "bass guitar"], eras: ["1960s", "1970s"] },
  { g: "indie rock", bpm: [110, 145], inst: ["jangly electric guitars", "driving bass", "live drums", "tambourine"], eras: ["1990s", "2000s", "contemporary"] },
  { g: "punk rock", bpm: [160, 190], inst: ["distorted power chords", "fast snare", "pick bass", "gang shouts"], eras: ["1970s", "1990s"] },
  { g: "pop punk", bpm: [160, 180], inst: ["palm-muted guitars", "double-time drums", "melodic bass"], eras: ["2000s"] },
  { g: "grunge", bpm: [90, 130], inst: ["sludgy distorted guitars", "loud-quiet dynamics", "heavy drums"], eras: ["1990s"] },
  { g: "heavy metal", bpm: [120, 180], inst: ["chugging guitar riffs", "double kick drums", "screaming guitar solo"], eras: ["1980s", "contemporary"] },
  { g: "alternative metal", bpm: [90, 140], inst: ["drop-tuned guitars", "groove-heavy drums", "atmospheric breakdown"] },
  { g: "shoegaze", bpm: [85, 120], inst: ["walls of distorted guitar", "washed-out drums", "reverb-soaked textures"], eras: ["1990s", "contemporary"] },
  { g: "post-rock", bpm: [80, 120], inst: ["delay-drenched guitars", "crescendoing drums", "glockenspiel", "bowed bass"], sings: false },
  { g: "blues", bpm: [70, 110], inst: ["slide guitar", "harmonica", "shuffle drums", "upright bass"], eras: ["1950s", "1960s"] },
  { g: "soul", bpm: [80, 110], inst: ["horn section", "hammond organ", "tight drums", "warm bass"], eras: ["1960s", "1970s"] },
  { g: "neo-soul", bpm: [70, 95], inst: ["Rhodes piano", "lazy swung drums", "fretless bass", "soft guitar licks"] },
  { g: "R&B", bpm: [65, 100], inst: ["smooth keys", "808 bass", "crisp snaps", "lush vocal harmonies"], eras: ["1990s", "contemporary"] },
  { g: "funk", bpm: [100, 115], inst: ["slap bass", "clavinet", "tight horn stabs", "wah guitar"], eras: ["1970s"] },
  { g: "disco", bpm: [115, 125], inst: ["four-on-the-floor kick", "octave bass", "string section", "rhythm guitar"], eras: ["1970s"] },
  { g: "gospel", bpm: [70, 120], inst: ["hammond organ", "choir", "piano", "hand claps"] },
  { g: "boom-bap hip hop", bpm: [85, 95], inst: ["dusty drum breaks", "sampled jazz chords", "deep bass", "scratches"], eras: ["1990s"] },
  { g: "trap", bpm: [130, 150], inst: ["rolling hi-hats", "booming 808s", "dark synth melody", "snare rolls"] },
  { g: "lo-fi hip hop", bpm: [70, 88], inst: ["dusty vinyl crackle", "mellow Rhodes chords", "boom-bap drums", "muted jazz guitar"], sings: false },
  { g: "drill", bpm: [138, 146], inst: ["sliding 808 bass", "syncopated hi-hats", "ominous piano"] },
  { g: "reggae", bpm: [70, 85], inst: ["offbeat skank guitar", "one-drop drums", "organ bubble", "heavy bass"], eras: ["1970s", "contemporary"] },
  { g: "dancehall", bpm: [90, 105], inst: ["digital riddim", "punchy kick", "synth stabs"] },
  { g: "reggaeton", bpm: [88, 100], inst: ["dembow drums", "deep bass", "plucked synths"] },
  { g: "salsa", bpm: [160, 200], inst: ["piano montuno", "congas and timbales", "brass section", "bass tumbao"] },
  { g: "bossa nova", bpm: [120, 140], inst: ["nylon-string guitar", "soft shaker", "upright bass", "flute melody"], eras: ["1960s"] },
  { g: "samba", bpm: [95, 105], inst: ["surdo drums", "cavaquinho", "tamborim", "whistles"] },
  { g: "cumbia", bpm: [85, 100], inst: ["accordion", "güira", "bass", "gaita flute"] },
  { g: "flamenco", bpm: [100, 140], inst: ["rasgueado guitar", "palmas handclaps", "cajón"] },
  { g: "afrobeats", bpm: [100, 115], inst: ["log drums", "syncopated percussion", "warm synth chords", "guitar licks"] },
  { g: "afrobeat", bpm: [100, 115], inst: ["interlocking guitars", "horn section riffs", "talking drum", "funky bassline"], eras: ["1970s"] },
  { g: "amapiano", bpm: [110, 115], inst: ["log drum bass", "shakers", "jazzy piano chords"] },
  { g: "highlife", bpm: [110, 130], inst: ["bright interlocking guitars", "horns", "light percussion"] },
  { g: "country", bpm: [80, 120], inst: ["pedal steel", "acoustic guitar", "fiddle", "brushed snare"], eras: ["1970s", "1990s", "contemporary"] },
  { g: "bluegrass", bpm: [120, 160], inst: ["banjo", "mandolin", "fiddle", "upright bass"] },
  { g: "americana", bpm: [80, 110], inst: ["acoustic guitar", "lap steel", "harmonica", "warm drums"] },
  { g: "indie folk", bpm: [80, 115], inst: ["fingerpicked guitar", "banjo", "soft harmonies", "stomp and clap"] },
  { g: "celtic folk", bpm: [100, 130], inst: ["tin whistle", "fiddle", "bodhrán", "acoustic guitar"] },
  { g: "singer-songwriter", bpm: [70, 105], inst: ["solo acoustic guitar", "gentle piano", "light percussion"] },
  { g: "jazz", bpm: [110, 180], inst: ["brushed snare", "walking upright bass", "piano comping", "muted trumpet"], eras: ["1950s", "1960s", "contemporary"] },
  { g: "big band swing", bpm: [140, 190], inst: ["brass section", "saxophones", "walking bass", "swing drums"], eras: ["1940s"] },
  { g: "jazz fusion", bpm: [100, 130], inst: ["electric piano", "fretless bass", "tight drums", "synth lead"], eras: ["1970s"] },
  { g: "house", bpm: [118, 126], inst: ["four-on-the-floor kick", "rolling bass", "open hi-hats", "piano stabs"] },
  { g: "deep house", bpm: [118, 124], inst: ["warm sub bass", "shimmering hats", "dub chords", "airy pads"] },
  { g: "techno", bpm: [126, 138], inst: ["pounding kick", "hypnotic synth loop", "metallic percussion", "dark rumble"], sings: false },
  { g: "trance", bpm: [132, 140], inst: ["supersaw leads", "rolling bassline", "big breakdown", "uplifting arpeggios"] },
  { g: "drum and bass", bpm: [170, 176], inst: ["chopped breakbeats", "reese bass", "atmospheric pads"] },
  { g: "dubstep", bpm: [138, 142], inst: ["wobble bass", "half-time drums", "sub drops"] },
  { g: "UK garage", bpm: [130, 136], inst: ["shuffled 2-step drums", "chopped vocal samples", "warm organ bass"] },
  { g: "synthwave", bpm: [95, 118], inst: ["analog arpeggiated synths", "gated reverb drums", "punchy synth bass", "soaring lead"] },
  { g: "trip hop", bpm: [80, 95], inst: ["slow broken beats", "vinyl-sampled strings", "deep sub bass", "scratches"], eras: ["1990s"] },
  { g: "chiptune", bpm: [130, 160], inst: ["square wave lead", "noise-channel drums", "triangle bass", "fast arpeggios"], sings: false },
  { g: "ambient", bpm: [60, 80], inst: ["evolving pads", "distant piano", "field recordings", "tape-saturated drones"], sings: false },
  { g: "cinematic orchestral", bpm: [70, 110], inst: ["swelling strings", "french horns", "timpani", "choir pads"], sings: false },
  { g: "classical piano", bpm: [60, 120], inst: ["solo grand piano"], sings: false },
  { g: "baroque", bpm: [90, 120], inst: ["harpsichord", "string quartet", "continuo cello"], sings: false },
  { g: "string quartet", bpm: [70, 120], inst: ["two violins", "viola", "cello"], sings: false },
  { g: "marching band", bpm: [110, 125], inst: ["snare line", "sousaphone", "trumpets", "cymbals"], sings: false },
  { g: "Bollywood", bpm: [90, 130], inst: ["tabla", "dholak", "strings", "sitar flourishes"] },
  { g: "Arabic pop", bpm: [90, 120], inst: ["oud", "darbuka", "qanun", "synth strings"] },
  { g: "Balkan brass", bpm: [130, 160], inst: ["trumpets", "tuba", "tapan drum", "clarinet"] },
  { g: "polka", bpm: [110, 130], inst: ["accordion", "tuba", "clarinet", "snare"] },
];

const MOODS = ["melancholic", "euphoric", "dreamy", "tense", "nostalgic", "playful", "triumphant", "brooding", "serene", "driving", "bittersweet", "mysterious"];
const PRODUCTION = ["wide stereo mix", "warm analog production", "crisp modern master", "lo-fi tape texture", "live room sound", "spacious reverb"];

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

/** What one press of "Surprise me" asks for, chosen before any model is asked. */
export interface Brief {
  genre: Genre;
  mood: string;
  /** Only for genres tied to particular decades. */
  era?: string;
  /** What the lyrics are about; only for a song that is sung. */
  theme?: string;
  /** Picked here: asked for a range, the model copies the range. */
  bpm: number;
  sing: boolean;
}

/** Draw a brief: a genre from the whole list, evenly. */
export function pickBrief(t: SurpriseTarget, rng: Rng = Math.random): Brief {
  const sing = t.vocals && rng() < 0.6;
  const genre = pick(
    GENRES.filter((g) => !sing || g.sings !== false),
    rng,
  );
  const era = genre.eras && rng() < 0.5 ? pick(genre.eras, rng) : undefined;
  return {
    genre,
    mood: pick(MOODS, rng),
    era,
    theme: sing ? pick(SPARKS, rng) : undefined,
    bpm: rint(genre.bpm[0], genre.bpm[1], rng),
    sing,
  };
}

/** An idea from the built-in lists, for when no language model answers. */
export function localSurprise(t: SurpriseTarget, rng: Rng = Math.random, brief: Brief = pickBrief(t, rng)): SurpriseIdea {
  const { genre, mood, era, sing, bpm } = brief;
  const style = `${era ? `${era} ` : ""}${genre.g}`;
  const composed = `${mood[0].toUpperCase()}${mood.slice(1)} ${style}, ${pickN(genre.inst, 2, rng).join(" and ")}, ${pick(PRODUCTION, rng)}, ${bpm} BPM`;
  return {
    prompt: sing ? `${composed}, ${pick(["female", "male"], rng)} vocals` : `${composed}, instrumental`,
    lyrics: sing ? pick(LYRICS, rng) : "",
    duration: clampDuration(pick([30, 45, 60, 90, 120, 180], rng), t),
  };
}

/** The instruction given to the language model for a brief. */
export function surpriseInstruction(t: SurpriseTarget, brief: Brief): string {
  const hi = Math.min(t.maxDuration, 180);
  const lo = Math.min(Math.max(t.minDuration, 20), hi);
  const { genre, mood, era, theme, sing, bpm } = brief;
  const style = `${era ? `${era} ` : ""}${genre.g}`;
  const lines = [
    sing
      ? `Write a style caption for a song generator. The song is ${style}, with a ${mood} mood.`
      : `Write a prompt for an instrumental music generator. The track is ${style}, with a ${mood} mood, and has no vocals.`,
    `Name instruments typical of ${genre.g} (for example ${genre.inst.join(", ")}) and the production sound. ` +
      "Add synth or ambient pads only if the genre really uses them.",
    `The tempo is ${bpm} BPM; write it as "${bpm} BPM".`,
  ];
  if (sing) {
    lines.push(
      "Describe the singer's voice in the caption. Also write original lyrics about " +
        `${theme}: 2 verses and a chorus with [Verse] and [Chorus] section tags, 16-24 lines in total.`,
    );
  }
  lines.push(
    `Suggest a duration between ${lo} and ${hi} seconds.`,
    `The prompt must be one line of at most 40 words that describes only the sound, must name the genre ${genre.g}, and must not mention the duration. ` +
      "Do not name real artists, bands or songs.",
    `Reply with only a JSON object with the keys "prompt", "duration"${sing ? ' and "lyrics"' : ""}.`,
  );
  return lines.join("\n");
}

/**
 * Read the model's reply; throws when it is unusable. With `bpm`, a
 * prompt that leaves out the tempo gets it appended.
 */
export function parseSurpriseReply(text: string, t: SurpriseTarget, sing: boolean, bpm?: number): SurpriseIdea {
  const clean = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const start = clean.indexOf("{");
  const end = clean.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("no JSON object in reply");
  const obj = JSON.parse(clean.slice(start, end + 1)) as Record<string, unknown>;
  let prompt = typeof obj.prompt === "string" ? obj.prompt.trim() : "";
  if (!prompt) throw new Error("reply has no prompt");
  if (bpm !== undefined && !/\d\s*BPM/i.test(prompt)) {
    prompt = `${prompt.replace(/[.\s]+$/, "")}, ${bpm} BPM`;
  }
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
  const brief = pickBrief(t);
  const { sing } = brief;
  const base = (opts.baseUrl ?? import.meta.env.VITE_ROUTER_BASE_URL ?? "").replace(/\/$/, "");
  const deadline = Date.now() + (opts.deadlineMs ?? 30_000);
  const instruction = surpriseInstruction(t, brief);
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
      return { idea: parseSurpriseReply(text, t, sing, brief.bpm), source: "model" };
    } catch {
      /* try the next model, then the built-in lists */
    } finally {
      clearTimeout(timer);
    }
  }
  return { idea: localSurprise(t, Math.random, brief), source: "builtin" };
}
