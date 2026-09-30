// Share links for the decision playground.
//
// The draft travels in the URL *fragment*, which browsers never send to a
// server: opening a shared link reveals nothing to anyone until the
// recipient presses Run. The payload is deflated (the editors are mostly
// repetitive JSON) and base64url-encoded behind a version number, so the
// format can change later without breaking links already handed out.

import { CHECKPOINTS, type Checkpoint, type Draft } from "./decisionRequest";

/** Fragment parameter carrying a shared draft. */
const PARAM = "p";
/** Current payload version. Bump it when the payload shape changes, and
 *  keep decoding the old ones. */
const VERSION = "1";

/**
 * Longest link offered. Browsers accept far longer URLs, but chat apps,
 * mail clients and link shorteners truncate well before that, and a
 * truncated link fails to decode. Past this the UI says so instead.
 */
export const MAX_SHARE_URL = 8000;

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(s: string): Uint8Array {
  if (!/^[A-Za-z0-9_-]*$/.test(s)) throw new Error("not base64url");
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Blob([bytes as BlobPart]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}

/** The draft as a URL fragment (`#p=1.<data>`). */
export async function encodeShare(draft: Draft): Promise<string> {
  const payload = JSON.stringify({
    s: draft.stateText,
    m: draft.stateMode,
    q: draft.questionsText,
    c: draft.checkpoint,
    ...(draft.maxLen ? { l: draft.maxLen } : {}),
    ...(draft.headMaxLen ? { h: draft.headMaxLen } : {}),
  });
  const packed = await pipe(new TextEncoder().encode(payload), new CompressionStream("deflate-raw"));
  return `#${PARAM}=${VERSION}.${toBase64Url(packed)}`;
}

/**
 * The draft a fragment carries, or null when it carries none, comes from
 * an unknown version, or is damaged in any way. Never throws: a broken
 * link must still open an ordinary playground.
 */
export async function decodeShare(hash: string): Promise<Draft | null> {
  try {
    const raw = new URLSearchParams(hash.replace(/^#/, "")).get(PARAM);
    if (!raw) return null;
    const dot = raw.indexOf(".");
    if (dot < 0 || raw.slice(0, dot) !== VERSION) return null;
    const bytes = await pipe(fromBase64Url(raw.slice(dot + 1)), new DecompressionStream("deflate-raw"));
    const p = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as Record<
      string,
      unknown
    >;
    if (typeof p.s !== "string" || typeof p.q !== "string") return null;
    return {
      stateText: p.s,
      stateMode: p.m === "json" ? "json" : "text",
      questionsText: p.q,
      checkpoint: CHECKPOINTS.includes(p.c as Checkpoint) ? (p.c as Checkpoint) : "auto",
      maxLen: typeof p.l === "string" ? p.l : undefined,
      headMaxLen: typeof p.h === "string" ? p.h : undefined,
    };
  } catch {
    return null;
  }
}
