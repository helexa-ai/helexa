// Browser fingerprint (FingerprintJS OSS) — best-effort, never auth.
//
// Computed ONLY at registration, where it is sent to the upstream as a
// multi-account abuse signal. Anonymous visitors are never fingerprinted:
// the chat page does not call this, anonymous local data is namespaced
// under the literal "anon" owner, and the anonymous send path carries no
// client identifier. Keeping the probe off the anonymous path is what
// lets the privacy page say "no tracking" without an asterisk (ePrivacy
// Art 5(3) treats fingerprinting like cookies; registration-time fraud
// prevention is the defensible narrow use).
//
// Cached in the Dexie `meta` store so it's computed at most once.

import FingerprintJS from "@fingerprintjs/fingerprintjs";
import { db } from "../data/db";

const META_KEY = "fingerprint";
let inFlight: Promise<string> | null = null;

export async function getFingerprint(): Promise<string> {
  const cached = await db.meta.get(META_KEY);
  if (cached && typeof cached.value === "string") return cached.value;
  if (inFlight) return inFlight;
  inFlight = (async () => {
    let id = "unknown";
    try {
      const fp = await FingerprintJS.load();
      const result = await fp.get();
      id = result.visitorId;
    } catch {
      // Fingerprinting is best-effort; fall back to a random local id so
      // registration still carries a stable per-browser signal.
      id = `local-${crypto.randomUUID()}`;
    }
    await db.meta.put({ key: META_KEY, value: id });
    return id;
  })();
  return inFlight;
}
