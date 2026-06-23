// Browser fingerprint (FingerprintJS OSS) — best-effort, never auth.
// Two uses: namespacing anonymous local data, and a soft client identifier
// the router can use to throttle the anonymous public path. Cached in the
// Dexie `meta` store so it's computed once.

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
      // anonymous data still has a stable namespace this session.
      id = `local-${crypto.randomUUID()}`;
    }
    await db.meta.put({ key: META_KEY, value: id });
    return id;
  })();
  return inFlight;
}
