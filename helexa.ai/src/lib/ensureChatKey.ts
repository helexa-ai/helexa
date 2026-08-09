import { accountApi } from "../api/account";
import { CHAT_API_KEY, CHAT_API_KEY_ID, db } from "../data/db";

/**
 * Provision this browser's chat key on demand.
 *
 * Signing in used to make the product worse: an anonymous visitor could
 * chat immediately, but a signed-in one hit a disabled composer until they
 * went and created an API key by hand. This closes that by minting one the
 * first time a signed-in browser needs it.
 *
 * Done here rather than at email verification on purpose. The key is stored
 * in IndexedDB, which is per-browser — and verification links are usually
 * opened wherever the mail is, often a different device from the one the
 * person signed up on. Minting at verification would leave the original
 * browser just as stuck, and the raw key is returned exactly once, so it
 * could never be recovered for it. Minting on demand instead covers every
 * device, a cleared cache, a reinstall and a private window, all with the
 * same code path.
 *
 * One key per browser is deliberate: it makes "revoke the laptop I lost" a
 * thing you can actually do, which a single shared key cannot express.
 */

/** Label the key so it is identifiable — and revocable — in the account UI. */
function deviceLabel(): string {
  const ua = navigator.userAgent;
  const browser =
    /Firefox\//.test(ua) ? "Firefox"
    : /Edg\//.test(ua) ? "Edge"
    : /Chrome\//.test(ua) ? "Chrome"
    : /Safari\//.test(ua) ? "Safari"
    : "browser";
  const os =
    /Android/.test(ua) ? "Android"
    : /iPhone|iPad/.test(ua) ? "iOS"
    : /Mac OS X/.test(ua) ? "macOS"
    : /Windows/.test(ua) ? "Windows"
    : /Linux/.test(ua) ? "Linux"
    : "";
  const when = new Date().toISOString().slice(0, 10);
  return `web chat — ${browser}${os ? ` on ${os}` : ""}, ${when}`;
}

/**
 * In-flight guard. React runs effects twice in StrictMode and the chat
 * re-renders freely; without this a single visit could mint several keys.
 * Module scope, so it is shared by every caller in the tab.
 */
let inFlight: Promise<void> | null = null;
/**
 * Stop retrying after a failure for the rest of the session. A failing
 * mint (offline, upstream down, session expired) must not turn into a
 * request loop against the account service — the UI falls back to the
 * existing "create a key" prompt, which still works by hand.
 */
let failed = false;

export function ensureChatKey(token: string): Promise<void> {
  if (failed) return Promise.resolve();
  if (inFlight) return inFlight;

  inFlight = (async () => {
    // Re-check inside the guard: another tab may have provisioned one
    // since this call was scheduled.
    const existing = await db.meta.get(CHAT_API_KEY);
    if (typeof existing?.value === "string" && existing.value) return;

    // percent/100 = bounded by the account allocation rather than a
    // separate hard cap, so the key spends the signup grant and nothing
    // more. The account's own budget stays the single ceiling.
    const created = await accountApi().createKey(token, deviceLabel(), "percent", 100);
    await db.meta.put({ key: CHAT_API_KEY, value: created.key });
    await db.meta.put({ key: CHAT_API_KEY_ID, value: created.id });
  })()
    .catch((e) => {
      // Deliberately quiet: the user still has the manual path, and an
      // error banner here would be noise for something they never asked
      // for. Visible in the console for support.
      failed = true;
      console.warn("could not provision a chat key automatically", e);
    })
    .finally(() => {
      inFlight = null;
    });

  return inFlight;
}
