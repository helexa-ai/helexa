import { useLiveQuery } from "dexie-react-hooks";
import { Navigate } from "react-router-dom";
import { useAuth } from "../auth/context";
import { db } from "../data/db";
import Landing from "./Landing";

/**
 * Decides what `/` is.
 *
 * The marketing landing is for people who have not used helexa. Anyone
 * signed in, or who already has conversations in this browser, gets the
 * workspace — showing a returning user an advert for the thing they are
 * already using is the mistake the previous version made, and it was
 * visible: a sidebar full of their own chats beside a pitch.
 *
 * `undefined` from the live query means "still reading IndexedDB"; render
 * nothing rather than flashing the landing at someone who is about to be
 * redirected past it.
 */
export default function Root() {
  const { status, accountId } = useAuth();
  const authed = status === "authed" && !!accountId;

  // Defaults to false so the landing renders immediately. Blocking on the
  // query to avoid a flash meant rendering nothing at all if it never
  // settled — a blank front door is a far worse failure than a returning
  // visitor seeing the landing for one frame before the redirect.
  const hasHistory =
    useLiveQuery(async () => (await db.conversations.count()) > 0, [], false) ?? false;

  if (authed || hasHistory) return <Navigate to="/chat" replace />;
  return <Landing />;
}
