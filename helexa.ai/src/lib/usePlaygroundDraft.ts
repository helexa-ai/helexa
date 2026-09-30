import { useEffect, useRef, useState } from "react";
import type { Draft } from "./decisionRequest";
import { decodeShare } from "./decisionShare";

/** Where the playground keeps the visitor's draft between visits. */
export const DRAFT_STORAGE_KEY = "hx.playground.draft";

function loadStored(): Draft | null {
  try {
    const raw = localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw) as Partial<Draft>;
    if (typeof d.stateText !== "string" || typeof d.questionsText !== "string") return null;
    return {
      stateText: d.stateText,
      stateMode: d.stateMode === "json" ? "json" : "text",
      questionsText: d.questionsText,
      checkpoint: d.checkpoint ?? "auto",
      maxLen: d.maxLen,
      headMaxLen: d.headMaxLen,
    };
  } catch {
    return null;
  }
}

/**
 * The playground's draft, restored and remembered.
 *
 * A share link wins: opening one shows the shared request, not whatever
 * the visitor last had open, and does not run it. Without one, the last
 * draft comes back from localStorage, so a reload never loses work. Every
 * later edit is saved. Storage failures (private windows, quota) are
 * ignored: the page still works, it just forgets.
 */
export function usePlaygroundDraft(fallback: Draft): {
  draft: Draft;
  setDraft: (d: Draft | ((prev: Draft) => Draft)) => void;
  /** True once a share link has been restored into the editors. */
  fromShare: boolean;
} {
  const [draft, setDraft] = useState<Draft>(() => loadStored() ?? fallback);
  const [fromShare, setFromShare] = useState(false);
  const restoring = useRef(typeof window !== "undefined" && window.location.hash.includes("p="));

  useEffect(() => {
    if (!restoring.current) return;
    let cancelled = false;
    void decodeShare(window.location.hash).then((shared) => {
      restoring.current = false;
      if (cancelled || !shared) return;
      setDraft(shared);
      setFromShare(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (restoring.current) return;
    try {
      localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(draft));
    } catch {
      /* storage unavailable: nothing to do */
    }
  }, [draft]);

  return { draft, setDraft, fromShare };
}
