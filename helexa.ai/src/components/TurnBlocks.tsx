import { useState } from "react";
import { useTranslation } from "react-i18next";
import { FaChevronRight, FaBrain, FaWrench } from "react-icons/fa6";
import type { TurnBlock } from "../data/db";
import { isOpen } from "../lib/turnBlockState";

/**
 * The assistant's working — reasoning and tool calls — shown above its
 * answer.
 *
 * Two things are in tension. All of this should be *available*, because a
 * model that reasons for two minutes and then answers has done work worth
 * inspecting, and a tool call the user cannot see is a claim they cannot
 * check. But none of it is what they asked for, so it must not bury the
 * answer.
 *
 * The resolution is that visibility follows activity:
 *
 *   * While a block is the live one — still streaming, no answer text yet
 *     — it is open. That span is otherwise dead air; a spinner would say
 *     "working", this says what it is working *on*.
 *   * The moment answer text starts, every block collapses to a one-line
 *     header. The thing the user asked for takes the space.
 *   * Completed turns load collapsed, so scrollback stays readable.
 *
 * A block the user opens or closes by hand stays that way. Explicit
 * intent outranks the automatic rule, including for the live block —
 * someone who closes a noisy reasoning stream should not have it reopen
 * itself on the next token.
 */
export function TurnBlocks({
  blocks,
  streaming,
  answerStarted,
}: {
  blocks: TurnBlock[];
  /** This turn is still being generated. */
  streaming: boolean;
  /** Answer text has begun arriving, so the working is now history. */
  answerStarted: boolean;
}) {
  const { t } = useTranslation();
  // Only blocks the user has touched appear here; everything else follows
  // the automatic rule. Keyed by index, which is stable because blocks
  // are only ever appended within a turn.
  const [overrides, setOverrides] = useState<Record<number, boolean>>({});

  if (blocks.length === 0) return null;

  return (
    <div className="hx-blocks">
      {blocks.map((b, i) => {
        const open = isOpen(i, blocks.length, streaming, answerStarted, overrides[i]);
        const body = b.kind === "reasoning" ? b.text : blockToolBody(b);
        return (
          <div key={i} className={`hx-block${open ? " hx-block-open" : ""}`}>
            <button
              type="button"
              className="hx-block-head"
              aria-expanded={open}
              onClick={() => setOverrides((o) => ({ ...o, [i]: !open }))}
            >
              <FaChevronRight
                className={`hx-block-caret${open ? " hx-block-caret-open" : ""}`}
                size={10}
                aria-hidden="true"
              />
              {b.kind === "reasoning" ? (
                <FaBrain size={11} aria-hidden="true" />
              ) : (
                <FaWrench size={11} aria-hidden="true" />
              )}
              <span className="hx-block-label">
                {b.kind === "reasoning"
                  ? t("chat:blockReasoning")
                  : t("chat:blockTool", { name: b.name })}
              </span>
              {/* A pending tool call is the one case where the header
                * alone should say something is happening: its body is
                * empty until the result lands, so an open block would
                * otherwise look like a stall. */}
              {b.kind === "tool" && b.result === undefined && streaming && (
                <span className="hx-block-running">{t("chat:blockRunning")}</span>
              )}
              {!open && (
                <span className="hx-block-peek">{peek(body)}</span>
              )}
            </button>
            {open && <div className="hx-block-body">{body}</div>}
          </div>
        );
      })}
    </div>
  );
}

/** Arguments, then the result once it arrives. */
function blockToolBody(b: Extract<TurnBlock, { kind: "tool" }>): string {
  const args = prettyArgs(b.args);
  return b.result === undefined ? args : `${args}\n\n${b.result}`;
}

/**
 * Tool arguments as sent, pretty-printed when they are JSON.
 *
 * Models emit malformed arguments often enough that this must never
 * throw — a UI that blanks because the model produced bad JSON hides the
 * very thing worth seeing.
 */
function prettyArgs(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

/**
 * The tail of a collapsed block, so a closed row still carries a hint of
 * its content.
 *
 * The tail rather than the head: while a block streams, the end is where
 * the model currently is. A head-anchored preview freezes on the first
 * few words and stops telling you anything.
 */
function peek(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length <= 80 ? flat : `…${flat.slice(-79)}`;
}
