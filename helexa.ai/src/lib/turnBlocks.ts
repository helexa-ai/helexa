import type { TurnBlock } from "../data/db";

/**
 * Building the ordered record of what the assistant did on the way to an
 * answer.
 *
 * Kept separate from `useChat` so the rules can be tested (#304). The
 * result-matching one in particular is subtle and fails silently: get it
 * wrong and the UI shows one search's answer under a different search's
 * query, with nothing to indicate anything went astray.
 *
 * Every function mutates the array in place. The caller owns it and
 * writes absolute snapshots to storage, matching how streamed content is
 * persisted — read-modify-write appends race each other when deltas
 * arrive faster than an IndexedDB round-trip.
 */

/**
 * Add a reasoning delta.
 *
 * Extends the open reasoning block, or starts one if the previous block
 * was something else. That is what preserves think → act → think order:
 * reasoning after a tool call is a *new* thought about the result, not a
 * continuation of the thought that led to the call.
 */
export function appendReasoning(blocks: TurnBlock[], text: string): void {
  const last = blocks[blocks.length - 1];
  if (last?.kind === "reasoning") last.text += text;
  else blocks.push({ kind: "reasoning", text });
}

/** Record a tool call. Its result arrives later, via `attachToolResult`. */
export function appendToolCall(blocks: TurnBlock[], name: string, args: string): void {
  blocks.push({ kind: "tool", name, args });
}

/**
 * Attach a result to the call it answers.
 *
 * Matches the **oldest** block with this tool name still awaiting a
 * result — first-in, first-out. `useChat` records calls in arrival order
 * and then executes them in that same order, so the first result belongs
 * to the first call.
 *
 * Scanning from the newest end instead is wrong in a way nothing would
 * report: with two `web_search` calls in one turn — the ordinary case,
 * not an edge one — the first query's results land under the second
 * query's heading and the UI looks entirely healthy. The first draft of
 * this did exactly that, and only writing the rule down as a testable
 * function made it visible.
 *
 * Matching by name as well as by pending-ness keeps interleaved tools
 * apart: ordering is guaranteed per tool, not globally.
 *
 * Returns whether a home was found. `false` means a result arrived for a
 * call that was never recorded — worth surfacing rather than dropping.
 */
export function attachToolResult(
  blocks: TurnBlock[],
  name: string,
  result: string,
): boolean {
  for (const b of blocks) {
    if (b.kind === "tool" && b.name === name && b.result === undefined) {
      b.result = result;
      return true;
    }
  }
  return false;
}
