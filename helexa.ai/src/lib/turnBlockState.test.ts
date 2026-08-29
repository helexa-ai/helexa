import { describe, expect, it } from "vitest";
import { isLive, isOpen } from "./turnBlockState";

/**
 * The visibility rule for the assistant's working, stated as behaviours
 * rather than as a truth table.
 *
 * These began as a throwaway node harness written while building
 * `TurnBlocks`, because the project had no runner (#304). Its first
 * version re-declared the rule instead of importing it — which checks
 * the description and not the code, and would have passed against any
 * implementation at all. Importing the real module is the whole point.
 */
describe("which blocks are open", () => {
  describe("while the turn is streaming and no answer has started", () => {
    it("opens the block being written", () => {
      expect(isOpen(0, 1, true, false, undefined)).toBe(true);
    });

    it("collapses an earlier block once a new one begins", () => {
      // Reasoning, then a tool call: the reasoning is finished work.
      expect(isOpen(0, 2, true, false, undefined)).toBe(false);
      expect(isOpen(1, 2, true, false, undefined)).toBe(true);
    });
  });

  it("collapses everything the moment answer text arrives", () => {
    // The block that was live one token ago is no longer live. This is
    // the behaviour the whole feature exists for: the answer takes the
    // space as soon as there is an answer.
    expect(isOpen(1, 2, true, true, undefined)).toBe(false);
    expect(isOpen(0, 2, true, true, undefined)).toBe(false);
  });

  it("loads completed turns collapsed", () => {
    expect(isOpen(0, 3, false, true, undefined)).toBe(false);
    expect(isOpen(2, 3, false, true, undefined)).toBe(false);
  });

  it("collapses a turn stopped mid-reasoning", () => {
    // Stop pressed while thinking: no answer ever arrives, but the turn
    // is over. Keyed on `streaming`, not on the presence of an answer —
    // otherwise an abandoned turn would sit permanently expanded.
    expect(isOpen(0, 1, false, false, undefined)).toBe(false);
  });

  describe("when the user has clicked", () => {
    it("keeps a finished block open", () => {
      expect(isOpen(0, 3, false, true, true)).toBe(true);
    });

    it("keeps the live block closed", () => {
      // Someone who closes a noisy reasoning stream must not have it
      // spring open again on the next token.
      expect(isOpen(1, 2, true, false, false)).toBe(false);
    });
  });
});

describe("isLive", () => {
  it("is true only for the final block of a streaming, answerless turn", () => {
    expect(isLive(2, 3, true, false)).toBe(true);
    expect(isLive(1, 3, true, false)).toBe(false);
    expect(isLive(2, 3, true, true)).toBe(false);
    expect(isLive(2, 3, false, false)).toBe(false);
  });

  it("is false when there are no blocks", () => {
    // count 0 makes `count - 1` negative; no index can match, and
    // nothing should claim to be live in an empty list.
    expect(isLive(0, 0, true, false)).toBe(false);
  });
});
