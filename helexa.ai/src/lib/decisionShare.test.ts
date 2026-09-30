import { describe, expect, it } from "vitest";
import { decodeShare, encodeShare } from "./decisionShare";
import type { Draft } from "./decisionRequest";

const draft: Draft = {
  stateText: '{"subject": "двойное списание", "body": "返金してください \' \\" \\u0000"}',
  stateMode: "json",
  questionsText: '{"10": {"type": "noul", "instructions": "Refund?"}, "2": {"type": "noul", "instructions": "x"}}',
  checkpoint: "multilingual",
  maxLen: "256",
};

describe("share links", () => {
  it("round-trips a draft exactly", async () => {
    const hash = await encodeShare(draft);
    expect(hash).toMatch(/^#p=1\.[A-Za-z0-9_-]+$/);
    expect(await decodeShare(hash)).toEqual({ ...draft, headMaxLen: undefined });
  });

  it("compresses repetitive editor contents", async () => {
    const big = { ...draft, questionsText: JSON.stringify({ q: { type: "noul", instructions: "x ".repeat(2000) } }) };
    const hash = await encodeShare(big);
    expect(hash.length).toBeLessThan(big.questionsText.length / 4);
  });

  it("falls back to auto for an unknown checkpoint", async () => {
    const hash = await encodeShare({ ...draft, checkpoint: "bogus" as Draft["checkpoint"] });
    expect((await decodeShare(hash))?.checkpoint).toBe("auto");
  });

  it.each([
    ["no fragment", ""],
    ["another parameter", "#x=1"],
    ["no version", "#p=abc"],
    ["not base64url", "#p=1.@@@"],
    ["not deflate", "#p=1.aGVsbG8"],
  ])("returns null for %s", async (_what, hash) => {
    expect(await decodeShare(hash)).toBeNull();
  });

  it("returns null for a valid payload under an unknown version", async () => {
    const hash = await encodeShare(draft);
    expect(await decodeShare(hash.replace("#p=1.", "#p=2."))).toBeNull();
  });

  it("returns null for a truncated link", async () => {
    const hash = await encodeShare(draft);
    expect(await decodeShare(hash.slice(0, hash.length - 12))).toBeNull();
  });
});
