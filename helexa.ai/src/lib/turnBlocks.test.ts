import { describe, expect, it } from "vitest";
import type { TurnBlock } from "../data/db";
import { appendReasoning, appendToolCall, attachToolResult } from "./turnBlocks";

describe("appendReasoning", () => {
  it("starts a block when there is nothing to extend", () => {
    const blocks: TurnBlock[] = [];
    appendReasoning(blocks, "let me ");
    expect(blocks).toEqual([{ kind: "reasoning", text: "let me " }]);
  });

  it("extends the open reasoning block across deltas", () => {
    // Reasoning arrives token by token; one block, not one per token.
    const blocks: TurnBlock[] = [];
    appendReasoning(blocks, "let me ");
    appendReasoning(blocks, "check ");
    appendReasoning(blocks, "that");
    expect(blocks).toEqual([{ kind: "reasoning", text: "let me check that" }]);
  });

  it("starts a new block for reasoning that follows a tool call", () => {
    // think → act → think is three blocks, not one interrupted one.
    // Merging them would claim the model reasoned continuously through a
    // call it was actually waiting on.
    const blocks: TurnBlock[] = [];
    appendReasoning(blocks, "I should search");
    appendToolCall(blocks, "web_search", '{"query":"a"}');
    appendReasoning(blocks, "those results suggest");

    expect(blocks.map((b) => b.kind)).toEqual(["reasoning", "tool", "reasoning"]);
    expect(blocks[0]).toEqual({ kind: "reasoning", text: "I should search" });
    expect(blocks[2]).toEqual({ kind: "reasoning", text: "those results suggest" });
  });
});

describe("attachToolResult", () => {
  it("fills in the call it answers", () => {
    const blocks: TurnBlock[] = [];
    appendToolCall(blocks, "web_search", '{"query":"helexa"}');
    expect(attachToolResult(blocks, "web_search", "one hit")).toBe(true);
    expect(blocks[0]).toEqual({
      kind: "tool",
      name: "web_search",
      args: '{"query":"helexa"}',
      result: "one hit",
    });
  });

  it("pairs repeated calls to one tool in order", () => {
    // The case that matters, and the one the first implementation got
    // backwards. Two searches in a turn is ordinary; results come back
    // in call order, so first result belongs to first call. Matching
    // newest-first silently files each answer under the wrong query and
    // the UI looks perfectly healthy.
    const blocks: TurnBlock[] = [];
    appendToolCall(blocks, "web_search", '{"query":"first"}');
    appendToolCall(blocks, "web_search", '{"query":"second"}');

    attachToolResult(blocks, "web_search", "results for first");
    attachToolResult(blocks, "web_search", "results for second");

    const tools = blocks.filter((b) => b.kind === "tool");
    expect(tools[0]).toMatchObject({
      args: '{"query":"first"}',
      result: "results for first",
    });
    expect(tools[1]).toMatchObject({
      args: '{"query":"second"}',
      result: "results for second",
    });
  });

  it("keeps interleaved tools apart", () => {
    // Ordering is guaranteed per tool, not globally, so a read_page
    // result must not be able to land on a pending web_search.
    const blocks: TurnBlock[] = [];
    appendToolCall(blocks, "web_search", '{"query":"q"}');
    appendToolCall(blocks, "read_page", '{"url":"https://example.com"}');

    attachToolResult(blocks, "read_page", "page text");

    // `not.toHaveProperty` rather than `toMatchObject({result: undefined})`
    // — the latter passes against an absent key *and* an explicit
    // undefined, so it would not distinguish "still pending" from
    // "answered with nothing".
    expect(blocks[0]).toMatchObject({ name: "web_search" });
    expect(blocks[0]).not.toHaveProperty("result");
    expect(blocks[1]).toMatchObject({ name: "read_page", result: "page text" });
  });

  it("reports a result with no matching call instead of dropping it", () => {
    const blocks: TurnBlock[] = [];
    appendToolCall(blocks, "web_search", "{}");
    attachToolResult(blocks, "web_search", "first");
    // Second result, only one call recorded: nowhere to put it.
    expect(attachToolResult(blocks, "web_search", "second")).toBe(false);
    expect(blocks[0]).toMatchObject({ result: "first" });
  });

  it("does not overwrite a result already attached", () => {
    const blocks: TurnBlock[] = [];
    appendToolCall(blocks, "web_search", "{}");
    appendToolCall(blocks, "web_search", "{}");
    attachToolResult(blocks, "web_search", "one");
    attachToolResult(blocks, "web_search", "two");
    expect(blocks.map((b) => (b.kind === "tool" ? b.result : null))).toEqual([
      "one",
      "two",
    ]);
  });
});
