import { afterEach, describe, expect, it, vi } from "vitest";
import { streamChatCompletion, type ToolCall } from "./chatClient";

/**
 * SSE framing.
 *
 * The parser buffers bytes and splits on a blank line, so the interesting
 * cases are all about where the network happens to cut the stream — which
 * is precisely what clicking around never reproduces, because a fast
 * local origin tends to deliver each frame whole (#304).
 */

/** A response whose body yields exactly these chunks, in order. */
function respondWith(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function frame(delta: Record<string, unknown>): string {
  return `data: ${JSON.stringify({ choices: [{ delta }] })}\n\n`;
}

/** Run a stream and collect everything the handlers saw. */
async function collect(chunks: string[]) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(respondWith(chunks)));
  const text: string[] = [];
  const reasoning: string[] = [];
  const tools: ToolCall[] = [];
  let done = false;
  let error: { code: string; message: string } | null = null;

  await streamChatCompletion(
    {
      model: "m",
      messages: [{ role: "user", content: "hi" }],
      signal: new AbortController().signal,
    },
    {
      onDelta: (t) => text.push(t),
      onReasoning: (t) => reasoning.push(t),
      onToolCall: (c) => tools.push(c),
      onDone: () => {
        done = true;
      },
      onError: (code, message) => {
        error = { code, message };
      },
    },
  );
  return { text: text.join(""), reasoning: reasoning.join(""), tools, done, error };
}

afterEach(() => vi.unstubAllGlobals());

describe("SSE framing", () => {
  it("delivers content deltas in order", async () => {
    const r = await collect([frame({ content: "he" }), frame({ content: "llo" }), "data: [DONE]\n\n"]);
    expect(r.text).toBe("hello");
    expect(r.done).toBe(true);
    expect(r.error).toBeNull();
  });

  it("reassembles a frame split across two reads", async () => {
    // The case unit tests exist for. A chunk boundary can fall anywhere,
    // including mid-JSON; the buffer must hold the partial frame rather
    // than trying to parse it.
    const whole = frame({ content: "split" });
    const cut = Math.floor(whole.length / 2);
    const r = await collect([whole.slice(0, cut), whole.slice(cut), "data: [DONE]\n\n"]);
    expect(r.text).toBe("split");
  });

  it("handles several frames arriving in one read", async () => {
    const r = await collect([
      frame({ content: "a" }) + frame({ content: "b" }) + frame({ content: "c" }),
      "data: [DONE]\n\n",
    ]);
    expect(r.text).toBe("abc");
  });

  it("separates reasoning from content", async () => {
    // `reasoning_content` must never leak into the answer: the whole
    // reason neuron emits it as its own field is that an unaware client
    // still renders clean output.
    const r = await collect([
      frame({ reasoning_content: "weighing " }),
      frame({ reasoning_content: "options" }),
      frame({ content: "the answer" }),
      "data: [DONE]\n\n",
    ]);
    expect(r.reasoning).toBe("weighing options");
    expect(r.text).toBe("the answer");
  });

  it("stops at [DONE] and ignores anything after it", async () => {
    const r = await collect([
      frame({ content: "kept" }),
      "data: [DONE]\n\n",
      frame({ content: "after the end" }),
    ]);
    expect(r.text).toBe("kept");
    expect(r.done).toBe(true);
  });

  it("skips a malformed frame without losing the stream", async () => {
    // One bad frame must not abort the turn — the tokens around it are
    // still the model's answer.
    const r = await collect([
      frame({ content: "before " }),
      "data: {not json\n\n",
      frame({ content: "after" }),
      "data: [DONE]\n\n",
    ]);
    expect(r.text).toBe("before after");
    expect(r.error).toBeNull();
  });

  it("ignores non-data lines", async () => {
    // Comments (keep-alives) and event: lines are legal SSE and carry
    // nothing this client needs.
    const r = await collect([
      ": keep-alive\n\n",
      "event: ping\ndata: {}\n\n",
      frame({ content: "ok" }),
      "data: [DONE]\n\n",
    ]);
    expect(r.text).toBe("ok");
  });

  it("surfaces a tool call once its name and id are present", async () => {
    const r = await collect([
      frame({
        tool_calls: [
          { id: "call_1", function: { name: "web_search", arguments: '{"query":"q"}' } },
        ],
      }),
      "data: [DONE]\n\n",
    ]);
    expect(r.tools).toHaveLength(1);
    expect(r.tools[0]).toMatchObject({
      id: "call_1",
      function: { name: "web_search", arguments: '{"query":"q"}' },
    });
  });
});
