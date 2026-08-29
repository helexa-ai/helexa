// Streaming chat client → the mesh router's OpenAI-compatible
// /v1/chat/completions (SSE). Parses the byte stream incrementally so tokens
// render as they arrive; surfaces the OpenAI error envelope's `code` so the
// UI can react (rate_limit_exceeded, insufficient_quota, invalid_api_key,
// context_length_exceeded). An AbortController powers the Stop button.

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** Assistant turns that requested tools (echoed back in the loop). */
  tool_calls?: ToolCall[];
  /** Tool-result turns: which call this answers. */
  tool_call_id?: string;
}

export interface StreamHandlers {
  onDelta: (text: string) => void;
  /**
   * A reasoning delta, from `choice.delta.reasoning_content`.
   *
   * Not in the OpenAI spec but the de-facto slot — DeepSeek, vLLM and
   * SGLang all use it, and neuron emits it by default rather than
   * folding reasoning into `content`, so a client that ignores the
   * field still shows a clean answer.
   *
   * Worth handling rather than dropping: on a hard prompt the model can
   * reason for minutes, and these are the only events on the wire for
   * that whole span. A UI that ignores them looks hung.
   */
  onReasoning?: (text: string) => void;
  /** A complete tool call arrived (neuron buffers the whole
   * `<tool_call>` block, so arguments are never fragmented). */
  onToolCall?: (call: ToolCall) => void;
  onUsage?: (prompt: number, completion: number) => void;
  onDone: () => void;
  onError: (code: string, message: string) => void;
}

export interface StreamOptions {
  baseUrl?: string;
  apiKey?: string; // bearer for authenticated requests; omitted = anonymous
  model: string;
  messages: ChatMessage[];
  /** OpenAI tools array; omitted = no tools offered. */
  tools?: readonly unknown[];
  /**
   * How hard to think, or `null`/absent for not at all.
   *
   * Two things travel together and both are needed. The
   * `x-include-thinking` header says the caller will display reasoning;
   * without it neuron couples generation to surfacing and the model
   * produces none at all (`reasoning_tokens=0` on every request, verified
   * against the live router). The `reasoning.effort` field then selects a
   * rung of the model's own ladder.
   *
   * Rung names are the model's, discovered from `/v1/models` rather than
   * hardcoded — `Qwen/Qwen3.8-27B` accepts exactly `low`, `medium` and
   * `xhigh` and its template raises on anything else.
   */
  reasoningEffort?: string | null;
  signal: AbortSignal;
}

const DEFAULT_BASE = import.meta.env.VITE_ROUTER_BASE_URL || "";

/** How long to wait for response HEADERS before declaring the origin dead.
 * Generous: an admission-queued request can legitimately hold ~30s before
 * the first byte. Without this, a misconfigured origin that swallows the
 * POST (e.g. a static host with no /v1 backend) hangs the UI in silence. */
const FIRST_BYTE_TIMEOUT_MS = 45_000;

export async function streamChatCompletion(
  opts: StreamOptions,
  h: StreamHandlers,
): Promise<void> {
  const base = (opts.baseUrl ?? DEFAULT_BASE).replace(/\/$/, "");
  let resp: Response;
  // Chain the caller's Stop signal with a first-byte timeout. `timedOut`
  // disambiguates our abort from the user's.
  let timedOut = false;
  const ctl = new AbortController();
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, FIRST_BYTE_TIMEOUT_MS);
  // The caller's signal must keep aborting ctl for the whole request —
  // headers AND body stream — so Stop works mid-generation. opts.signal
  // is per-send, so the listener's lifetime is naturally bounded.
  opts.signal.addEventListener("abort", () => ctl.abort(), { once: true });
  try {
    const headers: Record<string, string> = {
      "content-type": "application/json",
      accept: "text/event-stream",
    };
    if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`;
    // Only sent when on. Absent means "naïve client", which is the
    // server-side default and the behaviour every other OpenAI client
    // gets — worth preserving exactly, since this toggle exists partly
    // to demonstrate both paths.
    if (opts.reasoningEffort) headers["x-include-thinking"] = "true";
    resp = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: opts.model,
        messages: opts.messages,
        ...(opts.tools?.length ? { tools: opts.tools } : {}),
        // The shape neuron reads (`request.extra.reasoning.effort`).
        // Omitted entirely when off, so the request is byte-identical to
        // what a vanilla OpenAI client sends — which is the other half of
        // what this toggle demonstrates.
        ...(opts.reasoningEffort ? { reasoning: { effort: opts.reasoningEffort } } : {}),
        stream: true,
      }),
      signal: ctl.signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError" && !timedOut) return h.onDone();
    return h.onError(
      "network_error",
      timedOut
        ? "No response from the mesh — the endpoint may be misconfigured or down."
        : "Could not reach the mesh.",
    );
  } finally {
    // Headers arrived (or failed) — the first-byte deadline is done.
    // Body-stream pacing is the model's business, not a timeout's.
    clearTimeout(timer);
  }

  if (!resp.ok || !resp.body) {
    // Parse the OpenAI error envelope for the machine-readable code.
    let code = "api_error";
    let message = `Request failed (${resp.status}).`;
    try {
      const body = await resp.json();
      code = body?.error?.code ?? body?.error?.type ?? code;
      message = body?.error?.message ?? message;
    } catch {
      /* non-JSON body */
    }
    return h.onError(code, message);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of frame.split("\n")) {
          const trimmed = line.trimStart();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.slice(5).trim();
          if (data === "[DONE]") {
            return h.onDone();
          }
          try {
            const json = JSON.parse(data);
            const delta = json?.choices?.[0]?.delta?.content;
            if (typeof delta === "string" && delta) h.onDelta(delta);
            const reasoning = json?.choices?.[0]?.delta?.reasoning_content;
            if (typeof reasoning === "string" && reasoning && h.onReasoning) {
              h.onReasoning(reasoning);
            }
            const toolCalls = json?.choices?.[0]?.delta?.tool_calls;
            if (Array.isArray(toolCalls) && h.onToolCall) {
              for (const tc of toolCalls) {
                if (tc?.id && tc?.function?.name) {
                  h.onToolCall({
                    id: tc.id,
                    type: "function",
                    function: {
                      name: tc.function.name,
                      arguments: tc.function.arguments ?? "{}",
                    },
                  });
                }
              }
            }
            const usage = json?.usage;
            if (usage && h.onUsage) {
              h.onUsage(usage.prompt_tokens ?? 0, usage.completion_tokens ?? 0);
            }
          } catch {
            /* keep streaming past a non-JSON keepalive */
          }
        }
      }
    }
    h.onDone();
  } catch (e) {
    if ((e as Error).name === "AbortError") return h.onDone();
    h.onError("stream_error", "The response stream was interrupted.");
  }
}
