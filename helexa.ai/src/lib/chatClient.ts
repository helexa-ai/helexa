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
    resp = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: opts.model,
        messages: opts.messages,
        ...(opts.tools?.length ? { tools: opts.tools } : {}),
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
