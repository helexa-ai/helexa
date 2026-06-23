// Streaming chat client → the mesh router's OpenAI-compatible
// /v1/chat/completions (SSE). Parses the byte stream incrementally so tokens
// render as they arrive; surfaces the OpenAI error envelope's `code` so the
// UI can react (rate_limit_exceeded, insufficient_quota, invalid_api_key,
// context_length_exceeded). An AbortController powers the Stop button.

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface StreamHandlers {
  onDelta: (text: string) => void;
  onUsage?: (prompt: number, completion: number) => void;
  onDone: () => void;
  onError: (code: string, message: string) => void;
}

export interface StreamOptions {
  baseUrl?: string;
  apiKey?: string; // bearer for authenticated requests; omitted = anonymous
  model: string;
  messages: ChatMessage[];
  signal: AbortSignal;
}

const DEFAULT_BASE = import.meta.env.VITE_ROUTER_BASE_URL || "";

export async function streamChatCompletion(
  opts: StreamOptions,
  h: StreamHandlers,
): Promise<void> {
  const base = (opts.baseUrl ?? DEFAULT_BASE).replace(/\/$/, "");
  let resp: Response;
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
        stream: true,
      }),
      signal: opts.signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return h.onDone();
    return h.onError("network_error", "Could not reach the mesh.");
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
