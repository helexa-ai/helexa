// Orchestrates a single conversation: persists the user turn, opens a
// streaming assistant message, appends deltas to Dexie live, and finalizes
// on done/error. The UI re-renders via useLiveQuery on the messages table.
//
// With tools (#177) a turn becomes a small agentic loop: stream a round,
// and when the model requests web_search, execute it against the edge's
// /tools/web_search, feed the results back as a tool message, and stream
// the next round — up to MAX_TOOL_ROUNDS. The visible message accumulates
// across rounds; consulted sources persist as citations.

import { useRef, useState } from "react";
import type { MessageSource } from "../data/db";
import {
  addMessage,
  finalizeMessage,
  listMessages,
  renameConversation,
  setMessageContent,
} from "../data/repositories";
import {
  streamChatCompletion,
  type ChatMessage,
  type ToolCall,
} from "./chatClient";
import {
  executeReadPage,
  executeWebSearch,
  READ_PAGE_TOOL,
  WEB_SEARCH_TOOL,
} from "./searchTool";
import { buildSystemPrompt } from "./systemPrompt";

/** The tool activity currently executing, for the UI status line. */
export interface ToolActivity {
  kind: "search" | "read";
  detail: string;
}

export interface UseChat {
  streaming: boolean;
  activity: ToolActivity | null;
  error: { code: string; message: string } | null;
  send: (conversationId: string, text: string) => Promise<void>;
  stop: () => void;
}

const MAX_TOOL_ROUNDS = 4;

export function useChat(opts: {
  model: string;
  apiKey?: string;
  locale?: string;
}): UseChat {
  const [streaming, setStreaming] = useState(false);
  const [activity, setActivity] = useState<ToolActivity | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // The conversation id is an explicit argument, NOT hook state: the
  // first-ever send creates the conversation and calls send() in the same
  // tick, before any re-render — a closured id would still be null and the
  // message would silently vanish (the fresh-browser first-message bug).
  async function send(conversationId: string, text: string): Promise<void> {
    if (!conversationId || streaming || !text.trim()) return;
    setError(null);

    // History → request messages (before adding the new turn's assistant slot).
    // The system prompt is app config, not conversation content: it is sent
    // with every request but never persisted to the transcript (#178).
    const history = await listMessages(conversationId);
    const reqMessages: ChatMessage[] = [
      {
        role: "system",
        content: buildSystemPrompt(opts.model, opts.locale ?? "en", true),
      },
      ...history
        .filter((m) => m.status !== "error")
        .map((m) => ({ role: m.role, content: m.content })),
    ];
    reqMessages.push({ role: "user", content: text });

    await addMessage(conversationId, "user", text, "complete");
    // Title the conversation from its first user message.
    if (history.length === 0) {
      await renameConversation(conversationId, text.slice(0, 60));
    }
    const assistantId = await addMessage(conversationId, "assistant", "", "streaming");

    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true);

    // Streamed content accumulates here and is written to Dexie as
    // ABSOLUTE snapshots, coalesced to one in-flight write: per-delta
    // read-modify-write appends raced each other and dropped tokens
    // (fast GPU streams outpace the IndexedDB round-trip). Finalize is
    // chained behind the last content write so `status: complete` never
    // lands on partial content.
    let acc = "";
    let writing = false;
    let dirty = false;
    let flushed: Promise<void> = Promise.resolve();
    const flush = async () => {
      if (writing) {
        dirty = true;
        return;
      }
      writing = true;
      try {
        do {
          dirty = false;
          await setMessageContent(assistantId, acc);
        } while (dirty);
      } finally {
        writing = false;
      }
    };

    const sources: MessageSource[] = [];
    const seenUrls = new Set<string>();
    let failed = false;

    const finalize = (patch: Parameters<typeof finalizeMessage>[1]) => {
      const withSources = sources.length ? { ...patch, sources } : patch;
      void flushed.then(() => flush()).then(() => finalizeMessage(assistantId, withSources));
      setStreaming(false);
      setActivity(null);
    };
    const addSource = (title: string, url: string) => {
      if (url && !seenUrls.has(url)) {
        seenUrls.add(url);
        sources.push({ title, url });
      }
    };

    for (let round = 0; round < MAX_TOOL_ROUNDS + 1; round++) {
      const roundStart = acc.length;
      const toolCalls: ToolCall[] = [];
      // Only offer tools while budget remains for another round; the
      // last pass runs tool-less so the model must answer.
      const offerTools = round < MAX_TOOL_ROUNDS;

      await streamChatCompletion(
        {
          apiKey: opts.apiKey,
          model: opts.model,
          messages: reqMessages,
          tools: offerTools ? [WEB_SEARCH_TOOL, READ_PAGE_TOOL] : undefined,
          signal: controller.signal,
        },
        {
          onDelta: (t) => {
            acc += t;
            flushed = flush();
          },
          onToolCall: (call) => toolCalls.push(call),
          onUsage: (p, c) =>
            void finalizeMessage(assistantId, { promptTokens: p, completionTokens: c }),
          onDone: () => {},
          onError: (code, message) => {
            failed = true;
            finalize({ status: "error", errorCode: code });
            setError({ code, message });
          },
        },
      );
      if (failed || controller.signal.aborted) {
        if (!failed) finalize({ status: "complete" });
        return;
      }
      if (toolCalls.length === 0) {
        finalize({ status: "complete" });
        return;
      }

      // Tool round: echo the assistant turn (its text + the calls), run
      // each search, append the results, and go around again.
      const roundText = acc.slice(roundStart);
      reqMessages.push({
        role: "assistant",
        content: roundText,
        tool_calls: toolCalls,
      });
      for (const call of toolCalls) {
        let parsed: Record<string, unknown> = {};
        try {
          parsed = JSON.parse(call.function.arguments) ?? {};
        } catch {
          /* malformed arguments; tool-specific fallback below */
        }
        let content: string;
        if (call.function.name === "read_page") {
          const url = String(parsed.url ?? "");
          let host = url;
          try {
            host = new URL(url).hostname;
          } catch {
            /* leave raw */
          }
          setActivity({ kind: "read", detail: host });
          const r = await executeReadPage(url, controller.signal);
          if (r.source) addSource(r.source.title, r.source.url);
          content = r.content;
        } else {
          const args = {
            query: String(parsed.query ?? call.function.arguments),
            category: parsed.category as string | undefined,
            time_range: parsed.time_range as string | undefined,
          };
          setActivity({ kind: "search", detail: args.query });
          const r = await executeWebSearch(args, controller.signal);
          for (const res of r.results) addSource(res.title, res.url);
          content = r.content;
        }
        reqMessages.push({ role: "tool", tool_call_id: call.id, content });
      }
      setActivity(null);
      // Visual seam between the pre-search text and the answer round.
      if (acc.length > 0 && !acc.endsWith("\n\n")) {
        acc += acc.endsWith("\n") ? "\n" : "\n\n";
        flushed = flush();
      }
    }
    // Loop exhausted without a final answer (pathological): close out.
    finalize({ status: "complete" });
  }

  function stop(): void {
    abortRef.current?.abort();
    setStreaming(false);
    setActivity(null);
  }

  return { streaming, activity, error, send, stop };
}
