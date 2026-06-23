// Orchestrates a single conversation: persists the user turn, opens a
// streaming assistant message, appends deltas to Dexie live, and finalizes
// on done/error. The UI re-renders via useLiveQuery on the messages table.

import { useRef, useState } from "react";
import {
  addMessage,
  appendToMessage,
  finalizeMessage,
  listMessages,
  renameConversation,
} from "../data/repositories";
import { streamChatCompletion, type ChatMessage } from "./chatClient";

export interface UseChat {
  streaming: boolean;
  error: { code: string; message: string } | null;
  send: (text: string) => Promise<void>;
  stop: () => void;
}

export function useChat(
  conversationId: string | null,
  opts: { model: string; apiKey?: string },
): UseChat {
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function send(text: string): Promise<void> {
    if (!conversationId || streaming || !text.trim()) return;
    setError(null);

    // History → request messages (before adding the new turn's assistant slot).
    const history = await listMessages(conversationId);
    const reqMessages: ChatMessage[] = history
      .filter((m) => m.status !== "error")
      .map((m) => ({ role: m.role, content: m.content }));
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

    await streamChatCompletion(
      {
        apiKey: opts.apiKey,
        model: opts.model,
        messages: reqMessages,
        signal: controller.signal,
      },
      {
        onDelta: (t) => void appendToMessage(assistantId, t),
        onUsage: (p, c) =>
          void finalizeMessage(assistantId, { promptTokens: p, completionTokens: c }),
        onDone: () => {
          void finalizeMessage(assistantId, { status: "complete" });
          setStreaming(false);
        },
        onError: (code, message) => {
          void finalizeMessage(assistantId, { status: "error", errorCode: code });
          setError({ code, message });
          setStreaming(false);
        },
      },
    );
  }

  function stop(): void {
    abortRef.current?.abort();
    setStreaming(false);
  }

  return { streaming, error, send, stop };
}
