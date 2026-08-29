// IndexedDB (Dexie) — the ONLY home for chat history and project
// organisation. Nothing here is ever sent to a server (#69/#F3): the mesh
// serves inference, but conversations live exclusively in the browser.
//
// `owner` namespaces data: `"anon"` for the fingerprinted anonymous visitor,
// or an account id once signed in. On login, anonymous data can be claimed
// into the account (F4) — still purely client-side.

import Dexie, { type Table } from "dexie";

export interface Project {
  id: string;
  owner: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  archived: boolean;
  sortOrder: number;
}

export interface Conversation {
  id: string;
  owner: string;
  projectId: string | null; // null → "Unsorted"
  title: string;
  model: string;
  createdAt: number;
  updatedAt: number;
  pinned: boolean;
}

export type MessageRole = "system" | "user" | "assistant";
export type MessageStatus = "complete" | "streaming" | "error";

/** A web source consulted via the web_search tool (#177), rendered as
 * a citation under the assistant message. */
export interface MessageSource {
  title: string;
  url: string;
}

/**
 * Work the assistant did on the way to an answer: a stretch of reasoning,
 * or one tool call and its result.
 *
 * Kept as an ordered list rather than two separate fields because the
 * order is the story — a turn can reason, call a tool, reason about what
 * came back, and call another. Flattening that into "the reasoning" and
 * "the tool calls" would lose which thought preceded which call.
 *
 * `text` and `result` grow while streaming, so a block is also its own
 * progress indicator: a reasoning block with text and no following block
 * is what the model is doing *right now*.
 */
export type TurnBlock =
  | { kind: "reasoning"; text: string }
  | { kind: "tool"; name: string; args: string; result?: string };

export interface Message {
  id: string;
  conversationId: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  status: MessageStatus;
  errorCode?: string;
  promptTokens?: number;
  completionTokens?: number;
  sources?: MessageSource[];
  /**
   * Reasoning and tool calls, in the order they happened. Absent on user
   * turns and on assistant turns that answered directly.
   *
   * Not indexed, so no schema version bump: Dexie stores the whole
   * object and only the declared indexes constrain it. Messages written
   * before this field existed simply have none, which renders as an
   * answer with no working shown — correct, since none was captured.
   */
  blocks?: TurnBlock[];
}

/**
 * A generated image, kept locally like everything else here.
 *
 * Worth keeping more carefully than a chat message: it cost real GPU
 * time, and without the seed it cannot be reproduced even from the same
 * prompt. So the whole request is stored beside the pixels.
 *
 * `png` is a Blob rather than a base64 string — base64 inflates by a
 * third, and IndexedDB stores Blobs natively.
 */
export interface GeneratedImage {
  id: string;
  owner: string;
  prompt: string;
  negativePrompt?: string;
  model: string;
  width: number;
  height: number;
  seed?: number;
  steps?: number;
  guidanceScale?: number;
  png: Blob;
  /** Megapixel-steps billed (#202). */
  units?: number;
  createdAt: number;
}

/** Small key/value store: fingerprint, active conversation, anon usage. */
export interface Meta {
  key: string;
  value: unknown;
}

/**
 * Meta keys for this browser's provisioned API key.
 *
 * Constants rather than literals because the reader and the writers live
 * in different files, and a one-character drift between them is silent:
 * the key is minted and stored, the consumer looks for a name nobody
 * writes, and every signed-in visitor is quietly downgraded to
 * anonymous while being told to create a key by hand. That happened —
 * an unrelated commit renamed the read to `chat:chatApiKey`.
 */
export const CHAT_API_KEY = "chatApiKey";
export const CHAT_API_KEY_ID = "chatApiKeyId";

class HelexaDB extends Dexie {
  projects!: Table<Project, string>;
  conversations!: Table<Conversation, string>;
  messages!: Table<Message, string>;
  meta!: Table<Meta, string>;
  images!: Table<GeneratedImage, string>;

  constructor() {
    super("helexa");
    this.version(1).stores({
      // Indexes only — Dexie stores the whole object. Compound indexes
      // drive the common queries (by owner, by conversation in time order).
      projects: "id, owner, [owner+archived], updatedAt",
      conversations: "id, owner, projectId, [owner+projectId], updatedAt",
      messages: "id, conversationId, [conversationId+createdAt]",
      meta: "key",
    });
    // v2 adds generated images (#242). Additive: Dexie carries the
    // existing stores forward untouched, so an upgrade keeps history.
    this.version(2).stores({
      images: "id, owner, [owner+createdAt], createdAt",
    });
    // v3 replaces the square-only `size` with explicit dimensions, now
    // that portrait and landscape are offered. Images generated before
    // this were all square, so the old value is both the width and the
    // height — backfilled rather than dropped, because a stored image
    // cost real GPU time and cannot be regenerated without its seed.
    this.version(3)
      .stores({ images: "id, owner, [owner+createdAt], createdAt" })
      .upgrade((tx) =>
        tx
          .table<Record<string, unknown>>("images")
          .toCollection()
          .modify((img) => {
            if (typeof img.size === "number") {
              img.width = img.size;
              img.height = img.size;
              delete img.size;
            }
          }),
      );
  }
}

export const db = new HelexaDB();
