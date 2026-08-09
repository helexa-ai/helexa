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
  size: number;
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
  }
}

export const db = new HelexaDB();
