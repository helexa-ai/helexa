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

/** Small key/value store: fingerprint, active conversation, anon usage. */
export interface Meta {
  key: string;
  value: unknown;
}

class HelexaDB extends Dexie {
  projects!: Table<Project, string>;
  conversations!: Table<Conversation, string>;
  messages!: Table<Message, string>;
  meta!: Table<Meta, string>;

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
  }
}

export const db = new HelexaDB();
