// Typed CRUD + queries over the Dexie store. UI components use the
// `useLiveQuery` hook (dexie-react-hooks) with the list helpers here so the
// sidebar/thread react to writes automatically.

import Dexie from "dexie";
import {
  db,
  type Conversation,
  type Message,
  type MessageRole,
  type Project,
} from "./db";

function uuid(): string {
  return crypto.randomUUID();
}
function now(): number {
  return Date.now();
}

// ── projects ────────────────────────────────────────────────────────

export async function listProjects(owner: string): Promise<Project[]> {
  const rows = await db.projects.where({ owner }).toArray();
  return rows
    .filter((p) => !p.archived)
    .sort((a, b) => a.sortOrder - b.sortOrder || a.createdAt - b.createdAt);
}

export async function createProject(owner: string, name: string): Promise<string> {
  const id = uuid();
  const ts = now();
  await db.projects.add({
    id,
    owner,
    name,
    createdAt: ts,
    updatedAt: ts,
    archived: false,
    sortOrder: ts,
  });
  return id;
}

export async function renameProject(id: string, name: string): Promise<void> {
  await db.projects.update(id, { name, updatedAt: now() });
}

export async function archiveProject(id: string): Promise<void> {
  // Detach its conversations to "Unsorted" so nothing is orphaned.
  await db.transaction("rw", db.projects, db.conversations, async () => {
    await db.projects.update(id, { archived: true, updatedAt: now() });
    const convs = await db.conversations.where({ projectId: id }).toArray();
    await Promise.all(
      convs.map((c) => db.conversations.update(c.id, { projectId: null })),
    );
  });
}

// ── conversations ───────────────────────────────────────────────────

export async function listConversations(owner: string): Promise<Conversation[]> {
  const rows = await db.conversations.where({ owner }).toArray();
  return rows.sort(
    (a, b) => Number(b.pinned) - Number(a.pinned) || b.updatedAt - a.updatedAt,
  );
}

export async function createConversation(
  owner: string,
  model: string,
  projectId: string | null = null,
  title = "New chat",
): Promise<string> {
  const id = uuid();
  const ts = now();
  await db.conversations.add({
    id,
    owner,
    projectId,
    title,
    model,
    createdAt: ts,
    updatedAt: ts,
    pinned: false,
  });
  return id;
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await db.conversations.update(id, { title, updatedAt: now() });
}

export async function moveConversation(
  id: string,
  projectId: string | null,
): Promise<void> {
  await db.conversations.update(id, { projectId, updatedAt: now() });
}

export async function deleteConversation(id: string): Promise<void> {
  await db.transaction("rw", db.conversations, db.messages, async () => {
    await db.messages.where({ conversationId: id }).delete();
    await db.conversations.delete(id);
  });
}

// ── messages ────────────────────────────────────────────────────────

export async function listMessages(conversationId: string): Promise<Message[]> {
  return db.messages
    .where("[conversationId+createdAt]")
    .between([conversationId, Dexie.minKey], [conversationId, Dexie.maxKey])
    .toArray();
}

export async function addMessage(
  conversationId: string,
  role: MessageRole,
  content: string,
  status: Message["status"] = "complete",
): Promise<string> {
  const id = uuid();
  await db.messages.add({ id, conversationId, role, content, createdAt: now(), status });
  await db.conversations.update(conversationId, { updatedAt: now() });
  return id;
}

/** Overwrite a message's content with an absolute value.
 *
 * Deliberately NOT an append: read-modify-write per streamed delta loses
 * updates when tokens arrive faster than the IndexedDB round-trip (two
 * in-flight appends read the same base and one delta vanishes — the
 * "Swiss-cheese response" bug). The caller accumulates the full content
 * and writes snapshots; absolute writes are safe to coalesce. */
export async function setMessageContent(id: string, content: string): Promise<void> {
  await db.messages.update(id, { content });
}

export async function finalizeMessage(
  id: string,
  patch: Partial<Pick<Message, "status" | "errorCode" | "promptTokens" | "completionTokens">>,
): Promise<void> {
  await db.messages.update(id, patch);
}

/** Rewrite all `anon` data to `accountId` on first login (stays local). */
export async function claimAnonymousData(accountId: string): Promise<void> {
  await db.transaction("rw", db.projects, db.conversations, async () => {
    const projects = await db.projects.where({ owner: "anon" }).toArray();
    await Promise.all(
      projects.map((p) => db.projects.update(p.id, { owner: accountId })),
    );
    const convs = await db.conversations.where({ owner: "anon" }).toArray();
    await Promise.all(
      convs.map((c) => db.conversations.update(c.id, { owner: accountId })),
    );
  });
}
