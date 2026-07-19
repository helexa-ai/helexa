import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLiveQuery } from "dexie-react-hooks";
import { Alert, Form } from "react-bootstrap";
import { FaArrowUp, FaStop, FaBarsStaggered } from "react-icons/fa6";
import {
  LuCheck,
  LuFolderInput,
  LuFolderPlus,
  LuMessageSquarePlus,
  LuPencil,
  LuTrash2,
  LuX,
} from "react-icons/lu";
import { db } from "../data/db";
import {
  archiveProject,
  createConversation,
  createProject,
  deleteConversation,
  listConversations,
  listProjects,
  moveConversation,
  renameConversation,
  renameProject,
} from "../data/repositories";
import { useChat } from "../lib/useChat";
import { useAuth } from "../auth/context";
import { accountApi } from "../api/account";

const ANON_MODEL = import.meta.env.VITE_ANON_MODEL || "helexa/small";
const AUTH_MODEL = import.meta.env.VITE_DEFAULT_MODEL || "helexa/balanced";
const ANON_MESSAGE_CAP = 20;
const ANON_COUNT_KEY = "anonMessageCount";

/**
 * The chat workspace landing (`/`). Anonymous visitors are fingerprinted and
 * capped, streaming from the constrained public model with no bearer. Signed
 * in (F5), the workspace switches its IndexedDB owner to the account, lifts
 * the cap, uses the full default model, and sends the user's API key (stored
 * locally, never server-side) as the bearer. History always stays in the
 * browser.
 */
export default function Chat() {
  const { t, i18n } = useTranslation("chat");
  const { status, accountId } = useAuth();
  const authed = status === "authed" && !!accountId;
  const owner = authed ? accountId! : "anon";
  const model = authed ? AUTH_MODEL : ANON_MODEL;

  // The user's API key for authenticated chat — stored client-side only,
  // captured from the create-key modal ("use for chat on this device").
  const chatApiKey = useLiveQuery(
    async () => {
      const m = await db.meta.get("chatApiKey");
      return typeof m?.value === "string" ? m.value : undefined;
    },
    [],
    undefined,
  );

  const projects = useLiveQuery(() => listProjects(owner), [owner], []);
  const conversations = useLiveQuery(() => listConversations(owner), [owner], []);
  const [activeId, setActiveId] = useState<string | null>(null);
  // Phone-width screens render the sidebar as an off-canvas drawer;
  // this state only has visible effect under the 768px media query.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Topic (project) currently in inline-rename mode; a freshly created
  // topic drops straight into it so it gets a real name immediately.
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);

  // Reset the active conversation when the owner changes (login/logout).
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setActiveId(null);
  }, [owner]);

  const anonCount =
    useLiveQuery(async () => {
      const m = await db.meta.get(ANON_COUNT_KEY);
      return typeof m?.value === "number" ? m.value : 0;
    }, [], 0) ?? 0;
  // The cap only applies to anonymous visitors; signed-in users are gated by
  // their account allocation (enforced upstream), not a client counter.
  const capped = !authed && anonCount >= ANON_MESSAGE_CAP;

  // Anonymous grounding gate (#191): a server-driven flag so the operator
  // can kill anonymous web search with a config flip, no site rebuild.
  // Fail closed — until /api/features answers, anonymous sessions run
  // tool-less. Signed-in sessions always get tools.
  const [anonWebSearch, setAnonWebSearch] = useState(false);
  useEffect(() => {
    let cancelled = false;
    accountApi()
      .features()
      .then((f) => {
        if (!cancelled) setAnonWebSearch(f.anon_web_search);
      })
      .catch(() => {
        /* stay fail-closed */
      });
    return () => {
      cancelled = true;
    };
  }, []);
  // Signed in but no local key enabled for chat → can't send as yourself yet.
  const needsKey = authed && !chatApiKey;

  const messages = useLiveQuery(
    async () => {
      if (!activeId) return [];
      const { listMessages } = await import("../data/repositories");
      return listMessages(activeId);
    },
    [activeId],
    [],
  );

  const { streaming, activity, error, send, stop } = useChat({
    model,
    apiKey: authed ? chatApiKey : undefined,
    locale: i18n.language,
    toolsEnabled: authed || anonWebSearch,
  });
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight });
  }, [messages]);

  async function newChat(projectId: string | null = null) {
    const id = await createConversation(owner, model, projectId);
    setActiveId(id);
    setSidebarOpen(false);
  }

  function selectConversation(id: string) {
    setActiveId(id);
    setSidebarOpen(false);
  }

  async function onSend() {
    const text = draft.trim();
    if (!text || streaming || capped || needsKey) return;
    let convId = activeId;
    if (!convId) {
      convId = await createConversation(owner, model);
      setActiveId(convId);
    }
    setDraft("");
    if (!authed) {
      await db.meta.put({ key: ANON_COUNT_KEY, value: anonCount + 1 });
    }
    // Pass convId explicitly — on the first-ever message it was created
    // two lines up and no re-render has delivered it to the hook yet.
    await send(convId, text);
  }

  // Group conversations by project for the sidebar.
  const grouped = useMemo(() => {
    const byProject = new Map<string | null, typeof conversations>();
    for (const c of conversations ?? []) {
      const arr = byProject.get(c.projectId) ?? [];
      arr.push(c);
      byProject.set(c.projectId, arr);
    }
    return byProject;
  }, [conversations]);

  return (
    <div className="d-flex flex-grow-1" style={{ minHeight: 0 }}>
      {/* Sidebar — off-canvas drawer under 768px, static column above. */}
      {sidebarOpen && (
        <div
          className="hx-drawer-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      <aside className={`hx-chat-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="d-flex gap-2 justify-content-end">
          <button
            type="button"
            className="hx-icon-btn hx-sidebar-action"
            title={t("newChat")}
            aria-label={t("newChat")}
            onClick={() => void newChat()}
          >
            <LuMessageSquarePlus size={17} />
          </button>
          <button
            type="button"
            className="hx-icon-btn hx-sidebar-action"
            title={t("newProject")}
            aria-label={t("newProject")}
            onClick={() =>
              void createProject(owner, t("newProjectName")).then(setEditingProjectId)
            }
          >
            <LuFolderPlus size={17} />
          </button>
        </div>

        {(grouped.get(null) ?? []).length > 0 && (
          <div className="hx-group-label">{t("unsorted")}</div>
        )}
        {(grouped.get(null) ?? []).map((c) => (
          <ThreadRow
            key={c.id}
            conv={c}
            active={c.id === activeId}
            onSelect={selectConversation}
            projects={projects ?? []}
            onDeleted={() => setActiveId(null)}
            t={t}
          />
        ))}

        {(projects ?? []).map((p) => (
          <div key={p.id}>
            {editingProjectId === p.id ? (
              <InlineRename
                initial={p.name}
                onCommit={(name) => {
                  if (name.trim()) void renameProject(p.id, name.trim());
                  setEditingProjectId(null);
                }}
                onCancel={() => setEditingProjectId(null)}
                t={t}
              />
            ) : (
              <div className="hx-row hx-group-label d-flex align-items-center">
                <span className="text-truncate flex-grow-1">{p.name}</span>
                <span className="hx-row-actions">
                  <button
                    type="button"
                    className="hx-icon-btn hx-row-btn"
                    title={t("rename")}
                    aria-label={t("rename")}
                    onClick={() => setEditingProjectId(p.id)}
                  >
                    <LuPencil size={13} />
                  </button>
                  <button
                    type="button"
                    className="hx-icon-btn hx-row-btn"
                    title={t("delete")}
                    aria-label={t("delete")}
                    onClick={() => void archiveProject(p.id)}
                  >
                    <LuTrash2 size={13} />
                  </button>
                </span>
              </div>
            )}
            {(grouped.get(p.id) ?? []).map((c) => (
              <ThreadRow
                key={c.id}
                conv={c}
                active={c.id === activeId}
                onSelect={selectConversation}
                projects={projects ?? []}
                onDeleted={() => setActiveId(null)}
                t={t}
              />
            ))}
          </div>
        ))}
      </aside>

      {/* Main */}
      <section
        className="d-flex flex-column flex-grow-1 position-relative"
        style={{ minWidth: 0 }}
      >
        <button
          type="button"
          className="hx-icon-btn hx-sidebar-toggle"
          aria-label={t("sidebarToggle")}
          title={t("sidebarToggle")}
          onClick={() => setSidebarOpen(true)}
        >
          <FaBarsStaggered size={15} />
        </button>
        <div ref={threadRef} className="flex-grow-1 p-3 overflow-auto">
          {(messages ?? []).length === 0 ? (
            <div className="hx-chat-empty">
              <img src="/logo.png" alt="" aria-hidden="true" />
              <p>{t("emptyState")}</p>
            </div>
          ) : (
            (messages ?? []).map((m) => (
              <div
                key={m.id}
                className={`mb-3 d-flex ${m.role === "user" ? "justify-content-end" : "justify-content-start"}`}
              >
                <div
                  className={`hx-bubble ${m.role === "user" ? "hx-bubble-user" : ""}`}
                >
                  {m.content}
                  {m.status === "streaming" && activity && (
                    <span className="hx-searching">
                      {activity.kind === "search"
                        ? t("searching", { query: activity.detail })
                        : t("reading", { host: activity.detail })}
                    </span>
                  )}
                  {m.status === "streaming" && <span className="opacity-50"> ▋</span>}
                  {m.status === "error" && (
                    <span className="text-danger small"> ⚠ {m.errorCode}</span>
                  )}
                  {m.sources && m.sources.length > 0 && (
                    <div className="hx-sources">
                      <span className="hx-sources-label">{t("sources")}</span>
                      {m.sources.map((s) => (
                        <a
                          key={s.url}
                          className="hx-source-chip"
                          href={s.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          title={s.title}
                        >
                          {new URL(s.url).hostname.replace(/^www\./, "")}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {error && (
          <Alert variant="warning" className="m-2 py-2">
            {error.message}{" "}
            {error.code === "insufficient_quota" ? (
              // Hard balance exhausted → top up (authed) or sign up (anon).
              <a href={authed ? "/account" : "/register"}>
                {authed ? t("topUp") : t("signUp")}
              </a>
            ) : error.code === "rate_limit_exceeded" ? (
              <span className="text-muted">{t("rateLimited")}</span>
            ) : (
              !authed && <a href="/register">{t("signUp")}</a>
            )}
          </Alert>
        )}

        {capped && !error && (
          <Alert variant="info" className="m-2 py-2">
            {t("anonBanner")} <a href="/register">{t("signUp")}</a>
          </Alert>
        )}

        {needsKey && !error && (
          <Alert variant="info" className="m-2 py-2">
            {t("needsKey")} <a href="/account/keys">{t("manageKeysLink")}</a>
          </Alert>
        )}

        <Form
          className="hx-composer"
          onSubmit={(e) => {
            e.preventDefault();
            void onSend();
          }}
        >
          <Form.Control
            as="textarea"
            rows={1}
            value={draft}
            disabled={capped || needsKey}
            placeholder={
              capped ? t("anonBanner") : needsKey ? t("needsKey") : t("inputPlaceholder")
            }
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void onSend();
              }
            }}
          />
          {streaming ? (
            <button
              type="button"
              className="hx-send-btn hx-stop"
              onClick={stop}
              aria-label={t("stop")}
              title={t("stop")}
            >
              <FaStop size={13} />
            </button>
          ) : (
            <button
              type="submit"
              className="hx-send-btn"
              disabled={capped || needsKey || !draft.trim()}
              aria-label={t("send")}
              title={t("send")}
            >
              <FaArrowUp size={15} />
            </button>
          )}
        </Form>
      </section>
    </div>
  );
}

/** Inline single-field rename editor: Enter/check commits, Escape/x cancels. */
function InlineRename({
  initial,
  onCommit,
  onCancel,
  t,
}: {
  initial: string;
  onCommit: (name: string) => void;
  onCancel: () => void;
  t: (k: string) => string;
}) {
  const [value, setValue] = useState(initial);
  return (
    <div className="hx-inline-edit d-flex align-items-center gap-1">
      <input
        autoFocus
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onCommit(value);
          if (e.key === "Escape") onCancel();
        }}
      />
      <button
        type="button"
        className="hx-icon-btn hx-row-btn"
        title={t("rename")}
        aria-label={t("rename")}
        onClick={() => onCommit(value)}
      >
        <LuCheck size={13} />
      </button>
      <button
        type="button"
        className="hx-icon-btn hx-row-btn"
        aria-label={t("cancel")}
        title={t("cancel")}
        onClick={onCancel}
      >
        <LuX size={13} />
      </button>
    </div>
  );
}

/** One thread in the sidebar: select on click; hover (or touch) actions for
 * rename (inline), move-to-topic (small popover menu), and delete. */
function ThreadRow({
  conv,
  active,
  onSelect,
  projects,
  onDeleted,
  t,
}: {
  conv: { id: string; title: string; projectId: string | null };
  active: boolean;
  onSelect: (id: string) => void;
  projects: { id: string; name: string }[];
  onDeleted: () => void;
  t: (k: string) => string;
}) {
  const [editing, setEditing] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  if (editing) {
    return (
      <InlineRename
        initial={conv.title}
        onCommit={(name) => {
          if (name.trim()) void renameConversation(conv.id, name.trim());
          setEditing(false);
        }}
        onCancel={() => setEditing(false)}
        t={t}
      />
    );
  }

  const destinations: { id: string | null; name: string }[] = [
    ...(conv.projectId !== null ? [{ id: null, name: t("unsorted") }] : []),
    ...projects
      .filter((p) => p.id !== conv.projectId)
      .map((p) => ({ id: p.id as string | null, name: p.name })),
  ];

  return (
    <div className="hx-row position-relative d-flex align-items-center">
      <button
        type="button"
        onClick={() => onSelect(conv.id)}
        className={`hx-chat-item flex-grow-1 ${active ? "active" : ""}`}
      >
        {conv.title}
      </button>
      <span className="hx-row-actions">
        <button
          type="button"
          className="hx-icon-btn hx-row-btn"
          title={t("rename")}
          aria-label={t("rename")}
          onClick={() => setEditing(true)}
        >
          <LuPencil size={13} />
        </button>
        {destinations.length > 0 && (
          <button
            type="button"
            className="hx-icon-btn hx-row-btn"
            title={t("moveTo")}
            aria-label={t("moveTo")}
            onClick={() => setMenuOpen((v) => !v)}
          >
            <LuFolderInput size={13} />
          </button>
        )}
        <button
          type="button"
          className="hx-icon-btn hx-row-btn"
          title={t("delete")}
          aria-label={t("delete")}
          onClick={() => {
            if (window.confirm(t("deleteThreadConfirm"))) {
              void deleteConversation(conv.id).then(() => {
                if (active) onDeleted();
              });
            }
          }}
        >
          <LuTrash2 size={13} />
        </button>
      </span>
      {menuOpen && (
        <>
          <div
            className="hx-menu-backdrop"
            onClick={() => setMenuOpen(false)}
            aria-hidden="true"
          />
          <div className="hx-move-menu" role="menu">
            {destinations.map((d) => (
              <button
                key={d.id ?? "__unsorted"}
                type="button"
                role="menuitem"
                className="hx-move-item text-truncate"
                onClick={() => {
                  void moveConversation(conv.id, d.id);
                  setMenuOpen(false);
                }}
              >
                {d.name}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
