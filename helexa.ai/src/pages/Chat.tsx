import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
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
import Markdown from "../components/Markdown";
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
import { ensureChatKey } from "../lib/ensureChatKey";
import { accountApi } from "../api/account";

const ANON_MODEL = import.meta.env.VITE_ANON_MODEL || "helexa/small";
const AUTH_MODEL = import.meta.env.VITE_DEFAULT_MODEL || "helexa/balanced";
const ANON_MESSAGE_CAP = 20;
/** Remaining messages at which the anonymous visitor is forewarned. */
const ANON_WARN_AT = 5;
const ANON_COUNT_KEY = "anonMessageCount";
const SIDEBAR_KEY = "sidebarExpanded";

/**
 * The chat workspace landing (`/`). Anonymous visitors are fingerprinted and
 * capped, streaming from the constrained public model with no bearer. Signed
 * in (F5), the workspace switches its IndexedDB owner to the account, lifts
 * the cap, uses the full default model, and sends the user's API key (stored
 * locally, never server-side) as the bearer. History always stays in the
 * browser.
 */
export default function Chat() {
  const { t, i18n } = useTranslation(["chat", "mission", "common"]);
  const { status, accountId, token } = useAuth();
  const authed = status === "authed" && !!accountId;
  const owner = authed ? accountId! : "anon";
  const model = authed ? AUTH_MODEL : ANON_MODEL;

  // This browser's API key for authenticated chat — stored client-side
  // only, provisioned on demand (see ensureChatKey).
  //
  // `undefined` means "still reading IndexedDB", `null` means "read, and
  // there isn't one". Collapsing both to undefined would flash the
  // create-a-key banner on every load and fire provisioning before we
  // knew whether a key already existed.
  const chatApiKey = useLiveQuery<string | null, undefined>(
    async () => {
      const m = await db.meta.get("chat:chatApiKey");
      return typeof m?.value === "string" ? m.value : null;
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
  // Desktop sidebar: an icon rail by default, expandable to the thread
  // list. A permanently open 280px column of history is the wrong default
  // for a chat that most people arrive at with nothing in it, and it is
  // what made the workspace feel dated. The choice persists per browser.
  const sidebarExpanded =
    useLiveQuery(async () => {
      const m = await db.meta.get(SIDEBAR_KEY);
      return typeof m?.value === "boolean" ? m.value : false;
    }, [], false) ?? false;
  const toggleSidebar = (): void => {
    void db.meta.put({ key: SIDEBAR_KEY, value: !sidebarExpanded });
  };
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
  // Warn before the wall rather than at it. The count is rendered as bare
  // numerals — "16 / 20" — deliberately: a phrase like "4 messages left"
  // puts a counted noun after an interpolated number, which needs case and
  // plural agreement that varies with the value across the Slavic, Baltic,
  // Celtic and Semitic locales. Numerals carry the same information and
  // need no grammar, and toLocaleString gives locales that prefer their
  // own digits (fa, ar) the right ones.
  const anonNearLimit =
    !authed && !capped && ANON_MESSAGE_CAP - anonCount <= ANON_WARN_AT;

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
  // Signed in with no key on this browser: provision one rather than
  // making the user go and do it. chatApiKey is a live query, so the
  // composer unblocks by itself the moment it lands.
  useEffect(() => {
    if (!authed || !token || chatApiKey !== null) return;
    void ensureChatKey(token);
  }, [authed, token, chatApiKey]);

  // Only a dead end once provisioning has been tried and there is still
  // no key — then the manual path in the banner still applies.
  const needsKey = authed && chatApiKey === null;

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
    apiKey: authed ? (chatApiKey ?? undefined) : undefined,
    locale: i18n.language,
    toolsEnabled: authed || anonWebSearch,
  });
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);

  // A prompt handed over from the landing page: create the conversation and
  // send it, so the first thing a visitor typed becomes the first message in
  // their thread instead of something they retype. The history entry is
  // replaced immediately, so a refresh cannot send it a second time.
  const location = useLocation();
  const navigate = useNavigate();
  const handedOver = (location.state as { prompt?: string } | null)?.prompt;
  const handedOverRef = useRef(false);
  useEffect(() => {
    if (!handedOver || handedOverRef.current) return;
    handedOverRef.current = true;
    navigate(".", { replace: true, state: null });
    void (async () => {
      const convId = await createConversation(owner, model);
      setActiveId(convId);
      await send(convId, handedOver);
    })();
  }, [handedOver, navigate, owner, model, send]);

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
      <aside
        className={`hx-chat-sidebar ${sidebarOpen ? "open" : ""} ${
          sidebarExpanded ? "expanded" : "rail"
        }`}
      >
        <div className="hx-sidebar-actions">
          <button
            type="button"
            className="hx-icon-btn hx-sidebar-action hx-sidebar-expand"
            title={t("chat:sidebarToggle")}
            aria-label={t("chat:sidebarToggle")}
            aria-expanded={sidebarExpanded}
            onClick={toggleSidebar}
          >
            <FaBarsStaggered size={15} />
          </button>
          <button
            type="button"
            className="hx-icon-btn hx-sidebar-action"
            title={t("chat:newChat")}
            aria-label={t("chat:newChat")}
            onClick={() => void newChat()}
          >
            <LuMessageSquarePlus size={17} />
          </button>
          <button
            type="button"
            className="hx-icon-btn hx-sidebar-action"
            title={t("chat:newProject")}
            aria-label={t("chat:newProject")}
            onClick={() =>
              void createProject(owner, t("chat:newProjectName")).then(setEditingProjectId)
            }
          >
            <LuFolderPlus size={17} />
          </button>
        </div>

        {(grouped.get(null) ?? []).length > 0 && (
          <div className="hx-group-label">{t("chat:unsorted")}</div>
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
                    title={t("chat:rename")}
                    aria-label={t("chat:rename")}
                    onClick={() => setEditingProjectId(p.id)}
                  >
                    <LuPencil size={13} />
                  </button>
                  <button
                    type="button"
                    className="hx-icon-btn hx-row-btn"
                    title={t("chat:delete")}
                    aria-label={t("chat:delete")}
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
          aria-label={t("chat:sidebarToggle")}
          title={t("chat:sidebarToggle")}
          onClick={() => setSidebarOpen(true)}
        >
          <FaBarsStaggered size={15} />
        </button>
        <div ref={threadRef} className="flex-grow-1 p-3 overflow-auto">
          {(messages ?? []).length === 0 ? (
            <div className="hx-chat-empty">
              <img src="/logo.png" alt="" aria-hidden="true" />
              <p>{t("chat:emptyState")}</p>
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
                  {/* Assistant turns are markdown (#194); user turns stay
                    * literal — people type `*` and `_` as punctuation. */}
                  {m.role === "user" ? (
                    m.content
                  ) : (
                    <Markdown
                      content={m.content}
                      caret={m.status === "streaming"}
                      highlight={m.status !== "streaming"}
                      className={m.status === "streaming" ? "hx-md-streaming" : undefined}
                    />
                  )}
                  {m.status === "streaming" && activity && (
                    <span className="hx-searching">
                      {activity.kind === "search"
                        ? t("chat:searching", { query: activity.detail })
                        : t("chat:reading", { host: activity.detail })}
                    </span>
                  )}
                  {m.status === "error" && (
                    <span className="text-danger small"> ⚠ {m.errorCode}</span>
                  )}
                  {m.sources && m.sources.length > 0 && (
                    <div className="hx-sources">
                      <span className="hx-sources-label">{t("chat:sources")}</span>
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
                {authed ? t("chat:topUp") : t("chat:signUp")}
              </a>
            ) : error.code === "rate_limit_exceeded" ? (
              <span className="text-muted">{t("chat:rateLimited")}</span>
            ) : (
              !authed && <a href="/register">{t("chat:signUp")}</a>
            )}
          </Alert>
        )}

        {anonNearLimit && !error && (
          <Alert variant="info" className="m-2 py-2 d-flex align-items-center gap-2 flex-wrap">
            {/* dir="ltr" is load-bearing, not decoration: "/" is a
                bidi-neutral character, so in an RTL locale "3 / 20"
                reorders on screen to "20 / 3" and reads as the wrong way
                round. Isolating the counter keeps the numerator first in
                every script. */}
            <span className="hx-anon-count" dir="ltr">
              {anonCount.toLocaleString(i18n.language)} /{" "}
              {ANON_MESSAGE_CAP.toLocaleString(i18n.language)}
            </span>
            <span>{t("chat:anonNearLimit")}</span>
            <Link to="/auth?tab=signup">{t("chat:signUp")}</Link>
          </Alert>
        )}

        {capped && !error && (
          <Alert variant="info" className="m-2 py-2">
            {t("chat:anonBanner")}{" "}
            <Link to="/auth?tab=signup">{t("chat:signUp")}</Link>
          </Alert>
        )}

        {needsKey && !error && (
          <Alert variant="info" className="m-2 py-2">
            {t("chat:needsKey")} <a href="/account/keys">{t("chat:manageKeysLink")}</a>
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
              capped ? t("chat:anonBanner") : needsKey ? t("chat:needsKey") : t("chat:inputPlaceholder")
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
              aria-label={t("chat:stop")}
              title={t("chat:stop")}
            >
              <FaStop size={13} />
            </button>
          ) : (
            <button
              type="submit"
              className="hx-send-btn"
              disabled={capped || needsKey || !draft.trim()}
              aria-label={t("chat:send")}
              title={t("chat:send")}
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
        title={t("chat:rename")}
        aria-label={t("chat:rename")}
        onClick={() => onCommit(value)}
      >
        <LuCheck size={13} />
      </button>
      <button
        type="button"
        className="hx-icon-btn hx-row-btn"
        aria-label={t("chat:cancel")}
        title={t("chat:cancel")}
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
    ...(conv.projectId !== null ? [{ id: null, name: t("chat:unsorted") }] : []),
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
          title={t("chat:rename")}
          aria-label={t("chat:rename")}
          onClick={() => setEditing(true)}
        >
          <LuPencil size={13} />
        </button>
        {destinations.length > 0 && (
          <button
            type="button"
            className="hx-icon-btn hx-row-btn"
            title={t("chat:moveTo")}
            aria-label={t("chat:moveTo")}
            onClick={() => setMenuOpen((v) => !v)}
          >
            <LuFolderInput size={13} />
          </button>
        )}
        <button
          type="button"
          className="hx-icon-btn hx-row-btn"
          title={t("chat:delete")}
          aria-label={t("chat:delete")}
          onClick={() => {
            if (window.confirm(t("chat:deleteThreadConfirm"))) {
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
