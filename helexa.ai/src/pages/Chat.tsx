import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLiveQuery } from "dexie-react-hooks";
import { Alert, Badge, Button, Form } from "react-bootstrap";
import { db } from "../data/db";
import {
  createConversation,
  createProject,
  listConversations,
  listProjects,
} from "../data/repositories";
import { getFingerprint } from "../lib/fingerprint";
import { useChat } from "../lib/useChat";

const ANON_OWNER = "anon";
const ANON_MODEL = import.meta.env.VITE_ANON_MODEL || "helexa/small";
const ANON_MESSAGE_CAP = 20;
const ANON_COUNT_KEY = "anonMessageCount";

/**
 * The chat workspace landing (`/`). Anonymous + fingerprinted: history and
 * project organisation live entirely in IndexedDB; inference streams from
 * the mesh router with no bearer (the constrained anonymous model), capped
 * client-side with a sign-up nudge. Authenticated mode (bearer + full
 * models, owner = account) lands in F5.
 */
export default function Chat() {
  const { t } = useTranslation("chat");
  const owner = ANON_OWNER;

  // Namespace anonymous data to the fingerprint (best-effort) at mount.
  useEffect(() => {
    void getFingerprint();
  }, []);

  const projects = useLiveQuery(() => listProjects(owner), [owner], []);
  const conversations = useLiveQuery(() => listConversations(owner), [owner], []);
  const [activeId, setActiveId] = useState<string | null>(null);

  const anonCount =
    useLiveQuery(async () => {
      const m = await db.meta.get(ANON_COUNT_KEY);
      return typeof m?.value === "number" ? m.value : 0;
    }, [], 0) ?? 0;
  const capped = anonCount >= ANON_MESSAGE_CAP;

  const messages = useLiveQuery(
    async () => {
      if (!activeId) return [];
      const { listMessages } = await import("../data/repositories");
      return listMessages(activeId);
    },
    [activeId],
    [],
  );

  const { streaming, error, send, stop } = useChat(activeId, { model: ANON_MODEL });
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight });
  }, [messages]);

  async function newChat(projectId: string | null = null) {
    const id = await createConversation(owner, ANON_MODEL, projectId);
    setActiveId(id);
  }

  async function onSend() {
    const text = draft.trim();
    if (!text || streaming || capped) return;
    let convId = activeId;
    if (!convId) {
      convId = await createConversation(owner, ANON_MODEL);
      setActiveId(convId);
    }
    setDraft("");
    await db.meta.put({ key: ANON_COUNT_KEY, value: anonCount + 1 });
    await send(text);
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
      {/* Sidebar */}
      <aside
        className="border-end p-3 d-flex flex-column gap-2"
        style={{ width: 280, overflowY: "auto" }}
      >
        <div className="d-flex gap-2">
          <Button size="sm" variant="primary" onClick={() => void newChat()}>
            {t("newChat")}
          </Button>
          <Button
            size="sm"
            variant="outline-secondary"
            onClick={() => void createProject(owner, t("newProjectName"))}
          >
            {t("newProject")}
          </Button>
        </div>

        <ConversationGroup
          label={t("unsorted")}
          items={grouped.get(null) ?? []}
          activeId={activeId}
          onSelect={setActiveId}
        />
        {(projects ?? []).map((p) => (
          <ConversationGroup
            key={p.id}
            label={p.name}
            items={grouped.get(p.id) ?? []}
            activeId={activeId}
            onSelect={setActiveId}
          />
        ))}
      </aside>

      {/* Main */}
      <section className="d-flex flex-column flex-grow-1" style={{ minWidth: 0 }}>
        <div ref={threadRef} className="flex-grow-1 p-3 overflow-auto">
          {(messages ?? []).length === 0 ? (
            <div className="text-muted text-center mt-5">
              <Badge bg="secondary" className="mb-2">
                {t("badge")}
              </Badge>
              <p>{t("emptyState")}</p>
            </div>
          ) : (
            (messages ?? []).map((m) => (
              <div
                key={m.id}
                className={`mb-3 d-flex ${m.role === "user" ? "justify-content-end" : "justify-content-start"}`}
              >
                <div
                  className={`surface-elevated p-2 px-3 rounded-3 ${m.role === "user" ? "bg-body-tertiary" : ""}`}
                  style={{ maxWidth: "80%", whiteSpace: "pre-wrap" }}
                >
                  {m.content}
                  {m.status === "streaming" && <span className="opacity-50"> ▋</span>}
                  {m.status === "error" && (
                    <span className="text-danger small"> ⚠ {m.errorCode}</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {error && (
          <Alert variant="warning" className="m-2 py-2">
            {error.message}{" "}
            {(error.code === "insufficient_quota" ||
              error.code === "rate_limit_exceeded" ||
              capped) && <a href="/register">{t("signUp")}</a>}
          </Alert>
        )}

        {capped && !error && (
          <Alert variant="info" className="m-2 py-2">
            {t("anonBanner")} <a href="/register">{t("signUp")}</a>
          </Alert>
        )}

        <Form
          className="d-flex gap-2 p-2 border-top"
          onSubmit={(e) => {
            e.preventDefault();
            void onSend();
          }}
        >
          <Form.Control
            as="textarea"
            rows={1}
            value={draft}
            disabled={capped}
            placeholder={capped ? t("anonBanner") : t("inputPlaceholder")}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void onSend();
              }
            }}
          />
          {streaming ? (
            <Button variant="outline-danger" onClick={stop}>
              {t("stop")}
            </Button>
          ) : (
            <Button type="submit" variant="primary" disabled={capped || !draft.trim()}>
              {t("send")}
            </Button>
          )}
        </Form>
      </section>
    </div>
  );
}

function ConversationGroup({
  label,
  items,
  activeId,
  onSelect,
}: {
  label: string;
  items: { id: string; title: string }[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  if (items.length === 0) return null;
  return (
    <div>
      <div className="text-uppercase text-muted small fw-semibold mt-2 mb-1">
        {label}
      </div>
      {items.map((c) => (
        <button
          key={c.id}
          type="button"
          onClick={() => onSelect(c.id)}
          className={`btn btn-sm w-100 text-start text-truncate ${
            c.id === activeId ? "btn-secondary" : "btn-link text-body"
          }`}
        >
          {c.title}
        </button>
      ))}
    </div>
  );
}
