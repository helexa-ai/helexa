import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useLiveQuery } from "dexie-react-hooks";
import { useAuth } from "../auth/context";
import { CHAT_API_KEY, db } from "../data/db";
import { ensureChatKey } from "../lib/ensureChatKey";
import { DecisionError, runDecision } from "../lib/decisionClient";
import { buildRequest, issueFromError, type Draft, type Issue } from "../lib/decisionRequest";
import { usePlaygroundDraft } from "../lib/usePlaygroundDraft";
import { EXAMPLES, exampleDraft } from "../data/decisionExamples";
import ComposePane from "../components/playground/ComposePane";
import ResultsPane, { type RunState } from "../components/playground/ResultsPane";
import ExamplesPanel from "../components/playground/ExamplesPanel";
import { CodePanel, ShareButton } from "../components/playground/SharePanel";
import "../components/playground/page.css";

type Tab = "results" | "examples" | "code";
const TABS: Tab[] = ["results", "examples", "code"];

/** A first visit opens on the triage use case: several questions, one run. */
const FIRST_DRAFT: Draft = exampleDraft(EXAMPLES.find((e) => e.id === "triage") ?? EXAMPLES[0]);

/**
 * `/playground` — try a decision model (System-One, `/v1/systemone`) in the
 * browser (#356).
 *
 * Write a state and some typed questions, run them, and read calibrated
 * answers: every option's probability, how confident the model is, which
 * checkpoint answered and why, and what it cost. Anonymous use works like
 * the chat does — a decision is one short forward pass, and the edge's
 * per-address limit is the backstop; a signed-in browser spends through
 * its own key.
 */
export default function Playground() {
  const { t } = useTranslation("decisions");
  const { status, accountId, token } = useAuth();
  const authed = status === "authed" && !!accountId;

  const apiKey = useLiveQuery<string | null, undefined>(
    async () => {
      const m = await db.meta.get(CHAT_API_KEY);
      return typeof m?.value === "string" ? m.value : null;
    },
    [],
    undefined,
  );
  useEffect(() => {
    if (!authed || !token || apiKey !== null) return;
    void ensureChatKey(token);
  }, [authed, token, apiKey]);

  const { draft, setDraft } = usePlaygroundDraft(FIRST_DRAFT);
  const built = useMemo(() => buildRequest(draft), [draft]);
  const [run, setRun] = useState<RunState>({ status: "idle" });
  const [serverIssue, setServerIssue] = useState<Issue | null>(null);
  const [tab, setTab] = useState<Tab>("examples");
  const abort = useRef<AbortController | null>(null);

  const issues = built.ok ? (serverIssue ? [serverIssue] : []) : built.issues;

  function edit(patch: Partial<Draft>): void {
    setDraft((d) => ({ ...d, ...patch }));
    setServerIssue(null);
  }

  async function onRun(): Promise<void> {
    if (!built.ok || run.status === "running") return;
    const { body, questions } = built.request;
    const ctl = new AbortController();
    abort.current = ctl;
    setServerIssue(null);
    setRun({ status: "running" });
    setTab("results");
    try {
      const result = await runDecision({
        body,
        apiKey: authed ? (apiKey ?? undefined) : undefined,
        signal: ctl.signal,
      });
      setRun({ status: "done", result, body, questions });
    } catch (e) {
      const error = e instanceof DecisionError ? e : new DecisionError("network", String(e));
      setRun(error.kind === "cancelled" ? { status: "idle" } : { status: "error", error });
      setServerIssue(issueFromError(error));
    } finally {
      abort.current = null;
    }
  }

  return (
    <main className="app-main container-xxl py-4 hx-playground">
      <header className="hx-pg-page-head">
        <div>
          <h1 className="h4 mb-1">{t("page.title")}</h1>
          <p className="hx-pg-help mb-0">
            {t("page.lead")} <Link to="/docs/using/decisions">{t("page.docsLink")}</Link>
          </p>
        </div>
        <ShareButton draft={draft} />
      </header>

      <div className="hx-pg-grid">
        <ComposePane
          draft={draft}
          onChange={edit}
          issues={issues}
          busy={run.status === "running"}
          onRun={() => void onRun()}
          onCancel={() => abort.current?.abort()}
        />

        <section className="hx-pg-side" aria-label={t("page.side")}>
          <div className="hx-pg-tabs" role="tablist">
            {TABS.map((k) => (
              <button
                key={k}
                type="button"
                role="tab"
                aria-selected={tab === k}
                className={tab === k ? "active" : ""}
                onClick={() => setTab(k)}
              >
                {t(`page.tabs.${k}`)}
              </button>
            ))}
          </div>
          <div role="tabpanel">
            {tab === "results" && <ResultsPane run={run} />}
            {tab === "examples" && (
              <ExamplesPanel
                onPick={(d) => {
                  setDraft(d);
                  setServerIssue(null);
                  setRun({ status: "idle" });
                }}
              />
            )}
            {tab === "code" && <CodePanel body={built.ok ? built.request.body : null} />}
          </div>
        </section>
      </div>
    </main>
  );
}
