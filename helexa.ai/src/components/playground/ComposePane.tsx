import { useTranslation } from "react-i18next";
import type { KeyboardEvent } from "react";
import { FaPlay, FaStop } from "react-icons/fa6";
import type { QuestionType } from "../../lib/decisionClient";
import {
  CHECKPOINTS,
  addQuestion,
  formatJson,
  type Checkpoint,
  type Draft,
  type Issue,
  type IssueField,
  type StateMode,
} from "../../lib/decisionRequest";
import "./compose.css";

const PRIMITIVES: QuestionType[] = ["noul", "score", "choice"];

export interface ComposePaneProps {
  draft: Draft;
  onChange: (patch: Partial<Draft>) => void;
  /** Problems with the draft; Run is disabled while there are any. */
  issues: Issue[];
  busy: boolean;
  onRun: () => void;
  onCancel: () => void;
}

function IssueList({ issues, field }: { issues: Issue[]; field: IssueField }) {
  const { t } = useTranslation("decisions");
  const mine = issues.filter((i) => i.field === field);
  if (!mine.length) return null;
  return (
    <ul className="hx-pg-issues" role="alert" aria-label={t("issues.title")}>
      {mine.map((i, n) => (
        <li key={n}>{t(i.key, i.params)}</li>
      ))}
    </ul>
  );
}

/**
 * The left pane: the state, the questions, a picker that inserts each
 * primitive's template, the checkpoint choice, and Run.
 *
 * Validation is live — the page rebuilds the request on every edit — so
 * Run is only ever enabled for a request the server will accept on shape.
 */
export default function ComposePane({ draft, onChange, issues, busy, onRun, onCancel }: ComposePaneProps) {
  const { t } = useTranslation("decisions");
  const canRun = !busy && issues.length === 0;

  function onKeyDown(e: KeyboardEvent): void {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (canRun) onRun();
    }
  }

  // Adding a question rewrites the editor, so it is refused while the text
  // is not an object it can add to — never at the cost of what was typed.
  const addFailed = issues.some(
    (i) => i.key === "issues.questionsJson" || i.key === "issues.questionsNotObject",
  );

  return (
    <section className="hx-pg-compose" aria-label={t("compose.label")} onKeyDown={onKeyDown}>
      <div className="hx-pg-panel">
        <div className="hx-pg-panel-head">
          <label htmlFor="pg-state">{t("state.label")}</label>
          <div className="hx-pg-seg" role="group" aria-label={t("state.mode")}>
            {(["text", "json"] as StateMode[]).map((m) => (
              <button
                key={m}
                type="button"
                className={draft.stateMode === m ? "active" : ""}
                aria-pressed={draft.stateMode === m}
                onClick={() => onChange({ stateMode: m })}
              >
                {t(`state.${m}`)}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="pg-state"
          className={`hx-pg-editor${draft.stateMode === "json" ? " hx-pg-mono" : ""}`}
          rows={7}
          spellCheck={draft.stateMode === "text"}
          value={draft.stateText}
          placeholder={t("state.placeholder")}
          onChange={(e) => onChange({ stateText: e.target.value })}
        />
        <small className="hx-pg-help">{t(`state.help.${draft.stateMode}`)}</small>
        <IssueList issues={issues} field="state" />
      </div>

      <div className="hx-pg-panel">
        <div className="hx-pg-panel-head">
          <label htmlFor="pg-questions">{t("questions.label")}</label>
          <button
            type="button"
            className="hx-pg-link"
            onClick={() => {
              const f = formatJson(draft.questionsText);
              if (f !== null) onChange({ questionsText: f });
            }}
          >
            {t("questions.format")}
          </button>
        </div>
        <textarea
          id="pg-questions"
          className="hx-pg-editor hx-pg-mono"
          rows={14}
          spellCheck={false}
          dir="ltr"
          value={draft.questionsText}
          onChange={(e) => onChange({ questionsText: e.target.value })}
        />
        <small className="hx-pg-help">{t("questions.help")}</small>
        <IssueList issues={issues} field="questions" />

        <div className="hx-pg-add" role="group" aria-label={t("add.label")}>
          <span className="hx-pg-eyebrow">{t("add.label")}</span>
          <div className="hx-pg-add-grid">
            {PRIMITIVES.map((p) => (
              <button
                key={p}
                type="button"
                className="hx-pg-add-btn"
                disabled={addFailed}
                onClick={() => {
                  const next = addQuestion(draft.questionsText, p);
                  if (next !== null) onChange({ questionsText: next });
                }}
              >
                <span className={`hx-pg-type hx-pg-type-${p}`}>{p}</span>
                <strong>{t(`add.${p}.title`)}</strong>
                <span className="hx-pg-add-desc">{t(`add.${p}.desc`)}</span>
              </button>
            ))}
          </div>
          {addFailed && <small className="hx-pg-help">{t("add.invalidJson")}</small>}
        </div>
      </div>

      <IssueList issues={issues} field="budget" />

      <div className="hx-pg-runbar">
        <label className="hx-pg-checkpoint">
          <span>{t("checkpoint.label")}</span>
          <select
            value={draft.checkpoint}
            onChange={(e) => onChange({ checkpoint: e.target.value as Checkpoint })}
          >
            {CHECKPOINTS.map((c) => (
              <option key={c} value={c}>
                {t(`checkpoint.${c}`)}
              </option>
            ))}
          </select>
        </label>
        <div className="hx-pg-run">
          <span className="hx-pg-help" aria-hidden>
            {t("shortcut")}
          </span>
          {busy ? (
            <button type="button" className="hx-btn-ghost" onClick={onCancel}>
              <FaStop aria-hidden /> {t("cancel")}
            </button>
          ) : (
            <button type="button" className="hx-btn-primary" disabled={!canRun} onClick={onRun}>
              <FaPlay aria-hidden /> {t("run")}
            </button>
          )}
        </div>
      </div>
      <small className="hx-pg-help">{t("checkpoint.help")}</small>
    </section>
  );
}
