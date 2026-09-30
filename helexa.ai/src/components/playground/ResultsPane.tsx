import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { FaCopy, FaCheck } from "react-icons/fa6";
import type { DecisionError, DecisionResult, ParsedAnswer } from "../../lib/decisionClient";
import { isMap, parseOrdered, stringifyOrdered, type OJ, type OMap } from "../../lib/orderedJson";
import "./results.css";

export type RunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: DecisionResult; body: string; questions: OMap }
  | { status: "error"; error: DecisionError };

function usePercent(): (p: number) => string {
  const { i18n } = useTranslation();
  const fmt = new Intl.NumberFormat(i18n.language, { style: "percent", maximumFractionDigits: 1 });
  return (p: number) => fmt.format(p);
}

function asText(v: OJ | undefined): string | undefined {
  if (v === undefined || v === null) return undefined;
  return typeof v === "string" ? v : stringifyOrdered(v, 0);
}

/** Bar label and description for one option, from the question as sent. */
function describeOption(
  answer: ParsedAnswer,
  key: string,
  question: OJ | undefined,
): { label: string; description?: string } {
  const q = isMap(question) ? question : undefined;
  const crit = q?.get("criteria");
  if (answer.type === "noul") {
    const labels = q?.get("labels");
    const label = isMap(labels) ? asText(labels.get(key)) : undefined;
    // The server accepts true/false criteria keys in any case.
    const entry = isMap(crit) ? [...crit].find(([k]) => k.toLowerCase() === key) : undefined;
    const desc = entry ? asText(entry[1]) : undefined;
    return { label: label ?? key, description: desc };
  }
  if (answer.type === "score") {
    const legend = answer.options.find((o) => o.key === key)?.legend;
    return { label: legend ? `${key} · ${legend}` : key };
  }
  return { label: key, description: isMap(crit) ? asText(crit.get(key)) || undefined : undefined };
}

function AnswerCard({ answer, question, collapsed }: { answer: ParsedAnswer; question?: OJ; collapsed: boolean }) {
  const { t } = useTranslation("decisions");
  const pct = usePercent();
  const instructions = isMap(question) ? asText(question.get("instructions")) : undefined;
  const headline =
    answer.type === "choice"
      ? answer.choice
      : answer.type === "score"
        ? t("results.expectedScore", { score: (answer.score ?? 0).toFixed(2) })
        : t("results.pTrue", { p: pct(answer.noul ?? 0) });
  return (
    <article className="hx-pg-panel hx-pg-answer" aria-label={answer.qid}>
      <div className="hx-pg-answer-head">
        <span className={`hx-pg-type hx-pg-type-${answer.type}`}>{answer.type}</span>
        <code className="hx-pg-qid" dir="auto">
          {answer.qid}
        </code>
      </div>
      {instructions && (
        <p className="hx-pg-instructions" dir="auto">
          {instructions}
        </p>
      )}
      <p className="hx-pg-headline" dir="auto">
        {headline}
      </p>
      <ul className="hx-pg-bars">
        {answer.options.map((o) => {
          const { label, description } = describeOption(answer, o.key, question);
          const chosen = o.key === answer.chosen;
          return (
            <li key={o.key} className={chosen ? "chosen" : ""} aria-current={chosen || undefined}>
              <div className="hx-pg-bar-label">
                <span dir="auto">{label}</span>
                <span className="hx-pg-bar-pct">{pct(o.probability)}</span>
              </div>
              <div className="hx-pg-bar-track">
                <div className="hx-pg-bar-fill" style={{ width: `${Math.max(0, Math.min(1, o.probability)) * 100}%` }} />
              </div>
              {description && (
                <small className="hx-pg-help" dir="auto">
                  {description}
                </small>
              )}
            </li>
          );
        })}
      </ul>
      <dl className="hx-pg-conf">
        <div>
          <dt>{t("results.answerConfidence")}</dt>
          <dd>{pct(answer.answerConfidence)}</dd>
        </div>
        <div>
          <dt>{t("results.confidence")}</dt>
          <dd>{pct(answer.confidence)}</dd>
        </div>
      </dl>
      {collapsed && <p className="hx-pg-warn">{t("results.collapsed")}</p>}
    </article>
  );
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const { t } = useTranslation("decisions");
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="hx-pg-link"
      onClick={() =>
        void navigator.clipboard?.writeText(text).then(
          () => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          },
          () => undefined,
        )
      }
    >
      {copied ? <FaCheck aria-hidden /> : <FaCopy aria-hidden />} {copied ? t("copied") : label}
    </button>
  );
}

function pretty(text: string): string {
  try {
    return stringifyOrdered(parseOrdered(text));
  } catch {
    return text;
  }
}

function Results({ result, body, questions }: { result: DecisionResult; body: string; questions: OMap }) {
  const { t } = useTranslation("decisions");
  const pct = usePercent();
  const { response, timing } = result;
  const routing = response.routing;
  const collapsed = new Set(response.collapsed.map((c) => c.qid));
  const multilingual = routing?.model === "multilingual";
  return (
    <>
      <p className="hx-pg-help hx-pg-conf-help">{t("results.confidenceHelp")}</p>
      {response.answers.map((a) => (
        <AnswerCard key={a.qid} answer={a} question={questions.get(a.qid)} collapsed={collapsed.has(a.qid)} />
      ))}

      {routing && (
        <section className="hx-pg-panel hx-pg-routing" aria-label={t("routing.title")}>
          <h3 className="hx-pg-eyebrow">{t("routing.title")}</h3>
          <p className="hx-pg-checkpoint-line">
            <span className={`hx-pg-pill${multilingual ? " hx-pg-pill-ml" : ""}`}>{routing.model}</span>
            <code className="hx-pg-repo">{routing.repo}</code>
          </p>
          <p className="hx-pg-reason" dir="ltr">
            {routing.reason}
          </p>
          {routing.detection && (
            <>
              <p className="hx-pg-help mb-1">
                {t("routing.detected", {
                  script: routing.detection.script ?? "—",
                  language: routing.detection.language ?? t("routing.undecided"),
                })}
              </p>
              {routing.detection.scriptProfile.length > 0 && (
                <ul className="hx-pg-scripts" aria-label={t("routing.scripts")}>
                  {routing.detection.scriptProfile.map(([script, share]) => (
                    <li key={script}>
                      <span>{script}</span> <strong>{pct(share)}</strong>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
          {!routing.detection && <p className="hx-pg-help mb-0">{t("routing.pinned")}</p>}
        </section>
      )}

      <section className="hx-pg-stats" aria-label={t("usage.title")}>
        <div>
          <span className="hx-pg-stat">{response.inputTokens.toLocaleString()}</span>
          <span className="hx-pg-help">{t("usage.inputTokens")}</span>
        </div>
        {timing.serverMs !== undefined && (
          <div>
            <span className="hx-pg-stat">{t("usage.ms", { ms: timing.serverMs.toFixed(1) })}</span>
            <span className="hx-pg-help">{t("usage.server")}</span>
          </div>
        )}
        <div>
          <span className="hx-pg-stat">{t("usage.ms", { ms: timing.clientMs.toFixed(0) })}</span>
          <span className="hx-pg-help">{t("usage.roundTrip")}</span>
        </div>
      </section>
      <p className="hx-pg-help">{t("usage.help")}</p>

      {[
        ["raw.request", pretty(body)],
        ["raw.response", pretty(result.raw)],
      ].map(([key, text]) => (
        <details key={key} className="hx-pg-raw">
          <summary>{t(key)}</summary>
          <CopyButton text={text} label={t("copy")} />
          <pre className="hx-pg-pre" dir="ltr">
            <code>{text}</code>
          </pre>
        </details>
      ))}
    </>
  );
}

/** Seconds left of a Retry-After, ticking down once a second. */
function useCountdown(error: DecisionError): number | undefined {
  const [left, setLeft] = useState(error.retryAfter);
  useEffect(() => {
    if (error.retryAfter === undefined) return;
    const until = Date.now() + error.retryAfter * 1000;
    const id = setInterval(() => {
      const s = Math.max(0, Math.ceil((until - Date.now()) / 1000));
      setLeft(s);
      if (s === 0) clearInterval(id);
    }, 250);
    return () => clearInterval(id);
  }, [error]);
  return left;
}

export function ErrorPanel({ error }: { error: DecisionError }) {
  const { t } = useTranslation("decisions");
  const left = useCountdown(error);
  const waits = error.kind === "rate_limited" || error.kind === "unavailable";
  return (
    <section className="hx-pg-error" role="alert">
      <strong>{t(`errors.${error.kind}.title`)}</strong>
      <p className="mb-1">{t(`errors.${error.kind}.help`)}</p>
      {(error.kind === "validation" || error.kind === "too_large" || error.kind === "not_found") && (
        <p className="hx-pg-error-detail mb-1" dir="ltr">
          {error.questionId !== undefined && <code className="hx-pg-qid">{error.questionId}</code>} {error.message}
        </p>
      )}
      {waits &&
        (left !== undefined ? (
          <p className="hx-pg-countdown mb-1" aria-live="polite">
            {left > 0 ? t("errors.retryIn", { seconds: left }) : t("errors.retryNow")}
          </p>
        ) : error.kind === "rate_limited" ? (
          <p className="mb-1">{t("errors.rateLimitNote")}</p>
        ) : null)}
      {error.status !== undefined && (
        <small className="hx-pg-help" dir="ltr">
          HTTP {error.status} · {error.code}
        </small>
      )}
    </section>
  );
}

/** The right pane's results view: answers, routing, usage, raw, errors. */
export default function ResultsPane({ run }: { run: RunState }) {
  const { t } = useTranslation("decisions");
  return (
    <div className="hx-pg-results" aria-live="polite" aria-busy={run.status === "running"}>
      {run.status === "idle" && <p className="hx-pg-help">{t("results.empty")}</p>}
      {run.status === "running" && <p className="hx-pg-help">{t("results.running")}</p>}
      {run.status === "error" && <ErrorPanel error={run.error} />}
      {run.status === "done" && <Results result={run.result} body={run.body} questions={run.questions} />}
    </div>
  );
}
