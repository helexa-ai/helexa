import { useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FaCheck, FaCopy, FaLink } from "react-icons/fa6";
import type { Draft } from "../../lib/decisionRequest";
import { MAX_SHARE_URL, encodeShare } from "../../lib/decisionShare";
import { PUBLIC_API_BASE, SNIPPET_LANGS, snippet, type SnippetLang } from "../../lib/decisionSnippets";
import "./share.css";

/**
 * Copies a link to the current draft. The draft rides in the URL fragment,
 * so the link reveals nothing to any server, and opening it restores the
 * editors without running anything.
 */
export function ShareButton({ draft }: { draft: Draft }) {
  const { t } = useTranslation("decisions");
  const [state, setState] = useState<"idle" | "copied" | "tooLong" | "failed">("idle");

  async function share(): Promise<void> {
    const hash = await encodeShare(draft);
    const url = `${window.location.origin}${window.location.pathname}${hash}`;
    if (url.length > MAX_SHARE_URL) {
      setState("tooLong");
      return;
    }
    window.history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url);
      setState("copied");
    } catch {
      // The link is still in the address bar, ready to copy by hand.
      setState("failed");
    }
    setTimeout(() => setState((s) => (s === "copied" ? "idle" : s)), 2000);
  }

  return (
    <span className="hx-pg-share">
      <button type="button" className="hx-btn-ghost" onClick={() => void share()}>
        {state === "copied" ? <FaCheck aria-hidden /> : <FaLink aria-hidden />}{" "}
        {state === "copied" ? t("share.copied") : t("share.button")}
      </button>
      {state === "tooLong" && (
        <small className="hx-pg-warn" role="status">
          {t("share.tooLong")}
        </small>
      )}
      {state === "failed" && (
        <small className="hx-pg-help" role="status">
          {t("share.inAddressBar")}
        </small>
      )}
    </span>
  );
}

/** The current request as curl, Python and laya-SDK code. */
export function CodePanel({ body }: { body: string | null }) {
  const { t } = useTranslation("decisions");
  const [lang, setLang] = useState<SnippetLang>("curl");
  const [copied, setCopied] = useState(false);
  const code = body ? snippet(lang, body, PUBLIC_API_BASE) : null;
  return (
    <div className="hx-pg-code">
      <div className="hx-pg-seg" role="tablist" aria-label={t("code.language")}>
        {SNIPPET_LANGS.map((l) => (
          <button
            key={l}
            type="button"
            role="tab"
            aria-selected={lang === l}
            className={lang === l ? "active" : ""}
            onClick={() => setLang(l)}
          >
            {t(`code.${l}`)}
          </button>
        ))}
      </div>
      {code ? (
        <>
          <pre className="hx-pg-pre" dir="ltr">
            <code>{code}</code>
          </pre>
          <button
            type="button"
            className="hx-pg-link"
            onClick={() =>
              void navigator.clipboard?.writeText(code).then(
                () => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 1500);
                },
                () => undefined,
              )
            }
          >
            {copied ? <FaCheck aria-hidden /> : <FaCopy aria-hidden />} {copied ? t("copied") : t("copy")}
          </button>
        </>
      ) : (
        <p className="hx-pg-help">{t("code.fixFirst")}</p>
      )}
      <p className="hx-pg-help">{t("code.jev")}</p>
      <p className="hx-pg-help">
        {t("code.key")} <Link to="/account/keys">{t("code.keyLink")}</Link>
      </p>
    </div>
  );
}
