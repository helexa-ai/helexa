import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FaArrowUp } from "react-icons/fa6";
import { useAuth } from "../auth/context";

/**
 * `/` — the front door for someone who has not used helexa before.
 *
 * Prompt-first, and deliberately without the chat workspace's sidebar:
 * the previous version rendered a marketing block inside the chat view, so
 * a visitor met a column of somebody else's conversation history beside an
 * advert. Here the input is the page, and the proposition sits beneath it
 * for anyone who scrolls.
 *
 * Sending hands the draft to `/chat`, which creates the conversation and
 * sends it — so the first message a visitor types is the first message in
 * their thread, rather than something they have to retype.
 *
 * Everyone sees this, signed in or not, so that the brand link in the
 * header leads somewhere predictable. The only thing that varies is the
 * call to action: offering "create an account" to someone who already has
 * one is noise, so they get the way into the workspace instead.
 *
 * No new i18n keys. Every string here already exists and is translated in
 * all 42 shipped locales — the mission copy is reused verbatim so it stays
 * operator-written.
 */
export default function Landing() {
  const { t } = useTranslation(["chat", "mission", "common"]);
  const { status, accountId } = useAuth();
  const authed = status === "authed" && !!accountId;
  const navigate = useNavigate();
  const [draft, setDraft] = useState("");
  const boxRef = useRef<HTMLTextAreaElement>(null);

  function send(): void {
    const text = draft.trim();
    if (!text) return;
    // Router state rather than a query parameter: the prompt is the
    // visitor's own words and has no business in a URL, browser history or
    // a referrer header.
    navigate("/chat", { state: { prompt: text } });
  }

  return (
    <main className="hx-landing-page">
      <section className="hx-landing-hero">
        <img src="/logo.png" alt="" aria-hidden="true" className="hx-landing-mark" />
        <h1>{t("mission:hero.title")}</h1>

        <form
          className="hx-landing-composer"
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
        >
          <textarea
            ref={boxRef}
            rows={3}
            value={draft}
            placeholder={t("chat:inputPlaceholder")}
            aria-label={t("chat:inputPlaceholder")}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button
            type="submit"
            className="hx-landing-send"
            disabled={!draft.trim()}
            aria-label={t("chat:send")}
            title={t("chat:send")}
          >
            <FaArrowUp size={16} />
          </button>
        </form>

        <div className="hx-landing-cta">
          {authed ? (
            <Link to="/chat" className="hx-btn-primary">
              {t("common:nav.chat")}
            </Link>
          ) : (
            <Link to="/auth?tab=signup" className="hx-btn-primary">
              {t("common:nav.register")}
            </Link>
          )}
          <Link to="/mission">{t("common:nav.mission")}</Link>
        </div>
      </section>

      <section className="hx-landing-below">
        <span className="hx-landing-badge">{t("mission:hero.badge")}</span>
        <p className="hx-landing-lead">{t("mission:hero.lead")}</p>
        <ul className="hx-landing-points">
          {(["operators", "routing", "value"] as const).map((k) => (
            <li key={k}>
              <strong>{t(`mission:howItWorks.${k}.eyebrow`)}</strong>
              <span>{t(`mission:howItWorks.${k}.title`)}</span>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
