import React from "react";
import { useTranslation } from "react-i18next";
import { FaShieldHalved } from "react-icons/fa6";

import "../App.css";
import "../index.css";

/** Privacy contact address — single source so copy and mailto agree. */
const PRIVACY_EMAIL = "privacy@helexa.ai";

/**
 * Privacy
 *
 * The complete disclosure of what helexa stores, where, and why —
 * deliberately short, because the product is built to have little to
 * disclose. Shares the mission page's visual grammar (numbered
 * gradient eyebrows, hx cards, readable measure).
 */
const Privacy: React.FC = () => {
  const { t } = useTranslation("privacy");

  return (
    <main className="app-main container py-4">
      <section className="hx-section">
        <div className="mb-4">
          <span className="badge-accent">
            <FaShieldHalved size={11} />
            {t("title")}
          </span>
        </div>
        <h1 className="hx-display mb-3">{t("title")}</h1>
        <p className="text-muted small mb-4">{t("updated")}</p>
        <p className="lead mb-0" style={{ maxWidth: "46rem" }}>
          {t("intro")}
        </p>
      </section>

      <section className="hx-section">
        <div className="hx-card p-4" style={{ maxWidth: "46rem" }}>
          <h2 className="h5 mb-3">{t("summary.title")}</h2>
          <ul className="hx-list mb-0">
            <li>{t("summary.noCookies")}</li>
            <li>{t("summary.localHistory")}</li>
            <li>{t("summary.noAnonId")}</li>
          </ul>
        </div>
      </section>

      <section className="hx-section" style={{ maxWidth: "46rem" }}>
        <div className="hx-section-head">
          <span className="hx-eyebrow">01</span>
          <h2>{t("local.title")}</h2>
        </div>
        <p className="mb-3">{t("local.intro")}</p>
        <ul className="hx-list mb-0">
          <li>{t("local.item1")}</li>
          <li>{t("local.item2")}</li>
          <li>{t("local.item3")}</li>
          <li>{t("local.item4")}</li>
          <li>{t("local.item5")}</li>
        </ul>
      </section>

      <section className="hx-section" style={{ maxWidth: "46rem" }}>
        <div className="hx-section-head">
          <span className="hx-eyebrow">02</span>
          <h2>{t("account.title")}</h2>
        </div>
        <p className="mb-3">{t("account.intro")}</p>
        <ul className="hx-list mb-0">
          <li>{t("account.item1")}</li>
          <li>{t("account.item2")}</li>
          <li>{t("account.item3")}</li>
        </ul>
      </section>

      <section className="hx-section" style={{ maxWidth: "46rem" }}>
        <div className="hx-section-head">
          <span className="hx-eyebrow">03</span>
          <h2>{t("infra.title")}</h2>
        </div>
        <p className="mb-0">{t("infra.body")}</p>
      </section>

      <section className="hx-section" style={{ maxWidth: "46rem" }}>
        <div className="hx-section-head">
          <span className="hx-eyebrow">04</span>
          <h2>{t("rights.title")}</h2>
        </div>
        <p className="mb-0">{t("rights.body")}</p>
      </section>

      <section className="hx-section" style={{ maxWidth: "46rem" }}>
        <div className="hx-section-head">
          <span className="hx-eyebrow">05</span>
          <h2>{t("contact.title")}</h2>
        </div>
        <p className="mb-0">
          {t("contact.body", { email: "" })}
          <a href={`mailto:${PRIVACY_EMAIL}`}>{PRIVACY_EMAIL}</a>
        </p>
      </section>
    </main>
  );
};

export default Privacy;
