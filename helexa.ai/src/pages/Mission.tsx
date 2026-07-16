import React from "react";
import { Row, Col } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import {
  FaGithub,
  FaArrowUpRightFromSquare,
  FaCircleNodes,
  FaPeopleGroup,
  FaScaleBalanced,
  FaInfinity,
} from "react-icons/fa6";

import "../App.css";
import "../index.css";

/**
 * Mission
 *
 * Narrative homepage: hero → intent → why now → how it works →
 * principles → road ahead → CTA. Sections share one visual grammar —
 * a numbered gradient eyebrow, a tight title, and content on `hx-card`
 * surfaces — so the page reads as one system rather than stacked
 * widgets. The only accent is the helix gradient.
 */
const Mission: React.FC = () => {
  return (
    <main className="app-main container py-4">
      <HeroSection />
      <IntentSection />
      <WhyNowSection />
      <HowItWorksSection />
      <PrinciplesSection />
      <RoadAheadSection />
      <JoinMeshSection />
    </main>
  );
};

/** Shared section header: numbered gradient eyebrow + title. The number
 * is language-neutral, so it needs no i18n key. */
const SectionHead: React.FC<{ num: string; title: string }> = ({
  num,
  title,
}) => (
  <div className="hx-section-head">
    <span className="hx-eyebrow">{num}</span>
    <h2>{title}</h2>
  </div>
);

const HeroSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="hx-section">
      <Row className="align-items-center g-5">
        <Col lg={6}>
          <div className="mb-4">
            <span className="badge-accent">
              <span className="bg-cyan-500-dot" />
              {t("hero.badge")}
            </span>
          </div>
          <h1 className="hx-display mb-4">{t("hero.title")}</h1>
          <p className="lead mb-4">{t("hero.lead")}</p>
          <p className="text-muted small mb-0">{t("hero.subcopy")}</p>
        </Col>

        <Col lg={6}>
          <div className="hx-hero-visual">
            <img src="/logo.png" alt={t("hero.imageAlt")} />
            <div>
              <div className="hx-hero-wordmark">HELEXA</div>
              <div className="hx-hero-tagline">{t("hero.badge")}</div>
            </div>
          </div>
        </Col>
      </Row>
    </section>
  );
};

const IntentSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="hx-section">
      <SectionHead num="01" title={t("intent.title")} />
      <div style={{ maxWidth: "46rem" }}>
        <p className="mb-3">{t("intent.p1")}</p>
        <p className="mb-3">{t("intent.p2Intro")}</p>
        <ul className="hx-list mb-4">
          <li>{t("intent.bullet1")}</li>
          <li>{t("intent.bullet2")}</li>
          <li>{t("intent.bullet3")}</li>
          <li>{t("intent.bullet4")}</li>
        </ul>
        <p className="mb-0 fw-semibold">{t("intent.closing")}</p>
      </div>
    </section>
  );
};

const WhyNowSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="hx-section">
      <SectionHead num="02" title={t("whyNow.title")} />
      <Row className="g-4">
        <Col lg={6}>
          <div className="hx-card p-4 h-100">
            <h3 className="h5 mb-3">{t("whyNow.problemTitle")}</h3>
            <ul className="hx-list mb-0">
              <li>{t("whyNow.problemBullet1")}</li>
              <li>{t("whyNow.problemBullet2")}</li>
              <li>{t("whyNow.problemBullet3")}</li>
              <li>{t("whyNow.problemBullet4")}</li>
              <li>{t("whyNow.problemBullet5")}</li>
            </ul>
          </div>
        </Col>
        <Col lg={6}>
          <div className="hx-card p-4 h-100">
            <h3 className="h5 mb-3">{t("whyNow.opportunityTitle")}</h3>
            <p className="mb-3 text-muted">{t("whyNow.opportunityIntro")}</p>
            <ul className="hx-list mb-3">
              <li>{t("whyNow.opportunityBullet1")}</li>
              <li>{t("whyNow.opportunityBullet2")}</li>
              <li>{t("whyNow.opportunityBullet3")}</li>
              <li>{t("whyNow.opportunityBullet4")}</li>
              <li>{t("whyNow.opportunityBullet5")}</li>
            </ul>
            <p className="fw-semibold mb-0">{t("whyNow.opportunityClosing")}</p>
          </div>
        </Col>
      </Row>
    </section>
  );
};

const HowItWorksSection: React.FC = () => {
  const { t } = useTranslation("mission");

  const steps = [
    {
      num: "01",
      eyebrow: t("howItWorks.operators.eyebrow"),
      title: t("howItWorks.operators.title"),
      body: t("howItWorks.operators.body"),
    },
    {
      num: "02",
      eyebrow: t("howItWorks.routing.eyebrow"),
      title: t("howItWorks.routing.title"),
      body: t("howItWorks.routing.body"),
    },
    {
      num: "03",
      eyebrow: t("howItWorks.value.eyebrow"),
      title: t("howItWorks.value.title"),
      body: t("howItWorks.value.body"),
    },
  ];

  return (
    <section className="hx-section">
      <SectionHead num="03" title={t("howItWorks.title")} />
      <Row className="g-4">
        {steps.map((s) => (
          <Col md={4} key={s.num}>
            <div className="hx-card p-4 h-100">
              <div className="hx-step-num mb-3">{s.num}</div>
              <div className="text-uppercase text-muted small fw-semibold mb-2">
                {s.eyebrow}
              </div>
              <h3 className="h5 mb-3">{s.title}</h3>
              <p className="mb-0 text-muted">{s.body}</p>
            </div>
          </Col>
        ))}
      </Row>
    </section>
  );
};

const PrinciplesSection: React.FC = () => {
  const { t } = useTranslation("mission");

  const principles = [
    {
      icon: <FaCircleNodes size={15} />,
      title: t("principles.distributed.title"),
      body: t("principles.distributed.body"),
    },
    {
      icon: <FaPeopleGroup size={15} />,
      title: t("principles.participation.title"),
      body: t("principles.participation.body"),
    },
    {
      icon: <FaScaleBalanced size={15} />,
      title: t("principles.fairness.title"),
      body: t("principles.fairness.body"),
    },
    {
      icon: <FaInfinity size={15} />,
      title: t("principles.evolving.title"),
      body: t("principles.evolving.body"),
    },
  ];

  return (
    <section className="hx-section">
      <SectionHead num="04" title={t("principles.title")} />
      <Row className="g-4">
        {principles.map((p) => (
          <Col md={6} key={p.title}>
            <div className="hx-card p-4 h-100 d-flex gap-3">
              <div className="hx-card-icon">{p.icon}</div>
              <div>
                <h3 className="h5 mb-2">{p.title}</h3>
                <p className="mb-0 text-muted">{p.body}</p>
              </div>
            </div>
          </Col>
        ))}
      </Row>
    </section>
  );
};

const RoadAheadSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="hx-section">
      <Row className="g-5 align-items-center">
        <Col lg={6}>
          <SectionHead num="05" title={t("roadAhead.title")} />
          <p className="mb-3">{t("roadAhead.p1")}</p>
          <p className="mb-3">{t("roadAhead.p2")}</p>
          <p className="mb-3">{t("roadAhead.p3")}</p>
          <p className="mb-0">{t("roadAhead.p4")}</p>
        </Col>
        <Col lg={6}>
          <div className="hx-card p-4 d-flex flex-column align-items-start gap-3">
            <div className="d-flex align-items-center gap-3">
              <img
                src="/logo.png"
                alt="Helexa logo"
                width={40}
                height={40}
                style={{ borderRadius: "999px" }}
              />
              <div>
                <div className="small text-uppercase text-muted">
                  {t("roadAhead.card.eyebrow")}
                </div>
                <div className="fw-semibold">{t("roadAhead.card.title")}</div>
              </div>
            </div>
            <p className="mb-0 text-muted small">{t("roadAhead.card.body")}</p>
          </div>
        </Col>
      </Row>
    </section>
  );
};

const JoinMeshSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="hx-section">
      <div className="join-mesh-cta p-4 p-md-5 text-center">
        <div className="mb-3">
          <span className="badge-accent">
            <span className="bg-cyan-500-dot" />
            {t("joinMesh.badge")}
          </span>
        </div>
        <h2 className="mb-3">
          {t("joinMesh.title")}{" "}
          <span className="hx-gradient-text">{t("joinMesh.titleHighlight")}</span>
        </h2>
        <p className="lead mb-4 mx-auto" style={{ maxWidth: "42rem" }}>
          {t("joinMesh.lead")}
        </p>

        <div className="d-flex flex-wrap justify-content-center gap-3 mb-4">
          <a
            className="hx-btn-ghost"
            href="https://github.com/helexa-ai"
            target="_blank"
            rel="noreferrer"
          >
            <FaGithub />
            {t("joinMesh.ctaExploreCode")}
          </a>
          <a
            className="hx-btn-ghost"
            href="https://x.com/helexaai"
            target="_blank"
            rel="noreferrer"
          >
            {t("joinMesh.ctaJoinAnnouncements")}
            <FaArrowUpRightFromSquare size={13} />
          </a>
          <span className="hx-soon-pill">{t("joinMesh.ctaRunNode")}</span>
        </div>

        <p className="text-muted small mb-0">{t("joinMesh.footer")}</p>
      </div>
    </section>
  );
};

export default Mission;
