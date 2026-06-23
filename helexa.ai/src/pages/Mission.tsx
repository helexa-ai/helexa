import React from "react";
import { Row, Col, Button } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import {
  FaArrowRight,
  FaArrowLeft,
  FaGithub,
  FaArrowUpRightFromSquare,
  FaServer,
  FaNetworkWired,
  FaCoins,
  FaCircleNodes,
  FaPeopleGroup,
  FaScaleBalanced,
  FaInfinity,
} from "react-icons/fa6";
import DirectionalIcon from "../components/DirectionalIcon";

import "../App.css";
import "../index.css";

/**
 * Mission
 *
 * Narrative homepage implementing the conceptual structure from ideas.md:
 * hero → intent → why now → how it works → principles → road ahead → CTA
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

const HeroSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="mb-5">
      <Row className="align-items-center g-4">
        <Col lg={6}>
          <div className="mb-3">
            <span className="badge-accent d-inline-flex align-items-center gap-2">
              <span className="rounded-circle bg-cyan-500-dot" />
              {t("hero.badge")}
            </span>
          </div>
          <h1 className="mb-3" style={{ letterSpacing: "0.04em" }}>
            {t("hero.title")}
          </h1>
          <p className="lead mb-4">{t("hero.lead")}</p>

          <div className="d-flex flex-wrap gap-3 mb-4">
            <Button
              size="lg"
              variant="primary"
              className="d-inline-flex align-items-center gap-2"
              href="https://x.com/helexaai"
              target="_blank"
              rel="noreferrer"
            >
              {t("hero.ctaJoinMesh")}
              <DirectionalIcon
                direction="forward"
                ltrIcon={FaArrowRight}
                rtlIcon={FaArrowLeft}
              />
            </Button>
            <Button
              size="lg"
              variant="outline-secondary"
              className="d-inline-flex align-items-center gap-2"
              href="https://github.com/helexa-ai"
              target="_blank"
              rel="noreferrer"
            >
              <FaGithub />
              {t("hero.ctaFollowProject")}
            </Button>
          </div>

          <p className="text-muted small mb-0">{t("hero.subcopy")}</p>
        </Col>

        <Col lg={6}>
          <div className="hero-visual-wrapper position-relative">
            <div className="hero-banner surface-elevated overflow-hidden">
              {/* Use banner.png (dark #000618 background) as hero visual */}
              <img
                src="/banner.png"
                alt={t("hero.imageAlt")}
                className="img-fluid w-100"
                style={{
                  display: "block",
                  objectFit: "cover",
                  borderRadius: "1rem",
                }}
              />
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
    <section className="mb-5">
      <Row className="g-4">
        <Col lg={5}>
          <h2 className="mb-3">{t("intent.title")}</h2>
        </Col>
        <Col lg={7}>
          <p className="mb-3">{t("intent.p1")}</p>
          <p className="mb-3">{t("intent.p2Intro")}</p>
          <ul className="mb-3">
            <li>{t("intent.bullet1")}</li>
            <li>{t("intent.bullet2")}</li>
            <li>{t("intent.bullet3")}</li>
            <li>{t("intent.bullet4")}</li>
          </ul>
          <p className="mb-0 fw-semibold">{t("intent.closing")}</p>
        </Col>
      </Row>
    </section>
  );
};

const WhyNowSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="mb-5">
      <Row className="g-4">
        <Col lg={12}>
          <h2 className="mb-3">{t("whyNow.title")}</h2>
        </Col>
      </Row>
      <Row className="g-4">
        <Col lg={6}>
          <div className="surface-elevated p-4 h-100">
            <h3 className="h5 mb-3">{t("whyNow.problemTitle")}</h3>
            <ul className="mb-0">
              <li>{t("whyNow.problemBullet1")}</li>
              <li>{t("whyNow.problemBullet2")}</li>
              <li>{t("whyNow.problemBullet3")}</li>
              <li>{t("whyNow.problemBullet4")}</li>
              <li>{t("whyNow.problemBullet5")}</li>
            </ul>
          </div>
        </Col>
        <Col lg={6}>
          <div className="surface-elevated p-4 h-100">
            <h3 className="h5 mb-3">{t("whyNow.opportunityTitle")}</h3>
            <p className="mb-3">{t("whyNow.opportunityIntro")}</p>
            <ul className="mb-3">
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

  return (
    <section className="mb-5">
      <h2 className="mb-3">{t("howItWorks.title")}</h2>
      <Row className="g-4">
        <Col md={4}>
          <div className="surface-elevated p-4 h-100">
            <div className="mesh-icon">
              <FaServer size={16} />
            </div>
            <h3 className="h6 text-uppercase text-muted mb-2">
              {t("howItWorks.operators.eyebrow")}
            </h3>
            <h4 className="h5 mb-3">{t("howItWorks.operators.title")}</h4>
            <p className="mb-0">{t("howItWorks.operators.body")}</p>
          </div>
        </Col>
        <Col md={4}>
          <div className="surface-elevated p-4 h-100">
            <div className="mesh-icon">
              <FaNetworkWired size={16} />
            </div>
            <h3 className="h6 text-uppercase text-muted mb-2">
              {t("howItWorks.routing.eyebrow")}
            </h3>
            <h4 className="h5 mb-3">{t("howItWorks.routing.title")}</h4>
            <p className="mb-0">{t("howItWorks.routing.body")}</p>
          </div>
        </Col>
        <Col md={4}>
          <div className="surface-elevated p-4 h-100">
            <div className="mesh-icon">
              <FaCoins size={16} />
            </div>
            <h3 className="h6 text-uppercase text-muted mb-2">
              {t("howItWorks.value.eyebrow")}
            </h3>
            <h4 className="h5 mb-3">{t("howItWorks.value.title")}</h4>
            <p className="mb-0">{t("howItWorks.value.body")}</p>
          </div>
        </Col>
      </Row>
    </section>
  );
};

const PrinciplesSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="mb-5">
      <h2 className="mb-3">{t("principles.title")}</h2>
      <Row className="g-4">
        <Col md={6}>
          <div className="surface-elevated p-4 h-100 d-flex gap-3">
            <div className="principle-icon">
              <FaCircleNodes size={14} />
            </div>
            <div>
              <h3 className="h5 mb-2">{t("principles.distributed.title")}</h3>
              <p className="mb-0">{t("principles.distributed.body")}</p>
            </div>
          </div>
        </Col>
        <Col md={6}>
          <div className="surface-elevated p-4 h-100 d-flex gap-3">
            <div className="principle-icon">
              <FaPeopleGroup size={14} />
            </div>
            <div>
              <h3 className="h5 mb-2">{t("principles.participation.title")}</h3>
              <p className="mb-0">{t("principles.participation.body")}</p>
            </div>
          </div>
        </Col>
        <Col md={6}>
          <div className="surface-elevated p-4 h-100 d-flex gap-3">
            <div className="principle-icon">
              <FaScaleBalanced size={14} />
            </div>
            <div>
              <h3 className="h5 mb-2">{t("principles.fairness.title")}</h3>
              <p className="mb-0">{t("principles.fairness.body")}</p>
            </div>
          </div>
        </Col>
        <Col md={6}>
          <div className="surface-elevated p-4 h-100 d-flex gap-3">
            <div className="principle-icon">
              <FaInfinity size={14} />
            </div>
            <div>
              <h3 className="h5 mb-2">{t("principles.evolving.title")}</h3>
              <p className="mb-0">{t("principles.evolving.body")}</p>
            </div>
          </div>
        </Col>
      </Row>
    </section>
  );
};

const RoadAheadSection: React.FC = () => {
  const { t } = useTranslation("mission");

  return (
    <section className="mb-5">
      <Row className="g-4 align-items-center">
        <Col lg={6}>
          <h2 className="mb-3">{t("roadAhead.title")}</h2>
          <p className="mb-3">{t("roadAhead.p1")}</p>
          <p className="mb-3">{t("roadAhead.p2")}</p>
          <p className="mb-3">{t("roadAhead.p3")}</p>
          <p className="mb-0">{t("roadAhead.p4")}</p>
        </Col>
        <Col lg={6}>
          <div className="surface-elevated p-4 d-flex flex-column align-items-start gap-3">
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
    <section className="mb-5">
      <div className="surface-elevated p-4 p-md-5 text-center position-relative overflow-hidden join-mesh-cta">
        <div className="mb-3">
          <span className="badge-accent d-inline-flex align-items-center gap-2">
            <span className="rounded-circle bg-cyan-500-dot" />
            {t("joinMesh.badge")}
          </span>
        </div>
        <h2 className="mb-3">
          {t("joinMesh.title")}{" "}
          <span className="text-hot-pink">{t("joinMesh.titleHighlight")}</span>
        </h2>
        <p className="lead mb-4">{t("joinMesh.lead")}</p>

        <div className="d-flex flex-wrap justify-content-center gap-3 mb-4">
          <Button
            size="lg"
            variant="primary"
            className="d-inline-flex align-items-center gap-2"
            href="https://github.com/helexa-ai"
            target="_blank"
            rel="noreferrer"
          >
            {t("joinMesh.ctaRunNode")}
            <DirectionalIcon
              direction="forward"
              ltrIcon={FaArrowRight}
              rtlIcon={FaArrowLeft}
            />
          </Button>
          <Button
            size="lg"
            variant="outline-secondary"
            className="d-inline-flex align-items-center gap-2"
            href="https://x.com/helexaai"
            target="_blank"
            rel="noreferrer"
          >
            {t("joinMesh.ctaJoinAnnouncements")}
            <FaArrowUpRightFromSquare />
          </Button>
          <Button
            size="lg"
            variant="outline-secondary"
            className="d-inline-flex align-items-center gap-2"
            href="https://github.com/helexa-ai"
            target="_blank"
            rel="noreferrer"
          >
            <FaGithub />
            {t("joinMesh.ctaExploreCode")}
          </Button>
        </div>

        <p className="text-muted small mb-0">{t("joinMesh.footer")}</p>
      </div>
    </section>
  );
};

export default Mission;
