import React from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FaGithub } from "react-icons/fa6";

/**
 * Footer
 *
 * Slim theme-aware bar: copyright on one side, project links (GitHub)
 * on the other. Brand/service names stay untranslated.
 */
const Footer: React.FC = () => {
  const year = new Date().getFullYear();
  const { t } = useTranslation("common");

  return (
    <footer className="app-footer border-top py-3 mt-auto">
      <div className="container-fluid d-flex align-items-center justify-content-between flex-wrap gap-2 text-muted small px-4">
        <span>{t("footer.copyright", { year })}</span>
        <span className="d-inline-flex align-items-center gap-4">
          <Link to="/privacy">{t("footer.privacy")}</Link>
          <a
            href="https://github.com/helexa-ai"
            target="_blank"
            rel="noreferrer"
            className="d-inline-flex align-items-center gap-2"
          >
            <FaGithub size={15} />
            GitHub
          </a>
        </span>
      </div>
    </footer>
  );
};

export default Footer;
