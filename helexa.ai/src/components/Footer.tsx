import React from "react";
import { useTranslation } from "react-i18next";

/**
 * Footer
 *
 * Simple application footer used in the main layout.
 * Renders a subtle, theme-aware bar with copyright text.
 */
const Footer: React.FC = () => {
  const year = new Date().getFullYear();
  const { t } = useTranslation("common");

  return (
    <footer className="app-footer border-top py-3 mt-auto">
      <div className="container-fluid text-center text-muted small">
        <span>{t("footer.copyright", { year })}</span>
      </div>
    </footer>
  );
};

export default Footer;
