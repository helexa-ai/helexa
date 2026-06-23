import { useState } from "react";
import { useTranslation } from "react-i18next";

/**
 * Slim public-beta notice shown above the header when VITE_PUBLIC_BETA is
 * set. Dismissible for the session (sessionStorage) so it doesn't nag, but
 * returns on the next visit while the beta lasts.
 */
const SHOWN = import.meta.env.VITE_PUBLIC_BETA === "true";
const DISMISS_KEY = "helexa.betaDismissed";

export default function BetaBanner() {
  const { t } = useTranslation("common");
  const [hidden, setHidden] = useState(
    () => sessionStorage.getItem(DISMISS_KEY) === "1",
  );
  if (!SHOWN || hidden) return null;
  return (
    <div className="beta-banner d-flex align-items-center justify-content-center gap-2 px-3 py-1 small">
      <span>
        <strong>{t("beta.tag")}</strong> {t("beta.message")}
      </span>
      <button
        type="button"
        className="btn-close btn-close-white ms-2"
        style={{ fontSize: "0.6rem" }}
        aria-label={t("beta.dismiss")}
        onClick={() => {
          sessionStorage.setItem(DISMISS_KEY, "1");
          setHidden(true);
        }}
      />
    </div>
  );
}
