import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Alert, Spinner } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { accountApi } from "../../api/account";
import AuthCard from "../../components/AuthCard";

export default function VerifyEmail() {
  const { t } = useTranslation("account");
  const [params] = useSearchParams();
  const [state, setState] = useState<"verifying" | "ok" | "failed">("verifying");

  useEffect(() => {
    const token = params.get("token");
    // Keep all setState in async callbacks (no synchronous setState in the
    // effect body): a missing token resolves to a rejected promise.
    const run = token ? accountApi().verify(token) : Promise.reject(new Error("no token"));
    run.then(() => setState("ok")).catch(() => setState("failed"));
  }, [params]);

  return (
    <AuthCard title={t("verify.verifying")}>
      {state === "verifying" && (
        <p className="text-muted mb-0">
          <Spinner size="sm" className="me-2" />
          {t("verify.verifying")}
        </p>
      )}
      {state === "ok" && (
        <Alert variant="success" className="mb-0">
          {t("verify.ok")} <Link to="/login">{t("verify.toLogin")}</Link>
        </Alert>
      )}
      {state === "failed" && (
        <Alert variant="warning" className="mb-0">
          {t("verify.failed")} <Link to="/login">{t("verify.toLogin")}</Link>
        </Alert>
      )}
    </AuthCard>
  );
}
