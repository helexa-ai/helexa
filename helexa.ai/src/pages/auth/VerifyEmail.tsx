import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Alert, Container, Spinner } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { accountApi } from "../../api/account";

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
    <Container className="py-5 flex-grow-1" style={{ maxWidth: 480 }}>
      {state === "verifying" && (
        <p>
          <Spinner size="sm" className="me-2" />
          {t("verify.verifying")}
        </p>
      )}
      {state === "ok" && (
        <Alert variant="success">
          {t("verify.ok")} <Link to="/login">{t("verify.toLogin")}</Link>
        </Alert>
      )}
      {state === "failed" && (
        <Alert variant="warning">
          {t("verify.failed")} <Link to="/login">{t("verify.toLogin")}</Link>
        </Alert>
      )}
    </Container>
  );
}
