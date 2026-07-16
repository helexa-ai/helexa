import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Alert, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { accountApi } from "../../api/account";
import { ApiError } from "../../api/types";
import AuthCard from "../../components/AuthCard";

export default function ResetPassword() {
  const { t } = useTranslation("account");
  const [params] = useSearchParams();
  const [password, setPassword] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const token = params.get("token");
    if (!token) {
      setError(t("verify.failed"));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await accountApi().confirmReset(token, password);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("error.generic"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthCard title={t("reset.confirmTitle")}>
      {done ? (
        <Alert variant="success">
          {t("reset.ok")} <Link to="/login">{t("verify.toLogin")}</Link>
        </Alert>
      ) : (
        <>
          {error && <Alert variant="warning">{error}</Alert>}
          <Form onSubmit={submit}>
            <Form.Group className="mb-4">
              <Form.Label>{t("reset.newPassword")}</Form.Label>
              <Form.Control
                type="password"
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </Form.Group>
            <button
              type="submit"
              disabled={busy}
              className="hx-btn-primary w-100"
            >
              {t("reset.confirmSubmit")}
            </button>
          </Form>
        </>
      )}
    </AuthCard>
  );
}
