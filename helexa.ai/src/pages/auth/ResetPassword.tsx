import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Alert, Button, Container, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { accountApi } from "../../api/account";
import { ApiError } from "../../api/types";

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
    <Container className="py-5 flex-grow-1" style={{ maxWidth: 420 }}>
      <h1 className="h3 mb-4">{t("reset.confirmTitle")}</h1>
      {done ? (
        <Alert variant="success">
          {t("reset.ok")} <Link to="/login">{t("verify.toLogin")}</Link>
        </Alert>
      ) : (
        <>
          {error && <Alert variant="warning">{error}</Alert>}
          <Form onSubmit={submit}>
            <Form.Group className="mb-3">
              <Form.Label>{t("reset.newPassword")}</Form.Label>
              <Form.Control
                type="password"
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </Form.Group>
            <Button type="submit" disabled={busy} className="w-100">
              {t("reset.confirmSubmit")}
            </Button>
          </Form>
        </>
      )}
    </Container>
  );
}
