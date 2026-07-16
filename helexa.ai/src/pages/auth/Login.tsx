import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Alert, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/context";
import { ApiError } from "../../api/types";
import AuthCard from "../../components/AuthCard";

export default function Login() {
  const { t } = useTranslation("account");
  const { login } = useAuth();
  const nav = useNavigate();
  const [params] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      nav(params.get("next") || "/account", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("error.generic"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthCard title={t("login.title")}>
      {error && <Alert variant="warning">{error}</Alert>}
      <Form onSubmit={submit}>
        <Form.Group className="mb-3">
          <Form.Label>{t("login.email")}</Form.Label>
          <Form.Control
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group className="mb-4">
          <Form.Label>{t("login.password")}</Form.Label>
          <Form.Control
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </Form.Group>
        <button type="submit" disabled={busy} className="hx-btn-primary w-100">
          {t("login.submit")}
        </button>
      </Form>
      <p className="mt-4 small mb-0">
        <Link to="/register">{t("login.noAccount")}</Link>
      </p>
    </AuthCard>
  );
}
