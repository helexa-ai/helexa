import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Alert, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/context";
import { ApiError } from "../../api/types";

/**
 * The sign-in and sign-up form bodies, without any page chrome.
 *
 * Extracted so the unified `/auth` route can show both in tabs while the
 * legacy `/login` and `/register` routes keep working — one implementation,
 * not two that drift.
 *
 * Neither form renders its own "already have an account?" cross-link: on
 * the tabbed page the tabs are the switch, and duplicating it below the
 * button would be two controls for one job.
 */

export function SignInForm({ onDone }: { onDone?: () => void }) {
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
      onDone?.();
      nav(params.get("next") || "/account", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("error.generic"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      {error && <Alert variant="warning">{error}</Alert>}
      <Form onSubmit={submit}>
        <Form.Group className="mb-3">
          <Form.Label>{t("login.email")}</Form.Label>
          <Form.Control
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group className="mb-4">
          <Form.Label>{t("login.password")}</Form.Label>
          <Form.Control
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </Form.Group>
        <button type="submit" disabled={busy} className="hx-btn-primary w-100">
          {t("login.submit")}
        </button>
      </Form>
    </>
  );
}

export function SignUpForm() {
  const { t } = useTranslation("account");
  const { register } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await register(email, password);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("error.generic"));
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return <Alert variant="success">{t("register.checkEmail")}</Alert>;
  }

  return (
    <>
      {error && <Alert variant="warning">{error}</Alert>}
      <Form onSubmit={submit}>
        <Form.Group className="mb-3">
          <Form.Label>{t("register.email")}</Form.Label>
          <Form.Control
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group className="mb-4">
          <Form.Label>{t("register.password")}</Form.Label>
          <Form.Control
            type="password"
            autoComplete="new-password"
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </Form.Group>
        <button type="submit" disabled={busy} className="hx-btn-primary w-100">
          {t("register.submit")}
        </button>
      </Form>
    </>
  );
}
