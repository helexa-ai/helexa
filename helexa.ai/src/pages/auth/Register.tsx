import { useState } from "react";
import { Link } from "react-router-dom";
import { Alert, Button, Container, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/context";
import { ApiError } from "../../api/types";

export default function Register() {
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

  return (
    <Container className="py-5 flex-grow-1" style={{ maxWidth: 420 }}>
      <h1 className="h3 mb-4">{t("register.title")}</h1>
      {done ? (
        <Alert variant="success">{t("register.checkEmail")}</Alert>
      ) : (
        <>
          {error && <Alert variant="warning">{error}</Alert>}
          <Form onSubmit={submit}>
            <Form.Group className="mb-3">
              <Form.Label>{t("register.email")}</Form.Label>
              <Form.Control
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </Form.Group>
            <Form.Group className="mb-3">
              <Form.Label>{t("register.password")}</Form.Label>
              <Form.Control
                type="password"
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </Form.Group>
            <Button type="submit" disabled={busy} className="w-100">
              {t("register.submit")}
            </Button>
          </Form>
          <p className="mt-3 small">
            <Link to="/login">{t("register.haveAccount")}</Link>
          </p>
        </>
      )}
    </Container>
  );
}
