import { useState } from "react";
import { Alert, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { accountApi } from "../../api/account";
import AuthCard from "../../components/AuthCard";

export default function RequestReset() {
  const { t } = useTranslation("account");
  const [email, setEmail] = useState("");
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    // Always succeeds from the UI's view (no account enumeration).
    try {
      await accountApi().requestReset(email);
    } catch {
      /* swallow */
    }
    setDone(true);
    setBusy(false);
  }

  return (
    <AuthCard title={t("reset.requestTitle")}>
      {done ? (
        <Alert variant="info">{t("reset.requestDone")}</Alert>
      ) : (
        <Form onSubmit={submit}>
          <Form.Group className="mb-4">
            <Form.Label>{t("reset.email")}</Form.Label>
            <Form.Control
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </Form.Group>
          <button
            type="submit"
            disabled={busy}
            className="hx-btn-primary w-100"
          >
            {t("reset.requestSubmit")}
          </button>
        </Form>
      )}
    </AuthCard>
  );
}
