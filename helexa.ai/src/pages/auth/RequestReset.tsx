import { useState } from "react";
import { Alert, Button, Container, Form } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { accountApi } from "../../api/account";

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
    <Container className="py-5 flex-grow-1" style={{ maxWidth: 420 }}>
      <h1 className="h3 mb-4">{t("reset.requestTitle")}</h1>
      {done ? (
        <Alert variant="info">{t("reset.requestDone")}</Alert>
      ) : (
        <Form onSubmit={submit}>
          <Form.Group className="mb-3">
            <Form.Label>{t("reset.email")}</Form.Label>
            <Form.Control
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </Form.Group>
          <Button type="submit" disabled={busy} className="w-100">
            {t("reset.requestSubmit")}
          </Button>
        </Form>
      )}
    </Container>
  );
}
