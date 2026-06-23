import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Alert, Button, Card, Container, Form, ProgressBar } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/context";
import { accountApi } from "../../api/account";
import { ApiError, type AccountBalance } from "../../api/types";

export default function Dashboard() {
  const { t } = useTranslation("account");
  const { token, logout } = useAuth();
  const [balance, setBalance] = useState<AccountBalance | null>(null);
  const [code, setCode] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setBalance(await accountApi().account(token));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) logout();
      else setError(t("error.generic"));
    }
  }, [token, logout, t]);

  useEffect(() => {
    // load() is async; setState happens after await, not synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  async function redeem(e: React.FormEvent) {
    e.preventDefault();
    if (!token) return;
    setError(null);
    setMsg(null);
    try {
      setBalance(await accountApi().redeem(token, code.trim()));
      setCode("");
      setMsg(t("dashboard.redeemed"));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("error.generic"));
    }
  }

  const remaining = balance
    ? balance.allocation_total - balance.allocation_spent - balance.allocation_reserved
    : 0;
  const pct = balance && balance.allocation_total > 0
    ? Math.round(((balance.allocation_spent + balance.allocation_reserved) / balance.allocation_total) * 100)
    : 0;

  return (
    <Container className="py-5 flex-grow-1" style={{ maxWidth: 720 }}>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1 className="h3 mb-0">{t("dashboard.title")}</h1>
        <Button variant="outline-secondary" size="sm" onClick={logout}>
          {t("dashboard.logout")}
        </Button>
      </div>

      <Card className="surface-elevated mb-4">
        <Card.Body>
          <Card.Title className="h6 text-uppercase text-muted">
            {t("dashboard.balance")}
          </Card.Title>
          {balance && (
            <>
              <ProgressBar now={pct} className="my-3" />
              <div className="d-flex justify-content-between small">
                <span>{t("dashboard.total")}: {balance.allocation_total.toLocaleString()}</span>
                <span>{t("dashboard.spent")}: {balance.allocation_spent.toLocaleString()}</span>
                <span>{t("dashboard.reserved")}: {balance.allocation_reserved.toLocaleString()}</span>
                <span>{t("dashboard.remaining")}: {remaining.toLocaleString()}</span>
              </div>
            </>
          )}
          <Link to="/account/keys" className="btn btn-primary btn-sm mt-3">
            {t("dashboard.manageKeys")}
          </Link>
        </Card.Body>
      </Card>

      <Card className="surface-elevated">
        <Card.Body>
          <Card.Title className="h6">{t("dashboard.redeemTitle")}</Card.Title>
          {msg && <Alert variant="success" className="py-2">{msg}</Alert>}
          {error && <Alert variant="warning" className="py-2">{error}</Alert>}
          <Form onSubmit={redeem} className="d-flex gap-2">
            <Form.Control
              value={code}
              placeholder={t("dashboard.redeemPlaceholder")}
              onChange={(e) => setCode(e.target.value)}
            />
            <Button type="submit" disabled={!code.trim()}>
              {t("dashboard.redeem")}
            </Button>
          </Form>
        </Card.Body>
      </Card>
    </Container>
  );
}
