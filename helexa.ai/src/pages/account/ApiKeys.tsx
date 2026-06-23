import { useCallback, useEffect, useState } from "react";
import { Alert, Badge, Button, Container, Form, Modal, Table } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/context";
import { accountApi } from "../../api/account";
import { db } from "../../data/db";
import { ApiError, type ApiKeySummary, type CreatedKey } from "../../api/types";

type LimitKind = "percent" | "hardcap";

export default function ApiKeys() {
  const { t } = useTranslation("account");
  const { token, logout } = useAuth();
  const [keys, setKeys] = useState<ApiKeySummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Create-key form state.
  const [label, setLabel] = useState("");
  const [limitKind, setLimitKind] = useState<LimitKind>("percent");
  const [limitValue, setLimitValue] = useState(100);
  const [created, setCreated] = useState<CreatedKey | null>(null);
  const [copied, setCopied] = useState(false);
  const [usedForChat, setUsedForChat] = useState(false);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setKeys(await accountApi().listKeys(token));
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

  async function create(e: React.FormEvent) {
    e.preventDefault();
    if (!token) return;
    setError(null);
    try {
      const key = await accountApi().createKey(token, label, limitKind, limitValue);
      setCreated(key);
      setLabel("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("error.generic"));
    }
  }

  async function archive(id: string) {
    if (!token) return;
    await accountApi().archiveKey(token, id);
    await load();
  }

  return (
    <Container className="py-5 flex-grow-1" style={{ maxWidth: 860 }}>
      <h1 className="h3 mb-4">{t("keys.title")}</h1>
      {error && <Alert variant="warning">{error}</Alert>}

      <Form onSubmit={create} className="surface-elevated p-3 rounded-3 mb-4">
        <div className="row g-2 align-items-end">
          <div className="col">
            <Form.Label className="small">{t("keys.label")}</Form.Label>
            <Form.Control value={label} onChange={(e) => setLabel(e.target.value)} />
          </div>
          <div className="col">
            <Form.Label className="small">{t("keys.limitKind")}</Form.Label>
            <Form.Select
              value={limitKind}
              onChange={(e) => setLimitKind(e.target.value as LimitKind)}
            >
              <option value="percent">{t("keys.percent")}</option>
              <option value="hardcap">{t("keys.hardcap")}</option>
            </Form.Select>
          </div>
          <div className="col">
            <Form.Label className="small">{t("keys.value")}</Form.Label>
            <Form.Control
              type="number"
              min={0}
              value={limitValue}
              onChange={(e) => setLimitValue(Number(e.target.value))}
            />
          </div>
          <div className="col-auto">
            <Button type="submit">{t("keys.create")}</Button>
          </div>
        </div>
      </Form>

      {keys.length === 0 ? (
        <p className="text-muted">{t("keys.none")}</p>
      ) : (
        <Table responsive hover>
          <thead>
            <tr>
              <th>{t("keys.label")}</th>
              <th>Prefix</th>
              <th>{t("keys.limitKind")}</th>
              <th>{t("keys.usage")}</th>
              <th>{t("keys.status")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k.id}>
                <td>{k.label || "—"}</td>
                <td>
                  <code>{k.prefix}…</code>
                </td>
                <td>
                  {k.limit_kind === "percent" ? `${k.limit_value}%` : k.limit_value.toLocaleString()}
                </td>
                <td>{k.spent.toLocaleString()}</td>
                <td>
                  <Badge bg={k.status === "active" ? "success" : "secondary"}>{k.status}</Badge>
                </td>
                <td className="text-end">
                  {k.status === "active" && (
                    <Button size="sm" variant="outline-danger" onClick={() => void archive(k.id)}>
                      {t("keys.archive")}
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}

      {/* The raw key is shown exactly once. */}
      <Modal show={!!created} onHide={() => setCreated(null)} centered>
        <Modal.Header closeButton>
          <Modal.Title className="h6">{t("keys.createdTitle")}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Alert variant="warning" className="py-2">{t("keys.createdWarn")}</Alert>
          <div className="d-flex gap-2">
            <Form.Control readOnly value={created?.key ?? ""} />
            <Button
              variant="outline-secondary"
              onClick={() => {
                if (created) void navigator.clipboard.writeText(created.key);
                setCopied(true);
              }}
            >
              {copied ? t("keys.copied") : t("keys.copy")}
            </Button>
          </div>
          {/* Store the raw key locally (this browser only) so the chat can
              use it as your bearer — consistent with no server-side secrets. */}
          <Button
            variant="link"
            className="px-0 mt-2"
            onClick={async () => {
              if (!created) return;
              await db.meta.put({ key: "chatApiKey", value: created.key });
              await db.meta.put({ key: "chatApiKeyId", value: created.id });
              setUsedForChat(true);
            }}
          >
            {usedForChat ? t("keys.usedForChat") : t("keys.useForChat")}
          </Button>
        </Modal.Body>
      </Modal>
    </Container>
  );
}
