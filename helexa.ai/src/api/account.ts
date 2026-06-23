// Account API client over helexa-upstream's /web/v1 (B4/B5). The browser
// calls a same-origin `/api` prefix (vite-proxied in dev, nginx-routed in
// prod). A MockAccountApi behind VITE_USE_MOCK_ACCOUNT_API lets the
// dashboard be built/demoed before the upstream service is reachable.

import {
  ApiError,
  type AccountBalance,
  type ApiKeySummary,
  type CreatedKey,
  type Session,
} from "./types";

export interface AccountApi {
  register(email: string, password: string, fingerprint?: string): Promise<void>;
  verify(token: string): Promise<void>;
  login(email: string, password: string): Promise<Session>;
  requestReset(email: string): Promise<void>;
  confirmReset(token: string, newPassword: string): Promise<void>;
  account(token: string): Promise<AccountBalance>;
  listKeys(token: string): Promise<ApiKeySummary[]>;
  createKey(
    token: string,
    label: string,
    limitKind: "percent" | "hardcap",
    limitValue: number,
  ): Promise<CreatedKey>;
  archiveKey(token: string, id: string): Promise<void>;
  updateKeyLimit(
    token: string,
    id: string,
    limitKind: "percent" | "hardcap",
    limitValue: number,
  ): Promise<void>;
  redeem(token: string, code: string): Promise<AccountBalance>;
}

const BASE = (import.meta.env.VITE_ACCOUNT_BASE_URL || "/api").replace(/\/$/, "");

async function call<T>(
  path: string,
  init: RequestInit & { token?: string } = {},
): Promise<T> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (init.token) headers.authorization = `Bearer ${init.token}`;
  let resp: Response;
  try {
    resp = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the account service.");
  }
  if (resp.status === 204) return undefined as T;
  let body: unknown = null;
  try {
    body = await resp.json();
  } catch {
    /* empty body */
  }
  if (!resp.ok) {
    const err = (body as { error?: { code?: string; message?: string } })?.error;
    throw new ApiError(resp.status, err?.code ?? "error", err?.message ?? "Request failed.");
  }
  return body as T;
}

class RealAccountApi implements AccountApi {
  async register(email: string, password: string, fingerprint?: string) {
    await call("/register", {
      method: "POST",
      body: JSON.stringify({ email, password, fingerprint }),
    });
  }
  async verify(token: string) {
    await call("/verify", { method: "POST", body: JSON.stringify({ token }) });
  }
  login(email: string, password: string) {
    return call<Session>("/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }
  async requestReset(email: string) {
    await call("/password-reset/request", {
      method: "POST",
      body: JSON.stringify({ email }),
    });
  }
  async confirmReset(token: string, newPassword: string) {
    await call("/password-reset/confirm", {
      method: "POST",
      body: JSON.stringify({ token, new_password: newPassword }),
    });
  }
  account(token: string) {
    return call<AccountBalance>("/account", { token });
  }
  listKeys(token: string) {
    return call<{ keys: ApiKeySummary[] }>("/keys", { token }).then((r) => r.keys);
  }
  createKey(token: string, label: string, limit_kind: "percent" | "hardcap", limit_value: number) {
    return call<CreatedKey>("/keys", {
      method: "POST",
      token,
      body: JSON.stringify({ label, limit_kind, limit_value }),
    });
  }
  async archiveKey(token: string, id: string) {
    await call(`/keys/${id}/archive`, { method: "POST", token, body: "{}" });
  }
  async updateKeyLimit(
    token: string,
    id: string,
    limit_kind: "percent" | "hardcap",
    limit_value: number,
  ) {
    await call(`/keys/${id}/limit`, {
      method: "PATCH",
      token,
      body: JSON.stringify({ limit_kind, limit_value }),
    });
  }
  redeem(token: string, code: string) {
    return call<AccountBalance>("/redeem", {
      method: "POST",
      token,
      body: JSON.stringify({ code }),
    });
  }
}

// ── Mock (VITE_USE_MOCK_ACCOUNT_API) ────────────────────────────────
// Minimal in-memory account so the dashboard is fully developable offline.

class MockAccountApi implements AccountApi {
  private total = 1_000_000;
  private spent = 0;
  private reserved = 0;
  private keys: ApiKeySummary[] = [];
  private seq = 1;

  async register() {}
  async verify() {}
  async login(): Promise<Session> {
    return { token: "mock-token", expires_in: 604800 };
  }
  async requestReset() {}
  async confirmReset() {}
  async account(): Promise<AccountBalance> {
    return {
      account_id: "mock-account",
      allocation_total: this.total,
      allocation_spent: this.spent,
      allocation_reserved: this.reserved,
    };
  }
  async listKeys(): Promise<ApiKeySummary[]> {
    return [...this.keys];
  }
  async createKey(
    _t: string,
    label: string,
    limit_kind: "percent" | "hardcap",
    limit_value: number,
  ): Promise<CreatedKey> {
    const id = `mock-${this.seq++}`;
    const prefix = `sk-helexa-mock${this.seq}`;
    this.keys.push({
      id,
      prefix,
      label,
      status: "active",
      limit_kind,
      limit_value,
      spent: 0,
      reserved: 0,
      created_at: new Date().toISOString(),
    });
    return { id, key: `${prefix}-RAWSECRETSHOWNONCE`, prefix, limit_kind, limit_value };
  }
  async archiveKey(_t: string, id: string) {
    const k = this.keys.find((x) => x.id === id);
    if (k) k.status = "archived";
  }
  async updateKeyLimit(
    _t: string,
    id: string,
    limit_kind: "percent" | "hardcap",
    limit_value: number,
  ) {
    const k = this.keys.find((x) => x.id === id);
    if (k) {
      k.limit_kind = limit_kind;
      k.limit_value = limit_value;
    }
  }
  async redeem(_t: string, code: string): Promise<AccountBalance> {
    if (!code.startsWith("helexa-topup-")) {
      throw new ApiError(400, "bad_request", "invalid or already-redeemed code");
    }
    this.total += 500_000;
    return this.account();
  }
}

let instance: AccountApi | null = null;
export function accountApi(): AccountApi {
  if (!instance) {
    instance = import.meta.env.VITE_USE_MOCK_ACCOUNT_API
      ? new MockAccountApi()
      : new RealAccountApi();
  }
  return instance;
}
