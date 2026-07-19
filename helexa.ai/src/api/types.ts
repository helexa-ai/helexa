// Wire types for the helexa-upstream /web/v1 account API (B4/B5).

export interface ApiKeySummary {
  id: string;
  prefix: string;
  label: string;
  status: "active" | "archived";
  limit_kind: "percent" | "hardcap";
  limit_value: number;
  spent: number;
  reserved: number;
  created_at: string;
}

export interface CreatedKey {
  id: string;
  /** Raw secret — shown exactly once at creation. */
  key: string;
  prefix: string;
  limit_kind: "percent" | "hardcap";
  limit_value: number;
}

export interface AccountBalance {
  account_id: string;
  allocation_total: number;
  allocation_spent: number;
  allocation_reserved: number;
}

export interface Session {
  token: string;
  expires_in: number;
}

/**
 * Product feature gates (#191), served unauthenticated from
 * GET /web/v1/features. Operators flip these in helexa-upstream.toml
 * to change product behaviour without a site rebuild.
 */
export interface FeatureFlags {
  /** Offer web grounding tools to anonymous chat sessions. */
  anon_web_search: boolean;
}

/** Typed error carrying the backend's machine-readable code. */
export class ApiError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.code = code;
    this.status = status;
  }
}
