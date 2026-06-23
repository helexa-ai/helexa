-- helexa-upstream initial schema (#59): accounts, keys, ledger, top-up
-- codes, served-usage. The mesh-level authority's system of record.
--
-- Token amounts are BIGINT (i64) throughout; the cortex EntitlementProvider
-- carries u64 but mesh allocations sit comfortably inside i64 and Postgres
-- has no unsigned type.

CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── Users (web auth: email + password) ──────────────────────────────
CREATE TABLE users (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email                    CITEXT NOT NULL UNIQUE,
    password_hash            TEXT NOT NULL,                 -- argon2id PHC string
    email_verified           BOOLEAN NOT NULL DEFAULT FALSE,
    -- Browser fingerprint captured at registration (#abuse). Best-effort,
    -- client-supplied; the primary signal for silent multi-account
    -- detection. NULL when the client could not produce one.
    registration_fingerprint TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX users_registration_fingerprint_idx
    ON users (registration_fingerprint)
    WHERE registration_fingerprint IS NOT NULL;

-- Single-use email tokens for verification and password reset. Only the
-- sha256 of the emailed secret is stored.
CREATE TABLE email_tokens (
    token_hash BYTEA PRIMARY KEY,
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL CHECK (kind IN ('verify', 'reset')),
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ
);
CREATE INDEX email_tokens_user_idx ON email_tokens (user_id);

-- ── Accounts (the billable allocation ledger) ───────────────────────
CREATE TABLE accounts (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    allocation_total   BIGINT NOT NULL DEFAULT 0,
    allocation_spent   BIGINT NOT NULL DEFAULT 0,
    allocation_reserved BIGINT NOT NULL DEFAULT 0,
    -- 'deactivated' is the SILENT abuse flag: keys stop authorizing but no
    -- surface ever tells the user why (see resolve → 401).
    status             TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'deactivated')),
    -- This account shares a registration fingerprint with >= 1 other.
    fingerprint_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- The no-overshoot backstop to the atomic reserve UPDATE.
    CONSTRAINT accounts_no_overshoot
        CHECK (allocation_spent + allocation_reserved <= allocation_total),
    CONSTRAINT accounts_nonneg
        CHECK (allocation_spent >= 0 AND allocation_reserved >= 0)
);
CREATE INDEX accounts_owner_idx ON accounts (owner_user_id);

-- ── API keys (Principal.key_id = api_keys.id) ───────────────────────
CREATE TABLE api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id  UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    key_hash    BYTEA NOT NULL,                 -- sha256(raw key)
    key_prefix  TEXT NOT NULL,                  -- non-secret display prefix
    label       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'archived')),
    -- Per-key sub-cap: 'hardcap' = absolute tokens; 'percent' = % of the
    -- account's allocation_total (resolved to an absolute at reserve time).
    limit_kind  TEXT NOT NULL DEFAULT 'percent'
                CHECK (limit_kind IN ('percent', 'hardcap')),
    limit_value BIGINT NOT NULL DEFAULT 100,
    -- serde of cortex_core::entitlements::CapWindow (Balance | Rolling).
    cap_window  JSONB NOT NULL DEFAULT '{"kind":"balance"}'::jsonb,
    -- Per-key running ledger (mirrors the account ledger; Balance semantics
    -- in this migration — rolling-window reset lands with the authz API).
    key_spent    BIGINT NOT NULL DEFAULT 0,
    key_reserved BIGINT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT api_keys_key_nonneg
        CHECK (key_spent >= 0 AND key_reserved >= 0)
);
-- A raw key resolves only while active; the hash is unique among active keys.
CREATE UNIQUE INDEX api_keys_active_hash_idx
    ON api_keys (key_hash) WHERE status = 'active';
CREATE INDEX api_keys_account_idx ON api_keys (account_id);

-- ── Reservations (reserve → settle/release) ─────────────────────────
-- id is BIGSERIAL so it maps to the cortex Reservation.id (u64) verbatim,
-- with the Postgres sequence as the sole global authority.
CREATE TABLE reservations (
    id          BIGSERIAL PRIMARY KEY,
    account_id  UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    key_id      UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    reserved    BIGINT NOT NULL,
    actual      BIGINT,
    state       TEXT NOT NULL DEFAULT 'open'
                CHECK (state IN ('open', 'settled', 'released')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    settled_at  TIMESTAMPTZ
);
-- The sweeper scans open reservations by age.
CREATE INDEX reservations_open_idx
    ON reservations (created_at) WHERE state = 'open';

-- ── Top-up codes (hybrid allocation) ────────────────────────────────
CREATE TABLE top_up_codes (
    code_hash   BYTEA PRIMARY KEY,             -- sha256(raw code)
    value       BIGINT NOT NULL,              -- tokens this code grants
    denomination TEXT,                         -- human label (e.g. "small")
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    redeemed_by UUID REFERENCES accounts(id) ON DELETE SET NULL,
    redeemed_at TIMESTAMPTZ
);

-- ── Served-usage ledger (#58 reconciliation) ────────────────────────
-- Absolute per-(operator, account, key, period) served tokens, upserted by
-- each cortex; reconciliation rolls these up for operator compensation.
CREATE TABLE served_usage (
    operator_id   TEXT NOT NULL,
    account_id    UUID NOT NULL,
    key_id        UUID NOT NULL,
    period        DATE NOT NULL,
    served_tokens BIGINT NOT NULL DEFAULT 0,
    reconciled_at TIMESTAMPTZ,
    PRIMARY KEY (operator_id, account_id, key_id, period)
);

-- ── Web sessions (DB-backed; alt/complement to stateless JWT) ───────
CREATE TABLE sessions (
    token_hash BYTEA PRIMARY KEY,             -- sha256(session token)
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX sessions_user_idx ON sessions (user_id);
