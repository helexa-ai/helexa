-- helexa-angels schema — the investor portal behind angels.helexa.ai.
--
-- These tables live in a dedicated `angels` Postgres SCHEMA, not in
-- `public`, for one hard reason: helexa-angels and helexa-upstream share a
-- database, and both run `sqlx::migrate!`. sqlx writes its bookkeeping to
-- an unqualified `_sqlx_migrations`, so two migrators in one schema would
-- fight over the same table and mutually corrupt each other's version
-- history. Giving angels its own schema gives it its own migration table.
--
-- The connection runs with `search_path = angels, public`, so unqualified
-- names below land in `angels` while `public.users` remains reachable.
-- References to upstream-owned tables are written schema-qualified so the
-- dependency direction is impossible to misread: angels reads `users`; it
-- never alters anything upstream owns.

-- ── Rounds ──────────────────────────────────────────────────────────
-- A round carries its own framing (D6): "Early Access Programme" for the
-- Tenstorrent round, something else next time. Nothing in the schema
-- assumes equity, shares, or priced packages.
CREATE TABLE rounds (
    slug             TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    -- Displayed instead of the word "round" wherever the UI names it.
    framing_label    TEXT NOT NULL DEFAULT 'Early Access Programme',
    status           TEXT NOT NULL DEFAULT 'draft'
                     CHECK (status IN ('draft', 'open', 'closed')),
    -- TRUE: a valid invite grants access immediately. FALSE: the invite
    -- creates a `pending` grant an operator must approve first. Per-round
    -- because a later round carrying real financials may want the gate
    -- even though this one does not.
    auto_grant       BOOLEAN NOT NULL DEFAULT TRUE,
    -- Git SHA (or any opaque tag) of the content tree this round was last
    -- rendered from; stamped into every access_log row so "which version
    -- did they see?" is answerable.
    content_version  TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    opened_at        TIMESTAMPTZ,
    closed_at        TIMESTAMPTZ
);

-- ── Invites ─────────────────────────────────────────────────────────
-- Reusable by design (D5). The code is a DISTRIBUTION mechanism, not a
-- security boundary: it will be forwarded. Confidentiality rests on the
-- fact that redeeming it produces a NAMED account and every subsequent
-- document view is attributed. Only the hash is stored — the plaintext
-- exists solely in the link the operator sends.
CREATE TABLE invites (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code_hash   BYTEA NOT NULL UNIQUE,
    -- Non-secret display tag so the operator can tell codes apart in a
    -- listing without holding the plaintext.
    label       TEXT NOT NULL,
    round_slug  TEXT NOT NULL REFERENCES rounds(slug) ON DELETE CASCADE,
    -- NULL = unlimited uses. Reusability is the point; the cap is a
    -- blast-radius control for a code that escapes further than intended.
    max_uses    INTEGER,
    used_count  INTEGER NOT NULL DEFAULT 0,
    expires_at  TIMESTAMPTZ,
    revoked_at  TIMESTAMPTZ,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT invites_uses_sane CHECK (max_uses IS NULL OR max_uses > 0)
);
CREATE INDEX invites_round_idx ON invites (round_slug);

-- ── Grants ──────────────────────────────────────────────────────────
-- Access attaches to the USER, not the code. This is what makes the model
-- work: revoking an invite stops new grants, revoking a grant cuts off one
-- named person — something an unguessable content URL can never do.
CREATE TABLE grants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    round_slug  TEXT NOT NULL REFERENCES rounds(slug) ON DELETE CASCADE,
    invite_id   UUID REFERENCES invites(id) ON DELETE SET NULL,
    state       TEXT NOT NULL DEFAULT 'active'
                CHECK (state IN ('pending', 'active', 'revoked')),
    granted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_by TEXT,
    revoked_at  TIMESTAMPTZ,
    UNIQUE (user_id, round_slug)
);
CREATE INDEX grants_user_idx ON grants (user_id);
CREATE INDEX grants_round_idx ON grants (round_slug);

-- ── Sessions ────────────────────────────────────────────────────────
-- Deliberately NOT public.sessions. Credentials are shared with helexa.ai
-- (D2) but session realms are not: the public SPA keeps its token in
-- localStorage and renders markdown, runs a chat loop and fetches remote
-- pages. Honouring those tokens here would put confidential documents one
-- script injection away. This cookie is HttpOnly and host-only.
CREATE TABLE sessions (
    token_hash   BYTEA PRIMARY KEY,
    user_id      UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    issued_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    ip           TEXT,
    user_agent   TEXT
);
CREATE INDEX sessions_user_idx ON sessions (user_id);
CREATE INDEX sessions_expiry_idx ON sessions (expires_at);

-- ── Access log ──────────────────────────────────────────────────────
-- The answer to D3, and the reason this service is server-rendered rather
-- than an SPA against an API: one document render is one server request,
-- so this table records what was actually read. A client-side app could
-- fetch every document once and re-render offline, and the log would show
-- a single view.
--
-- Holds identifiable data about (probably) EU persons — see the retention
-- sweep in A5 and the portal privacy note.
CREATE TABLE access_log (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES public.users(id) ON DELETE SET NULL,
    -- Denormalised so the trail survives account deletion: who read what
    -- is a record we must keep even once the account is gone.
    user_email      TEXT,
    round_slug      TEXT,
    document_slug   TEXT,
    content_version TEXT,
    kind            TEXT NOT NULL DEFAULT 'view'
                    CHECK (kind IN ('view', 'download', 'export', 'denied')),
    viewed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    ip              TEXT,
    user_agent      TEXT
);
CREATE INDEX access_log_user_idx ON access_log (user_id);
CREATE INDEX access_log_round_idx ON access_log (round_slug, viewed_at DESC);
CREATE INDEX access_log_time_idx ON access_log (viewed_at DESC);

-- ── Expressions of interest ─────────────────────────────────────────
-- No payments here: taking six figures through a web form is a different
-- project with its own compliance surface. This records intent and routes
-- it to a human.
--
-- All three commercial axes are the INVESTOR's decision (operator, 2026-08-03):
-- who purchases (investor direct from Tenstorrent, or Bears Lairs EOOD on
-- their behalf), who hosts, and who covers maintenance and running costs.
-- Contracts are bespoke per investor to reflect the combination chosen,
-- so these columns capture a starting position, not a fixed product.
CREATE TABLE interest (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    round_slug      TEXT NOT NULL REFERENCES rounds(slug) ON DELETE CASCADE,
    package_ref     TEXT,
    purchaser       TEXT,
    hosting_choice  TEXT,
    running_costs   TEXT,
    message         TEXT,
    state           TEXT NOT NULL DEFAULT 'new'
                    CHECK (state IN ('new', 'acknowledged', 'closed')),
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX interest_round_idx ON interest (round_slug, submitted_at DESC);
