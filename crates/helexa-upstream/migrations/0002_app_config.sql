-- ── Application config (operator-tunable, admin-UI backed) ──────────
--
-- Settings that an operator should be able to change without a deploy
-- live here rather than in helexa-upstream.toml: the toml is for how the
-- process runs (listen address, database, SMTP), this table is for how
-- the product behaves (grants, thresholds, caps).
--
-- The table is deliberately self-describing. Every row carries the type,
-- bounds and human labelling a form needs, so the planned admin UI can
-- render and validate an editor for settings that did not exist when it
-- was written — adding a setting is an INSERT, not a UI change.
--
-- `value` is JSONB so a setting can later be a list or object without a
-- schema migration; `value_type` tells the UI (and the typed accessors
-- in config_store.rs) how to read it.
CREATE TABLE app_config (
    key          TEXT PRIMARY KEY,
    value        JSONB NOT NULL,
    -- 'integer' | 'boolean' | 'string' — how to interpret `value`.
    value_type   TEXT NOT NULL,
    -- Grouping for the admin UI (e.g. 'topup', 'grant', 'abuse').
    category     TEXT NOT NULL,
    -- Human-facing label + help text, so the UI needs no per-key strings.
    label        TEXT NOT NULL,
    description  TEXT,
    -- Optional guard rails for numeric settings; the UI renders them as
    -- input bounds and the accessor clamps to them.
    min_value    BIGINT,
    max_value    BIGINT,
    -- Audit: who changed it and when (NULL updated_by = shipped default).
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by   UUID REFERENCES users(id) ON DELETE SET NULL,

    CONSTRAINT app_config_value_type_known
        CHECK (value_type IN ('integer', 'boolean', 'string')),
    CONSTRAINT app_config_bounds_ordered
        CHECK (min_value IS NULL OR max_value IS NULL OR min_value <= max_value)
);

CREATE INDEX app_config_category_idx ON app_config (category, key);

-- ── Where a top-up code came from ───────────────────────────────────
-- Auto-issued codes are capped and rate-limited per account, which needs
-- them distinguishable from codes an operator minted by hand.
ALTER TABLE top_up_codes
    ADD COLUMN source TEXT NOT NULL DEFAULT 'manual';

CREATE INDEX top_up_codes_auto_idx
    ON top_up_codes (redeemed_by, redeemed_at)
    WHERE source = 'auto';

-- ── Shipped defaults for self-service top-ups ───────────────────────
-- Chosen so somebody evaluating helexa can keep going without operator
-- involvement, while a runaway (or a bot farming free grants) is bounded:
-- at 3 auto top-ups of 1M tokens with a 24h cooldown, the worst case is
-- 3M tokens over three days per verified account.
INSERT INTO app_config (key, value, value_type, category, label, description, min_value, max_value) VALUES
    ('topup.auto.enabled', 'true'::jsonb, 'boolean', 'topup',
     'Self-service top-ups',
     'Let an account grant itself more allocation when it is running low, without operator involvement.',
     NULL, NULL),
    ('topup.auto.threshold_pct', '75'::jsonb, 'integer', 'topup',
     'Request threshold (%)',
     'How much of its allocation an account must have used before it may request a top-up.',
     1, 100),
    ('topup.auto.grant_tokens', '1000000'::jsonb, 'integer', 'topup',
     'Tokens per top-up',
     'How much allocation each self-service top-up adds.',
     0, 1000000000),
    ('topup.auto.max_per_account', '3'::jsonb, 'integer', 'topup',
     'Maximum per account',
     'How many self-service top-ups one account may ever receive. Beyond this the account needs an operator-minted code.',
     0, 1000),
    ('topup.auto.cooldown_secs', '86400'::jsonb, 'integer', 'topup',
     'Cooldown (seconds)',
     'Minimum gap between one account''s self-service top-ups.',
     0, 2592000);
