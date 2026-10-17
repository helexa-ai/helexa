-- ── Per-model rates for non-token units (#386) ───────────────────────
--
-- The ledger counts tokens. Work metered in another unit — audio-seconds
-- for music generation today — is converted into ledger tokens at a
-- per-model rate kept here, so an operator can price a model, or change
-- its price, without a deploy. cortex fetches the table from
-- `/authz/v1/rates` and reserves and settles at the rate it read.
--
-- `model_id = '*'` is the default for a unit: it applies to every model
-- without a row of its own. A unit with no row at all, for the model or
-- for '*', is not priced, and cortex refuses that work rather than serving
-- it free.
CREATE TABLE model_rates (
    model_id         TEXT NOT NULL,
    -- 'audio_second' — one second of generated audio.
    unit             TEXT NOT NULL,
    -- Ledger tokens charged per unit.
    tokens_per_unit  BIGINT NOT NULL,
    description      TEXT,
    -- Audit: who changed it and when (NULL updated_by = shipped default).
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by       UUID REFERENCES users(id) ON DELETE SET NULL,

    PRIMARY KEY (model_id, unit),
    CONSTRAINT model_rates_unit_known CHECK (unit IN ('audio_second')),
    CONSTRAINT model_rates_positive CHECK (tokens_per_unit > 0)
);

-- Placeholder default while music generation is being built; set real
-- per-model prices with an UPDATE or per-model INSERT.
INSERT INTO model_rates (model_id, unit, tokens_per_unit, description)
VALUES ('*', 'audio_second', 1000, 'default audio rate (placeholder)');
