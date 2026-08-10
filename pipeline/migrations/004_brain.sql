-- Phase 3: the brain.
--
-- One table for both the teacher's judgements and the student's predictions,
-- separated by `source`. Keeping them together makes the comparison that
-- matters a single query: where does the student disagree with the teacher, and
-- is the disagreement systematic?
--
-- Storing predictions rather than computing them on demand is deliberate. These
-- feed node size, search ranking, and the agent's reasoning — all read paths
-- that must never touch a model.

CREATE TABLE IF NOT EXISTS repo_score (
    repo_id     BIGINT   NOT NULL REFERENCES repo(id) ON DELETE CASCADE,
    -- 0 = teacher (an LLM read the README), 1 = student (XGBoost predicted it).
    source      SMALLINT NOT NULL,

    maintenance          REAL,
    production_readiness REAL,
    specificity          REAL,
    learning_value       REAL,
    onboarding_ease      REAL,
    canonicity           REAL,

    -- One sentence saying what the software does. The teacher writes this for
    -- the sampled repositories; it is also the honest source for the UI's
    -- "brief", because the student can predict numbers but cannot write prose.
    summary     TEXT,
    flags       TEXT[] NOT NULL DEFAULT '{}',

    -- The content_hash the score was computed FROM. A README rewrite should
    -- invalidate the judgement of it; without this the brain slowly describes
    -- a corpus that no longer exists.
    scored_hash TEXT,
    model       TEXT,
    scored_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (repo_id, source)
);

CREATE INDEX IF NOT EXISTS repo_score_source_idx ON repo_score (source);
-- The product's central query: "good things in this niche", not "popular ones".
CREATE INDEX IF NOT EXISTS repo_score_canonicity_idx
    ON repo_score (canonicity DESC NULLS LAST) WHERE source = 1;
CREATE INDEX IF NOT EXISTS repo_score_production_idx
    ON repo_score (production_readiness DESC NULLS LAST) WHERE source = 1;

-- What produced a set of student scores. Without it, "why did every score
-- change" after a retrain is unanswerable — the feature set, the sample, and
-- the model version all move independently.
CREATE TABLE IF NOT EXISTS brain_run (
    id            BIGSERIAL PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    teacher_model TEXT,
    teacher_n     INTEGER,
    student_n     INTEGER,
    feature_names TEXT[],
    -- Held-out Spearman per dimension, and the popularity-leakage check.
    metrics       JSONB,
    notes         TEXT
);
