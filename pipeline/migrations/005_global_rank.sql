-- Rank against all of GitHub, not just the corpus.
--
-- PageRank answers "how important is this among the repositories we happen to
-- have embedded". That is the wrong question for a user, and it changes meaning
-- every time the corpus grows — which it now does continuously.

-- The measured survival function: how many public repositories have at least N
-- stars, sampled from the GitHub search API at a point in time.
--
-- A table rather than a constant because it is a MEASUREMENT, and every score
-- depends on it. Without the run recorded, "why did this repository's score
-- change" has two indistinguishable answers — the repository moved, or the
-- yardstick did. GitHub adds millions of repositories a year, so it does move.
CREATE TABLE IF NOT EXISTS star_scale_run (
    id           BIGSERIAL PRIMARY KEY,
    measured_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    thresholds   INTEGER[] NOT NULL,
    counts       BIGINT[]  NOT NULL,
    total_repos  BIGINT    NOT NULL,
    -- Rungs the API refused or rate-limited, repaired by monotonic_repair.
    -- A run with many repairs is a run to distrust.
    repaired     INTEGER   NOT NULL DEFAULT 0,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS repo_global_rank (
    repo_id         BIGINT PRIMARY KEY REFERENCES repo(id) ON DELETE CASCADE,

    -- 0-100 composite. Stars carry the SMALLEST weight of the five inputs
    -- (0.25 of 1.00) precisely so this does not become GitHub's own ranking
    -- with extra steps.
    score           REAL   NOT NULL,

    -- Position among all public repositories by stars alone. Kept separately
    -- because it is the number a user can sanity-check against intuition, and
    -- the only component with a measured empirical distribution behind it.
    star_rank       BIGINT NOT NULL,
    star_percentile REAL   NOT NULL,

    -- Per-component percentiles. Storing the breakdown is what makes a score
    -- arguable rather than oracular — "why is this 71" has an answer.
    components      JSONB  NOT NULL DEFAULT '{}'::jsonb,

    scale_id        BIGINT REFERENCES star_scale_run(id) ON DELETE SET NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS repo_global_rank_score_idx ON repo_global_rank (score DESC);
CREATE INDEX IF NOT EXISTS repo_global_rank_scale_idx ON repo_global_rank (scale_id);

COMMENT ON COLUMN repo_global_rank.score IS
    '0-100 composite over stars, dependents, pagerank ratio, criticality, velocity';
COMMENT ON COLUMN repo_global_rank.star_rank IS
    'Estimated position among all public GitHub repositories, by stars alone';
