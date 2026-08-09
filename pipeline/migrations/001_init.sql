-- GitGlobe Phase 1 schema.
--
-- Mirrors docs/ARCHITECTURE.md §4, with two additions the architecture doc did
-- not anticipate: `ingest_state` for resumable checkpoints, and the cleaner's
-- quality columns, because "why is this repo in the wrong cluster" is a
-- question you will ask in Phase 2 and can only answer if you kept the evidence.

CREATE TABLE IF NOT EXISTS repo (
    id              BIGSERIAL PRIMARY KEY,
    host            TEXT        NOT NULL DEFAULT 'github',
    full_name       TEXT        NOT NULL,
    description     TEXT,
    language        TEXT,
    topics          TEXT[]      NOT NULL DEFAULT '{}',

    stars           INTEGER     NOT NULL DEFAULT 0,
    -- Stars gained in the trailing 90 days. Velocity is what separates a live
    -- project from a 2015 project with 40k stars, and raw totals cannot.
    stars_90d       INTEGER,
    forks           INTEGER     NOT NULL DEFAULT 0,
    open_issues     INTEGER     NOT NULL DEFAULT 0,
    pushed_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ,

    license         TEXT,
    is_fork         BOOLEAN     NOT NULL DEFAULT FALSE,
    is_archived     BOOLEAN     NOT NULL DEFAULT FALSE,

    -- OSSF criticality score. A better importance signal than stars because it
    -- weighs dependents and maintenance activity, not popularity.
    criticality     REAL,

    -- Cleaner output and its provenance.
    readme_raw      TEXT,
    clean_text      TEXT,
    embedding_input TEXT,
    low_signal      BOOLEAN     NOT NULL DEFAULT FALSE,
    non_english     BOOLEAN     NOT NULL DEFAULT FALSE,
    clean_reduction REAL,
    dropped_sections TEXT[]     NOT NULL DEFAULT '{}',

    -- sha256 of embedding_input. Phase 2 skips anything whose hash is unchanged,
    -- which makes re-runs almost free.
    content_hash    TEXT,

    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (host, full_name)
);

CREATE INDEX IF NOT EXISTS repo_stars_idx        ON repo (stars DESC);
CREATE INDEX IF NOT EXISTS repo_criticality_idx  ON repo (criticality DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS repo_language_idx     ON repo (language);
CREATE INDEX IF NOT EXISTS repo_content_hash_idx ON repo (content_hash);
-- Partial index: Phase 2 only ever embeds the rows worth embedding.
CREATE INDEX IF NOT EXISTS repo_embeddable_idx   ON repo (id) WHERE NOT low_signal;

-- Package coordinates, so deps.dev edges (which are between *packages*) can be
-- resolved to repositories. One repo can publish several packages.
CREATE TABLE IF NOT EXISTS package (
    ecosystem   TEXT   NOT NULL,
    name        TEXT   NOT NULL,
    repo_id     BIGINT REFERENCES repo(id) ON DELETE CASCADE,
    PRIMARY KEY (ecosystem, name)
);
CREATE INDEX IF NOT EXISTS package_repo_idx ON package (repo_id);

CREATE TABLE IF NOT EXISTS edge (
    src     BIGINT   NOT NULL REFERENCES repo(id) ON DELETE CASCADE,
    dst     BIGINT   NOT NULL REFERENCES repo(id) ON DELETE CASCADE,
    -- 0 = dependency (deps.dev), 1 = semantic kNN (Phase 2).
    kind    SMALLINT NOT NULL DEFAULT 0,
    weight  REAL     NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src, dst, kind),
    CHECK (src <> dst)
);
CREATE INDEX IF NOT EXISTS edge_src_idx ON edge (src, kind);
CREATE INDEX IF NOT EXISTS edge_dst_idx ON edge (dst, kind);

-- Resumable checkpoints. A crash three hours into a 100k ingest must not mean
-- starting over, and GitHub's rate limit makes a restart genuinely expensive.
CREATE TABLE IF NOT EXISTS ingest_state (
    source      TEXT        PRIMARY KEY,
    cursor      TEXT,
    completed   BOOLEAN     NOT NULL DEFAULT FALSE,
    rows_seen   BIGINT      NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    detail      JSONB       NOT NULL DEFAULT '{}'::jsonb
);

-- A single row of quality metrics per run, so cleaner regressions are visible
-- as a trend rather than discovered by eye three phases later.
CREATE TABLE IF NOT EXISTS ingest_run (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    target_repos    INTEGER,
    repos_ingested  INTEGER,
    edges_ingested  INTEGER,
    low_signal_pct  REAL,
    mean_reduction  REAL,
    notes           TEXT
);
