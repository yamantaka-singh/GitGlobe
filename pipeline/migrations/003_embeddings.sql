-- Phase 2: embeddings, spherical projection, clusters.
--
-- Vectors are stored as BYTEA (raw little-endian float32), not pgvector.
--
-- pgvector would be the obvious choice if Postgres had to answer nearest-
-- neighbour queries. It does not. Every kNN query in this pipeline happens
-- inside UMAP, in one process, over one numpy array — and `similar_to` edges
-- fall out of the kNN graph UMAP already built. Adding pgvector would mean an
-- extension the stock postgres:16-alpine image does not ship, an index build
-- over a million rows, and a second copy of the vectors, all to accelerate a
-- query nothing issues.
--
-- BYTEA round-trips to numpy with `np.frombuffer(buf, np.float32)` at memcpy
-- speed and costs 4*dim + 4 bytes per row.

ALTER TABLE repo ADD COLUMN IF NOT EXISTS embedding      BYTEA;
ALTER TABLE repo ADD COLUMN IF NOT EXISTS embedding_dim  SMALLINT;
-- The hash the embedding was computed FROM. Compare against repo.content_hash:
-- equal means the vector is current, different means the README changed and it
-- needs re-embedding, null means it was never embedded. One column turns a
-- re-run from "spend the whole budget again" into "embed the delta".
ALTER TABLE repo ADD COLUMN IF NOT EXISTS embedded_hash  TEXT;
ALTER TABLE repo ADD COLUMN IF NOT EXISTS embedded_at    TIMESTAMPTZ;

-- Position on the unit sphere, from UMAP with output_metric='haversine'.
-- Stored as angles rather than xyz: it is what the tile format wants, it is two
-- numbers instead of three, and it cannot drift off the unit sphere.
ALTER TABLE repo ADD COLUMN IF NOT EXISTS theta  DOUBLE PRECISION;  -- [0, PI]
ALTER TABLE repo ADD COLUMN IF NOT EXISTS phi    DOUBLE PRECISION;  -- [0, 2PI)

-- HDBSCAN label, and the coarse domain the colour palette indexes.
-- cluster_id is fine-grained (hundreds); domain is the 12 the eye can tell
-- apart. -1 in cluster_id is HDBSCAN's noise label and is kept, not discarded:
-- "this repo belongs to no tight cluster" is a real and useful fact.
ALTER TABLE repo ADD COLUMN IF NOT EXISTS cluster_id  INTEGER;
ALTER TABLE repo ADD COLUMN IF NOT EXISTS domain      SMALLINT;

CREATE INDEX IF NOT EXISTS repo_needs_embedding_idx ON repo (id)
    WHERE NOT low_signal AND (embedded_hash IS NULL OR embedded_hash IS DISTINCT FROM content_hash);
CREATE INDEX IF NOT EXISTS repo_cluster_idx ON repo (cluster_id);

-- Cluster names and centroids. Populated by clustering; the LLM naming pass is
-- deferred, so `label` may be null and the UI falls back to the top languages
-- and topics — which are already descriptive enough to ship.
CREATE TABLE IF NOT EXISTS cluster (
    id           INTEGER PRIMARY KEY,
    label        TEXT,
    domain       SMALLINT,
    size         INTEGER NOT NULL DEFAULT 0,
    centroid     BYTEA,
    -- Spherical centre, so the camera can fly to a cluster without scanning
    -- every member.
    theta        DOUBLE PRECISION,
    phi          DOUBLE PRECISION,
    top_languages TEXT[] NOT NULL DEFAULT '{}',
    top_topics    TEXT[] NOT NULL DEFAULT '{}'
);

-- What produced the current projection. Without this, "why did every repo move"
-- after a re-run is unanswerable: UMAP is stochastic, and a different seed or
-- n_neighbors reshuffles the entire globe.
CREATE TABLE IF NOT EXISTS projection_run (
    id           BIGSERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    n_points     INTEGER,
    embed_model  TEXT,
    embed_dim    SMALLINT,
    umap_params  JSONB,
    seed         INTEGER,
    notes        TEXT
);
