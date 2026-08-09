-- Relatedness layers.
--
-- Three edge kinds, because "related" means three different things and
-- collapsing them loses the distinction the product is built on:
--
--   0  depends_on   A cannot work without B.      Directed. Sparse, precise.
--   1  similar_to   A is an alternative to B.     From embeddings.
--   2  used_with    People who use A use B.       From behaviour. THE missing one.
--
-- `used_with` is what answers "LangChain -> ChromaDB". Those two are not
-- similar (a framework and a database) and need not depend on each other. They
-- are used together, which is a behavioural fact and needs its own signal.

ALTER TABLE edge ADD COLUMN IF NOT EXISTS ppmi REAL;
ALTER TABLE edge ADD COLUMN IF NOT EXISTS observations INTEGER;

COMMENT ON COLUMN edge.kind IS '0=depends_on, 1=similar_to, 2=used_with';
COMMENT ON COLUMN edge.ppmi IS 'Positive pointwise mutual information, kind=2 only';
COMMENT ON COLUMN edge.observations IS 'Baskets containing both endpoints, kind=2 only';

-- Star events, the raw material for co-occurrence. Kept rather than aggregated
-- on the fly so the PPMI parameters can be retuned without re-querying
-- BigQuery, which is the expensive half.
CREATE TABLE IF NOT EXISTS star_event (
    actor       TEXT   NOT NULL,
    repo_id     BIGINT NOT NULL REFERENCES repo(id) ON DELETE CASCADE,
    starred_at  DATE,
    PRIMARY KEY (actor, repo_id)
);
CREATE INDEX IF NOT EXISTS star_event_actor_idx ON star_event (actor);
CREATE INDEX IF NOT EXISTS star_event_repo_idx  ON star_event (repo_id);

-- Per-repo relatedness summary, so the API never recomputes.
CREATE TABLE IF NOT EXISTS repo_relatedness (
    repo_id         BIGINT PRIMARY KEY REFERENCES repo(id) ON DELETE CASCADE,
    depends_on_n    INTEGER NOT NULL DEFAULT 0,
    dependents_n    INTEGER NOT NULL DEFAULT 0,
    used_with_n     INTEGER NOT NULL DEFAULT 0,
    similar_n       INTEGER NOT NULL DEFAULT 0,
    -- PageRank over the UNION of depends_on and used_with. Running it over
    -- dependencies alone leaves every unpackaged repository — awesome-lists,
    -- dotfiles, notebooks, most C++ — at the teleport floor and therefore
    -- indistinguishable from each other.
    rank            DOUBLE PRECISION
);
