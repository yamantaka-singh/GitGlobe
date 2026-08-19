-- Phase 4: the lexical arm of hybrid search.
--
-- `/search` matched with `WHERE full_name ILIKE '%q%' OR description ILIKE '%q%'`,
-- which is a substring test, not a search. It cannot match across word order, it
-- cannot stem, and because the pattern is the *whole* query string, any query of
-- more than one word matches only if those words appear contiguously and in that
-- order. Measured over the 30-query eval set that scored recall@10 = 0.017, with
-- 26 of 30 queries returning nothing at all.
--
-- A GIN index over the same text as a tsvector scored 0.229 with no empty
-- results. That is still well short of the 0.7 exit criterion — lexical matching
-- genuinely cannot answer "lightweight c++ web servers with minimal
-- dependencies" — but it is a real second input for the reciprocal rank fusion,
-- which until now was called with an empty list and was therefore decorative.
--
-- Expression index rather than a stored tsvector column: it keeps the corpus
-- schema owned by the pipeline, and the query planner uses it as long as the
-- expression here is character-for-character what `LEXICAL_SQL` computes. If
-- you change one, change both, or the search silently reverts to a seq scan
-- over 87k rows.
--
-- `replace(full_name, '/', ' ')` is load-bearing. Postgres classifies
-- `ohmyzsh/ohmyzsh` as a single `file` token, so the lexeme stored is the whole
-- `'ohmyzsh/ohmyzsh'` and the word `ohmyzsh` matches nothing — searching for a
-- repository by name returned an empty page. Splitting on the slash puts owner
-- and repo in as ordinary words. The eval set does not catch this: all 30 of
-- its queries describe what software does, and none looks one up by name.

-- Deliberately unweighted. Setting the name to weight A and the description to
-- B is the textbook move and it measured worse: descriptive recall@10 fell from
-- 0.221 to 0.086, because for "time series database" it promotes anything
-- *named* `*-database` above the repositories actually described as one. It
-- bought one extra name lookup out of six. `repo_name_idx` below serves name
-- lookup properly and costs the full-text arm nothing.

DROP INDEX IF EXISTS repo_fts_idx;

CREATE INDEX IF NOT EXISTS repo_fts_idx ON repo USING GIN ((
    to_tsvector('english',
        replace(coalesce(full_name, ''), '/', ' ') || ' ' || coalesce(description, ''))
));

-- The exact-name arm. `ts_rank` cannot express "this repository is literally
-- called that" — it found the canonical repo for "react", "vue", "linux" and
-- three others in 2 cases out of 6, and ranked it first in 1 — so name lookup
-- is a separate arm that reciprocal rank fusion merges, not a ranking tweak.
CREATE INDEX IF NOT EXISTS repo_name_idx ON repo (lower(split_part(full_name, '/', 2)));
