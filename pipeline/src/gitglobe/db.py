"""Postgres access.

Everything is an upsert keyed on `(host, full_name)`, which is what makes the
whole pipeline idempotent: re-running it is safe, a crash mid-run costs only the
current batch, and a partial ingest can be topped up rather than restarted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import asyncpg

log = logging.getLogger(__name__)

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def content_hash(text: str) -> str:
    """Cache key for Phase 2. Unchanged hash means the embedding is still valid."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RepoRow:
    full_name: str
    description: str
    language: str
    topics: list[str]
    stars: int
    forks: int
    open_issues: int
    pushed_at: Any
    created_at: Any
    license: str
    is_fork: bool
    is_archived: bool
    readme_raw: str
    clean_text: str
    embedding_input: str
    low_signal: bool
    non_english: bool
    clean_reduction: float
    dropped_sections: list[str]


class Database:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str, *, min_size: int = 2, max_size: int = 10) -> "Database":
        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def migrate(self) -> None:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            log.info("Applying %s", path.name)
            async with self.pool.acquire() as conn:
                await conn.execute(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ repos

    async def upsert_repos(self, rows: Sequence[RepoRow], *, host: str = "github") -> int:
        """Insert or update a batch.

        `executemany` in one transaction rather than a row at a time: at 100k
        rows the per-statement round trip dominates everything else.
        """
        if not rows:
            return 0

        sql = """
        INSERT INTO repo (
            host, full_name, description, language, topics,
            stars, forks, open_issues, pushed_at, created_at,
            license, is_fork, is_archived,
            readme_raw, clean_text, embedding_input,
            low_signal, non_english, clean_reduction, dropped_sections,
            content_hash, fetched_at
        ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21, now()
        )
        ON CONFLICT (host, full_name) DO UPDATE SET
            description      = EXCLUDED.description,
            language         = EXCLUDED.language,
            topics           = EXCLUDED.topics,
            stars            = EXCLUDED.stars,
            forks            = EXCLUDED.forks,
            open_issues      = EXCLUDED.open_issues,
            pushed_at        = EXCLUDED.pushed_at,
            created_at       = EXCLUDED.created_at,
            license          = EXCLUDED.license,
            is_fork          = EXCLUDED.is_fork,
            is_archived      = EXCLUDED.is_archived,
            readme_raw       = EXCLUDED.readme_raw,
            clean_text       = EXCLUDED.clean_text,
            embedding_input  = EXCLUDED.embedding_input,
            low_signal       = EXCLUDED.low_signal,
            non_english      = EXCLUDED.non_english,
            clean_reduction  = EXCLUDED.clean_reduction,
            dropped_sections = EXCLUDED.dropped_sections,
            content_hash     = EXCLUDED.content_hash,
            fetched_at       = now()
        """

        payload = [
            (
                host, r.full_name, r.description, r.language, r.topics,
                r.stars, r.forks, r.open_issues, r.pushed_at, r.created_at,
                r.license, r.is_fork, r.is_archived,
                r.readme_raw, r.clean_text, r.embedding_input,
                r.low_signal, r.non_english, r.clean_reduction, r.dropped_sections,
                content_hash(r.embedding_input),
            )
            for r in rows
        ]

        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(sql, payload)
        return len(payload)

    async def update_star_velocity(self, velocity: dict[str, int], *, host: str = "github") -> int:
        if not velocity:
            return 0
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "UPDATE repo SET stars_90d = $3 WHERE host = $1 AND full_name = $2",
                [(host, name, count) for name, count in velocity.items()],
            )
        return len(velocity)

    async def update_criticality(self, scores: dict[str, float], *, host: str = "github") -> int:
        if not scores:
            return 0
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "UPDATE repo SET criticality = $3 WHERE host = $1 AND full_name = $2",
                [(host, name, score) for name, score in scores.items()],
            )
        return len(scores)

    # --------------------------------------------------------------- packages

    async def upsert_packages(self, mappings: Iterable[tuple[str, str, str]], *, host: str = "github") -> int:
        """(ecosystem, package_name, repo_full_name) triples.

        Rows whose repository we do not hold are skipped by the join rather than
        rejected — deps.dev knows about far more repositories than we ingest.
        """
        rows = list(mappings)
        if not rows:
            return 0
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                INSERT INTO package (ecosystem, name, repo_id)
                SELECT $1, $2, r.id FROM repo r WHERE r.host = $4 AND r.full_name = $3
                ON CONFLICT (ecosystem, name) DO UPDATE SET repo_id = EXCLUDED.repo_id
                """,
                [(eco, pkg, repo, host) for eco, pkg, repo in rows],
            )
        return len(rows)

    async def package_names(self) -> list[str]:
        async with self.pool.acquire() as conn:
            return [r["name"] for r in await conn.fetch(
                "SELECT DISTINCT name FROM package WHERE repo_id IS NOT NULL"
            )]

    # ------------------------------------------------------------------ edges

    async def upsert_edges(self, edges: Iterable[tuple[str, str, str, float]]) -> int:
        """(ecosystem, src_package, dst_package, weight).

        Resolved to repo ids in SQL. Self-edges are filtered because a repo that
        publishes several packages will otherwise depend on itself.
        """
        rows = list(edges)
        if not rows:
            return 0
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                INSERT INTO edge (src, dst, kind, weight)
                SELECT ps.repo_id, pd.repo_id, 0, $4
                FROM package ps, package pd
                WHERE ps.ecosystem = $1 AND ps.name = $2
                  AND pd.ecosystem = $1 AND pd.name = $3
                  AND ps.repo_id IS NOT NULL AND pd.repo_id IS NOT NULL
                  AND ps.repo_id <> pd.repo_id
                ON CONFLICT (src, dst, kind) DO UPDATE SET weight = EXCLUDED.weight
                """,
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------ checkpoints

    async def checkpoint(self, source: str, cursor: str | None, *, rows_seen: int, completed: bool = False,
                         detail: dict | None = None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_state (source, cursor, rows_seen, completed, detail, updated_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, now())
                ON CONFLICT (source) DO UPDATE SET
                    cursor = EXCLUDED.cursor,
                    rows_seen = EXCLUDED.rows_seen,
                    completed = EXCLUDED.completed,
                    detail = EXCLUDED.detail,
                    updated_at = now()
                """,
                source, cursor, rows_seen, completed, json.dumps(detail or {}),
            )

    async def resume_point(self, source: str) -> tuple[str | None, int, bool]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT cursor, rows_seen, completed FROM ingest_state WHERE source = $1", source
            )
        if row is None:
            return None, 0, False
        return row["cursor"], row["rows_seen"], row["completed"]

    # -------------------------------------------------------------- reporting

    async def quality_report(self) -> dict[str, Any]:
        """The numbers Phase 1's exit criterion is judged on."""
        async with self.pool.acquire() as conn:
            return dict(await conn.fetchrow(
                """
                SELECT
                  COUNT(*)                                             AS repos,
                  COUNT(*) FILTER (WHERE clean_text <> '')             AS with_clean_text,
                  COUNT(*) FILTER (WHERE low_signal)                   AS low_signal,
                  COUNT(*) FILTER (WHERE non_english)                  AS non_english,
                  COUNT(*) FILTER (WHERE stars_90d IS NOT NULL)        AS with_velocity,
                  COUNT(*) FILTER (WHERE criticality IS NOT NULL)      AS with_criticality,
                  ROUND(AVG(clean_reduction)::numeric, 3)              AS mean_reduction,
                  ROUND(AVG(LENGTH(clean_text))::numeric, 0)           AS mean_clean_chars,
                  (SELECT COUNT(*) FROM edge WHERE kind = 0)           AS dependency_edges,
                  (SELECT COUNT(*) FROM package WHERE repo_id IS NOT NULL) AS mapped_packages
                FROM repo
                """
            ))

    async def suspect_readmes(self) -> dict[str, Any]:
        """Rows whose README looks like a symlink target rather than prose.

        A symlink blob's content is its target path: one short line, no spaces,
        containing a slash or a dot. Non-empty, so it passes every emptiness
        check, and it embeds a repository on the strength of its own filename.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                  COUNT(*) FILTER (
                    WHERE readme_raw <> ''
                      AND LENGTH(readme_raw) < 200
                      AND readme_raw NOT LIKE '%' || chr(10) || '%'
                      AND readme_raw NOT LIKE '% %'
                      AND (readme_raw LIKE '%/%' OR readme_raw LIKE '%.%')
                  ) AS symlink_like,
                  COUNT(*) FILTER (WHERE readme_raw = '' OR readme_raw IS NULL) AS empty_readme,
                  COUNT(*) AS total
                FROM repo
                """
            )
            examples = [dict(r) for r in await conn.fetch(
                """
                SELECT full_name, readme_raw
                FROM repo
                WHERE readme_raw <> ''
                  AND LENGTH(readme_raw) < 200
                  AND readme_raw NOT LIKE '%' || chr(10) || '%'
                  AND readme_raw NOT LIKE '% %'
                  AND (readme_raw LIKE '%/%' OR readme_raw LIKE '%.%')
                LIMIT 5
                """
            )]
        return {**dict(row), "examples": examples}

    async def rehydrate_readmes(self, recovered: dict[str, str], *, host: str = "github") -> int:
        """Re-clean and store READMEs fetched by the backfill pass.

        Cleaning happens here rather than in the caller so `readme_raw` and
        `clean_text` are always written together — the database is never in a
        state where one is fresh and the other is stale.
        """
        if not recovered:
            return 0
        from .clean.readme import clean_readme

        payload = []
        for full_name, raw in recovered.items():
            result = clean_readme(raw, name=full_name.split("/")[-1])
            payload.append((
                host, full_name, raw, result.text, result.embedding_input,
                result.low_signal, result.non_english, result.reduction,
                result.dropped_sections, content_hash(result.embedding_input),
            ))

        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                UPDATE repo SET
                    readme_raw = $3, clean_text = $4, embedding_input = $5,
                    low_signal = $6, non_english = $7, clean_reduction = $8,
                    dropped_sections = $9, content_hash = $10, fetched_at = now()
                WHERE host = $1 AND full_name = $2
                """,
                payload,
            )
        return len(payload)

    async def reset_checkpoints(self) -> int:
        """Clear shard checkpoints so the next run re-fetches everything.

        The repo rows themselves are left alone — upserts are idempotent, so a
        re-run overwrites them in place with corrected data. Needed after a
        fetch-side bug fix, because completed shards are otherwise skipped and
        the bad rows would never be revisited.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM ingest_state")
        return int(result.split()[-1]) if result else 0

    async def shard_progress(self) -> list[dict[str, Any]]:
        """Per-shard state. The single most useful view when a run looks wrong.

        A shard that completed with far fewer rows than its siblings usually
        means the star band is thinner than the plan assumed; one stuck
        incomplete with a cursor means the run was interrupted there.
        """
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(
                """
                SELECT source, rows_seen, completed, cursor IS NOT NULL AS has_cursor, updated_at
                FROM ingest_state
                ORDER BY completed, rows_seen DESC
                """
            )]

    async def distribution(self) -> dict[str, Any]:
        """Shape of what was ingested, not just the count."""
        async with self.pool.acquire() as conn:
            stars = dict(await conn.fetchrow(
                """
                SELECT MIN(stars) AS min_stars, MAX(stars) AS max_stars,
                       PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY stars) AS median_stars,
                       COUNT(*) FILTER (WHERE stars >= 10000) AS over_10k,
                       COUNT(*) FILTER (WHERE stars < 100)    AS under_100
                FROM repo
                """
            ))
            languages = [dict(r) for r in await conn.fetch(
                """
                SELECT COALESCE(NULLIF(language, ''), '(none)') AS language, COUNT(*) AS n
                FROM repo GROUP BY 1 ORDER BY n DESC LIMIT 12
                """
            )]
            reduction = [dict(r) for r in await conn.fetch(
                """
                SELECT width_bucket(clean_reduction, 0, 1, 10) AS decile, COUNT(*) AS n
                FROM repo WHERE clean_reduction IS NOT NULL
                GROUP BY 1 ORDER BY 1
                """
            )]
            worst = [dict(r) for r in await conn.fetch(
                """
                SELECT full_name, stars, clean_reduction, LENGTH(clean_text) AS chars
                FROM repo
                WHERE NOT low_signal AND clean_reduction IS NOT NULL
                ORDER BY clean_reduction ASC LIMIT 8
                """
            )]
        return {"stars": stars, "languages": languages,
                "reduction_deciles": reduction, "least_cleaned": worst}

    async def sample_clean_text(self, limit: int = 20) -> list[dict[str, Any]]:
        """A random sample for the manual review the exit criterion requires."""
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(
                """
                SELECT full_name, description, clean_reduction, low_signal,
                       LEFT(clean_text, 400) AS excerpt
                FROM repo
                WHERE NOT low_signal
                ORDER BY random()
                LIMIT $1
                """,
                limit,
            )]

    # ------------------------------------------------------------- phase 2

    async def backfill_content_hashes(self) -> int:
        """Compute `content_hash` for rows that predate it.

        Everything about re-run cost depends on this column: it is the only
        thing that distinguishes "already embedded, skip it" from "the README
        changed, pay again".
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, embedding_input FROM repo "
                "WHERE content_hash IS NULL AND embedding_input IS NOT NULL"
            )
            if not rows:
                return 0
            await conn.executemany(
                "UPDATE repo SET content_hash = $2 WHERE id = $1",
                [(r["id"], content_hash(r["embedding_input"])) for r in rows],
            )
        return len(rows)

    async def rows_needing_embedding(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Rows whose embedding is missing or stale.

        `IS DISTINCT FROM` rather than `<>`: a plain comparison against NULL is
        NULL, not TRUE, so never-embedded rows would be silently excluded and
        the first run would embed nothing at all.
        """
        query = """
            SELECT id, full_name, embedding_input, content_hash
            FROM repo
            WHERE NOT low_signal
              AND embedding_input IS NOT NULL
              AND length(embedding_input) > 0
              AND (embedded_hash IS NULL OR embedded_hash IS DISTINCT FROM content_hash)
            ORDER BY stars DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(query)]

    # ------------------------------------------------------------------ brain

    #: Teacher and student share `repo_score`, separated by this column.
    TEACHER = 0
    STUDENT = 1

    async def rows_for_teacher(self) -> list[dict[str, Any]]:
        """Candidates the teacher could rate, with everything the prompt needs.

        `days_since_push` is here because the sampler stratifies on it: rating
        4,000 rows drawn only from what is popular would teach the student that
        popularity is the signal, which is the one thing the rubric forbids.

        Deliberately NOT filtered by what has already been rated — `teach`
        samples from the full population and then skips what it holds, so an
        interrupted run resumes into the same stratified sample instead of
        drawing a fresh one from whatever is left.
        """
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(
                """
                SELECT r.id, r.full_name, r.description, r.language, r.topics,
                       r.license, r.clean_text, r.content_hash,
                       COALESCE(r.stars, 0) AS stars,
                       COALESCE(r.domain, 0) AS domain,
                       COALESCE(EXTRACT(EPOCH FROM (now() - r.pushed_at)) / 86400,
                                3650)::float AS days_since_push
                FROM repo r
                WHERE NOT r.low_signal
                  AND r.clean_text IS NOT NULL
                  AND length(r.clean_text) > 0
                ORDER BY r.id
                """
            )]

    async def brain_rows(self) -> list[dict[str, Any]]:
        """Every column `brain.features.build_features` reads, for every repo.

        Unfiltered on purpose: the teacher rates a sample, but the student must
        predict for the whole corpus, so this is the population — filtering here
        would silently shrink what the globe can show.

        README *lengths* are measured in SQL rather than fetching the text.
        `build_features` only ever calls `len()` on them, and at ~8,700 chars
        per row across `clean_text` and `readme_raw` the naive version moved
        most of a gigabyte through the driver — growing linearly with the
        corpus — to produce two floats per repository.
        """
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(
                """
                SELECT r.id, r.full_name, r.description, r.language, r.topics,
                       r.license, r.dropped_sections,
                       COALESCE(length(r.clean_text), 0) AS readme_chars,
                       COALESCE(length(r.readme_raw), 0) AS raw_readme_chars,
                       r.created_at, r.pushed_at, r.is_fork, r.is_archived,
                       r.low_signal, r.non_english, r.cluster_id, r.domain,
                       r.content_hash, r.stars, r.stars_90d, r.criticality,
                       -- Read by build_features and previously not selected, so
                       -- `forks`, `open_issues` and `clean_reduction` were all
                       -- constant zero — along with `fork_ratio` and
                       -- `issues_per_star`, which derive from them. Five dead
                       -- columns out of 55, silent because a missing key
                       -- becomes the neutral value rather than an error.
                       r.forks, r.open_issues, r.clean_reduction
                FROM repo r
                ORDER BY r.id
                """
            )]

    async def graph_features(self) -> tuple:
        """(rank, in_degree, out_degree, similar_degree) dicts keyed by repo id.

        Degrees are split by edge kind because they mean different things: an
        in-degree over `depends_on` is a fact about the world, while one over
        `similar_to` is a fact about our own k-NN step. Merging them would let
        the corpus vote on its own importance.

        Queries `edge` directly rather than reusing `relatedness_edges`, which
        filters `similar_to` out on purpose for PageRank — reusing it here would
        leave `similar_degree` silently zero for every repository.
        """
        in_deg: dict = {}
        out_deg: dict = {}
        similar: dict = {}
        async with self.pool.acquire() as conn:
            ranks = {
                int(r["repo_id"]): float(r["rank"] or 0.0)
                for r in await conn.fetch("SELECT repo_id, rank FROM repo_relatedness")
            }
            for row in await conn.fetch("SELECT src, dst, kind FROM edge"):
                src, dst = int(row["src"]), int(row["dst"])
                if int(row["kind"]) == 1:      # similar_to is undirected
                    similar[src] = similar.get(src, 0) + 1
                    similar[dst] = similar.get(dst, 0) + 1
                else:
                    out_deg[src] = out_deg.get(src, 0) + 1
                    in_deg[dst] = in_deg.get(dst, 0) + 1
        return ranks, in_deg, out_deg, similar

    async def store_scores(self, scores: dict[int, dict], *, source: int,
                           model: str = "", hashes: dict[int, str] | None = None) -> int:
        """Upsert teacher or student scores.

        Upsert rather than insert so a re-run repairs rows instead of failing on
        the primary key — which matters because `rate_many` checkpoints every
        200 rows and a resumed run will re-touch the tail of the last batch.
        """
        if not scores:
            return 0
        hashes = hashes or {}
        payload = [
            (rid, source,
             s.get("maintenance"), s.get("production_readiness"), s.get("specificity"),
             s.get("learning_value"), s.get("onboarding_ease"), s.get("canonicity"),
             s.get("summary"), list(s.get("flags") or []),
             hashes.get(rid), model)
            for rid, s in scores.items()
        ]
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                INSERT INTO repo_score (
                    repo_id, source, maintenance, production_readiness, specificity,
                    learning_value, onboarding_ease, canonicity, summary, flags,
                    scored_hash, model
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (repo_id, source) DO UPDATE SET
                    maintenance = EXCLUDED.maintenance,
                    production_readiness = EXCLUDED.production_readiness,
                    specificity = EXCLUDED.specificity,
                    learning_value = EXCLUDED.learning_value,
                    onboarding_ease = EXCLUDED.onboarding_ease,
                    canonicity = EXCLUDED.canonicity,
                    summary = EXCLUDED.summary,
                    flags = EXCLUDED.flags,
                    scored_hash = EXCLUDED.scored_hash,
                    model = EXCLUDED.model,
                    scored_at = now()
                """,
                payload,
            )
        return len(payload)

    async def scores(self, *, source: int) -> dict[int, dict[str, Any]]:
        """Stored scores by repo id. Rows whose README changed are excluded.

        A judgement of a README that no longer exists is worse than no
        judgement: it is wrong and it looks current. `scored_hash` is what makes
        that detectable, and this is the only place it is enforced.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.repo_id, s.maintenance, s.production_readiness, s.specificity,
                       s.learning_value, s.onboarding_ease, s.canonicity, s.summary, s.flags
                FROM repo_score s JOIN repo r ON r.id = s.repo_id
                WHERE s.source = $1
                  AND (s.scored_hash IS NULL OR s.scored_hash = r.content_hash)
                """,
                source,
            )
        return {int(r["repo_id"]): dict(r) for r in rows}

    async def start_brain_run(self, **fields) -> int:
        keys = ", ".join(fields)
        holes = ", ".join(f"${i + 1}" for i in range(len(fields)))
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval(
                f"INSERT INTO brain_run ({keys}) VALUES ({holes}) RETURNING id",
                *fields.values(),
            ))

    async def finish_brain_run(self, run_id: int, *, metrics: dict | None = None,
                               student_n: int = 0, notes: str = "") -> None:
        import json

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE brain_run
                   SET finished_at = now(), metrics = $2, student_n = $3, notes = $4
                 WHERE id = $1
                """,
                run_id, json.dumps(metrics or {}), student_n, notes,
            )

    async def store_embeddings(self, vectors: dict[int, bytes], dim: int) -> int:
        """Persist a batch. `embedded_hash` is copied from the CURRENT
        `content_hash`, which is what makes the next run skip these rows."""
        if not vectors:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    UPDATE repo
                       SET embedding = $2, embedding_dim = $3,
                           embedded_hash = content_hash, embedded_at = now()
                     WHERE id = $1
                    """,
                    [(rid, blob, dim) for rid, blob in vectors.items()],
                )
        return len(vectors)

    async def embedded_rows(self) -> list[dict[str, Any]]:
        """Everything with a current embedding, for projection and clustering."""
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(
                """
                SELECT id, full_name, embedding, embedding_dim, stars,
                       criticality, low_signal, is_archived, is_fork
                FROM repo
                WHERE embedding IS NOT NULL
                ORDER BY id
                """
            )]

    async def store_projection(
        self,
        positions: dict[int, tuple[float, float]],
        clusters: dict[int, tuple[int, int]] | None = None,
    ) -> int:
        """Write theta/phi, and optionally cluster_id/domain, in one pass."""
        if not positions:
            return 0
        clusters = clusters or {}
        payload = [
            (rid, theta, phi, *clusters.get(rid, (None, None)))
            for rid, (theta, phi) in positions.items()
        ]
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    UPDATE repo
                       SET theta = $2, phi = $3,
                           cluster_id = COALESCE($4, cluster_id),
                           domain     = COALESCE($5, domain)
                     WHERE id = $1
                    """,
                    payload,
                )
        return len(payload)

    async def store_star_scale(self, thresholds: list, counts: list, *,
                               total_repos: int, repaired: int = 0, notes: str = "") -> int:
        """Record a measured star survival function, return its run id.

        A run rather than an overwrite because every global score is computed
        against one of these. Without the history, "why did this score change"
        cannot distinguish the repository moving from the yardstick moving.
        """
        if len(thresholds) != len(counts):
            raise ValueError(f"thresholds/counts differ: {len(thresholds)}, {len(counts)}")
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval(
                """
                INSERT INTO star_scale_run (thresholds, counts, total_repos, repaired, notes)
                VALUES ($1, $2, $3, $4, $5) RETURNING id
                """,
                [int(t) for t in thresholds], [int(c) for c in counts],
                int(total_repos), int(repaired), notes,
            ))

    async def latest_star_scale(self) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, thresholds, counts, total_repos, measured_at, repaired "
                "FROM star_scale_run ORDER BY measured_at DESC LIMIT 1"
            )
        return dict(row) if row else None

    async def store_global_ranks(self, ranks: dict, scale_id: int | None) -> int:
        """Write the 0-100 global composite for every scored repository."""
        if not ranks:
            return 0
        payload = [
            (int(rid), float(r.score), int(r.star_rank), float(r.star_percentile),
             json.dumps(r.components), scale_id)
            for rid, r in ranks.items()
        ]
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO repo_global_rank
                        (repo_id, score, star_rank, star_percentile, components,
                         scale_id, computed_at)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, now())
                    ON CONFLICT (repo_id) DO UPDATE SET
                        score = EXCLUDED.score,
                        star_rank = EXCLUDED.star_rank,
                        star_percentile = EXCLUDED.star_percentile,
                        components = EXCLUDED.components,
                        scale_id = EXCLUDED.scale_id,
                        computed_at = EXCLUDED.computed_at
                    """,
                    payload,
                )
        return len(payload)

    async def global_ranks(self) -> dict[int, float]:
        """repo_id -> 0-100 composite, for the tile builder and the HUD."""
        async with self.pool.acquire() as conn:
            return {int(r["repo_id"]): float(r["score"])
                    for r in await conn.fetch("SELECT repo_id, score FROM repo_global_rank")}

    async def store_ranks(self, ranks: dict[int, float]) -> int:
        if not ranks:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO repo_relatedness (repo_id, rank) VALUES ($1, $2)
                    ON CONFLICT (repo_id) DO UPDATE SET rank = EXCLUDED.rank
                    """,
                    list(ranks.items()),
                )
        return len(ranks)

    async def replace_similar_edges(self, edges: list[tuple[int, int, float]]) -> int:
        """Swap in a fresh `similar_to` layer.

        Deleted first because kNN edges are entirely derived: a re-projection
        invalidates all of them at once, and leaving the old ones would union
        two different maps' notions of similarity.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM edge WHERE kind = 1")
                if edges:
                    await conn.executemany(
                        "INSERT INTO edge (src, dst, kind, weight) VALUES ($1, $2, 1, $3) "
                        "ON CONFLICT (src, dst, kind) DO UPDATE SET weight = EXCLUDED.weight",
                        edges,
                    )
        return len(edges)

    async def relatedness_edges(self) -> list[tuple[int, int, float, int]]:
        """Edges PageRank runs over: `depends_on` (0) and `used_with` (2).

        `similar_to` is excluded on purpose. It is a kNN graph, so every node
        has roughly k neighbours by construction and it carries almost no
        information about importance — including it would smear the ranking
        towards uniform. See docs/RELATEDNESS.md.
        """
        async with self.pool.acquire() as conn:
            return [
                (r["src"], r["dst"], float(r["w"]), r["kind"])
                for r in await conn.fetch(
                    "SELECT src, dst, kind, COALESCE(ppmi, weight) AS w "
                    "FROM edge WHERE kind IN (0, 2)"
                )
            ]

    async def world_rows(self) -> list[dict[str, Any]]:
        """Everything the tile writer needs, in one query.

        Only rows with a position: an embedded repository that has not been
        projected has no place on the globe yet.

        `criticality` and `stars_90d` are here because `stage_calibrate` scores
        from these rows and `signals_for` reads both. They were missing once,
        and because that function turns an absent key into the neutral value
        rather than raising, the effect was not an error: 84,434 backfilled
        criticality scores were silently discarded on read and the composite
        came back byte-identical. Any column the score reads must be selected
        here, or it fails quietly and looks like the data was wrong.
        """
        async with self.pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(
                """
                SELECT r.id, r.full_name, r.theta, r.phi, r.domain, r.cluster_id,
                       r.stars, r.low_signal, r.is_archived, r.is_fork,
                       r.criticality, r.stars_90d,
                       COALESCE(rr.rank, 0.0) AS rank,
                       gr.score AS global_score, gr.star_rank, gr.star_percentile,
                       -- Mean of whatever dimensions the student was allowed to
                       -- keep. avg() over unnest skips NULLs, so a dimension
                       -- that failed its baseline check is absent from the mean
                       -- rather than counted as zero.
                       (SELECT avg(v) FROM unnest(ARRAY[
                            rs.maintenance, rs.production_readiness, rs.specificity,
                            rs.learning_value, rs.onboarding_ease, rs.canonicity
                        ]) v WHERE v IS NOT NULL) AS brain_score
                FROM repo r
                LEFT JOIN repo_relatedness rr ON rr.repo_id = r.id
                LEFT JOIN repo_global_rank gr ON gr.repo_id = r.id
                LEFT JOIN repo_score rs ON rs.repo_id = r.id AND rs.source = 1
                WHERE r.theta IS NOT NULL AND r.phi IS NOT NULL
                ORDER BY r.id
                """
            )]

    async def store_clusters(self, entries: list[dict[str, Any]]) -> int:
        if not entries:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO cluster (id, label, domain, size, theta, phi)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO UPDATE SET
                        label = EXCLUDED.label, domain = EXCLUDED.domain,
                        size  = EXCLUDED.size,  theta  = EXCLUDED.theta,
                        phi   = EXCLUDED.phi
                    """,
                    [
                        (e["id"], e.get("label"), e.get("domain"), e.get("size", 0),
                         e.get("theta"), e.get("phi"))
                        for e in entries
                    ],
                )
        return len(entries)

    async def start_projection_run(self, **fields) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO projection_run (n_points, embed_model, embed_dim, umap_params, seed)
                VALUES ($1, $2, $3, $4, $5) RETURNING id
                """,
                fields.get("n_points"), fields.get("embed_model"), fields.get("embed_dim"),
                json.dumps(fields.get("umap_params") or {}), fields.get("seed"),
            )

    async def finish_projection_run(self, run_id: int, notes: str = "") -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE projection_run SET finished_at = now(), notes = $2 WHERE id = $1",
                run_id, notes,
            )

    async def find_repos(self, names: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Look up specific repositories by full name — for the spot-check."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, full_name, theta, phi, domain, cluster_id, embedding, embedding_dim
                FROM repo WHERE lower(full_name) = ANY($1::text[])
                """,
                [n.lower() for n in names],
            )
        return {r["full_name"].lower(): dict(r) for r in rows}

    async def upsert_repo_edges(
        self, edges: Iterable[tuple[str, str, float]], *, host: str = "github"
    ) -> int:
        """(src_full_name, dst_full_name, weight) -> kind=0 edges.

        Takes repository names because deps.dev resolves packages to projects
        inside BigQuery now. The package-keyed `upsert_edges` still exists for
        the `package` table, but the edge path no longer depends on that join
        being correct on our side.
        """
        rows = list(edges)
        if not rows:
            return 0
        written = 0
        async with self.pool.acquire() as conn, conn.transaction():
            # Chunked: 2M+ rows in one executemany builds an enormous parameter
            # list in the driver and can outlive the statement timeout.
            for start in range(0, len(rows), 10_000):
                chunk = rows[start:start + 10_000]
                await conn.executemany(
                    """
                    INSERT INTO edge (src, dst, kind, weight)
                    SELECT rs.id, rd.id, 0, $3
                    FROM repo rs, repo rd
                    WHERE rs.host = $4 AND rs.full_name = $1
                      AND rd.host = $4 AND rd.full_name = $2
                      AND rs.id <> rd.id
                    ON CONFLICT (src, dst, kind) DO UPDATE SET weight = EXCLUDED.weight
                    """,
                    [(s, d, float(w), host) for s, d, w in chunk],
                )
                written += len(chunk)
        return written

    async def package_repo_map(self) -> list[tuple[str, str, str]]:
        """(ecosystem, package_name, repo_full_name) for every mapped package.

        This is the 1.4 TiB deps.dev scan, already paid for and persisted. The
        edge query uploads it back to BigQuery rather than re-deriving it.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT p.ecosystem, p.name, r.full_name "
                "FROM package p JOIN repo r ON r.id = p.repo_id "
                "WHERE p.repo_id IS NOT NULL"
            )
        return [(r["ecosystem"], r["name"], r["full_name"]) for r in rows]

    async def repo_name_case_map(self, *, host: str = "github") -> dict[str, str]:
        """lowercase full_name -> stored full_name.

        deps.dev stores `ProjectName` in its own casing and GitHub's API returns
        another. Joining on the raw strings drops every repository whose owner
        capitalises differently — silently, as missing edges rather than an
        error.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT full_name FROM repo WHERE host = $1", host)
        return {r["full_name"].lower(): r["full_name"] for r in rows}

    async def edge_counts(self) -> dict[str, int]:
        """Edges by kind, and how many repositories have any at all.

        This exists because it was missing, and its absence let a globe with
        ZERO edges pass `gitglobe status` while the TypeScript verifier caught
        it. PageRank over an empty graph returns 1/n for every node — it sums
        to 1, every rank is positive, and every consistency check passes. The
        only symptom is `max/min = 1x`. A status command that cannot see the
        edge table cannot see the most consequential thing about the graph.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT kind, count(*) AS n FROM edge GROUP BY kind")
            by_kind = {int(r["kind"]): int(r["n"]) for r in rows}
            connected = await conn.fetchval(
                "SELECT count(DISTINCT id) FROM ("
                "  SELECT src AS id FROM edge WHERE kind IN (0, 2)"
                "  UNION SELECT dst FROM edge WHERE kind IN (0, 2)) t"
            )
            packages = await conn.fetchval("SELECT count(*) FROM package WHERE repo_id IS NOT NULL")
            stars = await conn.fetchval("SELECT count(*) FROM star_event")
        return {
            "depends_on": by_kind.get(0, 0),
            "similar_to": by_kind.get(1, 0),
            "used_with": by_kind.get(2, 0),
            "rankable_edges": by_kind.get(0, 0) + by_kind.get(2, 0),
            "connected_repos": int(connected or 0),
            "mapped_packages": int(packages or 0),
            "star_events": int(stars or 0),
        }

    async def upsert_star_events(self, events: Iterable[tuple[str, str, Any]], *, host: str = "github") -> int:
        """(full_name, actor, starred_at). Raw events, not pairs.

        Kept raw so the PPMI parameters can be retuned without re-querying
        BigQuery, which is the expensive half by orders of magnitude.
        """
        rows = list(events)
        if not rows:
            return 0
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                INSERT INTO star_event (actor, repo_id, starred_at)
                SELECT $2, r.id, $3 FROM repo r
                WHERE r.host = $4 AND r.full_name = $1
                ON CONFLICT (actor, repo_id) DO NOTHING
                """,
                [(name, actor, at, host) for name, actor, at in rows],
            )
        return len(rows)

    async def star_baskets(self, min_size: int = 2, max_size: int = 400) -> list[list[str]]:
        """One basket of repository names per actor.

        Bounded on both ends in SQL rather than in Python: an actor who starred
        5,000 repositories is browsing, not choosing, and would alone contribute
        12.5 million pairs to the co-occurrence count.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT array_agg(r.full_name) AS items
                FROM star_event s JOIN repo r ON r.id = s.repo_id
                GROUP BY s.actor
                HAVING count(*) BETWEEN $1 AND $2
                """,
                min_size, max_size,
            )
        return [list(r["items"]) for r in rows]

    async def replace_used_with_edges(
        self, pairs: Iterable[tuple[str, str, float, float]], *, host: str = "github"
    ) -> int:
        """(name_a, name_b, ppmi, observations) -> kind=2 edges, both directions.

        Written symmetrically because `used_with` has no direction — "people who
        use A also use B" is the same fact as its converse — and the CSR builder
        would otherwise record a direction bit that means nothing.
        """
        rows = list(pairs)
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM edge WHERE kind = 2")
            if not rows:
                return 0
            await conn.executemany(
                """
                INSERT INTO edge (src, dst, kind, weight, ppmi, observations)
                SELECT ra.id, rb.id, 2, $3, $3, $4
                FROM repo ra, repo rb
                WHERE ra.host = $5 AND ra.full_name = $1
                  AND rb.host = $5 AND rb.full_name = $2
                  AND ra.id <> rb.id
                ON CONFLICT (src, dst, kind) DO UPDATE
                  SET weight = EXCLUDED.weight, ppmi = EXCLUDED.ppmi,
                      observations = EXCLUDED.observations
                """,
                [(a, b, w, int(n), host) for a, b, w, n in rows]
                + [(b, a, w, int(n), host) for a, b, w, n in rows],
            )
        return len(rows)

    async def phase2_report(self) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            return dict(await conn.fetchrow(
                """
                SELECT
                  count(*)                                              AS repos,
                  count(*) FILTER (WHERE NOT low_signal)                AS embeddable,
                  count(embedding)                                      AS embedded,
                  count(*) FILTER (WHERE embedded_hash IS DISTINCT FROM content_hash
                                     AND NOT low_signal
                                     AND embedding_input IS NOT NULL)   AS stale,
                  count(theta)                                          AS projected,
                  count(domain)                                         AS with_domain,
                  count(*) FILTER (WHERE cluster_id >= 0)               AS clustered,
                  count(DISTINCT cluster_id) FILTER (WHERE cluster_id >= 0) AS clusters
                FROM repo
                """
            ))
