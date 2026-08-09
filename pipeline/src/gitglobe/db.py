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
