"""REPO_SELECT must return exactly one row per repo, even when both a
teacher (source=0) and a display-fill (source=2) `repo_score` row exist.

Needs a live DATABASE_URL — this is a property of the actual join against the
actual schema, not a pure function. `test_search_contract.py` covers what can
be checked without one; this covers what can't.

The bug this guards: `repo_score`'s primary key is (repo_id, source), so
`source IN (0, 2)` on a plain JOIN condition matches BOTH rows when a repo has
one of each, silently duplicating that repo in the result set. Discovered by
reasoning about the schema while wiring on-demand summaries, not by seeing it
happen — this test is what makes sure it doesn't, quietly, later.

Run directly: `python api/tests/test_repo_select_no_duplicates.py`.
"""

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import asyncpg

from gitglobe_api.main import REPO_SELECT

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gitglobe:gitglobe@localhost:5433/gitglobe")


async def test_lateral_join_survives_a_double_source_row():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        async with conn.transaction():
            repo_id = await conn.fetchval("SELECT id FROM repo LIMIT 1")

            # Force the exact collision REPO_SELECT's LATERAL subquery exists
            # to prevent: one teacher row, one display-fill row, same repo.
            await conn.execute(
                "INSERT INTO repo_score (repo_id, source, summary, model) "
                "VALUES ($1, 0, 'teacher summary', 'test') "
                "ON CONFLICT (repo_id, source) DO UPDATE SET summary = EXCLUDED.summary",
                repo_id,
            )
            await conn.execute(
                "INSERT INTO repo_score (repo_id, source, summary, model) "
                "VALUES ($1, 2, 'display-fill summary', 'test') "
                "ON CONFLICT (repo_id, source) DO UPDATE SET summary = EXCLUDED.summary",
                repo_id,
            )

            rows = await conn.fetch(REPO_SELECT.format(predicate="r.id = $1"), repo_id)

            assert len(rows) == 1, (
                f"REPO_SELECT returned {len(rows)} rows for one repo with both a "
                f"source=0 and source=2 score — the LATERAL join is duplicating "
                f"rows instead of picking one."
            )
            assert rows[0]["summary"] == "teacher summary", (
                "REPO_SELECT must prefer source=0 (the careful original sample) "
                f"over source=2 (batch or on-demand fill); got {rows[0]['summary']!r}"
            )

            # Transaction never commits — nothing written stays written.
            raise _Rollback
    except _Rollback:
        pass
    finally:
        await conn.close()


class _Rollback(Exception):
    """Forces the transaction block above to roll back, not to signal failure."""


async def main():
    await test_lateral_join_survives_a_double_source_row()
    print("ok  test_lateral_join_survives_a_double_source_row")
    print("\nall repo_select checks passed")


if __name__ == "__main__":
    asyncio.run(main())
