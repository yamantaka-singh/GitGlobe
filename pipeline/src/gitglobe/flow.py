"""The Phase 1 ingest flow.

Ordered so that each stage is useful on its own and safe to re-run:

    1. repositories + cleaning   (GitHub GraphQL — the long pole)
    2. star velocity             (GH Archive)
    3. criticality               (OSSF)
    4. package -> repo mapping   (deps.dev)
    5. dependency edges          (deps.dev)

Stages 2-5 all enrich rows that stage 1 already wrote, so a run that is
interrupted after stage 1 still leaves a usable database. That ordering is
deliberate: the expensive, rate-limited stage runs first and its output is
independently valuable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.cache_policies import NO_CACHE

from .clean.readme import clean_readme
from .db import Database, RepoRow
from .ingest.github import GitHubIngest
from .ingest.plan import (
    EXPECTED_YIELD,
    SEARCH_RESULT_CAP,
    band_query,
    language_queries,
    plan_for_target,
)
from .settings import Settings

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    repos: int
    edges: int
    packages: int


@task(retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
async def ingest_repositories(db: Database, settings: Settings, target: int) -> int:
    """Fetch, clean, and store repositories.

    Cleaning happens inline rather than as a later pass. It is pure CPU work on
    text we already hold, and doing it here means the raw README and its cleaned
    form are written in the same transaction — so the database is never in a
    state where `readme_raw` is populated and `clean_text` is not.
    """
    logger = get_run_logger()
    plan = plan_for_target(target)
    logger.info("Plan: %d star bands, %d queries worst case", len(plan.bands()), len(plan.queries()))

    state = _Progress(target=target)

    async with GitHubIngest(settings.github_tokens) as gh:
        for band in plan.bands():
            if state.total >= target:
                break

            # Run the band unfiltered first. Only a band that comes back AT the
            # result cap is genuinely truncated and has anything left for the
            # language axis to recover. The previous version precomputed the
            # full cross product and spent 47% of its queries on empty
            # (band, language) pairs that could never have held anything.
            #
            # A band that fails outright is SKIPPED, not fatal. GitHub returns
            # 502 for queries it finds too expensive, and letting one band abort
            # the task means Prefect restarts the whole loop — which then hits
            # the same band and fails the same way, forever. The checkpoint
            # records nothing, so a later run retries it cleanly.
            try:
                hit_cap = await _drain(gh, db, band_query(band), state, logger)
            except Exception as exc:  # noqa: BLE001 - one band is not the run
                state.failed_bands.append(band)
                logger.warning("Band %s failed (%s) — skipping, will retry next run", band, exc)
                continue

            if plan.languages and hit_cap and state.total < target:
                logger.info("Band %s hit the cap — splitting by language", band)
                for sub in language_queries(band):
                    if state.total >= target:
                        break
                    try:
                        await _drain(gh, db, sub, state, logger)
                    except Exception as exc:  # noqa: BLE001
                        state.failed_bands.append(sub)
                        logger.warning("Sub-query failed (%s) — skipping", exc)
            elif not hit_cap:
                logger.info("Band %s exhausted in one query (%d rows)", band, state.last_shard_rows)

            await _backfill_readmes(gh, db, state, logger)

    logger.info(
        "Ingested %d repos in %d queries (%d shards already complete, "
        "%d READMEs recovered by backfill, %d bands skipped after failure)",
        state.total, state.queries_issued, state.shards_skipped,
        state.readme_recovered, len(state.failed_bands),
    )
    if state.failed_bands:
        logger.warning("Bands to retry next run: %s", state.failed_bands[:10])
    return state.total


async def _backfill_readmes(gh, db: Database, state: _Progress, logger) -> None:
    """Fetch the READMEs the inline pass could not resolve.

    Runs per band rather than once at the end, so an interrupted run still keeps
    the repositories it recovered. Best-effort throughout — a failed backfill
    leaves a repo with an empty README, which is a known state the cleaner
    already flags, not a corrupt one.
    """
    if not state.pending_readmes:
        return
    pending = list(state.pending_readmes.items())
    state.pending_readmes.clear()

    recovered = await gh.fetch_readmes(pending)
    if not recovered:
        return

    rows = await db.rehydrate_readmes(recovered)
    state.readme_recovered += rows
    logger.info("Recovered %d READMEs (symlink targets and unguessed filenames)", rows)


@dataclass
class _Progress:
    """Run-level counters, so the flow can report efficiency, not just totals."""

    target: int
    total: int = 0
    queries_issued: int = 0
    shards_skipped: int = 0
    last_shard_rows: int = 0
    #: full_name -> path, for repositories whose README needs a targeted fetch.
    pending_readmes: dict = field(default_factory=dict)
    readme_recovered: int = 0
    failed_bands: list = field(default_factory=list)


async def _drain(gh, db: Database, query: str, state: _Progress, logger) -> bool:
    """Run one query to exhaustion. Returns True if it hit the result cap.

    Hitting the cap is the signal that the shard is truncated and needs
    splitting; anything less means we have seen everything it holds.
    """
    cursor, seen, completed = await db.resume_point(query)
    if completed:
        state.total += seen
        state.shards_skipped += 1
        state.last_shard_rows = seen
        return seen >= EXPECTED_YIELD

    if cursor:
        logger.info("Resuming %s at %d rows", query, seen)

    remaining = state.target - state.total
    async for batch, next_cursor in gh.search(query, limit=min(SEARCH_RESULT_CAP, remaining), after=cursor):
        state.queries_issued += 1
        rows = [_clean(record) for record in batch]
        written = await db.upsert_repos(rows)
        seen += written
        state.total += written
        for record in batch:
            if getattr(record, "readme_needs_backfill", False) and record.readme_path:
                state.pending_readmes[record.full_name] = record.readme_path

        # Checkpoint AFTER the write, never before. Checkpointing first is how a
        # crash silently loses a page.
        await db.checkpoint(query, next_cursor, rows_seen=seen)

        if state.total >= state.target:
            break

    await db.checkpoint(query, None, rows_seen=seen, completed=True)
    state.last_shard_rows = seen
    return seen >= EXPECTED_YIELD


def _clean(record) -> RepoRow:
    result = clean_readme(
        record.readme,
        name=record.full_name.split("/")[-1],
        description=record.description,
        language=record.language,
        topics=record.topics,
    )
    return RepoRow(
        full_name=record.full_name,
        description=record.description,
        language=record.language,
        topics=record.topics,
        stars=record.stars,
        forks=record.forks,
        open_issues=record.open_issues,
        pushed_at=record.pushed_at,
        created_at=record.created_at,
        license=record.license,
        is_fork=record.is_fork,
        is_archived=record.is_archived,
        readme_raw=record.readme,
        clean_text=result.text,
        embedding_input=result.embedding_input,
        low_signal=result.low_signal,
        non_english=result.non_english,
        clean_reduction=result.reduction,
        dropped_sections=result.dropped_sections,
    )


@task(retries=1, retry_delay_seconds=60, cache_policy=NO_CACHE)
async def enrich_star_velocity(db: Database, settings: Settings) -> int:
    from .ingest.bigquery import BigQueryConfig, BigQueryExtractor, trailing_months

    logger = get_run_logger()
    async with db.pool.acquire() as conn:
        names = [r["full_name"] for r in await conn.fetch("SELECT full_name FROM repo")]
    if not names:
        return 0

    extractor = BigQueryExtractor(BigQueryConfig(project=settings.gcp_project))
    velocity = extractor.star_velocity(names, months=trailing_months(3))
    updated = await db.update_star_velocity(velocity)
    logger.info("Star velocity for %d/%d repos", updated, len(names))
    return updated


@task(retries=1, retry_delay_seconds=60, cache_policy=NO_CACHE)
async def enrich_criticality(db: Database, settings: Settings) -> int:
    from .ingest.criticality import fetch_criticality_scores

    logger = get_run_logger()
    async with db.pool.acquire() as conn:
        names = [r["full_name"] for r in await conn.fetch("SELECT full_name FROM repo")]
    scores = await fetch_criticality_scores(names)
    updated = await db.update_criticality(scores)
    logger.info("Criticality for %d/%d repos", updated, len(names))
    return updated


@task(retries=1, retry_delay_seconds=60, cache_policy=NO_CACHE)
async def ingest_dependency_graph(db: Database, settings: Settings) -> tuple[int, int]:
    from .ingest.bigquery import BigQueryConfig, BigQueryExtractor

    logger = get_run_logger()
    extractor = BigQueryExtractor(BigQueryConfig(project=settings.gcp_project))

    mappings = extractor.package_to_repo()
    packages = await db.upsert_packages(
        (m["ecosystem"], m["package_name"], m["full_name"])
        for m in mappings
        if m.get("full_name")
    )
    logger.info("Mapped %d packages", packages)

    names = await db.package_names()
    edges = await db.upsert_edges(
        (e["ecosystem"], e["src_package"], e["dst_package"], float(e["weight"]))
        for e in extractor.dependency_edges(names)
    )
    logger.info("Wrote %d dependency edges", edges)
    return packages, edges


@flow(name="gitglobe-ingest")
async def ingest_flow(
    target: int = 5_000,
    *,
    skip_bigquery: bool = False,
) -> IngestResult:
    logger = get_run_logger()
    settings = Settings.from_env()

    db = await Database.connect(settings.database_url)
    try:
        await db.migrate()

        repos = await ingest_repositories(db, settings, target)
        packages = edges = 0

        if not skip_bigquery:
            await enrich_star_velocity(db, settings)
            await enrich_criticality(db, settings)
            packages, edges = await ingest_dependency_graph(db, settings)

        report = await db.quality_report()
        logger.info("Quality report: %s", report)
        return IngestResult(repos=repos, edges=edges, packages=packages)
    finally:
        await db.close()
