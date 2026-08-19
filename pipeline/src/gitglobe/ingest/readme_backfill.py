"""Fill in repositories the corpus knows about but has no text for.

Two different holes, one pass:

**Rows with no `clean_text`.** `teach` selects on `clean_text IS NOT NULL`, so
with the column empty the brain has nothing to read and every summary is absent.
This is the state a database restored from vectors or from a metadata-only dump
is in.

**Nodes with no row at all.** The rendered globe carries 198,731 named nodes
while `repo` and Qdrant hold 87,227 — a sampled check found 81 of 200 tile names
present. The other ~111,000 draw on the globe, respond to a click, and then fall
through `/repo` to the unauthenticated GitHub API at 60 requests an hour, so they
show nothing. The tile name files are the authority on what the globe displays,
so they are the work list here rather than the `repo` table.

**GraphQL, batched, because REST cannot afford this.** REST's `/readme` costs one
request per repository against 5,000/hour: 111,000 rows is seventeen hours. A
GraphQL query carrying 25 aliased `repository` lookups costs **1 point** total —
measured against the `rateLimit` field the query returns, not assumed — which
puts the whole corpus inside a couple of hours' budget on one token.

**Multiple tokens multiply that budget, not just parallelism.** GitHub's 5,000
points/hour is scoped per authenticated account, verified here by hitting
`/user` with each of the three configured tokens and getting three different
logins back. Three tokens is three independent 5,000-point buckets, so the same
run finishes in a third of the time. `TokenPool` already exists for exactly this
— the normal `ingest` path uses it — so it is reused rather than re-implemented
here; a second round-robin over the same tokens would be the kind of duplicate
this codebase already had to fix once (`Database.rehydrate_readmes`).

Nothing is parsed or stored here. The node shape matches `build_search_query`
exactly, so `_to_record` → `_clean` → `upsert_repos` is the same chain a normal
ingest runs: it creates the missing rows with full metadata *and* fills
`clean_text` on the existing ones, in one path, with `content_hash` computed by
the function Phase 2 checks against.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
from dataclasses import dataclass

from .github import TokenPool
from .readme_select import INLINE_CANDIDATES

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

#: 30 aliases still cost 1 point, so the ceiling is response *size*, not rate
#: limit: every alias can carry a full README and 100 of them drew a 502 from
#: GitHub's gateway. 25 leaves headroom for unusually large READMEs.
BATCH = 25

#: Concurrent in-flight queries. GitHub throttles abusive bursts separately from
#: the points budget and these responses are large; four is polite.
DEFAULT_CONCURRENCY = 4

#: Write every this many repositories rather than once at the end. At corpus
#: scale the accumulate-then-store version held roughly 700 MB of README text in
#: memory and lost all of it if anything failed. Flushing also makes the run
#: resumable in the only way that matters — work already paid for stays paid for.
CHECKPOINT_EVERY = 2_000


@dataclass
class BackfillStats:
    attempted: int = 0
    fetched: int = 0
    missing: int = 0
    failed: int = 0
    stored: int = 0
    points_used: int = 0
    #: Fetched a README but the cleaner reduced it to nothing — usually a page
    #: that is almost entirely links. Counted separately because it is the
    #: difference between "we got text" and "`teach` can read it", and reporting
    #: only `fetched` made a run that stored no usable text look successful.
    emptied: int = 0

    def summary(self) -> str:
        return (
            f"{self.stored:,} stored of {self.attempted:,} attempted "
            f"({self.fetched:,} with usable text, {self.emptied:,} cleaned to nothing, "
            f"{self.missing:,} gone or no README, {self.failed:,} failed, "
            f"{self.points_used:,} rate-limit points)"
        )


def tile_names(tiles: pathlib.Path) -> list[str]:
    """Every repository the globe draws, most significant first.

    The `names-N.json` files are already ordered by tier — `names-0` is the top
    few thousand — so reading them in order gives an importance ordering for
    free. That matters because coverage is partial by design and the
    repositories someone is most likely to open should be filled first.
    """
    names: list[str] = []
    for path in sorted(tiles.glob("names-*.json")):
        data = json.loads(path.read_text())
        names.extend(data if isinstance(data, list) else list(data.get("names", data)))
    return [n for n in names if isinstance(n, str) and "/" in n]


async def pending(db, tiles: pathlib.Path | None, limit: int) -> list[str]:
    """What still needs fetching, in priority order.

    **Done means "we fetched and cleaned it", not "it has text".** `content_hash`
    is written by `upsert_repos` for every row that goes through the chain, so it
    is the honest marker. Keying on a non-empty `clean_text` instead looks
    equivalent and is not: the cleaner legitimately reduces some READMEs to
    nothing — `_drop_link_dumps` empties a README that is mostly links, which is
    159 of the first 6,127 processed here — and those rows would be handed back
    on every run, re-fetched, re-emptied, and re-stored forever. The job would
    never converge and each re-run would spend rate limit re-learning the same
    thing.

    Falls back to the `repo` table when the tiles are not on disk, so this still
    does something useful in a checkout without a built globe.
    """
    async with db.pool.acquire() as conn:
        done = {
            r["full_name"]
            for r in await conn.fetch("SELECT full_name FROM repo WHERE content_hash IS NOT NULL")
        }
        names = tile_names(tiles) if tiles and tiles.is_dir() else [
            r["full_name"] for r in await conn.fetch(
                "SELECT full_name FROM repo WHERE content_hash IS NULL ORDER BY stars DESC"
            )
        ]

    seen: set[str] = set()
    todo = []
    for name in names:
        if name in done or name in seen:
            continue
        seen.add(name)
        todo.append(name)
        if len(todo) >= limit:
            break
    return todo


def build_query(names: list[str]) -> str:
    """One query fetching full metadata plus README candidates for many repos.

    Deliberately the same node shape as `build_search_query`, so `_to_record`
    and `select_readme` read the result without knowing which produced it —
    including the symlink handling that keeps monorepos from being dropped.
    """
    readme_fields = " ".join(
        f'{alias}: object(expression: "HEAD:{path}") {{ ... on Blob {{ text }} }}'
        for alias, path in INLINE_CANDIDATES.items()
    )
    parts = []
    for i, full_name in enumerate(names):
        owner, _, name = full_name.partition("/")
        owner = owner.replace("\\", "\\\\").replace('"', '\\"')
        name = name.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(
            f'r{i}: repository(owner: "{owner}", name: "{name}") {{'
            f" nameWithOwner description primaryLanguage {{ name }}"
            f" repositoryTopics(first: 20) {{ nodes {{ topic {{ name }} }} }}"
            f" stargazerCount forkCount issues(states: OPEN) {{ totalCount }}"
            f" pushedAt createdAt licenseInfo {{ spdxId }} isFork isArchived"
            f' root_tree: object(expression: "HEAD:") {{ ... on Tree {{ entries {{ name type mode }} }} }}'
            f" {readme_fields} }}"
        )
    return "query {\n rateLimit { cost remaining resetAt }\n " + "\n ".join(parts) + "\n}"


async def backfill(db, tokens, *, limit: int, tiles: pathlib.Path | None = None,
                   concurrency: int = DEFAULT_CONCURRENCY) -> BackfillStats:
    """Fetch and store metadata and READMEs for repositories still missing them.

    `tokens` is one or more GitHub tokens. Each gets its own 5,000 point/hour
    budget — verified by checking `/user` returns a different login per token
    here, not assumed — so `TokenPool` (already used by the normal `ingest`
    path) round-robins across them and only sleeps once every token is spent.
    """
    import httpx

    from ..flow import _clean
    from .github import _parse_ts, _to_record

    names = await pending(db, tiles, limit)
    if not names:
        return BackfillStats()

    pool = TokenPool([tokens] if isinstance(tokens, str) else tokens)
    stats = BackfillStats(attempted=len(names))
    batches = [names[i:i + BATCH] for i in range(0, len(names), BATCH)]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    buffer: list = []

    async def flush() -> None:
        """Persist and clear. Caller holds the lock."""
        if not buffer:
            return
        rows, buffer[:] = list(buffer), []
        stats.stored += await db.upsert_repos(rows)
        log.info("checkpoint: %s stored so far", f"{stats.stored:,}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def one(batch: list[str]) -> None:
            # Every failure is contained here. `gather` propagates the first
            # exception and cancels its siblings, so one bad batch would
            # otherwise discard everything not yet checkpointed.
            try:
                token = await pool.acquire()
                async with semaphore:
                    response = await client.post(
                        GRAPHQL_URL,
                        headers={"Authorization": f"bearer {token.value}"},
                        json={"query": build_query(batch)},
                    )
                if response.status_code != 200:
                    stats.failed += len(batch)
                    log.warning("batch of %d: HTTP %s", len(batch), response.status_code)
                    return
                data = (response.json() or {}).get("data") or {}
                if not data:
                    stats.failed += len(batch)
                    return

                # Same fields `GitHubIngest._post` reads, so this token's budget
                # is tracked exactly like the normal ingest path's — the next
                # `pool.acquire()` sees it and skips this token once it is spent
                # rather than the whole run stopping over one exhausted token.
                rate = data.get("rateLimit") or {}
                stats.points_used += int(rate.get("cost") or 0)
                if "remaining" in rate:
                    token.remaining = rate["remaining"]
                    reset = _parse_ts(rate.get("resetAt"))
                    token.reset_at = reset.timestamp() if reset else time.time() + 3600

                fresh = []
                for i in range(len(batch)):
                    node = data.get(f"r{i}")
                    # A null node is a repository that was renamed, deleted or
                    # made private since the layout was built. Nothing to do.
                    if not node or not node.get("nameWithOwner"):
                        stats.missing += 1
                        continue
                    record = _to_record(node)
                    row = _clean(record)
                    # Counted on the cleaned text, not the raw README. `teach`
                    # selects on `clean_text`, so a raw README the cleaner
                    # emptied is not progress and must not be reported as any.
                    if row.clean_text:
                        stats.fetched += 1
                    elif record.readme:
                        stats.emptied += 1
                    else:
                        stats.missing += 1
                    fresh.append(row)

                async with lock:
                    buffer.extend(fresh)
                    if len(buffer) >= CHECKPOINT_EVERY:
                        await flush()
            except Exception as exc:
                stats.failed += len(batch)
                log.warning("batch of %d failed: %s", len(batch), exc)

        await asyncio.gather(*(one(b) for b in batches))

    async with lock:
        await flush()
    return stats
