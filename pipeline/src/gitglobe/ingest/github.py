"""GitHub GraphQL ingest.

GraphQL rather than REST, for one reason: a single query returns 100
repositories *with their READMEs inline*. The REST equivalent is one call for
the search page plus one per repository for the README — 101 requests for the
same 100 rows. At 100k repositories that is the difference between an afternoon
and a week.

Three things make this survivable in practice:

* **A token pool.** GitHub's GraphQL limit is 5,000 points/hour per token. One
  token gets roughly 100k repos/hour; several get you there proportionally
  faster, and the pool rotates away from any token that is exhausted.
* **Checkpoints.** Every page writes its cursor to `ingest_state`. A crash at
  hour three resumes at hour three.
* **Backoff that respects the server.** Secondary rate limits return
  `Retry-After`; ignoring it is how an integration gets blocked outright.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable

import httpx

from .plan import SearchPlan, plan_for_target, star_shards  # noqa: F401  (re-exported)
from .readme_select import INLINE_CANDIDATES, select_readme

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

# READMEs live under several names and cases. GraphQL has no "any of these"
# selector, so each is aliased and the first non-null wins.
# Derived from INLINE_CANDIDATES so there is exactly one list of filenames.
# Which candidate wins is decided in readme_select — NOT by order: a repository
# can carry a stub README.md that symlinks to the real README.rst.
_README_ALIASES = {alias: f"HEAD:{path}" for alias, path in INLINE_CANDIDATES.items()}

# Repositories per page, and the floor it may shrink to.
#
# This is a *cost* dial, not a throughput dial. Each repository on a page also
# resolves its README candidates and its root tree, so a page of 50 asks GitHub
# to open a few hundred files in one query. Push it too far and the answer is a
# 502, which is slower than any page size would have been.
DEFAULT_PAGE_SIZE = max(1, min(100, int(os.getenv("GITHUB_PAGE_SIZE", "50"))))
MIN_PAGE_SIZE = 5


def build_search_query(page_size: int) -> str:
    """The search query at a given page size.

    Built per call rather than once at import, so the client can shrink the page
    when GitHub starts refusing and grow it back when it stops.
    """
    readme_fields = "\n        ".join(
        f'{alias}: object(expression: "HEAD:{path}") {{ ... on Blob {{ text }} }}'
        for alias, path in INLINE_CANDIDATES.items()
    )
    return """
query($q: String!, $after: String) {
  rateLimit { cost remaining resetAt }
  search(query: $q, type: REPOSITORY, first: %d, after: $after) {
    repositoryCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Repository {
        nameWithOwner
        description
        primaryLanguage { name }
        repositoryTopics(first: 20) { nodes { topic { name } } }
        stargazerCount
        forkCount
        issues(states: OPEN) { totalCount }
        pushedAt
        createdAt
        licenseInfo { spdxId }
        isFork
        isArchived
        root_tree: object(expression: "HEAD:") {
          ... on Tree { entries { name type mode } }
        }
        %s
      }
    }
  }
}
""" % (page_size, readme_fields)


#: Fetch specific README paths for repositories the inline pass could not
#: resolve — a symlink target, or a filename only the tree knew about.
def build_backfill_query(items: list[tuple[str, str, str]]) -> str:
    """`items` is [(alias, "owner/name", "path/to/README.md"), ...]."""
    fields = []
    for alias, full_name, path in items:
        owner, name = full_name.split("/", 1)
        escaped = path.replace('"', '\\"')
        fields.append(
            f'{alias}: repository(owner: "{owner}", name: "{name}") {{'
            f' object(expression: "HEAD:{escaped}") {{ ... on Blob {{ text }} }} }}'
        )
    return "query {\n  rateLimit { cost remaining resetAt }\n  " + "\n  ".join(fields) + "\n}"


@dataclass
class RepoRecord:
    full_name: str
    description: str
    language: str
    topics: list[str]
    stars: int
    forks: int
    open_issues: int
    pushed_at: datetime | None
    created_at: datetime | None
    license: str
    is_fork: bool
    is_archived: bool
    readme: str
    #: Which file the README came from, or which one still needs fetching.
    readme_path: str = ""
    #: True when the root tree names a README we could not fetch inline.
    readme_needs_backfill: bool = False


@dataclass
class _Token:
    value: str
    remaining: int = 5000
    reset_at: float = 0.0

    @property
    def usable(self) -> bool:
        return self.remaining > 100 or time.time() >= self.reset_at


class TokenPool:
    """Round-robin over tokens, skipping any that are spent.

    Sleeping on an exhausted token while another sits idle is the most common
    way a multi-token ingest ends up no faster than a single-token one.
    """

    def __init__(self, tokens: Iterable[str]) -> None:
        self._tokens = [_Token(t) for t in tokens if t]
        if not self._tokens:
            raise ValueError("At least one GITHUB_TOKEN is required")
        self._index = 0

    async def acquire(self) -> _Token:
        for _ in range(len(self._tokens)):
            token = self._tokens[self._index % len(self._tokens)]
            self._index += 1
            if token.usable:
                return token

        soonest = min(self._tokens, key=lambda t: t.reset_at)
        wait = max(1.0, soonest.reset_at - time.time())
        log.warning("All %d tokens exhausted; sleeping %.0fs", len(self._tokens), wait)
        await asyncio.sleep(wait)
        return soonest


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _to_record(node: dict[str, Any]) -> RepoRecord:
    choice = select_readme(node)
    topics = [
        t["topic"]["name"]
        for t in (node.get("repositoryTopics") or {}).get("nodes", [])
        if t.get("topic")
    ]
    return RepoRecord(
        full_name=node["nameWithOwner"],
        description=node.get("description") or "",
        language=((node.get("primaryLanguage") or {}) or {}).get("name") or "",
        topics=topics,
        stars=node.get("stargazerCount") or 0,
        forks=node.get("forkCount") or 0,
        open_issues=((node.get("issues") or {}) or {}).get("totalCount") or 0,
        pushed_at=_parse_ts(node.get("pushedAt")),
        created_at=_parse_ts(node.get("createdAt")),
        license=((node.get("licenseInfo") or {}) or {}).get("spdxId") or "",
        is_fork=bool(node.get("isFork")),
        is_archived=bool(node.get("isArchived")),
        readme=choice.text,
        readme_path=choice.path,
        readme_needs_backfill=choice.needs_backfill,
    )


class GitHubIngest:
    def __init__(self, tokens: Iterable[str], *, timeout: float = 60.0) -> None:
        self.pool = TokenPool(tokens)
        # Shrinks on 502/504, recovers slowly on success. A 502 from GitHub's
        # GraphQL endpoint almost always means "that query was too expensive",
        # and retrying the identical query is how a run spends twenty minutes
        # failing the same way.
        self.page_size = DEFAULT_PAGE_SIZE
        self._consecutive_ok = 0
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "GitGlobe-Pipeline/0.1"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GitHubIngest":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _post(self, query: str, variables: dict[str, Any], *, attempt: int = 0) -> dict[str, Any]:
        token = await self.pool.acquire()
        try:
            response = await self._client.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers={"Authorization": f"bearer {token.value}"},
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            if attempt >= 6:
                raise
            await self._backoff(attempt, reason=str(exc))
            return await self._post(query, variables, attempt=attempt + 1)

        # Secondary rate limit. The server tells us how long to wait; ignoring
        # Retry-After is how an integration gets blocked outright.
        if response.status_code in (403, 429):
            retry_after = float(response.headers.get("Retry-After", 0) or 0)
            token.remaining = 0
            token.reset_at = time.time() + max(retry_after, 60)
            if attempt >= 6:
                response.raise_for_status()
            await asyncio.sleep(max(retry_after, 1.0))
            return await self._post(query, variables, attempt=attempt + 1)

        if response.status_code >= 500:
            # Shrink the page before retrying. The query is the problem, so
            # backing off without changing it just fails more slowly.
            if self.page_size > MIN_PAGE_SIZE:
                self.page_size = max(MIN_PAGE_SIZE, self.page_size // 2)
                self._consecutive_ok = 0
                log.warning("HTTP %d — page size down to %d", response.status_code, self.page_size)
            if attempt >= 6:
                response.raise_for_status()
            await self._backoff(attempt, reason=f"HTTP {response.status_code}")
            return await self._post(query, variables, attempt=attempt + 1)

        response.raise_for_status()

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            if attempt >= 6:
                raise
            await self._backoff(attempt, reason=f"JSON Decode Error: {exc}")
            return await self._post(query, variables, attempt=attempt + 1)

        # Recover the page size gradually. Jumping straight back to the full
        # size after one success just re-triggers the failure.
        self._consecutive_ok += 1
        if self._consecutive_ok >= 5 and self.page_size < DEFAULT_PAGE_SIZE:
            self.page_size = min(DEFAULT_PAGE_SIZE, self.page_size * 2)
            self._consecutive_ok = 0
            log.info("Recovered — page size up to %d", self.page_size)

        limit = (payload.get("data") or {}).get("rateLimit")
        if limit:
            token.remaining = limit["remaining"]
            reset = _parse_ts(limit["resetAt"])
            token.reset_at = reset.timestamp() if reset else time.time() + 3600

        if payload.get("errors"):
            messages = [e.get("message", "") for e in payload["errors"]]
            # A partial response with per-node errors is normal — a repository
            # can vanish between the search index and the fetch. Only bail when
            # nothing came back at all.
            if payload.get("data") is None:
                raise RuntimeError(f"GraphQL error: {messages}")
            log.debug("Partial GraphQL errors: %s", messages[:3])

        return payload["data"]

    @staticmethod
    async def _backoff(attempt: int, *, reason: str) -> None:
        """Exponential, with jitter added rather than multiplied.

        Full jitter (`base * random()`) can return almost zero, which is what
        the first version did — the logs showed retries 0.3s apart against a
        service that was already returning 502. Half the base plus half of it
        jittered keeps the delay meaningful while still de-synchronising
        concurrent workers.
        """
        base = min(15.0, 2.0 ** (attempt + 1))
        delay = base * (0.5 + 0.5 * random.random())
        log.warning("Retry %d in %.1fs (%s)", attempt + 1, delay, reason)
        await asyncio.sleep(delay)

    async def search(
        self,
        query: str,
        *,
        limit: int,
        after: str | None = None,
    ) -> AsyncIterator[tuple[list[RepoRecord], str | None]]:
        """Yield (batch, cursor) pairs.

        The cursor is yielded *with* its batch so the caller can checkpoint only
        after the rows are durably written. Checkpointing before the write is
        how you silently lose a page on the next crash.

        GitHub's search API caps any single query at 1,000 results, which is why
        the flow shards by star range rather than paginating one query to 100k.
        """
        seen = 0
        cursor = after
        # Power of 10 rule 2. `seen < limit` only terminates if `seen` grows,
        # and it does not when the API returns an empty page — which it can do
        # under load while still reporting `hasNextPage`. Without a page cap
        # that is an unbounded loop against a rate-limited paid API.
        # SEARCH_RESULT_CAP/page_size is the most pages that can ever be useful.
        pages = 0
        max_pages = max(1, -(-limit // max(1, self.page_size))) + 2

        while seen < limit and pages < max_pages:
            pages += 1
            data = await self._post(build_search_query(self.page_size), {"q": query, "after": cursor})
            search = data["search"]
            nodes = [n for n in search["nodes"] if n]
            batch = [_to_record(n) for n in nodes]
            seen += len(batch)
            cursor = search["pageInfo"]["endCursor"]

            yield batch, cursor

            if not search["pageInfo"]["hasNextPage"] or not batch:
                return


    async def fetch_readmes(self, items: list[tuple[str, str]], *, batch: int = 25) -> dict[str, str]:
        """Fetch specific README paths. `items` is [(full_name, path), ...].

        The second pass. Repositories arrive here for one of two reasons: their
        root README is a symlink into a monorepo package (zod, vuetify, unocss,
        certbot), or the file is spelled in a way the three inline candidates do
        not cover. Both are recoverable, and both are significant repositories —
        dropping them would quietly bias the corpus against monorepos.
        """
        out: dict[str, str] = {}
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            aliased = [(f"r{i}", full_name, path) for i, (full_name, path) in enumerate(chunk)]
            try:
                data = await self._post(build_backfill_query(aliased), {})
            except Exception as exc:
                log.warning("README backfill batch failed (%s) — continuing", exc)
                continue
            for alias, full_name, _path in aliased:
                blob = ((data.get(alias) or {}).get("object") or {})
                text = blob.get("text") or ""
                if text:
                    out[full_name] = text
        return out
