"""Measure GitHub's star distribution, then score every repository against it.

Thirty search requests, free, inside the normal rate limit. The result is the
empirical survival function — the exact number of public repositories at or above
each star threshold — which is what turns "top 2% of our corpus" into "roughly
one in fifty thousand repositories on GitHub".

Re-measure every few months. The distribution drifts slowly, but a scale
measured in 2024 quoted as fact in 2027 is worse than no scale, because it looks
just as authoritative.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .global_scale import (
    GITHUB_PUBLIC_REPOS,
    STAR_LADDER,
    StarScale,
    Weights,
    dependents_percentile,
    monotonic_repair,
    star_magnitude,
)

log = logging.getLogger(__name__)

SEARCH_URL = "https://api.github.com/search/repositories"

#: GitHub's search endpoint allows 30 authenticated requests per minute — a
#: different, much tighter budget than the 5,000/hour core limit. The ladder is
#: 29 rungs, so one pass fits in a minute with room to retry.
SEARCH_REQUESTS_PER_MINUTE = 28
MAX_RETRIES = 4


async def measure_star_scale(
    tokens: list[str],
    ladder: list | None = None,
    *,
    total_repos: int = GITHUB_PUBLIC_REPOS,
) -> StarScale:
    """Walk the ladder and record how many repositories clear each threshold."""
    import httpx

    ladder = ladder or STAR_LADDER
    if not tokens:
        raise ValueError("a GitHub token is required; search is not anonymous-friendly")

    thresholds: list = []
    counts: list = []
    interval = 60.0 / SEARCH_REQUESTS_PER_MINUTE

    async with httpx.AsyncClient(timeout=30.0) as client:
        for index, threshold in enumerate(ladder):
            # Sequential and paced on purpose. Search is 30 requests/minute, and
            # a burst earns a 403 that costs more time than the pacing.
            if index:
                await asyncio.sleep(interval)
            count = await _count_at_least(client, tokens[index % len(tokens)], threshold)
            if count is None:
                log.warning("No count for stars>=%d; skipping that rung", threshold)
                continue
            thresholds.append(threshold)
            counts.append(count)
            # `,` is a str.format/f-string grouping option, NOT a printf flag.
            # "%15,d" raises inside logging, which swallows it as a handler
            # error and prints a traceback per rung instead of the measurement.
            log.info("stars >= %-7d %15s repositories", threshold, f"{count:,}")

    # Counts must fall as the threshold rises. Where they do not, a request was
    # truncated — drop it rather than smoothing a failed measurement into the
    # scale, where it would silently distort every rank derived from it.
    thresholds, counts = monotonic_repair(thresholds, counts)

    scale = StarScale(
        thresholds=thresholds,
        counts=counts,
        total_repos=total_repos,
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    scale.validate()
    return scale


async def _count_at_least(client, token: str, threshold: int) -> int | None:
    """`total_count` for `stars:>=threshold`. None if it could not be read."""
    import httpx

    params = {"q": f"stars:>={threshold}", "per_page": 1}
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(SEARCH_URL, params=params, headers=headers)
            if response.status_code == 200:
                return int(response.json().get("total_count", 0))
            # 403 here is secondary rate limiting, not authorisation.
            if response.status_code in (403, 429):
                wait = float(response.headers.get("retry-after", 0)) or min(2**attempt, 30)
                log.info("Search rate limited; waiting %.0fs", wait)
                await asyncio.sleep(wait)
                continue
            log.warning("stars>=%d returned HTTP %d", threshold, response.status_code)
            return None
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            await asyncio.sleep(min(2**attempt, 30) * (0.5 + 0.5 * random.random()))
            log.debug("Retry %d after %s", attempt + 1, type(exc).__name__)
    return None


@dataclass
class RepoSignals:
    """Everything the composite score reads, all on absolute scales."""

    stars: float = 0.0
    dependents: float = 0.0
    #: PageRank relative to our corpus, expressed as a multiple of the mean so
    #: it does not change meaning when the corpus grows.
    pagerank_ratio: float = 1.0
    criticality: float = 0.0
    stars_90d: float = 0.0


@dataclass
class GlobalRank:
    """A repository's standing among all public repositories."""

    score: float                # 0-100 composite
    star_rank: int              # position by stars alone
    star_percentile: float
    components: dict = field(default_factory=dict)

    def describe(self, scale: StarScale) -> str:
        return f"{self.score:.1f}/100 · {scale.describe(self.star_rank)}"


def composite_score(
    signals: RepoSignals,
    scale: StarScale,
    weights: Weights | None = None,
) -> GlobalRank:
    """Blend absolute signals into one 0-100 score.

    Every component is on a scale that means the same thing outside this corpus:

    * **stars** — measured global percentile, the only component with a real
      empirical distribution behind it. It supplies the units.
    * **dependents** — a raw count. Five hundred is five hundred anywhere.
    * **pagerank_ratio** — a multiple of the corpus mean, which is scale-free in
      a way a raw PageRank value is not: the raw number shrinks as 1/n purely
      because the corpus grew.
    * **criticality** — OSSF's score, already 0-1 and already absolute.
    * **velocity** — stars gained recently, as a share of the global rate.

    Stars carry the *smallest* meaningful weight. They anchor the scale because
    theirs is the distribution we can measure; giving them the largest share
    would reproduce GitHub's own ranking with extra steps, which is the thing
    this project exists to improve on.
    """
    weights = weights or Weights()
    total_weight = weights.total()
    if total_weight <= 0:
        raise ValueError("weights sum to zero")

    # Absolute magnitude, NOT scale.percentile. Every repository in this corpus
    # sits above GitHub's 99.8th percentile, so the percentile spread across the
    # whole corpus was 0.0016 — it added a flat ~25 points and ordered nothing.
    # The scale still supplies `star_rank` below, which is the honest measured
    # number to show a user; it is just not a usable ranking signal in here.
    star_pct = star_magnitude(signals.stars)
    dep_pct = dependents_percentile(signals.dependents)
    # A ratio of 1.0 is exactly average; 100x average maps near the top. log
    # because PageRank, like everything else here, is power-law distributed.
    rank_pct = min(1.0, max(0.0, math.log10(max(signals.pagerank_ratio, 0.01) + 0.1) / 3.0 + 0.34))
    crit = min(1.0, max(0.0, signals.criticality))
    # Same ruler as stars, for the same reason: a percentile of an annualised
    # star rate saturates just as hard. x4 turns 90 days into a yearly rate.
    vel_pct = star_magnitude(signals.stars_90d * 4) if signals.stars_90d else 0.0

    components = {
        "stars": star_pct,
        "dependents": dep_pct,
        "pagerank": rank_pct,
        "criticality": crit,
        "velocity": vel_pct,
    }
    blended = (
        weights.stars * star_pct
        + weights.dependents * dep_pct
        + weights.pagerank * rank_pct
        + weights.criticality * crit
        + weights.velocity * vel_pct
    ) / total_weight

    return GlobalRank(
        score=round(blended * 100, 2),
        # Both of these are the MEASURED scale, not the ruler used above. They
        # are what a user is shown — "#4,312 of ~420M" — and that claim has an
        # empirical survival function behind it. Setting star_percentile from
        # `star_pct` instead, as an earlier version of this did, replaced a
        # measurement with a presentation choice and reported 0.376 for a
        # repository genuinely in the top 0.01%.
        star_rank=scale.rank_of(signals.stars),
        star_percentile=round(scale.percentile(signals.stars), 6),
        components={k: round(v, 4) for k, v in components.items()},
    )
