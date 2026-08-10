"""Rank against all of GitHub, not against the 87,000 repositories we happen to hold.

Every number in this pipeline so far has been a **within-corpus percentile**.
"Top 2%" has meant top 2% of what Phase 1 chose to ingest — and Phase 1 sharded
by star bands starting around 50 stars, which is already the top fraction of a
percent of GitHub. So a repository sitting at the *bottom* of our corpus is
still, globally, unusually popular. Presenting an in-corpus percentile as a rank
is not a small distortion; it is off by orders of magnitude, and it is off in the
flattering direction.

**The distribution is measured, not modelled.** GitHub's search API reports
`total_count` for any query, so `stars:>=1000` returns the exact number of public
repositories with at least a thousand stars. Walking a ladder of thresholds gives
the empirical survival function directly — about thirty requests, free, no
scraping, no assumed power law. Where a power law *is* used it is only to
interpolate between measured rungs, and the rungs are dense enough that the
interpolation barely matters.

That distinction is the whole point. The published figure for GitHub's size
(~420M public repositories) is a citation; the shape of the star distribution is
not published anywhere. Fitting an invented Pareto to it would produce
confident-looking ranks with nothing underneath.

What this module does NOT claim: that stars measure quality. They do not, which
is why `composite_score` blends the measured star percentile with dependent
count and the brain's absolute rubric. The star ladder supplies the *scale* — a
way to say "roughly one in fifty thousand" — and the other signals supply the
judgement.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: Public repositories on GitHub. Used only to turn a percentile into "#N of M",
#: which is a presentational choice, not an input to any score. Verify against
#: https://github.com/search and the GitHub blog before quoting it in the UI.
GITHUB_PUBLIC_REPOS = 420_000_000

#: Star thresholds to measure. Log-spaced because the distribution is, and dense
#: at the bottom where almost every repository lives — the region that decides
#: whether "top 5%" means anything.
STAR_LADDER = [
    0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 250, 400, 600, 1_000,
    1_500, 2_500, 4_000, 6_000, 10_000, 15_000, 25_000, 40_000, 60_000,
    100_000, 150_000, 250_000,
]

#: GitHub's search API caps `total_count` reporting for very broad queries. A
#: rung that returns this is a ceiling, not a measurement, and is discarded.
SEARCH_COUNT_CEILING = 1_000_000_000


@dataclass
class StarScale:
    """Measured survival function: how many public repos have >= N stars."""

    #: Ascending star thresholds.
    thresholds: list = field(default_factory=list)
    #: `counts[i]` = repositories with at least `thresholds[i]` stars.
    counts: list = field(default_factory=list)
    total_repos: int = GITHUB_PUBLIC_REPOS
    measured_at: str = ""

    def validate(self) -> None:
        if len(self.thresholds) != len(self.counts):
            raise ValueError(
                f"thresholds/counts differ: {len(self.thresholds)}, {len(self.counts)}"
            )
        if len(self.thresholds) < 3:
            raise ValueError("need at least three rungs to interpolate between")
        if self.thresholds != sorted(self.thresholds):
            raise ValueError("thresholds must ascend")
        # A survival function only ever decreases. If it does not, a rung was
        # rate-limited or truncated and the whole scale is untrustworthy.
        for i in range(1, len(self.counts)):
            if self.counts[i] > self.counts[i - 1]:
                raise ValueError(
                    f"count rises at {self.thresholds[i]} stars "
                    f"({self.counts[i - 1]} -> {self.counts[i]}); a rung failed"
                )

    def repos_at_least(self, stars: float) -> float:
        """Estimated number of public repos with at least `stars` stars.

        Log-log linear between measured rungs, which is exact for a power law
        and close enough between rungs this dense for anything else.
        """
        self.validate()
        stars = max(0.0, float(stars))
        if stars <= self.thresholds[0]:
            return float(self.counts[0])
        if stars >= self.thresholds[-1]:
            # Extrapolating past the top rung would invent precision about the
            # handful of repositories above it. Hold the last measurement.
            return float(self.counts[-1])

        i = bisect.bisect_right(self.thresholds, stars) - 1
        x0, x1 = self.thresholds[i], self.thresholds[i + 1]
        y0, y1 = self.counts[i], self.counts[i + 1]
        if y0 <= 0 or y1 <= 0:
            return float(y0)
        # +1 keeps log defined at the zero-star rung.
        lx0, lx1, lx = math.log(x0 + 1), math.log(x1 + 1), math.log(stars + 1)
        span = lx1 - lx0
        if span <= 0:
            return float(y0)
        t = (lx - lx0) / span
        return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))

    def percentile(self, stars: float) -> float:
        """Fraction of all public repositories this beats, in [0, 1]."""
        above = self.repos_at_least(stars)
        return max(0.0, min(1.0, 1.0 - above / max(self.total_repos, 1)))

    def rank_of(self, stars: float) -> int:
        """Approximate global position — 1 is the most-starred repository."""
        return max(1, int(round(self.repos_at_least(stars))))

    def describe(self, stars: float) -> str:
        """How this reads in the UI."""
        rank = self.rank_of(stars)
        share = rank / max(self.total_repos, 1)
        if share < 1e-6:
            rarity = f"top 1 in {int(1 / max(share, 1e-12)):,}"
        else:
            rarity = f"top {share * 100:.3g}%"
        return f"#{rank:,} of ~{self.total_repos // 1_000_000}M · {rarity}"

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "counts": self.counts,
            "totalRepos": self.total_repos,
            "measuredAt": self.measured_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StarScale":
        scale = cls(
            thresholds=list(data["thresholds"]),
            counts=list(data["counts"]),
            total_repos=int(data.get("totalRepos", GITHUB_PUBLIC_REPOS)),
            measured_at=str(data.get("measuredAt", "")),
        )
        scale.validate()
        return scale


def monotonic_repair(thresholds: list, counts: list) -> tuple[list, list]:
    """Drop rungs that break monotonicity, keeping the measurement honest.

    A survival function cannot rise. When it appears to, the cause is a
    rate-limited or truncated response, not a real feature of GitHub — and
    silently smoothing it over would bake a failed request into every rank the
    scale ever produces. Dropping the rung loses a little resolution and keeps
    the remaining measurements true.
    """
    if len(thresholds) != len(counts):
        raise ValueError(f"lengths differ: {len(thresholds)}, {len(counts)}")

    kept_t: list = []
    kept_c: list = []
    for threshold, count in sorted(zip(thresholds, counts)):
        if count <= 0 or count >= SEARCH_COUNT_CEILING:
            continue
        if kept_c and count > kept_c[-1]:
            log.warning(
                "Discarding rung at %d stars: count %d exceeds the previous "
                "rung's %d, so the request was truncated.",
                threshold, count, kept_c[-1],
            )
            continue
        kept_t.append(int(threshold))
        kept_c.append(int(count))
    return kept_t, kept_c


@dataclass
class Weights:
    """How much each absolute signal contributes.

    These are a product judgement, not a measurement, and they are stated here
    rather than scattered so they can be argued with and tuned in one place.

    `stars` is the smallest weight that still anchors the scale. It is the only
    signal with a measured global distribution, so it sets the units — but it is
    also the signal this project exists to move past, and giving it the largest
    share would rebuild GitHub's own ranking with extra steps.
    """

    stars: float = 0.25
    dependents: float = 0.35
    pagerank: float = 0.20
    criticality: float = 0.10
    velocity: float = 0.10

    def total(self) -> float:
        return self.stars + self.dependents + self.pagerank + self.criticality + self.velocity


def dependents_percentile(count: float) -> float:
    """Absolute scale for "how many things depend on this".

    Dependent counts are power-law distributed and — unlike a within-corpus
    percentile — mean the same thing regardless of who is measuring. Five
    hundred dependents is five hundred dependents.

    The anchors are deliberately coarse because the underlying claim is coarse:
    1 dependent is meaningfully different from 0, 10 from 1, and 10,000 from
    1,000, but 5,300 and 5,400 are not.
    """
    if count <= 0:
        return 0.0
    # log10 mapped so 1 -> 0.30, 10 -> 0.50, 100 -> 0.70, 1k -> 0.85, 10k -> 0.95
    anchors = [(1, 0.30), (10, 0.50), (100, 0.70), (1_000, 0.85), (10_000, 0.95), (100_000, 1.0)]
    if count <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if count <= x1:
            t = (math.log10(count) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
            return y0 + t * (y1 - y0)
    return 1.0
