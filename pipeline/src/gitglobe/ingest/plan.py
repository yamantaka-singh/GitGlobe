"""Search planning — which queries to run, in what order.

Pure logic, no I/O, no third-party imports. Separated from `github.py` so the
part with real edge cases can be tested without a network, a token, or httpx
installed.

## The 1,000-result wall

GitHub's search API returns at most 1,000 results for any query, however you
paginate. Asking for 100,000 repositories therefore means asking ~100 different
questions, not one question a hundred times.

## Why the cross product was wrong

The first version precomputed every (star band x language) pair up front. A live
run showed why that fails: **47% of completed shards returned zero rows**, mean
yield 22.5 per query, projecting to ~18,700 repositories against a target of
100,000.

The reason is that the two axes are not independent in density. At 400,000+
stars there are a few dozen repositories total, so 25 of the 26 language queries
for that band are guaranteed empty. The cross product pays full price for them
anyway.

The fix is to **split on demand**. Run the band unfiltered first; only if it
comes back at the 1,000-result cap — meaning it is genuinely truncated — is
there anything for the language axis to recover. Dense bands get split, sparse
bands cost one query.

## Why one axis is not enough

The obvious axis is stars: monotonic, cheap to filter, roughly power-law
distributed, so bands can widen geometrically. But it runs out. Geometric bands
from 50 to 400,000 stars give only about 30 shards — a ceiling of 30,000
repositories, well short of 100k, and nowhere near the 1M the project targets.

Narrowing the bands does not fix it. Below a few hundred stars the population is
in the millions, so a band would blow through the cap however thin you slice it.

So the plan is two-dimensional: **stars x language**. Language is a near
partition of the corpus, roughly 25 values cover the vast majority of
repositories, and it is orthogonal to stars — which is exactly what a second
sharding axis needs to be. 30 star bands x 26 language values is ~780 queries
and a ceiling of 780,000.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

#: GitHub will not return more than this for a single query.
SEARCH_RESULT_CAP = 1_000

#: Assume a shard yields somewhat less than the cap — bands are never perfectly
#: sized, and planning against the theoretical maximum leaves no margin.
EXPECTED_YIELD = 800

#: The languages that carry most of GitHub. `None` is the catch-all for repos
#: with no detected language, which is a large and genuinely distinct group
#: (documentation, dotfiles, awesome-lists, data).
LANGUAGE_SHARDS: tuple[str, ...] = (
    "JavaScript", "Python", "Java", "TypeScript", "C++", "C", "C#", "PHP", "Go",
    "Ruby", "Rust", "Kotlin", "Swift", "Shell", "Scala", "R", "Dart", "Lua",
    "Perl", "Haskell", "Elixir", "Julia", "HTML", "CSS", "Jupyter Notebook",
)


def star_shards(
    low: int = 50,
    high: int = 400_000,
    step_factor: float = 1.35,
) -> list[str]:
    """Contiguous, non-overlapping star bands, ascending.

    `low` defaults to 50 rather than 0 deliberately. Below roughly 50 stars the
    population is tens of millions and the READMEs are overwhelmingly empty or
    templated — the band would consume the entire ingest budget and return
    almost nothing but low-signal rows.
    """
    if low < 1 or high <= low:
        raise ValueError(f"Need 1 <= low < high, got low={low} high={high}")
    if step_factor <= 1.0:
        raise ValueError(f"step_factor must exceed 1.0, got {step_factor}")

    shards: list[str] = []
    lower = low
    while lower < high:
        upper = max(lower + 1, int(lower * step_factor))
        if upper >= high:
            break
        shards.append(f"stars:{lower}..{upper - 1}")
        lower = upper
    shards.append(f"stars:{lower}..{high - 1}")
    shards.append(f"stars:>={high}")
    return shards


def parse_shard(shard: str) -> tuple[int, int | None]:
    """`(low, high_inclusive)`; `high` is None for the open-ended top band."""
    if (m := re.fullmatch(r"stars:(\d+)\.\.(\d+)", shard)):
        return int(m.group(1)), int(m.group(2))
    if (m := re.fullmatch(r"stars:>=(\d+)", shard)):
        return int(m.group(1)), None
    raise ValueError(f"Unrecognised shard: {shard!r}")


def band_query(band: str, base_filters: str = "is:public archived:false") -> str:
    return f"{base_filters} {band} sort:stars-desc"


def language_queries(band: str, base_filters: str = "is:public archived:false") -> list[str]:
    """Sub-queries for a band that hit the result cap.

    The unfiltered query has already been run at this point, so it is not
    repeated — but repositories with no detected language were only reachable
    through it, and those are a large group (docs, dotfiles, awesome-lists,
    datasets). They are recovered by the `-language:` exclusion query.
    """
    queries = [f'{base_filters} {band} language:"{lang}" sort:stars-desc' for lang in LANGUAGE_SHARDS]
    excluded = " ".join(f'-language:"{lang}"' for lang in LANGUAGE_SHARDS)
    queries.append(f"{base_filters} {band} {excluded} sort:stars-desc")
    return queries


@dataclass
class SearchPlan:
    """The full set of queries for a run.

    Use :func:`plan_for_target` rather than constructing directly — it picks the
    axes needed to actually reach the number you asked for, instead of silently
    returning a third of it.
    """

    #: `archived:false` because an archived repository's README describes
    #: something that no longer exists, and it would still occupy a node.
    base_filters: str = "is:public archived:false"
    star_bands: list[str] = field(default_factory=star_shards)
    #: Empty means "do not shard by language" — one query per star band.
    languages: tuple[str, ...] = ()

    def bands(self) -> list[str]:
        """Star bands, descending.

        Most important repositories first, so a run cut short still holds the
        repos that matter rather than a random slice of the long tail — which is
        what makes the 5k proof run a useful dataset rather than a sample.
        """
        return list(reversed(self.star_bands))

    def queries(self) -> list[str]:
        """Every query the plan *could* issue. Upper bound, not a schedule.

        The live run issues far fewer: language sub-queries are only reached for
        bands that actually hit the result cap.
        """
        out: list[str] = []
        for band in self.bands():
            out.append(band_query(band, self.base_filters))
            if self.languages:
                out.extend(language_queries(band, self.base_filters))
        return out

    @property
    def reachable(self) -> int:
        """Realistic upper bound, using EXPECTED_YIELD rather than the raw cap."""
        return len(self.queries()) * EXPECTED_YIELD

    def covers(self, target: int) -> bool:
        return self.reachable >= target


def plan_for_target(target: int, *, low: int = 50, high: int = 400_000) -> SearchPlan:
    """The cheapest plan that can actually return `target` repositories.

    Adds axes only when needed. Sharding by language multiplies the query count
    by 26, and each query is a round trip — so a 5,000-repo proof run should not
    pay for it.
    """
    bands = star_shards(low=low, high=high)

    plain = SearchPlan(star_bands=bands)
    if plain.covers(target):
        return plain

    with_language = SearchPlan(star_bands=bands, languages=LANGUAGE_SHARDS)
    if with_language.covers(target):
        return with_language

    # Beyond roughly 780k, stars x language is exhausted too and a third axis
    # (creation-date ranges) is required. Say so plainly rather than returning a
    # plan that quietly delivers half of what was asked for.
    raise ValueError(
        f"Cannot reach {target:,} repositories with stars x language "
        f"(ceiling ~{with_language.reachable:,}). Add a third sharding axis — "
        f"created-date ranges are the usual next one."
    )


def estimated_queries(target: int) -> int:
    """How many API round trips a target implies, for cost estimation."""
    return math.ceil(target / EXPECTED_YIELD)
