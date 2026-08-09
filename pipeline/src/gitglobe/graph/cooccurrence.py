"""Co-occurrence relatedness — the "used together" signal.

## The gap this fills

GitGlobe had two ways to relate repositories, and neither captures the thing the
product actually promises.

**Dependency** (deps.dev) answers *"A cannot work without B"*. Precise, but very
sparse: it only exists for packaged software. Awesome-lists, dotfiles, ML model
repos, tutorials, most C++ projects and nearly every Jupyter notebook have no
package identity at all, so they have zero dependency edges. In the first live
ingest, `(none)`, `C++` and `Jupyter Notebook` together were ~15% of repositories
and will be almost entirely edge-less.

**Semantic similarity** (embedding kNN) answers *"A is an alternative to B"*.
The eight nearest neighbours of `express` are eight other web frameworks. Useful
for "show me alternatives", and actively unhelpful for "show me the ecosystem" —
it returns competitors, not collaborators.

The promise — *"LangChain → ChromaDB"* — is neither. Those two are not similar
(one is a framework, one is a database) and there need be no dependency between
them. They are **used together**. That is a behavioural fact, not a textual or
structural one, and it needs its own signal.

## Why PMI and not raw co-occurrence

Count co-occurrences directly and the top result for everything is `react`,
because `react` co-occurs with everything. Popularity swamps association.

Pointwise mutual information divides that out:

    PMI(a, b) = log( P(a, b) / (P(a) * P(b)) )

It asks whether a and b appear together *more than chance would predict*, given
how common each already is. Positive PMI (PPMI, clamped at zero) is the standard
form; negative values mean "these co-occur less than chance", which is real
information but not something to draw an arc for.

## The two sources

* **Co-starring** — from GH Archive `WatchEvent`. Users who starred A also
  starred B. Strong signal, covers every repository regardless of packaging,
  which is exactly where the dependency graph is blind.
* **Co-dependency** — packages appearing together in the same manifest. Narrower,
  but higher precision for the software-stack question.

Pure functions, no I/O. The maths is the part that goes subtly wrong, so it is
the part that is tested.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

#: Users with more baskets than this are dropped entirely. Someone who has
#: starred 5,000 repositories is not expressing a preference, and they would
#: contribute 12.5M pairs on their own.
MAX_BASKET_SIZE = 400

#: Below this, a basket says nothing about association.
MIN_BASKET_SIZE = 2

#: A pair seen fewer times than this is noise, however high its PMI. Two obscure
#: repos sharing one user produces a spectacular PMI score and means nothing.
MIN_PAIR_COUNT = 3


@dataclass(frozen=True)
class RelatedPair:
    a: str
    b: str
    #: Positive pointwise mutual information. Higher = more surprising together.
    ppmi: float
    #: How many baskets contained both. PPMI alone is not enough — see
    #: MIN_PAIR_COUNT.
    count: int

    def normalised(self) -> tuple[str, str]:
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


def basket_weight(size: int) -> float:
    """How much one basket's opinion counts.

    A user who starred 8 repositories is making a strong statement about each;
    a user who starred 300 is browsing. Weighting by `1/log` rather than `1/n`
    is deliberate — `1/n` punishes large baskets so hard that only tiny ones
    matter, and tiny baskets are the noisiest.
    """
    return 1.0 / math.log(size + 1.0)


def pairs_from_basket(items: Sequence[str]) -> Iterator[tuple[str, str]]:
    """Every unordered pair in a basket, deduplicated and canonically ordered."""
    unique = sorted(set(items))
    for i, a in enumerate(unique):
        for b in unique[i + 1 :]:
            yield a, b


def co_occurrence(
    baskets: Iterable[Sequence[str]],
    *,
    max_basket: int = MAX_BASKET_SIZE,
    min_basket: int = MIN_BASKET_SIZE,
) -> tuple[Counter, Counter, float]:
    """Weighted pair and item counts across all baskets.

    Returns `(pair_counts, item_counts, total_weight)`, which is everything PPMI
    needs. Kept separate from the PPMI step so the expensive pass over the data
    happens once and can be checkpointed.
    """
    pair_counts: Counter = Counter()
    item_counts: Counter = Counter()
    total = 0.0

    for basket in baskets:
        unique = set(basket)
        size = len(unique)
        if size < min_basket or size > max_basket:
            continue

        weight = basket_weight(size)
        total += weight
        for item in unique:
            item_counts[item] += weight
        for pair in pairs_from_basket(list(unique)):
            pair_counts[pair] += weight

    return pair_counts, item_counts, total


def ppmi(
    pair_counts: Counter,
    item_counts: Counter,
    total_weight: float,
    *,
    min_pair_count: float = MIN_PAIR_COUNT,
    smoothing: float = 0.75,
) -> list[RelatedPair]:
    """Positive pointwise mutual information for every surviving pair.

    `smoothing` raises the marginal probabilities to a power < 1, which is the
    context-distribution smoothing from Levy & Goldberg (2015). It damps the
    bias PMI has toward rare items: without it, two repositories that appear
    almost nowhere but happen to share a basket score higher than any genuine
    association in the corpus.
    """
    if total_weight <= 0:
        return []

    # Smoothed marginal denominator, computed once.
    smoothed_total = sum(count**smoothing for count in item_counts.values())
    if smoothed_total <= 0:
        return []

    out: list[RelatedPair] = []
    for (a, b), joint in pair_counts.items():
        if joint < min_pair_count:
            continue
        p_ab = joint / total_weight
        p_a = item_counts[a] / total_weight
        p_b = (item_counts[b] ** smoothing) / smoothed_total
        if p_a <= 0 or p_b <= 0:
            continue
        value = math.log(p_ab / (p_a * p_b))
        if value > 0:
            out.append(RelatedPair(a=a, b=b, ppmi=value, count=joint))

    out.sort(key=lambda p: p.ppmi, reverse=True)
    return out


def top_k_per_item(pairs: Sequence[RelatedPair], k: int = 12) -> dict[str, list[RelatedPair]]:
    """Keep each item's strongest associations.

    A global threshold does not work: popular repositories have thousands of
    pairs above any cut, obscure ones have none, and you end up with a graph
    where the hubs are hairballs and the tail is disconnected. Top-k per item
    gives every node the same budget.
    """
    by_item: dict[str, list[RelatedPair]] = defaultdict(list)
    for pair in pairs:
        by_item[pair.a].append(pair)
        by_item[pair.b].append(pair)
    return {
        item: sorted(ps, key=lambda p: p.ppmi, reverse=True)[:k]
        for item, ps in by_item.items()
    }


def mutual_top_k(pairs: Sequence[RelatedPair], k: int = 12) -> list[RelatedPair]:
    """Only pairs that are in *each other's* top k.

    An asymmetric edge — B is one of A's strongest associations but A is nowhere
    near B's — is nearly always popularity leaking back in: every small React
    component library counts `react` among its top associations, and `react`
    counts none of them. Requiring the relationship to be mutual removes that
    whole class of edge, and it is what turns the result from a star graph into
    a structure worth drawing.
    """
    top = top_k_per_item(pairs, k)
    kept: list[RelatedPair] = []
    for pair in pairs:
        if pair in top.get(pair.a, []) and pair in top.get(pair.b, []):
            kept.append(pair)
    return kept
