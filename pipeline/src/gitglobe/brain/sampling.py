"""Choosing which repositories the teacher rates.

This is the quietest place in the brain to get something badly wrong.

The obvious sample is "the top 4,000 by stars". It is also the worst. The
student would then only ever see popular repositories, learn that everything is
production-ready and canonical, and score the tail — which is 99% of GitHub and
the entire reason this product exists — by extrapolating far outside anything it
was trained on. The metrics would look excellent, because the held-out set is
drawn from the same skewed pool.

So the sample is stratified across the axes that actually matter:

* **Popularity band** — log-spaced, because stars are power-law. The teacher
  must see 12-star repositories, and see enough of them.
* **Domain** — the twelve territories. A student that never saw a compiler
  cannot judge one.
* **Activity** — recently pushed versus dormant, so `maintenance` has range to
  learn from rather than one value.

Within a cell the draw is random. Across cells it is deliberately *not*
proportional: proportional allocation reproduces the corpus skew, which is the
thing being corrected for. Rare cells get over-sampled relative to their size,
and the student is told nothing about cell sizes so it cannot learn the
sampling scheme as a shortcut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Upper bounds, log-spaced. The bottom two bands hold most of GitHub and are
#: where the product has to be good, so they are never allowed to be rare in
#: the sample.
STAR_BANDS = [10, 50, 200, 1_000, 5_000, 25_000, math.inf]

#: Days since last push. "Alive", "slowing", "dormant", "abandoned".
ACTIVITY_BANDS = [30, 180, 730, math.inf]


def band_of(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges) - 1


@dataclass
class Stratum:
    key: tuple
    members: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.members)


@dataclass
class SampleResult:
    indices: np.ndarray
    strata: dict
    allocation: dict
    #: Cells that exist in the corpus but could not be sampled from.
    empty_strata: list = field(default_factory=list)

    def summary(self) -> str:
        filled = sum(1 for v in self.allocation.values() if v > 0)
        return (
            f"{len(self.indices):,} sampled across {filled} of "
            f"{len(self.strata)} strata"
        )


def stratify(
    stars: np.ndarray,
    domain: np.ndarray,
    days_since_push: np.ndarray,
) -> dict:
    """Group row indices into (star band, domain, activity band) cells."""
    strata: dict = {}
    for i in range(len(stars)):
        key = (
            band_of(float(stars[i]), STAR_BANDS),
            int(domain[i]),
            band_of(float(days_since_push[i]), ACTIVITY_BANDS),
        )
        strata.setdefault(key, Stratum(key)).members.append(i)
    return strata


def allocate(strata: dict, total: int, *, floor: int = 3) -> dict:
    """How many to draw from each cell.

    Allocation is proportional to the **square root** of cell size, not to size
    itself. Proportional allocation would hand most of the budget to the biggest
    cell and reproduce exactly the skew we are trying to correct; equal
    allocation would over-weight cells holding a handful of repositories.
    Square-root sits between the two and is the standard compromise.

    Every non-empty cell gets at least `floor`, because a cell the teacher never
    sees is a region of the map the student is guessing about.
    """
    if total <= 0 or not strata:
        return {}

    sizes = {k: len(v) for k, v in strata.items() if len(v) > 0}
    if not sizes:
        return {}

    weights = {k: math.sqrt(n) for k, n in sizes.items()}
    weight_total = sum(weights.values())

    allocation = {}
    for key, weight in weights.items():
        want = int(round(total * weight / weight_total))
        # Cell size is the OUTER bound. Writing `max(floor, min(want, size))`
        # lets the floor win over the cap, so a cell holding one repository is
        # allocated three — which the sampler then silently clips, leaving the
        # allocation dict claiming coverage that was never drawn. The coverage
        # report reads that dict and reported 247% sampling of the rarest band.
        allocation[key] = min(max(floor, want), sizes[key])

    # Trim or top up to hit `total` exactly, always taking from and giving to
    # the largest cells so the floor on rare cells survives.
    order = sorted(allocation, key=lambda k: -sizes[k])
    _rebalance(allocation, sizes, order, total, floor)
    return allocation


def _rebalance(
    allocation: dict, sizes: dict, order: list, total: int, floor: int
) -> None:
    """Nudge `allocation` toward summing to `total`, in place.

    Power of 10 rule 2 — every loop needs a bound you can point at. The earlier
    version nested `for` inside `while`, recomputed `sum()` each pass, and
    relied on a `changed` flag. It did terminate, but only by an argument you
    had to reconstruct by reading, and the repeated `sum()` made it quadratic
    in the number of strata.

    Now the running total is carried, so each pass provably moves at least one
    unit or stops, and `max_passes` turns the bound from an inference into a
    fact. Split out of `allocate` because rule 2 pushed that function past rule
    4's page limit — the fix for one rule must not quietly break another.
    """
    current = sum(allocation.values())
    # One unit moved per pass at worst, and the gap can never exceed `total`.
    max_passes = total + len(order) + 1

    for direction, at_limit in (
        (-1, lambda key: allocation[key] <= floor),
        (+1, lambda key: allocation[key] >= sizes[key]),
    ):
        passes = 0
        while passes < max_passes:
            if (current > total) if direction < 0 else (current < total):
                passes += 1
            else:
                break
            moved = False
            for key in order:
                if (current <= total) if direction < 0 else (current >= total):
                    break
                if at_limit(key):
                    continue
                allocation[key] += direction
                current += direction
                moved = True
            if not moved:
                break  # every cell is at its floor, or every cell is full
        if passes >= max_passes:
            raise RuntimeError(
                f"_rebalance hit its {max_passes}-pass ceiling without "
                "converging. Unreachable unless the loop condition is wrong."
            )


def stratified_sample(
    stars: np.ndarray,
    domain: np.ndarray,
    days_since_push: np.ndarray,
    *,
    total: int = 4_000,
    seed: int = 42,
    floor: int = 3,
) -> SampleResult:
    """Pick the rows the teacher will rate."""
    strata = stratify(stars, domain, days_since_push)
    allocation = allocate(strata, total, floor=floor)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key, count in allocation.items():
        members = strata[key].members
        if count >= len(members):
            chosen.extend(members)
        else:
            chosen.extend(rng.choice(members, size=count, replace=False).tolist())

    # Sorted so the teacher's work order is deterministic and resumable: an
    # interrupted run restarts at the same place rather than re-rating a
    # different random subset.
    return SampleResult(
        indices=np.array(sorted(chosen), dtype=np.int64),
        strata=strata,
        allocation=allocation,
        empty_strata=[k for k, v in strata.items() if len(v) == 0],
    )


def plan_teaching(rows: list, already: set, *, total: int = 4_000, seed: int = 42):
    """Which rows the teacher should rate next. Returns `(todo, sample)`.

    Sample from the FULL population, then subtract what is already rated. The
    obvious alternative — filter to unrated rows and sample from those — looks
    equivalent and is not: every resume would draw a fresh stratified sample
    from a shrinking pool, so an interrupted 4,000-row run ends up rating far
    more than 4,000 rows, and the strata drift a little further from the corpus
    on each restart. Both effects are invisible in the output and cost money.

    Pure so this is testable without a database, which is the whole reason it
    lives here rather than inside the stage.
    """
    if not rows:
        return [], None
    sample = stratified_sample(
        np.array([r["stars"] for r in rows], dtype=np.float64),
        np.array([r["domain"] for r in rows], dtype=np.int64),
        np.array([r["days_since_push"] for r in rows], dtype=np.float64),
        total=total, seed=seed,
    )
    chosen = [rows[i] for i in sample.indices]
    return [r for r in chosen if r["id"] not in already], sample


def train_test_split(
    indices: np.ndarray,
    strata: dict,
    *,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the teacher's labels, holding out proportionally within each cell.

    A plain random split leaves the test set with a different mix of cells from
    the training set purely by chance, and at four thousand rows that chance is
    not small. The held-out score would then be measuring the difference in mix
    as much as the model — flattering or damning it at random between runs.
    """
    rng = np.random.default_rng(seed)
    position = {idx: i for i, idx in enumerate(indices)}
    train, test = [], []

    for stratum in strata.values():
        present = [position[m] for m in stratum.members if m in position]
        if not present:
            continue
        shuffled = rng.permutation(present)
        # At least one test row per cell once a cell has two, so held-out
        # coverage matches training coverage.
        n_test = max(1, int(round(len(shuffled) * test_fraction))) if len(shuffled) > 1 else 0
        test.extend(shuffled[:n_test].tolist())
        train.extend(shuffled[n_test:].tolist())

    return np.array(sorted(train), dtype=np.int64), np.array(sorted(test), dtype=np.int64)


def coverage_report(strata: dict, allocation: dict) -> str:
    """What the teacher will and will not see. Read before spending money."""
    lines = [f"{len(strata)} strata, {sum(allocation.values()):,} to rate"]
    by_star: dict = {}
    for (star_band, _, _), stratum in strata.items():
        entry = by_star.setdefault(star_band, [0, 0])
        entry[0] += len(stratum)
        entry[1] += allocation.get(stratum.key, 0)

    labels = ["<10", "10-49", "50-199", "200-999", "1k-5k", "5k-25k", "25k+"]
    lines.append(f"  {'stars':>10} {'corpus':>10} {'sampled':>9} {'rate':>8}")
    for band in sorted(by_star):
        corpus, sampled = by_star[band]
        label = labels[band] if band < len(labels) else str(band)
        lines.append(
            f"  {label:>10} {corpus:>10,} {sampled:>9,} "
            f"{(sampled / corpus if corpus else 0):>7.2%}"
        )
    lines.append(
        "  Sampling rate should FALL as stars rise — the tail is where the "
        "product has to work and where the corpus has the most rows."
    )
    return "\n".join(lines)
