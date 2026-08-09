"""PageRank over the union of `depends_on` and `used_with`.

**Why the union and not dependencies alone.** A dependency graph only contains
repositories that publish a package. Awesome-lists, dotfiles, notebooks,
datasets, most C++ and nearly all research code publish nothing, so a
dependency-only PageRank leaves every one of them sitting at the teleport floor
`(1-d)/n` — mathematically fine, and useless, because it makes tens of thousands
of repositories exactly equally important and therefore unsortable. Node size,
LOD band and label priority all read this number. `used_with` comes from
behaviour rather than packaging, so it reaches those repositories.

`similar_to` is deliberately excluded. It is a kNN graph: every node has roughly
k neighbours by construction, so it carries almost no information about
importance — it would mostly add a uniform smear and dilute the two signals that
mean something.

Implementation notes that matter:

* **Dangling mass is redistributed, not dropped.** A node with no out-edges
  sends its rank nowhere; ignore that and total rank shrinks every iteration, so
  ranks become incomparable across runs of different sizes. Repositories that
  depend on nothing are extremely common, so this is the normal case here, not
  an edge case.
* **Weights are row-normalised.** Raw PPMI on an out-edge is a property of the
  pair, not a share of the source's attention.
* **No scipy.** `np.bincount` over the edge list is the same computation as a
  CSR matvec and keeps the pipeline installable with numpy alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DAMPING = 0.85
DEFAULT_TOLERANCE = 1e-10
DEFAULT_MAX_ITER = 200


@dataclass
class PageRankResult:
    rank: np.ndarray
    iterations: int
    delta: float
    converged: bool

    def top(self, k: int) -> np.ndarray:
        """Indices of the k highest-ranked nodes, best first."""
        k = min(k, len(self.rank))
        idx = np.argpartition(-self.rank, k - 1)[:k] if k else np.empty(0, np.int64)
        return idx[np.argsort(-self.rank[idx])]


def pagerank(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> PageRankResult:
    """Weighted PageRank by power iteration. `rank` sums to 1."""
    if n <= 0:
        return PageRankResult(np.zeros(0), 0, 0.0, True)

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if len(src) != len(dst):
        raise ValueError(f"src has {len(src)} entries, dst has {len(dst)}")
    if len(src) and (src.max() >= n or dst.max() >= n or src.min() < 0 or dst.min() < 0):
        raise ValueError("edge endpoints must be in [0, n)")

    w = np.ones(len(src)) if weight is None else np.asarray(weight, dtype=np.float64)
    if (w < 0).any():
        raise ValueError("negative edge weights make PageRank meaningless")

    out_strength = np.bincount(src, weights=w, minlength=n)
    dangling = out_strength == 0
    # Share of the source's outgoing attention this edge carries.
    share = np.zeros(len(src))
    live = out_strength[src] > 0
    share[live] = w[live] / out_strength[src][live]

    rank = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n

    delta = 0.0
    for iteration in range(1, max_iter + 1):
        # Dangling mass must be collected BEFORE the push, from the current
        # vector. Collect it after and you are redistributing the wrong
        # iteration's rank — which converges to something plausible-looking and
        # subtly wrong.
        leaked = damping * rank[dangling].sum() / n
        contribution = rank[src] * share
        pushed = np.bincount(dst, weights=contribution, minlength=n)

        nxt = teleport + leaked + damping * pushed
        delta = float(np.abs(nxt - rank).sum())
        rank = nxt
        if delta < tolerance:
            return PageRankResult(rank, iteration, delta, True)

    return PageRankResult(rank, max_iter, delta, False)


def combine_edges(
    layers: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge weighted edge layers into one edge list, scaling each layer.

    Each layer is `(src, dst, weight, scale)`. Layers are scaled rather than
    concatenated raw because their weights are not commensurable: a dependency
    edge weight is a count, a `used_with` weight is a PPMI in roughly [0, 8].
    Concatenating them unscaled silently hands whichever has the larger
    numbers control of the ranking.

    Each layer's weights are normalised to mean 1 first, so `scale` means
    "how much this layer counts", not "how big its numbers happen to be".
    """
    if not layers:
        empty_i = np.empty(0, np.int64)
        return empty_i, empty_i, np.empty(0, np.float64)

    all_src, all_dst, all_w = [], [], []
    for src, dst, weight, scale in layers:
        if len(src) == 0:
            continue
        w = np.asarray(weight, dtype=np.float64)
        mean = w.mean()
        if mean > 0:
            w = w / mean
        all_src.append(np.asarray(src, np.int64))
        all_dst.append(np.asarray(dst, np.int64))
        all_w.append(w * scale)

    if not all_src:
        empty_i = np.empty(0, np.int64)
        return empty_i, empty_i, np.empty(0, np.float64)
    return np.concatenate(all_src), np.concatenate(all_dst), np.concatenate(all_w)


def importance_order(rank: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """Indices from most to least important, PageRank first, `tiebreak` second.

    **PageRank alone cannot sort the tail.** Every repository with no in-edges
    lands on exactly the teleport floor `(1-d)/n` — not approximately, exactly.
    On a realistic corpus that is around 80% of all rows sharing one identical
    float. Sorting by rank alone leaves that 80% in arbitrary order, and sizing
    by it renders them all at one radius: a flat, uniform field over most of the
    globe, which is precisely the failure the size channel exists to avoid.

    `tiebreak` is a second signal — stars, criticality — that carries real
    information about exactly those rows. It is used only to order rows that
    PageRank ties, so it never overrides the primary signal.
    """
    rank = np.asarray(rank, dtype=np.float64)
    if tiebreak is None:
        return np.argsort(-rank, kind="stable")
    # lexsort's LAST key is primary, so rank is passed last.
    return np.lexsort((-np.asarray(tiebreak, dtype=np.float64), -rank))


def to_display_size(rank: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """PageRank to a normalised node radius in [0, 1].

    PageRank is power-law distributed over four or five orders of magnitude.
    Map it linearly onto radius and every node below the top hundred renders at
    literally the same size — the field looks uniform and the visual carries no
    information. So: rank-normalise rather than min-max.

    Rank-normalising is what keeps the field evenly spread whatever the
    underlying distribution does. A min-max on the log still bunches, because
    the log of a power law is exponential-ish: measured on a Pareto sample, a
    linear map put 19,992 of 20,000 nodes in the bottom decile, and a log
    min-max still put 65% there.
    """
    n = len(rank)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    order = importance_order(rank, tiebreak)
    percentile = np.empty(n)
    # `order` is best-first, so reverse it: the top node gets 1.0.
    percentile[order] = np.arange(n - 1, -1, -1) / (n - 1)

    # Rows equal on BOTH signals must get the same size — two repositories with
    # identical data rendering at different radii is a visible lie. Rows that
    # differ on the tiebreak are genuinely different and keep their own size.
    keys = np.asarray(rank, np.float64) if tiebreak is None else np.column_stack(
        [np.asarray(rank, np.float64), np.asarray(tiebreak, np.float64)]
    )
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    if len(counts) < n:
        percentile = (np.bincount(inverse, weights=percentile) / np.bincount(inverse))[inverse]
    return percentile


def banded_display_size(
    rank: np.ndarray,
    band_sizes: list[int],
    tiebreak: np.ndarray | None = None,
    *,
    band_ranges: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Node radius that spreads out *within* each LOD band as well as across.

    A single global percentile is correct in aggregate and useless up close:
    the top 2% of nodes occupy the top 2% of any percentile scale, so band 0 —
    the first thing anyone sees — spans a size range of 0.02 and every hub
    renders at the same radius. Inside that band there is a thousandfold range
    of PageRank, and none of it is visible.

    So each band gets its own slice of the radius range and its members spread
    across the whole slice. The result is still monotone overall, because every
    band-0 node outranks every band-1 node by construction. What changes is that
    "which band" and "where in the band" both become legible.

    Input must already be sorted best-first — the same order the tiles use.
    """
    n = len(rank)
    if n == 0:
        return np.zeros(0)

    ranges = band_ranges or _default_band_ranges(len(band_sizes))
    size = np.zeros(n)
    start = 0
    for count, (lo, hi) in zip(band_sizes, ranges):
        stop = start + count
        if count > 0:
            local = to_display_size(
                np.asarray(rank[start:stop], np.float64),
                None if tiebreak is None else np.asarray(tiebreak)[start:stop],
            )
            size[start:stop] = lo + (hi - lo) * local
        start = stop
    return size


def _default_band_ranges(n_bands: int) -> list[tuple[float, float]]:
    """Radius slice per band, widest at the top.

    Band 0 gets the most room because it is the layer that is always drawn and
    always in focus; the tail shares a narrow slice because at that density the
    eye reads position and colour, not radius.
    """
    if n_bands <= 1:
        return [(0.0, 1.0)]
    presets = {2: [(0.45, 1.0), (0.0, 0.45)], 3: [(0.62, 1.0), (0.30, 0.62), (0.0, 0.30)]}
    if n_bands in presets:
        return presets[n_bands]
    edges = np.linspace(0.0, 1.0, n_bands + 1)[::-1]
    return [(float(edges[i + 1]), float(edges[i])) for i in range(n_bands)]
