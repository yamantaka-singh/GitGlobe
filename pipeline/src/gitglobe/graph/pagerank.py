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


def to_display_size(rank: np.ndarray) -> np.ndarray:
    """PageRank to a normalised node radius in [0, 1].

    PageRank is power-law distributed over four or five orders of magnitude.
    Map it linearly onto radius and every node below the top hundred renders at
    literally the same size — the field looks uniform and the visual carries no
    information. So: log first, then rank-normalise.

    Rank-normalising (rather than min-max on the log) is what keeps the field
    evenly spread whatever the underlying distribution does. A min-max on the
    log still bunches, because the log of a power law is exponential-ish.
    """
    n = len(rank)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    logged = np.log1p(np.maximum(rank, 0.0) / max(rank.max(), 1e-30) * 1e6)
    order = np.argsort(logged, kind="stable")
    percentile = np.empty(n)
    percentile[order] = np.arange(n) / (n - 1)

    # Ties must not become different sizes: two repos with identical rank
    # rendering at different radii is a visible lie about the data.
    unique, inverse = np.unique(logged, return_inverse=True)
    if len(unique) < n:
        means = np.bincount(inverse, weights=percentile) / np.bincount(inverse)
        percentile = means[inverse]
    return percentile
