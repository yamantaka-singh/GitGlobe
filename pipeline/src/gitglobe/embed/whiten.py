"""Isotropic transformation — undoing the cone that LLM embeddings live in.

Random repository pairs in this corpus have cosine similarity **0.6473**. If the
embedding used its space properly that number would sit near zero. It does not
because large language models push every vector into a narrow cone: there is a
dominant direction shared by all of them, roughly "this is a GitHub
repository", and it swamps the axes that distinguish one repository from
another.

The cost is dynamic range. With random pairs at 0.65 and the maximum at 1.0,
every meaningful distinction has to fit in the remaining 0.35 — so a genuinely
coherent cluster can only ever score a "lift" of a few hundredths, and the
metric looks broken when the data is merely compressed.

**The fix (Mu & Viswanath, 2018, "All-but-the-Top"): subtract the mean, then
project out the leading principal components, then renormalise.** The first
component is the cone's axis. Removing it recentres the manifold on the origin
and restores the full range.

`recommend_components` measures how many to strip rather than assuming. The
paper suggests around d/100 — about 7 for 768 dimensions — but the right number
is a property of the corpus, and it is cheap to measure.

**What this does and does not fix.** It repairs measurement, and it should help
any model reading these vectors as features. It changes the UMAP projection much
less than one might expect, because UMAP normalises distances locally per point
(each point's own nearest neighbour sets its scale), so a globally shared
direction largely cancels there already. That is why the spotcheck passed on
un-whitened vectors: the map was fine, the ruler was compressed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Mu & Viswanath suggest d/100. Used only as the default upper bound for the
#: search; `recommend_components` picks the actual number from the data.
DEFAULT_MAX_COMPONENTS = 8

#: Rows sampled when fitting. The leading components of a 768-dimensional cone
#: are extremely stable, so the full corpus buys nothing over a large sample.
FIT_SAMPLE = 20_000


@dataclass
class Whitener:
    """A fitted isotropic transform. Fit once, apply everywhere.

    Holding the basis matters for the same reason `reduce_embeddings` holds its
    PCA basis: refitting on a different subset gives different axes, so the
    "same" vector would land somewhere else depending on which batch it arrived
    in.
    """

    mean: np.ndarray
    axes: np.ndarray          # (k, d) components removed
    components: int
    baseline_before: float
    baseline_after: float

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")
        if vectors.shape[1] != len(self.mean):
            raise ValueError(
                f"whitener was fitted on {len(self.mean)} dimensions, "
                f"input has {vectors.shape[1]}"
            )
        centred = vectors - self.mean
        if self.components:
            centred = centred - (centred @ self.axes.T) @ self.axes
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        return (centred / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "components": self.components,
            "baseline_before": round(self.baseline_before, 4),
            "baseline_after": round(self.baseline_after, 4),
        }


def mean_pairwise(vectors: np.ndarray, rng, pairs: int = 40_000) -> float:
    """Mean cosine between random distinct pairs — the isotropy measure."""
    n = len(vectors)
    if n < 2:
        return 0.0
    i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    keep = i != j
    return float(np.mean(np.einsum("ij,ij->i", vectors[i[keep]], vectors[j[keep]])))


def recommend_components(
    vectors: np.ndarray,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    *,
    seed: int = 0,
) -> tuple:
    """How many leading components to strip. Measured, not assumed.

    Returns `(best_k, table)` where `table` lists the baseline similarity that
    results from removing each number of components. The best k is the smallest
    one that brings the baseline near zero — stripping more after that removes
    genuine signal for no further isotropy gain.
    """
    rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, dtype=np.float64)
    sample = vectors
    if len(vectors) > FIT_SAMPLE:
        sample = vectors[rng.choice(len(vectors), FIT_SAMPLE, replace=False)]

    mean = sample.mean(axis=0)
    centred = sample - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)

    table = []
    for k in range(0, max_components + 1):
        projected = centred - ((centred @ vt[:k].T) @ vt[:k] if k else 0.0)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        unit = projected / np.where(norms == 0, 1.0, norms)
        table.append((k, mean_pairwise(unit, np.random.default_rng(seed))))

    # Smallest k whose baseline is within 0.02 of zero. Beyond that point the
    # remaining components carry meaning, not the cone.
    best = next((k for k, b in table if abs(b) < 0.02), table[-1][0])
    return best, table


def fit(vectors: np.ndarray, components: int | None = None, *, seed: int = 0) -> Whitener:
    """Fit the transform. `components=None` measures the right number."""
    rng = np.random.default_rng(seed)
    vectors64 = np.asarray(vectors, dtype=np.float64)
    if vectors64.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {vectors64.shape}")

    table = None
    if components is None:
        components, table = recommend_components(vectors64, seed=seed)
        log.info("Isotropy scan: %s", [(k, round(b, 4)) for k, b in table])

    sample = vectors64
    if len(vectors64) > FIT_SAMPLE:
        sample = vectors64[rng.choice(len(vectors64), FIT_SAMPLE, replace=False)]
    mean = sample.mean(axis=0)
    centred = sample - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = vt[:components] if components else np.zeros((0, vectors64.shape[1]))

    unit = vectors64 / np.maximum(
        np.linalg.norm(vectors64, axis=1, keepdims=True), 1e-12
    )
    before = mean_pairwise(unit, np.random.default_rng(seed))

    whitener = Whitener(
        mean=mean.astype(np.float32), axes=axes.astype(np.float32),
        components=int(components), baseline_before=before, baseline_after=0.0,
    )
    whitener.baseline_after = mean_pairwise(
        whitener.apply(vectors64), np.random.default_rng(seed)
    )
    log.info(
        "Whitening: removed %d components, random-pair similarity %.4f -> %.4f "
        "(usable range %.0f%% -> %.0f%% of [0, 1])",
        whitener.components, whitener.baseline_before, whitener.baseline_after,
        100 * (1 - whitener.baseline_before), 100 * (1 - whitener.baseline_after),
    )
    return whitener
