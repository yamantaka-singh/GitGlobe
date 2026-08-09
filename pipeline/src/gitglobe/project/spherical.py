"""Spherical projection: 768-d embeddings to points on S².

**UMAP optimises ON the sphere.** `output_metric="haversine"` makes the loss
function itself spherical, so neighbour relationships are preserved in the
geometry we actually render. The tempting alternative — project to 3D, then
normalise onto the sphere — optimises for a space nobody looks at and then
throws away the dimension it spent its effort on. Points near the origin get
flung to arbitrary directions, and the radial structure UMAP worked to build is
discarded wholesale.

**A sphere has no edge, which is the point.** Every 2D layout has a boundary,
and a boundary is a lie: it says "nothing beyond here", when what is actually
beyond is the other side of the map. On a sphere, every direction leads
somewhere, so panning never dead-ends.

Two things about UMAP's haversine output that are easy to get wrong:

* **The output is unbounded.** UMAP does not clamp θ to [0, π]; it optimises
  freely and lets values run past the poles. Feeding those into the tile format
  directly clamps them, which piles points onto the poles. `wrap_to_sphere`
  reflects them properly instead — going past a pole means coming down the far
  side, with φ shifted by π.
* **UMAP's convention is Z-up, ours is Y-up.** We reuse θ and φ unchanged, which
  is a relabelling of axes — a rotation. A sphere of repositories has no
  preferred orientation, so a rotation changes nothing that means anything.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

TAU = 2.0 * math.pi

#: How many points each neighbour sees. Low values fragment the map into
#: hundreds of disconnected islands; high values smear everything into one
#: blob and lose the fine structure that makes browsing worthwhile. 30 is
#: UMAP's usual middle ground for corpora of this size.
DEFAULT_N_NEIGHBORS = 30

#: How tightly points may pack. On a sphere the total area is fixed, so this
#: mostly controls how much whitespace separates clusters.
DEFAULT_MIN_DIST = 0.05

DEFAULT_SEED = 42


@dataclass
class ProjectionParams:
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    min_dist: float = DEFAULT_MIN_DIST
    metric: str = "cosine"
    seed: int = DEFAULT_SEED
    n_epochs: int | None = None

    def to_dict(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "seed": self.seed,
            "n_epochs": self.n_epochs,
            "output_metric": "haversine",
        }


@dataclass
class ProjectionResult:
    theta: np.ndarray
    phi: np.ndarray
    knn_indices: np.ndarray | None = None
    knn_distances: np.ndarray | None = None
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.theta)

    def to_xyz(self) -> np.ndarray:
        """Y-up unit vectors, matching the tile format and the point shader."""
        st = np.sin(self.theta)
        return np.column_stack([st * np.cos(self.phi), np.cos(self.theta), st * np.sin(self.phi)])


def wrap_to_sphere(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold arbitrary (θ, φ) onto the canonical θ∈[0,π], φ∈[0,2π).

    Clamping θ would be wrong. θ = π + 0.1 does not mean "at the south pole", it
    means "0.1 past the south pole" — which is a real location, on the opposite
    side of the sphere. Clamping stacks every overshooting point onto the pole
    and produces a visible spike of nodes there; reflecting sends them where
    they belong.
    """
    theta = np.mod(np.asarray(theta, dtype=np.float64), TAU)
    phi = np.asarray(phi, dtype=np.float64).copy()

    over = theta > math.pi
    theta = np.where(over, TAU - theta, theta)
    phi = np.where(over, phi + math.pi, phi)

    phi = np.mod(np.mod(phi, TAU) + TAU, TAU)
    return theta, phi


def project(
    vectors: np.ndarray,
    params: ProjectionParams | None = None,
    *,
    keep_knn: bool = True,
) -> ProjectionResult:
    """Embed unit vectors onto S² with UMAP.

    `keep_knn` retains the kNN graph UMAP built along the way. It is the exact
    thing the `similar_to` edges need, and recomputing it afterwards would be
    the single most expensive operation in Phase 2 done twice.
    """
    import umap

    params = params or ProjectionParams()
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n = len(vectors)
    if n <= params.n_neighbors:
        raise ValueError(
            f"{n} points is too few for n_neighbors={params.n_neighbors}. "
            "Lower n_neighbors, or embed more repositories first."
        )

    log.info(
        "UMAP: %d x %d, n_neighbors=%d, min_dist=%.3f, output_metric=haversine",
        n, vectors.shape[1], params.n_neighbors, params.min_dist,
    )
    if n > 400_000:
        log.warning(
            "CPU UMAP at %d points takes hours and needs ~%.1f GB. "
            "Consider cuML on a GPU VM if this becomes the bottleneck.",
            n, n * vectors.shape[1] * 4 / 1e9 * 3,
        )

    reducer = umap.UMAP(
        n_components=2,
        output_metric="haversine",
        metric=params.metric,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        n_epochs=params.n_epochs,
        random_state=params.seed,
        verbose=True,
    )
    embedding = reducer.fit_transform(vectors)

    theta, phi = wrap_to_sphere(embedding[:, 0], embedding[:, 1])
    return ProjectionResult(
        theta=theta,
        phi=phi,
        knn_indices=getattr(reducer, "_knn_indices", None) if keep_knn else None,
        knn_distances=getattr(reducer, "_knn_dists", None) if keep_knn else None,
        params=params.to_dict(),
    )


def coverage(theta: np.ndarray, phi: np.ndarray, *, bins: int = 24) -> dict:
    """How well the points cover the sphere.

    A projection that collapsed — everything in one hemisphere, or a ring around
    the equator — still produces valid θ and φ and renders without error. It is
    only obvious as "the globe looks wrong", by which point the cause is three
    stages back. This measures it directly.

    Bins are equal-AREA (uniform in cos θ), not equal-angle. Equal-angle
    latitude bands near the poles cover almost no surface, so a perfectly
    uniform sphere would look badly under-filled there and the check would
    report a problem that is not there.
    """
    n = len(theta)
    if n == 0:
        return {"points": 0, "occupied_fraction": 0.0, "gini": 0.0, "max_cell_fraction": 0.0}

    lat_bin = np.clip(((np.cos(theta) + 1) / 2 * bins).astype(int), 0, bins - 1)
    lon_bin = np.clip((phi / TAU * bins * 2).astype(int), 0, bins * 2 - 1)
    counts = np.bincount(lat_bin * bins * 2 + lon_bin, minlength=bins * bins * 2)

    occupied = float((counts > 0).mean())
    ordered = np.sort(counts)
    index = np.arange(1, len(ordered) + 1)
    total = ordered.sum()
    gini = float((2 * (index * ordered).sum()) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered))

    return {
        "points": n,
        "cells": len(counts),
        "occupied_fraction": round(occupied, 4),
        "gini": round(gini, 4),
        "max_cell_fraction": round(float(counts.max()) / n, 4),
        "empty_cells": int((counts == 0).sum()),
    }


def assess_coverage(stats: dict) -> list[str]:
    """Human-readable complaints about a projection. Empty means it looks fine."""
    problems = []
    if stats["points"] < 100:
        return problems
    if stats["occupied_fraction"] < 0.5:
        problems.append(
            f"only {stats['occupied_fraction']:.0%} of the sphere has any points — "
            "the projection has collapsed onto part of the surface"
        )
    if stats["max_cell_fraction"] > 0.10:
        problems.append(
            f"one cell holds {stats['max_cell_fraction']:.1%} of all points — "
            "likely a pole spike from clamping instead of wrapping"
        )
    if stats["gini"] > 0.85:
        problems.append(
            f"density Gini {stats['gini']:.2f} — almost everything is in a few cells"
        )
    return problems


def knn_to_edges(
    indices: np.ndarray,
    distances: np.ndarray,
    repo_ids: np.ndarray,
    *,
    k: int = 8,
    max_distance: float = 0.45,
) -> list[tuple[int, int, float]]:
    """UMAP's kNN graph to `similar_to` edges, as (src, dst, weight).

    Two filters, both load-bearing:

    * **A distance ceiling.** kNN always returns k neighbours, even for a
      repository with no real relatives — it just returns the least-distant
      strangers. Without a ceiling, every isolated repo acquires eight
      confidently-wrong "similar" edges.
    * **Mutuality.** A is in B's top-k far more often than the reverse, because
      hubs appear in everyone's list. Requiring both directions is what stops a
      handful of popular repositories from being "similar" to the entire corpus.

    These edges are stored but deliberately NOT drawn as arcs: two repositories
    that are semantically similar are already adjacent on the globe, so an arc
    between them re-states what position already says while adding clutter.
    """
    if indices is None or distances is None or len(indices) == 0:
        return []

    n = len(indices)
    # Column 0 is the point itself.
    neighbours = indices[:, 1 : k + 1]
    dists = distances[:, 1 : k + 1]

    candidates: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, d in zip(neighbours[i], dists[i]):
            if j < 0 or j >= n or j == i or d > max_distance:
                continue
            candidates[(i, int(j))] = float(d)

    edges = []
    for (i, j), d in candidates.items():
        if i < j and (j, i) in candidates:
            edges.append((int(repo_ids[i]), int(repo_ids[j]), round(1.0 - d, 4)))
    return edges
