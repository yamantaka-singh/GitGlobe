"""Clusters and domains — the territories on the map.

Two levels, because they answer different questions:

* **cluster** — fine-grained, hundreds of them, from HDBSCAN. "This is the
  Rust async runtime neighbourhood." Used for labels and fly-to.
* **domain** — twelve, from spherical k-means over the cluster centres. This is
  the `Uint8` colour field in the tile. Twelve is roughly the limit of
  categorical colours a person can tell apart at a glance; a hundred distinct
  hues is a hundred shades of nothing.

**Clustering happens on the sphere, not in 768-d.** That looks like the weaker
choice and is deliberate. The colour field has to be spatially contiguous —
the user asked for territories, and a territory speckled with three other
colours is not a territory, it is noise. Cluster in 768-d and you get
semantically pure groups scattered across the globe wherever UMAP's projection
was imperfect; cluster on the sphere and contiguity is guaranteed by
construction. UMAP already did the semantic work, and this is the surface people
actually look at.

This is the same class of bug as assigning domains by `i % 12`, which put every
domain everywhere and broke fly-to-domain along with the territory rendering.

The cost is real and worth naming: where UMAP misplaced a repository, this
inherits the mistake and assigns it to its neighbourhood rather than its
meaning. `cluster_purity` measures exactly that, so it stays a known quantity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

TAU = 2.0 * math.pi

N_DOMAINS = 12

#: Smallest group HDBSCAN will call a cluster. Too low and the map fragments
#: into thousands of two-repo "clusters" that cannot be named or navigated;
#: too high and genuine niches get swallowed into their larger neighbours.
DEFAULT_MIN_CLUSTER_SIZE = 60
DEFAULT_MIN_SAMPLES = 10


def to_unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), np.cos(theta), st * np.sin(phi)])


def to_angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.clip(vectors[:, 1], -1.0, 1.0)
    phi = np.arctan2(vectors[:, 2], vectors[:, 0])
    return np.arccos(y), np.where(phi < 0, phi + TAU, phi)


def spherical_mean(vectors: np.ndarray) -> np.ndarray:
    """Mean direction of unit vectors, renormalised back onto the sphere.

    The arithmetic mean of points on a sphere sits inside it; for a well-
    concentrated group, normalising that back out is the right centre. A group
    spread over an entire hemisphere has a mean near the origin and no
    meaningful centre at all, so the degenerate case falls back to the first
    member rather than normalising numerical noise into a random direction.
    """
    if len(vectors) == 0:
        return np.array([0.0, 1.0, 0.0])
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-9:
        # The fallback must be renormalised too: callers pass weighted vectors,
        # so `vectors[0]` is not a unit vector and returning it raw would put a
        # k-means centre off the sphere.
        first = vectors[0]
        first_norm = float(np.linalg.norm(first))
        return first / first_norm if first_norm > 1e-12 else np.array([0.0, 1.0, 0.0])
    return mean / norm


def spherical_kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    weights: np.ndarray | None = None,
    seed: int = 42,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means on S² with cosine similarity. Returns (labels, centres).

    Seeded with farthest-point sampling rather than at random. Random seeding on
    a sphere reliably drops two centres into the same dense region, and the
    resulting territories are lopsided in a way that survives every iteration —
    the same failure that made an earlier version of the globe put several
    domains on top of each other.
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(0, np.int32), np.zeros((0, 3))
    k = min(k, n)

    rng = np.random.default_rng(seed)
    centres = np.empty((k, 3))
    centres[0] = vectors[rng.integers(n)]
    # Farthest-point: each new centre is the point least similar to anything
    # chosen so far, which spreads the initial centres over the whole sphere.
    best_similarity = vectors @ centres[0]
    for i in range(1, k):
        centres[i] = vectors[int(np.argmin(best_similarity))]
        best_similarity = np.maximum(best_similarity, vectors @ centres[i])

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    # Sentinel, NOT zeros. Initialise to zeros and the first assignment for k=1
    # is trivially "unchanged", so the loop breaks before a single centre is
    # ever averaged and the seed points are returned as the answer.
    labels = np.full(n, -1, np.int32)
    for _ in range(max_iter):
        similarity = vectors @ centres.T
        new_labels = np.argmax(similarity, axis=1).astype(np.int32)
        converged = np.array_equal(new_labels, labels)
        labels = new_labels

        for i in range(k):
            members = labels == i
            if not members.any():
                # An empty cluster stays empty forever unless it is reseeded,
                # and an unused domain means one palette colour never appears.
                # The worst-fitting point is where a centre is most needed.
                worst = int(np.argmin(similarity[np.arange(n), labels]))
                centres[i] = vectors[worst]
                continue
            # spherical_mean normalises, so the weighted sum's scale cancels —
            # only the direction sum(w_i * v_i) matters.
            centres[i] = spherical_mean(vectors[members] * w[members, None])

        if converged:
            break
    return labels, centres


@dataclass
class ClusterResult:
    cluster_id: np.ndarray
    domain: np.ndarray
    centres: np.ndarray
    cluster_sizes: dict = field(default_factory=dict)
    noise_count: int = 0
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        real = len(self.cluster_sizes)
        return (
            f"{real} clusters, {self.noise_count} unclustered "
            f"({self.noise_count / max(len(self.cluster_id), 1):.1%}), "
            f"{len(np.unique(self.domain))} domains"
        )


def cluster(
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_domains: int = N_DOMAINS,
    seed: int = 42,
) -> ClusterResult:
    """HDBSCAN on the sphere, then domains over the cluster centres."""
    import hdbscan

    n = len(theta)
    if n == 0:
        return ClusterResult(np.zeros(0, np.int32), np.zeros(0, np.uint8), np.zeros((0, 3)))

    # HDBSCAN's haversine metric wants (latitude, longitude) in radians.
    # theta is measured from the pole, so latitude is PI/2 - theta. Get this
    # backwards and clusters come out mirrored about the equator — which looks
    # entirely plausible and is completely wrong.
    lat_lon = np.column_stack([math.pi / 2 - theta, phi - math.pi])

    log.info("HDBSCAN on %d points (min_cluster_size=%d)", n, min_cluster_size)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="haversine",
        core_dist_n_jobs=-1,
    ).fit_predict(lat_lon)

    vectors = to_unit_vectors(theta, phi)
    real = np.unique(labels[labels >= 0])
    sizes = {int(c): int((labels == c).sum()) for c in real}

    if len(real) == 0:
        # No structure found. Domains straight from the points, so the map is
        # still coloured and navigable instead of uniformly grey.
        log.warning("HDBSCAN found no clusters — assigning domains directly")
        domain, centres = spherical_kmeans(vectors, n_domains, seed=seed)
        return ClusterResult(
            labels.astype(np.int32), domain.astype(np.uint8), centres,
            noise_count=int((labels < 0).sum()),
            params={"min_cluster_size": min_cluster_size, "no_clusters": True},
        )

    # Domains from the cluster CENTRES, weighted by cluster size — not from the
    # points. Running k-means over a million points lets one crowded region
    # claim several domains purely because it is crowded, while sparse regions
    # of the map share one.
    centroids = np.array([spherical_mean(vectors[labels == c]) for c in real])
    weights = np.array([sizes[int(c)] for c in real], dtype=np.float64)
    cluster_domain, centres = spherical_kmeans(
        centroids, n_domains, weights=weights, seed=seed
    )

    domain_of = {int(c): int(d) for c, d in zip(real, cluster_domain)}
    domain = np.empty(n, dtype=np.uint8)
    clustered = labels >= 0
    domain[clustered] = [domain_of[int(c)] for c in labels[clustered]]

    # Noise points get the nearest domain centre. They are real repositories in
    # real places; leaving them uncoloured would punch holes in every territory.
    if (~clustered).any():
        domain[~clustered] = np.argmax(vectors[~clustered] @ centres.T, axis=1)

    return ClusterResult(
        cluster_id=labels.astype(np.int32),
        domain=domain,
        centres=centres,
        cluster_sizes=sizes,
        noise_count=int((~clustered).sum()),
        params={
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_domains": n_domains,
            "seed": seed,
        },
    )


def _mean_pairwise_similarity(vectors: np.ndarray, rng, pairs: int = 20_000) -> float:
    """Mean cosine similarity between random distinct pairs.

    Deliberately NOT similarity-to-the-group's-own-mean. That estimator is
    biased by group size — the empirical mean of m random unit vectors in d
    dimensions has expected concentration ~1/sqrt(m*d), so it falls as the group
    grows. Since clusters are always smaller than the corpus, comparing a
    cluster's concentration to the corpus concentration reports every cluster as
    tight regardless of whether it is. Measured on pure 8-d noise, a 60-member
    "cluster" scored 21x the 20,000-point baseline.

    Mean pairwise similarity has no such bias: it is flat in m, so a comparison
    between a small cluster and a large corpus is a fair one.
    """
    m = len(vectors)
    if m < 2:
        return 0.0
    i = rng.integers(0, m, pairs)
    j = rng.integers(0, m, pairs)
    distinct = i != j
    if not distinct.any():
        return 0.0
    return float(np.mean(np.einsum("ij,ij->i", vectors[i[distinct]], vectors[j[distinct]])))


def cluster_purity(cluster_id: np.ndarray, vectors: np.ndarray, sample: int = 20_000) -> dict:
    """How semantically tight the spatial clusters actually are.

    This is the honest accounting for clustering on the sphere instead of in
    768-d. A cluster whose members' embeddings barely agree is a group UMAP
    placed together without a good reason, and it will read as a mislabelled
    region on the map.

    `lift` is the headline number: within-cluster similarity minus corpus
    similarity. Zero means the spatial clusters carry no semantic signal at all
    and the territories are decoration. `ratio` is reported too, but only where
    the baseline is far enough from zero for a ratio to mean anything.
    """
    real = np.unique(cluster_id[cluster_id >= 0])
    empty = {"clusters": 0, "mean_within": 0.0, "baseline": 0.0, "lift": 0.0, "ratio": None}
    if len(real) == 0 or len(vectors) == 0:
        return empty

    rng = np.random.default_rng(0)
    idx = rng.choice(len(vectors), size=min(sample, len(vectors)), replace=False)
    baseline = _mean_pairwise_similarity(vectors[idx], rng)

    within = []
    for c in real:
        members = vectors[cluster_id == c]
        if len(members) < 2:
            continue
        within.append(_mean_pairwise_similarity(members, rng, pairs=4_000))
    if not within:
        return empty

    mean_within = float(np.mean(within))
    return {
        "clusters": len(real),
        "mean_within": round(mean_within, 4),
        "baseline": round(baseline, 4),
        "lift": round(mean_within - baseline, 4),
        # A ratio against a near-zero baseline is a division by noise.
        "ratio": round(mean_within / baseline, 3) if abs(baseline) > 0.02 else None,
        "weakest": round(float(np.min(within)), 4),
    }
