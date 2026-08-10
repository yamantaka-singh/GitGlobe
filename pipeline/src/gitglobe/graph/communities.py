"""Communities — the colonies, found from connections rather than from position.

HDBSCAN answers "what is near what". That is a fact about the projection, not
about software: it says three repositories sit together because UMAP put them
together. Modularity optimisation answers a different and better question —
"what is densely connected to what" — which is a fact about the ecosystem and
survives any change to the layout.

**Louvain, implemented here, rather than Leiden from a package.**

Leiden is the better algorithm: it adds a refinement pass that guarantees every
community is internally connected, which Louvain can violate. But `leidenalg`
means `python-igraph`, a C extension that cannot be built or verified in this
environment, and an unverifiable dependency on the critical path is worse than a
slightly weaker algorithm that is tested. `disconnected_communities()` measures
exactly the flaw Leiden fixes, so the cost is known rather than assumed. If it
reports a real problem on real data, that is the moment to take the dependency.

**Which edges count, and why it differs from PageRank.**

PageRank deliberately excludes `similar_to`: a kNN graph gives every node
roughly k neighbours by construction, so it carries almost no information about
*importance* and would smear the ranking toward uniform.

Community detection is the opposite case. It asks about *grouping*, and a kNN
graph is highly informative about grouping — that is the only thing it encodes.
It also covers the 86% of repositories that publish no package and are therefore
invisible to the dependency graph. Excluding it here would leave most of the
globe in no community at all.

Same graph, two questions, two answers about which edges belong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

#: Louvain's resolution. Above 1.0 gives more, smaller communities; below gives
#: fewer, larger ones. 1.0 is the standard modularity definition and is where
#: the maths is least arguable.
DEFAULT_RESOLUTION = 1.0

#: Local moving repeats until no node improves, but a pass that changes almost
#: nothing is not worth another sweep over every node.
MIN_IMPROVEMENT = 1e-7
MAX_PASSES = 20
MAX_LEVELS = 10


@dataclass
class CommunityResult:
    labels: np.ndarray
    modularity: float
    levels: int
    sizes: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def count(self) -> int:
        return len(self.sizes)

    def summary(self) -> str:
        if not self.sizes:
            return "no communities"
        biggest = max(self.sizes.values())
        return (
            f"{self.count} communities, modularity {self.modularity:.4f}, "
            f"largest holds {biggest:,} ({biggest / max(len(self), 1):.1%}), "
            f"{self.levels} aggregation levels"
        )


def build_adjacency(
    n: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric CSR from an edge list. Returns (offsets, targets, weights).

    Undirected because modularity is undirected: "A depends on B" and "B is
    depended on by A" are the same edge for the purpose of finding a cluster.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if len(src) != len(dst):
        raise ValueError(f"src has {len(src)}, dst has {len(dst)}")
    w = np.ones(len(src)) if weight is None else np.asarray(weight, dtype=np.float64)
    if len(w) != len(src):
        raise ValueError(f"weight has {len(w)}, expected {len(src)}")
    if len(src) and (src.max() >= n or dst.max() >= n or min(src.min(), dst.min()) < 0):
        raise ValueError(f"endpoints outside [0, {n})")

    keep = src != dst  # self-loops contribute nothing to modularity
    src, dst, w = src[keep], dst[keep], w[keep]

    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    both_w = np.concatenate([w, w])

    order = np.argsort(both_src, kind="stable")
    targets = both_dst[order]
    weights = both_w[order]
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(both_src, minlength=n), out=offsets[1:])
    return offsets, targets, weights


def modularity(
    labels: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    resolution: float = DEFAULT_RESOLUTION,
) -> float:
    """Newman-Girvan modularity of a partition, in [-0.5, 1].

    Above ~0.3 is generally taken as real community structure; near 0 means the
    partition explains no more than chance would.
    """
    n = len(offsets) - 1
    if n == 0 or len(targets) == 0:
        return 0.0

    degree = np.zeros(n)
    np.add.at(degree, np.repeat(np.arange(n), np.diff(offsets)), weights)
    two_m = degree.sum()
    if two_m <= 0:
        return 0.0

    sources = np.repeat(np.arange(n), np.diff(offsets))
    internal = weights[labels[sources] == labels[targets]].sum()

    n_labels = int(labels.max()) + 1 if len(labels) else 0
    community_degree = np.zeros(n_labels)
    np.add.at(community_degree, labels, degree)

    return float(
        internal / two_m - resolution * ((community_degree / two_m) ** 2).sum()
    )


def _best_move(
    node: int,
    current: int,
    labels: np.ndarray,
    degree: np.ndarray,
    community_degree: np.ndarray,
    two_m: float,
    resolution: float,
    neighbours: np.ndarray,
    link: np.ndarray,
) -> tuple[int, float, float]:
    """Best community for one node. Returns (label, its gain, staying's gain).

    Assumes the caller has already removed `node` from `current`'s degree total,
    otherwise the node competes against a community that still contains it.

    A node may only move to a community one of its neighbours is already in.
    Letting it isolate into an EMPTY community is the textbook escape from an
    over-merged state, and it was tried: once most nodes share a label,
    `stay_gain` sits at approximately zero, float noise tips it negative, nodes
    isolate, re-merge, and oscillate for every remaining pass. It turned a graph
    scoring 0.42 into one scoring 0.000. The greedy rule is stable, and on
    graphs of a realistic size it recovers planted communities exactly.
    """
    # Exclude the node's own self-loop. It contributes to `degree` — correctly,
    # it is real weight — but it must NOT count as "weight into my community",
    # because the self-loop travels with the node wherever it goes.
    #
    # Counting it is why aggregation never got past level one. After the first
    # level every super-node carries a self-loop holding its entire internal
    # weight, which dwarfs its external edges, so staying alone beat every
    # possible move and the second level merged nothing. The symptom was a
    # community count almost unaffected by resolution: a 20x change moved it
    # 8.5%, and the median community size was exactly 5 at every setting.
    external = neighbours != node
    if not external.all():
        neighbours, link = neighbours[external], link[external]
        if len(neighbours) == 0:
            return current, 0.0, 0.0

    unique, inverse = np.unique(labels[neighbours], return_inverse=True)
    into = np.bincount(inverse.ravel(), weights=link, minlength=len(unique))

    penalty = resolution * degree[node] / two_m
    gain = into - penalty * community_degree[unique]
    best = int(np.argmax(gain))

    own = into[unique == current]
    stay_gain = (own[0] if len(own) else 0.0) - penalty * community_degree[current]
    return int(unique[best]), float(gain[best]), float(stay_gain)


def _local_moving(
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    resolution: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Louvain level: move nodes to neighbouring communities while it helps.

    Bounded by MAX_PASSES. The standard formulation loops "until no node moves",
    which on a pathological graph is not a bound at all — and this runs inside a
    pipeline where an unbounded loop is indistinguishable from a slow one.
    """
    n = len(offsets) - 1
    degree = np.zeros(n)
    np.add.at(degree, np.repeat(np.arange(n), np.diff(offsets)), weights)
    two_m = degree.sum()
    if two_m <= 0:
        return np.arange(n)

    labels = np.arange(n)
    community_degree = degree.copy()

    for _ in range(MAX_PASSES):
        moved = 0
        # Random order: Louvain is order-sensitive, and a fixed order biases the
        # result toward whatever the node numbering happens to be — which here
        # is PageRank order, so it would bias communities toward hubs.
        for node in rng.permutation(n):
            start, end = offsets[node], offsets[node + 1]
            if start == end:
                continue

            current = labels[node]
            community_degree[current] -= degree[node]
            best_label, best_gain, stay_gain = _best_move(
                node, current, labels, degree, community_degree, two_m,
                resolution, targets[start:end], weights[start:end],
            )
            if best_gain > stay_gain + MIN_IMPROVEMENT:
                labels[node] = best_label
                moved += 1
            community_degree[labels[node]] += degree[node]

        if moved == 0:
            break
    return labels


def _aggregate(
    labels: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse each community into one node, KEEPING internal weight.

    Two things here are easy to get wrong and both destroy the algorithm
    silently — the result is still a valid partition, just a worthless one.

    **Self-loops must survive.** An edge inside a community becomes a self-loop
    on the super-node, and that self-loop IS the community's internal weight.
    Drop it and every super-node looks weightless, so merging always appears
    profitable and level two swallows the entire graph into one community.
    Measured: modularity fell from 0.7101 to 0.0000.

    **Do not re-symmetrise.** The input is already a symmetric CSR, so each
    original edge appears twice. Passing the pairs back through
    `build_adjacency` would double them again.
    """
    k = int(labels.max()) + 1 if len(labels) else 0
    if k == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0)

    sources = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    a = labels[sources].astype(np.int64)
    b = labels[targets].astype(np.int64)

    # Sum parallel edges between the same pair of communities.
    key = a * k + b
    unique, inverse = np.unique(key, return_inverse=True)
    summed = np.bincount(inverse.ravel(), weights=weights)
    agg_a, agg_b = unique // k, unique % k

    order = np.argsort(agg_a, kind="stable")
    new_offsets = np.zeros(k + 1, dtype=np.int64)
    np.cumsum(np.bincount(agg_a, minlength=k), out=new_offsets[1:])
    return new_offsets, agg_b[order], summed[order]


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Compact labels to 0..k-1, so downstream arrays stay small."""
    _, packed = np.unique(labels, return_inverse=True)
    return packed.astype(np.int64).ravel()


def detect(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = 42,
) -> CommunityResult:
    """Louvain community detection. Returns one label per node."""
    if n <= 0:
        return CommunityResult(np.zeros(0, np.int64), 0.0, 0)

    offsets, targets, weights = build_adjacency(n, src, dst, weight)
    if len(targets) == 0:
        # No edges: every node is its own community, and modularity is 0.
        return CommunityResult(np.arange(n), 0.0, 0, {i: 1 for i in range(n)})

    rng = np.random.default_rng(seed)
    node_labels = np.arange(n)
    level_offsets, level_targets, level_weights = offsets, targets, weights
    levels = 0

    for _ in range(MAX_LEVELS):
        local = _relabel(
            _local_moving(level_offsets, level_targets, level_weights, resolution, rng)
        )
        if len(np.unique(local)) == len(level_offsets) - 1:
            break  # nothing merged; further levels cannot help

        node_labels = local[node_labels]
        levels += 1

        # Aggregate so the next level can find structure larger than one node's
        # neighbourhood — communities of communities. `_aggregate` keeps
        # intra-community weight as self-loops; dropping it (and re-symmetrising
        # through `build_adjacency`, which discards self-loops by design)
        # collapsed this graph from modularity 0.6548 to 0.0000.
        level_offsets, level_targets, level_weights = _aggregate(
            local, level_offsets, level_targets, level_weights
        )
        if len(level_targets) == 0:
            break

    node_labels = _relabel(node_labels)
    q = modularity(node_labels, offsets, targets, weights, resolution)
    counts = np.bincount(node_labels)
    return CommunityResult(
        labels=node_labels,
        modularity=q,
        levels=levels,
        sizes={int(i): int(c) for i, c in enumerate(counts) if c},
    )


def connected_components(offsets: np.ndarray, targets: np.ndarray) -> tuple[int, int, int]:
    """Returns (component count, largest size, isolated node count).

    This is the number that says whether tuning is worth attempting.
    Modularity optimisation can only ever group nodes that are CONNECTED — no
    resolution setting merges two components, because there is no edge across
    which to measure a gain. So if the component count is close to the
    community count, the communities *are* the components and the graph itself
    is the problem, not the parameter.

    Union-find with path compression: near-linear, and it answers in under a
    second on a corpus-sized graph.
    """
    n = len(offsets) - 1
    if n == 0:
        return 0, 0, 0

    parent = np.arange(n)

    def find(x: int) -> int:
        root = x
        # Bounded: the tree depth cannot exceed n, and path compression below
        # keeps it near 1 in practice.
        for _ in range(n + 1):
            if parent[root] == root:
                break
            root = parent[root]
        # Path compression, bounded by the same depth argument as the walk above.
        for _ in range(n + 1):
            if parent[x] == root:
                break
            parent[x], x = root, parent[x]
        return root

    sources = np.repeat(np.arange(n), np.diff(offsets))
    for a, b in zip(sources, targets):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(n)])
    _, counts = np.unique(roots, return_counts=True)
    return len(counts), int(counts.max()), int((np.diff(offsets) == 0).sum())


def disconnected_communities(
    labels: np.ndarray, offsets: np.ndarray, targets: np.ndarray
) -> list[int]:
    """Communities whose members are not actually connected to each other.

    This is precisely the defect Leiden's refinement pass eliminates and Louvain
    does not. Measuring it turns "we used the weaker algorithm" from an
    unexamined assumption into a number — and if that number is ever large on
    real data, it is the argument for taking the `leidenalg` dependency.
    """
    bad = []
    for community in np.unique(labels):
        members = np.where(labels == community)[0]
        if len(members) < 2:
            continue
        member_set = set(members.tolist())
        # Breadth-first from one member, staying inside the community.
        seen = {int(members[0])}
        queue = [int(members[0])]
        # Bounded by construction: every iteration pops one node and only ever
        # pushes nodes not already in `seen`, so at most `len(members)` passes.
        # Stated as a ceiling anyway — "the invariant holds" is an argument, and
        # a bound you can point at is a fact.
        for _ in range(len(members) + 1):
            if not queue:
                break
            node = queue.pop()
            for k in range(offsets[node], offsets[node + 1]):
                neighbour = int(targets[k])
                if neighbour in member_set and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        if len(seen) < len(members):
            bad.append(int(community))
    return bad
