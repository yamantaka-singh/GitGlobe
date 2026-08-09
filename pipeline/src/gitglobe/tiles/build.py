"""Build the world: Postgres rows to the files the browser fetches.

Output, into `web/public/tiles/`:

    manifest.json     what exists, how big, and how it was made
    band-0.bin        the top 2% by PageRank
    band-1.bin        the next 18%
    band-2.bin        everything else
    names-N.json      band-aligned repository names
    graph.bin         PageRank + undirected CSR + the ambient backbone

**Node id is rank position.** Sort every repository by PageRank descending; that
ordinal is the graph's node index, and `ordinal + 1` is the tile's `repoId`
(0 is reserved for "nothing picked"). One ordering does four jobs at once:

* band membership is a contiguous slice, so LOD is `bands[0..k]` with no lookup;
* tile-to-graph is `repoId - 1`, not a hash map with a million entries;
* the picking id space is dense, so a 24-bit colour covers 16M nodes;
* rank is monotonic in node id, which `verify-world.ts` asserts and which makes
  "the top N repositories" a slice rather than a sort.

The synthetic generator (`web/scripts/gen-world.ts`) already uses this
convention. Matching it means `npm run verify` — 30 checks written against the
synthetic world — validates the real one unchanged.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..graph.pagerank import DEFAULT_DAMPING, PageRankResult, to_display_size
from .format import (
    FLAG_ARCHIVED,
    FLAG_FORK,
    FLAG_LOW_SIGNAL,
    BandSpec,
    GraphArrays,
    Manifest,
    ManifestBand,
    TilePoints,
    build_undirected_csr,
    encode_graph,
    encode_tile,
)

log = logging.getLogger(__name__)

LAYOUT_VERSION = 2

#: Arcs drawn with no repository selected — the "there is a network here"
#: layer. Bounded by what the arc shader can draw at 60 fps, not by what the
#: data contains.
DEFAULT_AMBIENT_ARCS = 900


@dataclass
class WorldInput:
    """One row per repository, straight out of Postgres. All arrays parallel."""

    repo_id: np.ndarray  # database id, used only to remap edges
    full_name: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    rank: np.ndarray
    domain: np.ndarray
    cluster_id: np.ndarray
    low_signal: np.ndarray
    is_archived: np.ndarray
    is_fork: np.ndarray

    def __len__(self) -> int:
        return len(self.repo_id)

    def validate(self) -> None:
        n = len(self)
        for name in (
            "full_name", "theta", "phi", "rank", "domain",
            "cluster_id", "low_signal", "is_archived", "is_fork",
        ):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"{name} has {got} rows, expected {n}")
        if n == 0:
            raise ValueError("nothing to build — run embed and project first")
        if len(np.unique(self.repo_id)) != n:
            raise ValueError("repo_id must be unique")
        for name in ("theta", "phi", "rank"):
            if np.isnan(getattr(self, name)).any():
                raise ValueError(f"{name} contains NaN — projection did not complete")


@dataclass
class BuildResult:
    manifest: Manifest
    ordinal_of: dict
    bytes_written: int = 0
    files: list = field(default_factory=list)


def pack_flags(low_signal: np.ndarray, archived: np.ndarray, fork: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(low_signal), dtype=np.uint8)
    flags |= np.where(low_signal, FLAG_LOW_SIGNAL, 0).astype(np.uint8)
    flags |= np.where(archived, FLAG_ARCHIVED, 0).astype(np.uint8)
    flags |= np.where(fork, FLAG_FORK, 0).astype(np.uint8)
    return flags


def rank_order(rank: np.ndarray) -> np.ndarray:
    """Indices sorted by rank descending, ties broken deterministically.

    `kind="stable"` matters: without it, two runs over the same data can order
    equal-ranked repositories differently, every node id shifts, and the whole
    globe appears to have been rebuilt from scratch when nothing changed.
    """
    return np.argsort(-np.asarray(rank, dtype=np.float64), kind="stable")


def select_ambient(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    rank: np.ndarray,
    *,
    limit: int = DEFAULT_AMBIENT_ARCS,
    max_per_node: int = 4,
) -> np.ndarray:
    """Pick the backbone arcs drawn when nothing is selected.

    Scored by edge weight times the geometric mean of both endpoints' rank, so
    the arcs shown are important edges between important nodes — not the
    heaviest edge between two repositories nobody has heard of.

    `max_per_node` is the part that makes it look like a network rather than a
    hairball: without it, the top few hubs claim every slot and the ambient
    layer renders as a handful of starbursts.
    """
    if len(src) == 0 or limit <= 0:
        return np.zeros(0, dtype=np.uint32)

    score = np.asarray(weight, np.float64) * np.sqrt(rank[src] * rank[dst])
    used: dict[int, int] = {}
    chosen = []
    for index in np.argsort(-score):
        a, b = int(src[index]), int(dst[index])
        if used.get(a, 0) >= max_per_node or used.get(b, 0) >= max_per_node:
            continue
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
        chosen.append((a, b))
        if len(chosen) >= limit:
            break
    return np.array(chosen, dtype=np.uint32).reshape(-1)


def build_world(
    world: WorldInput,
    *,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    pagerank_result: PageRankResult | None,
    out_dir: Path,
    domains: list[str],
    clusters: list[dict] | None = None,
    bands: BandSpec | None = None,
    seed: int = 42,
    ambient_arcs: int = DEFAULT_AMBIENT_ARCS,
) -> BuildResult:
    """Write every tile artifact. `edges` are (src, dst, weight) in DB ids."""
    world.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(world)
    order = rank_order(world.rank)

    # Database id -> node ordinal. Built once; every edge remap reads it.
    ordinal_of = {int(rid): i for i, rid in enumerate(world.repo_id[order])}

    theta = world.theta[order]
    phi = world.phi[order]
    rank = np.asarray(world.rank, np.float64)[order]
    size = to_display_size(rank)
    domain = world.domain[order].astype(np.uint8)
    flags = pack_flags(world.low_signal[order], world.is_archived[order], world.is_fork[order])
    names = world.full_name[order]

    # ---- tiles ----------------------------------------------------------
    spec = bands or BandSpec()
    sizes = spec.sizes(n)
    manifest_bands: list[ManifestBand] = []
    files: list[Path] = []
    total_bytes = 0
    start = 0

    for band_index, count in enumerate(sizes):
        stop = start + count
        points = TilePoints(
            theta=theta[start:stop],
            phi=phi[start:stop],
            # Ordinal + 1: repoId 0 means "nothing picked" in the pick buffer.
            repo_id=(np.arange(start, stop, dtype=np.uint32) + 1),
            size=size[start:stop],
            domain=domain[start:stop],
            flags=flags[start:stop],
        )
        blob = encode_tile(points, layout_version=LAYOUT_VERSION, lod_band=band_index)
        tile_file = f"band-{band_index}.bin"
        (out_dir / tile_file).write_bytes(blob)

        names_file = f"names-{band_index}.json"
        (out_dir / names_file).write_text(
            json.dumps([str(x) for x in names[start:stop]], separators=(",", ":"))
        )

        manifest_bands.append(
            ManifestBand(band=band_index, count=count, bytes=len(blob), file=tile_file, names=names_file)
        )
        files += [out_dir / tile_file, out_dir / names_file]
        total_bytes += len(blob)
        log.info("band %d: %d points, %.1f MB", band_index, count, len(blob) / 1e6)
        start = stop

    # ---- graph ----------------------------------------------------------
    db_src, db_dst, weight = edges
    graph_meta = None

    if len(db_src):
        # Edges reference repositories that may not have made it into the
        # projection — anything unembedded, or filtered as low signal. Dropping
        # them here rather than failing keeps a partial dataset buildable, which
        # is what makes the 5k proof run possible at all.
        keep = np.array(
            [int(a) in ordinal_of and int(b) in ordinal_of for a, b in zip(db_src, db_dst)]
        )
        dropped = int((~keep).sum())
        if dropped:
            log.info("Dropped %d edges pointing outside the projected set", dropped)

        src = np.array([ordinal_of[int(a)] for a in db_src[keep]], dtype=np.uint32)
        dst = np.array([ordinal_of[int(b)] for b in db_dst[keep]], dtype=np.uint32)
        w = np.asarray(weight, np.float64)[keep]
    else:
        src = dst = np.zeros(0, np.uint32)
        w = np.zeros(0)

    self_loops = src == dst
    if self_loops.any():
        # A self-loop makes the CSR asymmetry check fail and renders as an arc
        # from a node to itself, which is a zero-length line.
        log.info("Dropped %d self-loops", int(self_loops.sum()))
        src, dst, w = src[~self_loops], dst[~self_loops], w[~self_loops]

    # Normalise weights into [0, 1] for the 15-bit weight field.
    if len(w):
        span = w.max() - w.min()
        norm_w = (w - w.min()) / span if span > 0 else np.full(len(w), 0.5)
    else:
        norm_w = w

    offsets, targets, weights = build_undirected_csr(n, src, dst, norm_w)
    ambient = select_ambient(src, dst, norm_w, rank, limit=ambient_arcs)

    graph = GraphArrays(
        rank=rank.astype(np.float32),
        offsets=offsets,
        targets=targets,
        weights=weights,
        ambient=ambient,
        layout_version=LAYOUT_VERSION,
    )
    blob = encode_graph(graph)
    (out_dir / "graph.bin").write_bytes(blob)
    files.append(out_dir / "graph.bin")
    total_bytes += len(blob)

    degree = np.diff(offsets.astype(np.int64))
    pr = pagerank_result
    graph_meta = {
        "file": "graph.bin",
        "bytes": len(blob),
        "directedEdges": int(len(src)),
        "csrEntries": int(len(targets)),
        "ambientArcs": int(len(ambient) // 2),
        "pagerank": {
            "damping": DEFAULT_DAMPING,
            "iterations": pr.iterations if pr else 0,
            "converged": bool(pr.converged) if pr else False,
            "delta": float(pr.delta) if pr else 0.0,
        },
        "degree": {
            "mean": round(float(degree.mean()), 3) if n else 0.0,
            "max": int(degree.max()) if n else 0,
            "p50": int(np.percentile(degree, 50)) if n else 0,
            "p99": int(np.percentile(degree, 99)) if n else 0,
        },
    }
    log.info(
        "graph: %d directed edges, %d csr entries, %d ambient arcs, %.1f MB",
        len(src), len(targets), len(ambient) // 2, len(blob) / 1e6,
    )

    manifest = Manifest(
        layout_version=LAYOUT_VERSION,
        total=n,
        bands=manifest_bands,
        domains=domains,
        clusters=clusters or [],
        graph=graph_meta,
        synthetic=False,
        seed=seed,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    files.append(out_dir / "manifest.json")

    return BuildResult(manifest=manifest, ordinal_of=ordinal_of, bytes_written=total_bytes, files=files)


def cluster_manifest_entries(
    cluster_id: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    domain: np.ndarray,
    labels: dict | None = None,
) -> list[dict]:
    """Per-cluster centres and spreads, for camera fly-to and hub labels.

    `kappa` is the von Mises-Fisher concentration, estimated from the mean
    resultant length. The camera uses it to choose a framing distance: a tight
    cluster wants a close shot, a diffuse one wants a wide one. Using a fixed
    distance instead means half the fly-tos overshoot and half undershoot.
    """
    entries = []
    for c in np.unique(cluster_id[cluster_id >= 0]):
        members = cluster_id == c
        st = np.sin(theta[members])
        vectors = np.column_stack(
            [st * np.cos(phi[members]), np.cos(theta[members]), st * np.sin(phi[members])]
        )
        mean = vectors.mean(axis=0)
        r = float(np.linalg.norm(mean))
        centre = mean / r if r > 1e-9 else vectors[0]

        # Banerjee's estimator for S^2 (p = 3).
        kappa = r * (3 - r * r) / (1 - r * r) if r < 0.9999 else 1e4

        y = float(np.clip(centre[1], -1.0, 1.0))
        phi_c = float(math.atan2(centre[2], centre[0]))
        entries.append({
            "id": int(c),
            "label": (labels or {}).get(int(c), f"cluster-{int(c)}"),
            "domain": int(np.bincount(domain[members]).argmax()),
            "theta": round(math.acos(y), 6),
            "phi": round(phi_c + 2 * math.pi if phi_c < 0 else phi_c, 6),
            "kappa": round(float(kappa), 3),
            "size": int(members.sum()),
        })
    return entries
