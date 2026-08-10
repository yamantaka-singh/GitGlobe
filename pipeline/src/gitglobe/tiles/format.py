"""Binary tile and graph writers.

**The contract is with `web/src/tile/format.ts` and `web/src/graph/format.ts`,
and it is byte-exact.** Python writes these files; TypeScript reads them. If the
two ever disagree the globe does not error — it renders garbage, or nothing,
with no clue why. `tests/test_tile_format.py` guards this by writing a tile here
and decoding it with the actual TypeScript decoder under Node.

Layout, little-endian throughout (see the TS files for the reasoning):

    tile:  16-byte header, then structure-of-arrays
           thetaQ Int16 | phiQ Uint16 | repoId Uint32
           sizeQ Uint16 | domain Uint8 | flags Uint8      = 12 bytes/point

    graph: 24-byte header, then
           rank Float32[n] | offsets Uint32[n+1] | targets Uint32[e]
           ambient Uint32[2a] | weights Uint16[e]

Every 4-byte array precedes the single 2-byte array, which keeps each typed
array naturally aligned for any n, e, a. Reorder them and `new Uint32Array(buf,
offset, len)` throws on odd inputs only — a bug that passes every test until
real data arrives.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TAU = 2.0 * math.pi

TILE_MAGIC = 0x47475431  # 'GGT1'
TILE_HEADER_BYTES = 16
BYTES_PER_POINT = 12

GRAPH_MAGIC = 0x47474731  # 'GGG1'
GRAPH_HEADER_BYTES = 24

THETA_QUANT = 32767  # theta in [0, PI]
PHI_QUANT = 65535  # phi   in [0, TAU)
SIZE_QUANT = 65535  # size  in [0, 1]

FLAG_LOW_SIGNAL = 1 << 0
FLAG_ARCHIVED = 1 << 1
FLAG_FORK = 1 << 2

WEIGHT_OUTGOING = 0x8000
#: Bits 13-14 carry the edge kind, so arcs can be coloured by what the
#: relationship IS rather than all rendering identically. The DB has had
#: `edge.kind` since migration 002 (0=depends_on, 1=similar_to, 2=used_with);
#: the tile builder simply discarded it, which is why every arc was one colour.
#:
#: Taking two bits from the weight drops it from 32,767 levels to 8,191. The
#: weight is a normalised 0-1 ribbon opacity, so even 256 levels would be
#: indistinguishable - this costs nothing real and avoids widening the array,
#: which would change every downstream offset.
KIND_SHIFT = 13
KIND_MASK = 0x6000
MAX_KIND = 3
WEIGHT_MASK = 0x1FFF


def quantise_theta(theta: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(theta, 0.0, math.pi) / math.pi * THETA_QUANT).astype(np.int16)


def quantise_phi(phi: np.ndarray) -> np.ndarray:
    """Wrap into [0, TAU) before scaling.

    Rounding can land exactly on PHI_QUANT + 1 for phi just under TAU, which
    silently wraps to 0 in a Uint16 and teleports the point to the seam.
    """
    wrapped = np.mod(np.mod(phi, TAU) + TAU, TAU)
    return np.minimum(np.rint(wrapped / TAU * PHI_QUANT), PHI_QUANT).astype(np.uint16)


def quantise_size(size: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(size, 0.0, 1.0) * SIZE_QUANT).astype(np.uint16)


def unit_to_spherical(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors to (theta, phi) in three.js Y-up convention.

    theta from +Y, phi around the XZ plane. Must match `directionAt` in
    format.ts and the point shader's reconstruction, or the nodes slide against
    the planet surface underneath them.
    """
    y = np.clip(vectors[:, 1], -1.0, 1.0)
    theta = np.arccos(y)
    phi = np.arctan2(vectors[:, 2], vectors[:, 0])
    phi = np.where(phi < 0, phi + TAU, phi)
    return theta, phi


@dataclass
class TilePoints:
    """One LOD band's worth of points, already ordered."""

    theta: np.ndarray
    phi: np.ndarray
    repo_id: np.ndarray
    size: np.ndarray
    domain: np.ndarray
    flags: np.ndarray

    def __len__(self) -> int:
        return len(self.theta)

    def validate(self) -> None:
        n = len(self)
        for name in ("phi", "repo_id", "size", "domain", "flags"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"{name} has {got} entries, expected {n}")
        if n and int(self.repo_id.min()) == 0:
            # 0 is reserved: the pick buffer is cleared to black and decodes 0
            # as "nothing hovered".
            raise ValueError("repo_id 0 is reserved for 'nothing picked'")


def encode_tile(points: TilePoints, *, layout_version: int, lod_band: int) -> bytes:
    points.validate()
    n = len(points)

    header = struct.pack("<IIHHI", TILE_MAGIC, n, layout_version, lod_band, 0)
    body = b"".join(
        arr.tobytes()
        for arr in (
            quantise_theta(points.theta),
            quantise_phi(points.phi),
            points.repo_id.astype(np.uint32),
            quantise_size(points.size),
            points.domain.astype(np.uint8),
            points.flags.astype(np.uint8),
        )
    )

    blob = header + body
    expected = TILE_HEADER_BYTES + BYTES_PER_POINT * n
    if len(blob) != expected:
        raise AssertionError(f"tile is {len(blob)} bytes, format requires {expected}")
    return blob


@dataclass
class GraphArrays:
    """Undirected CSR plus PageRank and the pre-selected backbone."""

    rank: np.ndarray  # float32[n]
    offsets: np.ndarray  # uint32[n+1]
    targets: np.ndarray  # uint32[e]
    weights: np.ndarray  # uint16[e], bit15 = outgoing, bits13-14 = kind
    ambient: np.ndarray  # uint32[2a], flat (src, dst) pairs
    #: Bumped 1 -> 2 when kind moved into bits 13-14. A version-1 file used all
    #: 15 low bits for weight, so reading one with a version-2 decoder yields
    #: plausible-looking arcs in arbitrary colours — wrong, but not obviously
    #: wrong. The reader rejects mismatched versions so a stale graph.bin fails
    #: loudly instead of quietly lying.
    layout_version: int = 2

    def validate(self) -> None:
        n = len(self.rank)
        if len(self.offsets) != n + 1:
            raise ValueError(f"offsets must be n+1 ({n + 1}), got {len(self.offsets)}")
        if len(self.weights) != len(self.targets):
            raise ValueError("weights and targets must be the same length")
        if len(self.ambient) % 2:
            raise ValueError("ambient must hold whole (src, dst) pairs")
        if int(self.offsets[-1]) != len(self.targets):
            raise ValueError(
                f"offsets end at {int(self.offsets[-1])} but there are {len(self.targets)} targets"
            )


def encode_graph(graph: GraphArrays) -> bytes:
    graph.validate()
    n = len(graph.rank)
    e = len(graph.targets)
    a = len(graph.ambient) // 2

    header = struct.pack("<IIIIH6x", GRAPH_MAGIC, n, e, a, graph.layout_version)
    body = b"".join(
        arr.tobytes()
        for arr in (
            graph.rank.astype(np.float32),
            graph.offsets.astype(np.uint32),
            graph.targets.astype(np.uint32),
            graph.ambient.astype(np.uint32),
            graph.weights.astype(np.uint16),
        )
    )

    blob = header + body
    expected = GRAPH_HEADER_BYTES + 4 * n + 4 * (n + 1) + 4 * e + 8 * a + 2 * e
    if len(blob) != expected:
        raise AssertionError(f"graph is {len(blob)} bytes, format requires {expected}")
    return blob


def pack_weight_and_kind(
    weight: Sequence[float] | np.ndarray,
    kind: Sequence[int] | np.ndarray | None,
    count: int,
) -> np.ndarray:
    """Quantised weight in bits 0-12, edge kind in bits 13-14.

    Split out of `build_undirected_csr` because adding kind support pushed that
    function to 63 code lines, and the Power of 10 rule 4 check refused it. The
    packing is a genuinely separable concern, so this is a better shape anyway.
    """
    w = np.clip(
        np.asarray(weight, dtype=np.float64) * WEIGHT_MASK, 0, WEIGHT_MASK
    ).astype(np.uint16)

    if kind is None:
        return w

    k = np.asarray(kind, dtype=np.int64)
    if len(k) != count:
        raise ValueError(f"kind has {len(k)} entries, src has {count}")
    # Rule 7. An out-of-range kind would silently overflow into the direction
    # bit and reverse the arrow — corruption that still decodes and still
    # renders, which is the worst kind.
    if len(k) and (int(k.max()) > MAX_KIND or int(k.min()) < 0):
        raise ValueError(
            f"edge kind must be 0..{MAX_KIND}, got {int(k.min())}..{int(k.max())}"
        )
    return w | (k.astype(np.uint16) << KIND_SHIFT)


def build_undirected_csr(
    n: int,
    src: Sequence[int] | np.ndarray,
    dst: Sequence[int] | np.ndarray,
    weight: Sequence[float] | np.ndarray,
    kind: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed edge list to undirected CSR, each edge in both rows.

    The renderer's only query is "everything this repository touches", which
    wants both directions available in one O(1) lookup. Direction survives in
    the weight's high bit, and the edge kind in bits 13-14.

    `kind` defaults to zeros so existing callers keep working, but a caller that
    HAS kinds and forgets to pass them gets a silently monochrome graph — so
    `build.py` passes them explicitly.
    """
    src = np.asarray(src, dtype=np.uint32)
    dst = np.asarray(dst, dtype=np.uint32)
    w = pack_weight_and_kind(weight, kind, len(src))

    # Power of 10 rule 7 — validate every parameter. Three parallel arrays, and
    # a length mismatch here does NOT raise: `zip` stops at the shortest, so
    # edges silently vanish and the CSR is quietly incomplete. An endpoint past
    # `n` is worse — it writes into another node's row and corrupts the graph
    # in a way that still decodes and still renders.
    if not (len(src) == len(dst) == len(w)):
        raise ValueError(
            f"src/dst/weight lengths differ: {len(src)}, {len(dst)}, {len(w)}"
        )
    if len(src) and (int(src.max()) >= n or int(dst.max()) >= n):
        raise ValueError(
            f"edge endpoint out of range for {n} nodes "
            f"(max src {int(src.max())}, max dst {int(dst.max())})"
        )

    degree = np.zeros(n, dtype=np.uint32)
    np.add.at(degree, src, 1)
    np.add.at(degree, dst, 1)

    offsets = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum(degree, out=offsets[1:])

    targets = np.zeros(offsets[-1], dtype=np.uint32)
    weights = np.zeros(offsets[-1], dtype=np.uint16)
    cursor = offsets[:n].copy()

    for i in range(len(src)):
        a, b = int(src[i]), int(dst[i])
        targets[cursor[a]] = b
        weights[cursor[a]] = w[i] | WEIGHT_OUTGOING
        cursor[a] += 1
        targets[cursor[b]] = a
        weights[cursor[b]] = w[i]
        cursor[b] += 1

    return offsets, targets, weights


@dataclass
class BandSpec:
    """How the ranked node list is split into LOD bands."""

    fractions: tuple[float, ...] = (0.02, 0.18, 0.80)

    def sizes(self, total: int) -> list[int]:
        sizes = [round(total * f) for f in self.fractions]
        # The last band absorbs rounding so the bands always sum to `total`.
        sizes[-1] = total - sum(sizes[:-1])
        if any(s < 0 for s in sizes):
            raise ValueError(f"band fractions produce negative sizes for total={total}")
        return sizes


@dataclass
class ManifestBand:
    band: int
    count: int
    bytes: int
    file: str
    #: Band-aligned repository names. Entry i is the name of point i in this
    #: band's tile, so no id lookup is needed on the client.
    names: str | None = None


@dataclass
class Manifest:
    """The manifest `web/src/tile/loader.ts` reads.

    Field names are camelCase because they cross into TypeScript. Changing one
    here without changing `TileManifest` there produces `undefined` at runtime,
    not a build error — the loader's own version check is the only thing that
    catches format drift, and it only covers `layoutVersion`.
    """

    layout_version: int
    total: int
    bands: list[ManifestBand] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    graph: dict | None = None
    synthetic: bool = False
    seed: int = 0
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "layoutVersion": self.layout_version,
            "generatedAt": self.generated_at,
            "seed": self.seed,
            "total": self.total,
            "synthetic": self.synthetic,
            "bands": [
                {
                    "band": b.band,
                    "count": b.count,
                    "bytes": b.bytes,
                    "file": b.file,
                    **({"names": b.names} if b.names else {}),
                }
                for b in self.bands
            ],
            "domains": self.domains,
            "graph": self.graph,
            "clusters": self.clusters,
        }
