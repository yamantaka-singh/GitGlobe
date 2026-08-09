"""Cross-language format tests: Python writes, TypeScript reads.

This is the single most important test in Phase 2.

Python produces the tiles; the browser decodes them with
`web/src/tile/format.ts`. If the two ever disagree about a byte, the globe does
not raise — it renders garbage, or an empty sphere, with nothing in the console
to explain it. Unit-testing the Python encoder against the Python decoder would
prove only that Python agrees with itself.

So these tests write real bytes and hand them to the **actual TypeScript
decoder** running under Node. Any drift between the two implementations fails
here, loudly, with a diff.

Skipped if Node is unavailable — the rest of the suite must still run anywhere.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gitglobe.tiles.format import (  # noqa: E402
    TAU,
    BandSpec,
    GraphArrays,
    TilePoints,
    build_undirected_csr,
    encode_graph,
    encode_tile,
    quantise_phi,
    quantise_theta,
    unit_to_spherical,
)

WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"
NODE = shutil.which("node")


def node_available() -> bool:
    if not NODE or not (WEB_SRC / "tile" / "format.ts").exists():
        return False
    # Type stripping is what lets us import the .ts decoder directly.
    probe = subprocess.run(
        [NODE, "--experimental-strip-types", "-e", "console.log('ok')"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0


@unittest.skipUnless(node_available(), "node with --experimental-strip-types required")
class TestTileReadableByTypeScript(unittest.TestCase):
    """Decode Python's output with the decoder the browser actually uses."""

    def decode_with_ts(self, blob: bytes) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tile_path = Path(tmp) / "band.bin"
            tile_path.write_bytes(blob)
            script = Path(tmp) / "decode.mjs"
            script.write_text(f"""
import {{ readFileSync }} from 'node:fs';
import {{ decodeTile, dequantiseTheta, dequantisePhi, dequantiseSize, directionAt }}
  from '{WEB_SRC / "tile" / "format.ts"}';

const buf = readFileSync({str(tile_path)!r});
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const tile = decodeTile(ab);

const norms = [];
for (let i = 0; i < tile.count; i++) {{
  const [x, y, z] = directionAt(tile, i);
  norms.push(Math.hypot(x, y, z));
}}

console.log(JSON.stringify({{
  count: tile.count,
  layoutVersion: tile.layoutVersion,
  lodBand: tile.lodBand,
  theta: Array.from(tile.thetaQ).map(dequantiseTheta),
  phi: Array.from(tile.phiQ).map(dequantisePhi),
  repoId: Array.from(tile.repoId),
  size: Array.from(tile.sizeQ).map(dequantiseSize),
  domain: Array.from(tile.domain),
  flags: Array.from(tile.flags),
  worstNormError: Math.max(...norms.map((n) => Math.abs(n - 1)), 0),
}}));
""")
            result = subprocess.run(
                [NODE, "--experimental-strip-types", str(script)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                self.fail(f"TypeScript decoder rejected Python's bytes:\n{result.stderr}")
            return json.loads(result.stdout)

    def test_a_realistic_tile_round_trips(self) -> None:
        rng = np.random.default_rng(7)
        n = 500
        # Uniform on the sphere: inverse-CDF on cos(theta). Uniform theta would
        # clump at the poles and hide errors that only show at the equator.
        theta = np.arccos(2 * rng.random(n) - 1)
        phi = rng.random(n) * TAU

        points = TilePoints(
            theta=theta,
            phi=phi,
            repo_id=np.arange(1, n + 1, dtype=np.uint32),
            size=rng.random(n),
            domain=rng.integers(0, 12, n).astype(np.uint8),
            flags=rng.integers(0, 4, n).astype(np.uint8),
        )
        decoded = self.decode_with_ts(encode_tile(points, layout_version=3, lod_band=1))

        self.assertEqual(decoded["count"], n)
        self.assertEqual(decoded["layoutVersion"], 3)
        self.assertEqual(decoded["lodBand"], 1)
        self.assertEqual(decoded["repoId"], list(range(1, n + 1)))
        self.assertEqual(decoded["domain"], points.domain.tolist())
        self.assertEqual(decoded["flags"], points.flags.tolist())

        # Angles survive a quantisation round trip through two languages.
        for i in range(n):
            self.assertAlmostEqual(decoded["theta"][i], theta[i], delta=1e-4)
            self.assertAlmostEqual(decoded["phi"][i], phi[i], delta=2e-4)
            self.assertAlmostEqual(decoded["size"][i], points.size[i], delta=1e-4)

        self.assertLess(decoded["worstNormError"], 1e-4)

    def test_byte_length_matches_the_format(self) -> None:
        for n in (0, 1, 3, 1000):
            points = TilePoints(
                theta=np.zeros(n), phi=np.zeros(n),
                repo_id=np.arange(1, n + 1, dtype=np.uint32),
                size=np.zeros(n), domain=np.zeros(n, np.uint8), flags=np.zeros(n, np.uint8),
            )
            with self.subTest(n=n):
                self.assertEqual(len(encode_tile(points, layout_version=1, lod_band=0)), 16 + 12 * n)

    def test_poles_and_the_phi_seam(self) -> None:
        """The values most likely to overflow their field or wrap wrongly."""
        theta = np.array([0.0, math.pi, math.pi / 2, math.pi / 2, math.pi / 2])
        phi = np.array([0.0, 0.0, TAU - 1e-9, TAU, -0.1])
        points = TilePoints(
            theta=theta, phi=phi,
            repo_id=np.arange(1, 6, dtype=np.uint32),
            size=np.array([0.0, 1.0, 0.5, 0.5, 0.5]),
            domain=np.zeros(5, np.uint8), flags=np.zeros(5, np.uint8),
        )
        decoded = self.decode_with_ts(encode_tile(points, layout_version=1, lod_band=0))

        self.assertAlmostEqual(decoded["theta"][0], 0.0, places=4)
        self.assertAlmostEqual(decoded["theta"][1], math.pi, places=4)
        # phi = TAU wraps to 0; phi = -0.1 wraps to TAU - 0.1.
        self.assertAlmostEqual(decoded["phi"][3], 0.0, places=3)
        self.assertAlmostEqual(decoded["phi"][4], TAU - 0.1, places=3)
        self.assertLess(decoded["worstNormError"], 1e-4)


@unittest.skipUnless(node_available(), "node with --experimental-strip-types required")
class TestGraphReadableByTypeScript(unittest.TestCase):
    def test_graph_round_trips_and_stays_aligned(self) -> None:
        # Odd n, e and a on purpose: misordered arrays only break alignment for
        # some values, so even counts would pass a broken layout.
        n, a = 7, 3
        src = np.array([0, 1, 2, 3, 5], dtype=np.uint32)
        dst = np.array([1, 2, 3, 4, 6], dtype=np.uint32)
        offsets, targets, weights = build_undirected_csr(n, src, dst, np.linspace(0.1, 0.9, len(src)))

        graph = GraphArrays(
            rank=np.full(n, 1 / n, dtype=np.float32),
            offsets=offsets, targets=targets, weights=weights,
            ambient=np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
            layout_version=2,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.bin"
            path.write_bytes(encode_graph(graph))
            script = Path(tmp) / "decode.mjs"
            script.write_text(f"""
import {{ readFileSync }} from 'node:fs';
import {{ decodeGraph, neighboursOf, degreeOf }} from '{WEB_SRC / "graph" / "format.ts"}';
const buf = readFileSync({str(path)!r});
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const g = decodeGraph(ab);
console.log(JSON.stringify({{
  nodeCount: g.nodeCount, edgeCount: g.edgeCount, ambientCount: g.ambientCount,
  layoutVersion: g.layoutVersion,
  offsets: Array.from(g.offsets),
  ambient: Array.from(g.ambient),
  n0: neighboursOf(g, 0).map((x) => [x.node, x.outgoing]),
  n1: neighboursOf(g, 1).map((x) => x.node).sort(),
  deg0: degreeOf(g, 0),
  rankSum: Array.from(g.rank).reduce((s, v) => s + v, 0),
}}));
""")
            result = subprocess.run(
                [NODE, "--experimental-strip-types", str(script)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                self.fail(f"TypeScript rejected Python's graph bytes:\n{result.stderr}")
            decoded = json.loads(result.stdout)

        self.assertEqual(decoded["nodeCount"], n)
        self.assertEqual(decoded["edgeCount"], 2 * len(src))
        self.assertEqual(decoded["ambientCount"], a)
        self.assertEqual(decoded["layoutVersion"], 2)
        self.assertEqual(decoded["offsets"], offsets.tolist())
        self.assertAlmostEqual(decoded["rankSum"], 1.0, places=5)

        # Direction survives the weight's high bit across the language boundary.
        self.assertEqual(decoded["n0"], [[1, True]])
        self.assertEqual(decoded["n1"], [0, 2])
        self.assertEqual(decoded["deg0"], {"in": 0, "out": 1})


class TestQuantisationBounds(unittest.TestCase):
    """Runs everywhere — no Node needed."""

    def test_no_quantised_value_overflows_its_field(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.random(50_000) * math.pi
        phi = rng.random(50_000) * TAU
        self.assertTrue((quantise_theta(theta) >= 0).all())
        self.assertTrue((quantise_theta(theta) <= 32767).all())
        self.assertTrue((quantise_phi(phi) <= 65535).all())

    def test_phi_just_below_tau_does_not_wrap_to_zero(self) -> None:
        # Rounding can land on 65536, which a uint16 silently turns into 0 —
        # teleporting the point across the seam.
        self.assertLessEqual(int(quantise_phi(np.array([TAU - 1e-12]))[0]), 65535)

    def test_spherical_conversion_is_y_up(self) -> None:
        vectors = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        theta, phi = unit_to_spherical(vectors)
        self.assertAlmostEqual(theta[0], 0.0)          # +Y is the north pole
        self.assertAlmostEqual(theta[1], math.pi)      # -Y is the south pole
        self.assertAlmostEqual(theta[2], math.pi / 2)  # +X is on the equator
        self.assertAlmostEqual(phi[2], 0.0)

    def test_conversion_round_trips(self) -> None:
        rng = np.random.default_rng(3)
        v = rng.normal(size=(2000, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        theta, phi = unit_to_spherical(v)
        st = np.sin(theta)
        back = np.column_stack([st * np.cos(phi), np.cos(theta), st * np.sin(phi)])
        self.assertLess(np.abs(back - v).max(), 1e-9)


class TestBandSpec(unittest.TestCase):
    def test_bands_always_sum_to_the_total(self) -> None:
        spec = BandSpec()
        for total in (0, 1, 7, 4999, 100_000, 999_983):
            with self.subTest(total=total):
                self.assertEqual(sum(spec.sizes(total)), total)

    def test_band_zero_is_the_smallest(self) -> None:
        sizes = BandSpec().sizes(100_000)
        self.assertEqual(sizes, [2000, 18000, 80000])


class TestEncoderRejectsBadInput(unittest.TestCase):
    def test_repo_id_zero_is_refused(self) -> None:
        # 0 decodes as "nothing hovered" in the pick buffer, so a real node
        # holding it would be permanently unselectable.
        points = TilePoints(
            theta=np.zeros(2), phi=np.zeros(2), repo_id=np.array([0, 1], np.uint32),
            size=np.zeros(2), domain=np.zeros(2, np.uint8), flags=np.zeros(2, np.uint8),
        )
        with self.assertRaises(ValueError):
            encode_tile(points, layout_version=1, lod_band=0)

    def test_mismatched_array_lengths_are_refused(self) -> None:
        points = TilePoints(
            theta=np.zeros(3), phi=np.zeros(2), repo_id=np.array([1, 2, 3], np.uint32),
            size=np.zeros(3), domain=np.zeros(3, np.uint8), flags=np.zeros(3, np.uint8),
        )
        with self.assertRaises(ValueError):
            encode_tile(points, layout_version=1, lod_band=0)

    def test_inconsistent_csr_is_refused(self) -> None:
        graph = GraphArrays(
            rank=np.zeros(3, np.float32),
            offsets=np.array([0, 1, 2], np.uint32),  # should be length 4
            targets=np.zeros(2, np.uint32), weights=np.zeros(2, np.uint16),
            ambient=np.zeros(0, np.uint32),
        )
        with self.assertRaises(ValueError):
            encode_graph(graph)


if __name__ == "__main__":
    unittest.main(verbosity=2)
