"""End-to-end: Python builds a world, the TypeScript verifier judges it.

`web/scripts/verify-world.ts` is 30 checks written against the synthetic
generator — quantisation overflow, id collisions, a CSR that points off the end
of itself, PageRank that stopped summing to 1. Every one applies to the real
pipeline output too.

Re-implementing those checks in Python would mean two definitions of "correct",
free to drift. Instead the Python writer produces a world and the *same*
verifier the web build uses is run over it. If Python and TypeScript ever
disagree about the format, this fails.
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

from gitglobe.graph.pagerank import pagerank  # noqa: E402
from gitglobe.tiles.build import (  # noqa: E402
    WorldInput,
    build_world,
    cluster_manifest_entries,
    pack_flags,
    rank_order,
    select_ambient,
)
from gitglobe.tiles.format import FLAG_ARCHIVED, FLAG_FORK, FLAG_LOW_SIGNAL, TAU  # noqa: E402

WEB = Path(__file__).resolve().parents[2] / "web"
VERIFIER = WEB / "scripts" / "verify-world.ts"
NODE = shutil.which("node")

DOMAINS = [f"domain-{i}" for i in range(12)]


def node_available() -> bool:
    if not NODE or not VERIFIER.exists():
        return False
    probe = subprocess.run(
        [NODE, "--experimental-strip-types", "-e", "console.log(1)"], capture_output=True
    )
    return probe.returncode == 0


def synthetic_world(n=3000, n_edges=9000, seed=1):
    """A corpus shaped like the real one: clustered, scale-free, mostly dangling."""
    rng = np.random.default_rng(seed)

    # Twelve concentrated blobs, so domains are contiguous the way real
    # clusters are — uniform noise would hide domain-assignment bugs.
    poles = rng.normal(size=(12, 3))
    poles /= np.linalg.norm(poles, axis=1, keepdims=True)
    which = rng.integers(0, 12, n)
    vectors = poles[which] + rng.normal(scale=0.25, size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    theta = np.arccos(np.clip(vectors[:, 1], -1, 1))
    phi = np.mod(np.arctan2(vectors[:, 2], vectors[:, 0]), TAU)

    # Preferential attachment: edges land on low-numbered nodes, which is what
    # gives PageRank orders of magnitude to spread over.
    src = rng.integers(0, n, n_edges)
    dst = (rng.pareto(1.0, n_edges) * 12).astype(int) % n
    keep = src != dst
    src, dst = src[keep], dst[keep]
    weight = rng.random(len(src)) * 4

    repo_ids = rng.permutation(np.arange(5_000_000, 5_000_000 + n))  # non-contiguous DB ids
    result = pagerank(n, src, dst, weight)

    world = WorldInput(
        repo_id=repo_ids,
        full_name=np.array([f"org{i % 40}/repo-{i}" for i in range(n)]),
        theta=theta,
        phi=phi,
        rank=result.rank,
        domain=which.astype(np.uint8),
        cluster_id=which.astype(np.int32),
        low_signal=rng.random(n) < 0.1,
        is_archived=rng.random(n) < 0.05,
        is_fork=np.zeros(n, bool),
    )
    edges = (repo_ids[src], repo_ids[dst], weight)
    return world, edges, result


@unittest.skipUnless(node_available(), "node with --experimental-strip-types required")
class TestVerifiedByTypeScript(unittest.TestCase):
    def test_a_python_built_world_passes_every_check(self) -> None:
        world, edges, result = synthetic_world()
        with tempfile.TemporaryDirectory() as tmp:
            build_world(
                world,
                edges=edges,
                pagerank_result=result,
                out_dir=Path(tmp),
                domains=DOMAINS,
                clusters=cluster_manifest_entries(
                    world.cluster_id, world.theta, world.phi, world.domain
                ),
            )
            proc = subprocess.run(
                [NODE, "--experimental-strip-types", str(VERIFIER), tmp],
                capture_output=True, text=True, cwd=str(WEB),
            )
        if proc.returncode != 0:
            failed = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("FAIL")]
            self.fail("verify-world.ts rejected Python's output:\n" + "\n".join(failed or [proc.stderr]))

    def test_the_verifier_actually_catches_a_broken_world(self) -> None:
        """A test that always passes is worse than no test. Prove it can fail."""
        world, edges, result = synthetic_world()
        with tempfile.TemporaryDirectory() as tmp:
            build_world(
                world, edges=edges, pagerank_result=result,
                out_dir=Path(tmp), domains=DOMAINS,
            )
            # Claim more points than the tiles contain.
            path = Path(tmp) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["total"] += 7
            path.write_text(json.dumps(manifest))

            proc = subprocess.run(
                [NODE, "--experimental-strip-types", str(VERIFIER), tmp],
                capture_output=True, text=True, cwd=str(WEB),
            )
        self.assertNotEqual(proc.returncode, 0, "verifier passed a world it should have rejected")


class TestNodeOrdering(unittest.TestCase):
    def test_rank_order_is_descending_and_stable(self) -> None:
        rank = np.array([0.1, 0.5, 0.5, 0.2])
        order = rank_order(rank)
        np.testing.assert_array_equal(order, [1, 2, 3, 0])

    def test_equal_ranks_keep_input_order_across_runs(self) -> None:
        # Unstable sorting reshuffles every node id between runs, so the whole
        # globe looks rebuilt when nothing changed.
        rank = np.full(500, 0.002)
        np.testing.assert_array_equal(rank_order(rank), np.arange(500))


class TestFlags(unittest.TestCase):
    def test_bits_combine(self) -> None:
        flags = pack_flags(
            np.array([True, False, True]),
            np.array([False, True, True]),
            np.array([False, False, True]),
        )
        self.assertEqual(flags[0], FLAG_LOW_SIGNAL)
        self.assertEqual(flags[1], FLAG_ARCHIVED)
        self.assertEqual(flags[2], FLAG_LOW_SIGNAL | FLAG_ARCHIVED | FLAG_FORK)
        self.assertEqual(flags.dtype, np.uint8)


class TestAmbientSelection(unittest.TestCase):
    def test_no_hub_can_claim_every_slot(self) -> None:
        # Without a per-node cap the top hub takes all of them and the ambient
        # layer renders as one starburst instead of a network.
        n_edges = 200
        src = np.zeros(n_edges, np.uint32)          # node 0 is in every edge
        dst = np.arange(1, n_edges + 1, dtype=np.uint32)
        rank = np.full(n_edges + 1, 0.001)
        rank[0] = 0.5
        ambient = select_ambient(src, dst, np.ones(n_edges), rank, limit=50, max_per_node=4)
        self.assertEqual(len(ambient) // 2, 4)

    def test_high_rank_edges_are_preferred(self) -> None:
        src = np.array([0, 2], np.uint32)
        dst = np.array([1, 3], np.uint32)
        rank = np.array([0.4, 0.4, 0.001, 0.001])
        ambient = select_ambient(src, dst, np.ones(2), rank, limit=1)
        self.assertEqual(ambient.tolist(), [0, 1])

    def test_no_edges(self) -> None:
        empty = np.zeros(0, np.uint32)
        self.assertEqual(len(select_ambient(empty, empty, np.zeros(0), np.zeros(3))), 0)


class TestBuildBehaviour(unittest.TestCase):
    def test_repo_ids_are_ordinal_plus_one_across_bands(self) -> None:
        # The whole scheme rests on this: tile repoId - 1 == graph node index.
        world, edges, result = synthetic_world(n=1500)
        with tempfile.TemporaryDirectory() as tmp:
            out = build_world(
                world, edges=edges, pagerank_result=result,
                out_dir=Path(tmp), domains=DOMAINS,
            )
            seen = []
            for band in out.manifest.bands:
                blob = (Path(tmp) / band.file).read_bytes()
                count = int.from_bytes(blob[4:8], "little")
                ids = np.frombuffer(blob, np.uint32, count=count, offset=16 + 4 * count)
                seen.append(ids)
            all_ids = np.concatenate(seen)
        np.testing.assert_array_equal(all_ids, np.arange(1, len(world) + 1))

    def test_edges_to_unprojected_repos_are_dropped_not_fatal(self) -> None:
        # A partial dataset must still build — that is what makes the 5k proof
        # run possible before everything is embedded.
        world, (src, dst, w), result = synthetic_world(n=800)
        src = np.concatenate([src, np.array([999_999_999])])
        dst = np.concatenate([dst, np.array([world.repo_id[0]])])
        w = np.concatenate([w, np.array([1.0])])
        with tempfile.TemporaryDirectory() as tmp:
            out = build_world(
                world, edges=(src, dst, w), pagerank_result=result,
                out_dir=Path(tmp), domains=DOMAINS,
            )
        self.assertEqual(out.manifest.total, 800)

    def test_names_files_align_with_the_tiles(self) -> None:
        world, edges, result = synthetic_world(n=1200)
        with tempfile.TemporaryDirectory() as tmp:
            out = build_world(
                world, edges=edges, pagerank_result=result,
                out_dir=Path(tmp), domains=DOMAINS,
            )
            names = []
            for band in out.manifest.bands:
                entries = json.loads((Path(tmp) / band.names).read_text())
                self.assertEqual(len(entries), band.count)
                names += entries
        # Band order is rank order, so the highest-ranked repo is first.
        self.assertEqual(names[0], world.full_name[rank_order(world.rank)[0]])
        self.assertEqual(len(set(names)), len(names))

    def test_a_world_with_no_edges_still_builds(self) -> None:
        world, _, result = synthetic_world(n=600)
        empty = (np.zeros(0), np.zeros(0), np.zeros(0))
        with tempfile.TemporaryDirectory() as tmp:
            out = build_world(
                world, edges=empty, pagerank_result=result,
                out_dir=Path(tmp), domains=DOMAINS,
            )
        self.assertEqual(out.manifest.graph["csrEntries"], 0)
        self.assertEqual(out.manifest.graph["ambientArcs"], 0)

    def test_nan_positions_are_refused(self) -> None:
        # UMAP that failed halfway leaves NaN. Encoding it produces a tile that
        # decodes fine and renders points at undefined locations.
        world, edges, result = synthetic_world(n=400)
        world.theta[3] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_world(
                    world, edges=edges, pagerank_result=result,
                    out_dir=Path(tmp), domains=DOMAINS,
                )


class TestClusterEntries(unittest.TestCase):
    def test_centre_lands_inside_a_tight_cluster(self) -> None:
        rng = np.random.default_rng(3)
        v = np.array([0.0, 0.0, 1.0]) + rng.normal(scale=0.05, size=(400, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        theta = np.arccos(np.clip(v[:, 1], -1, 1))
        phi = np.mod(np.arctan2(v[:, 2], v[:, 0]), TAU)
        entries = cluster_manifest_entries(
            np.zeros(400, np.int32), theta, phi, np.zeros(400, np.uint8)
        )
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]["theta"], math.pi / 2, delta=0.1)
        self.assertGreater(entries[0]["kappa"], 50)  # tight

    def test_a_diffuse_cluster_has_low_kappa(self) -> None:
        # The camera reads kappa to pick a framing distance. A tight and a
        # diffuse cluster must not report the same number.
        rng = np.random.default_rng(6)
        theta = np.arccos(2 * rng.random(2000) - 1)
        phi = rng.random(2000) * TAU
        entry = cluster_manifest_entries(
            np.zeros(2000, np.int32), theta, phi, np.zeros(2000, np.uint8)
        )[0]
        self.assertLess(entry["kappa"], 5)

    def test_noise_label_is_excluded(self) -> None:
        self.assertEqual(
            cluster_manifest_entries(
                np.array([-1, -1]), np.zeros(2), np.zeros(2), np.zeros(2, np.uint8)
            ),
            [],
        )

    def test_phi_is_in_range(self) -> None:
        rng = np.random.default_rng(8)
        theta = np.arccos(2 * rng.random(900) - 1)
        phi = rng.random(900) * TAU
        ids = rng.integers(0, 9, 900).astype(np.int32)
        for entry in cluster_manifest_entries(ids, theta, phi, np.zeros(900, np.uint8)):
            self.assertGreaterEqual(entry["phi"], 0.0)
            self.assertLess(entry["phi"], TAU + 1e-6)
            self.assertGreaterEqual(entry["theta"], 0.0)
            self.assertLessEqual(entry["theta"], math.pi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
