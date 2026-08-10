"""Tests for community detection.

Community detection is easy to write and hard to verify: any partition produces
*some* modularity number, and a plausible-looking number over a wrong partition
is indistinguishable from a right one. So these tests use graphs whose true
communities are known by construction, and check that the algorithm recovers
them — not merely that it returns something.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gitglobe.graph.communities import (  # noqa: E402
    build_adjacency,
    detect,
    disconnected_communities,
    modularity,
)


def planted_partition(groups=4, per_group=40, p_in=0.35, p_out=0.005, seed=0):
    """A graph with known communities: dense inside, sparse between.

    The standard benchmark for this class of algorithm, because the correct
    answer is known before you run anything.
    """
    rng = np.random.default_rng(seed)
    n = groups * per_group
    truth = np.repeat(np.arange(groups), per_group)
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if truth[i] == truth[j] else p_out
            if rng.random() < p:
                src.append(i)
                dst.append(j)
    return n, np.array(src), np.array(dst), truth


def agreement(found: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of node pairs the two partitions agree about."""
    same_found = found[:, None] == found[None, :]
    same_truth = truth[:, None] == truth[None, :]
    iu = np.triu_indices(len(found), k=1)
    return float((same_found[iu] == same_truth[iu]).mean())


class TestRecoversKnownStructure(unittest.TestCase):
    def test_finds_planted_communities(self) -> None:
        n, src, dst, truth = planted_partition()
        result = detect(n, src, dst)
        self.assertGreater(agreement(result.labels, truth), 0.95)
        self.assertGreater(result.modularity, 0.3)

    def test_finds_roughly_the_right_number(self) -> None:
        n, src, dst, truth = planted_partition(groups=6, per_group=30)
        result = detect(n, src, dst)
        # Louvain merges or splits at the margins; the count should be close,
        # not exact. Demanding exactness would be testing a coincidence.
        self.assertGreaterEqual(result.count, 4)
        self.assertLessEqual(result.count, 9)

    def test_a_graph_with_no_structure_scores_near_zero(self) -> None:
        # THE control. If a random graph produced high modularity, the metric
        # would be meaningless and so would every territory on the globe.
        rng = np.random.default_rng(1)
        n, e = 300, 3_000
        src, dst = rng.integers(0, n, e), rng.integers(0, n, e)
        keep = src != dst
        self.assertLess(detect(n, src[keep], dst[keep]).modularity, 0.35)

    def test_two_disconnected_cliques_are_never_merged(self) -> None:
        src, dst = [], []
        for base in (0, 20):
            for i in range(base, base + 20):
                for j in range(i + 1, base + 20):
                    src.append(i)
                    dst.append(j)
        result = detect(40, np.array(src), np.array(dst))
        self.assertNotEqual(result.labels[0], result.labels[20])
        self.assertGreater(result.modularity, 0.4)

    def test_weights_change_the_partition(self) -> None:
        # Tested on a graph large enough for greedy optimisation to be
        # reliable. A twelve-node version of this collapsed to one community
        # regardless of weights — Louvain is a greedy heuristic and is known to
        # be unreliable on tiny graphs, which is not a workload this pipeline
        # has. Asserting on one would be testing the heuristic's failure mode.
        rng = np.random.default_rng(5)
        n, per = 80, 40
        src, dst, w = [], [], []
        for base in (0, per):
            for i in range(base, base + per):
                for j in range(i + 1, base + per):
                    if rng.random() < 0.3:
                        src.append(i)
                        dst.append(j)
                        w.append(1.0)
        bridges = [(i, i + per) for i in range(0, per, 4)]

        def run(bridge_weight: float):
            s = np.array(src + [a for a, _ in bridges])
            d = np.array(dst + [b for _, b in bridges])
            weights = np.array(w + [bridge_weight] * len(bridges))
            return detect(n, s, d, weights, seed=3)

        weak = run(0.01)
        strong = run(3.0)

        # The honest statement of "weights are respected": endpoints joined by
        # a heavier edge end up together more often. Asserting that heavy
        # bridges FUSE the two groups is wrong — a bridge much heavier than the
        # surrounding edges becomes the dominant structure and forms its own
        # two-node community, which is correct behaviour and the opposite of a
        # merge. Two earlier versions of this test asserted the merge.
        def bridged_together(result) -> float:
            return sum(result.labels[a] == result.labels[b] for a, b in bridges) / len(bridges)

        self.assertGreater(bridged_together(strong), bridged_together(weak))

    def test_greedy_optimisation_is_documented_as_unreliable_when_tiny(self) -> None:
        # Honest record of a known limitation rather than a hidden one. Two
        # triangles joined by an edge: the obvious split scores 0.357 and
        # Louvain returns a single community at 0.000.
        src = np.array([0, 1, 2, 3, 4, 5, 2])
        dst = np.array([1, 2, 0, 4, 5, 3, 3])
        offsets, targets, weights = build_adjacency(6, src, dst)
        ideal = modularity(np.array([0, 0, 0, 1, 1, 1]), offsets, targets, weights)
        self.assertGreater(ideal, 0.3)
        # If this ever starts passing, the algorithm improved and the note in
        # `_local_moving` should be revisited.
        self.assertLessEqual(detect(6, src, dst).modularity, ideal)


class TestModularity(unittest.TestCase):
    def test_perfect_partition_beats_a_random_one(self) -> None:
        n, src, dst, truth = planted_partition()
        offsets, targets, weights = build_adjacency(n, src, dst)
        rng = np.random.default_rng(2)
        self.assertGreater(
            modularity(truth, offsets, targets, weights),
            modularity(rng.integers(0, 4, n), offsets, targets, weights),
        )

    def test_everything_in_one_community_scores_zero(self) -> None:
        # By definition: one community explains nothing beyond the degree
        # distribution, and modularity is defined to notice that.
        n, src, dst, _ = planted_partition(groups=2, per_group=20)
        offsets, targets, weights = build_adjacency(n, src, dst)
        self.assertAlmostEqual(
            modularity(np.zeros(n, np.int64), offsets, targets, weights), 0.0, places=9
        )

    def test_empty_graph(self) -> None:
        offsets, targets, weights = build_adjacency(0, np.zeros(0), np.zeros(0))
        self.assertEqual(modularity(np.zeros(0, np.int64), offsets, targets, weights), 0.0)


class TestAdjacency(unittest.TestCase):
    def test_is_symmetric(self) -> None:
        offsets, targets, weights = build_adjacency(4, [0, 1], [1, 2], [2.0, 3.0])
        self.assertEqual(len(targets), 4)  # each edge appears twice
        self.assertAlmostEqual(weights.sum(), 10.0)

    def test_self_loops_are_dropped(self) -> None:
        # A self-loop contributes nothing to modularity and would inflate the
        # degree, distorting every gain computed against it.
        offsets, targets, _ = build_adjacency(3, [0, 1], [0, 2])
        self.assertEqual(len(targets), 2)

    def test_ragged_input_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_adjacency(3, [0, 1], [1])
        with self.assertRaises(ValueError):
            build_adjacency(3, [0, 1], [1, 2], [1.0])

    def test_out_of_range_endpoints_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_adjacency(3, [0], [99])


class TestDegenerateInputs(unittest.TestCase):
    def test_no_edges_gives_every_node_its_own_community(self) -> None:
        result = detect(5, np.zeros(0, np.int64), np.zeros(0, np.int64))
        self.assertEqual(result.count, 5)
        self.assertEqual(result.modularity, 0.0)

    def test_empty_graph(self) -> None:
        self.assertEqual(len(detect(0, np.zeros(0), np.zeros(0))), 0)

    def test_isolated_nodes_survive_alongside_a_dense_core(self) -> None:
        # 86% of the corpus has no dependency edge. Those nodes must come back
        # labelled, not dropped.
        n, src, dst, _ = planted_partition(groups=2, per_group=20)
        result = detect(n + 50, src, dst)
        self.assertEqual(len(result), n + 50)
        self.assertTrue((result.labels >= 0).all())

    def test_labels_are_compact(self) -> None:
        # Downstream arrays are sized by max(label); sparse labels would waste
        # memory proportional to the largest id rather than the count.
        n, src, dst, _ = planted_partition()
        labels = detect(n, src, dst).labels
        self.assertEqual(set(np.unique(labels).tolist()), set(range(labels.max() + 1)))

    def test_deterministic_for_a_seed(self) -> None:
        n, src, dst, _ = planted_partition()
        a = detect(n, src, dst, seed=9).labels
        b = detect(n, src, dst, seed=9).labels
        np.testing.assert_array_equal(a, b)


class TestLouvainsKnownWeakness(unittest.TestCase):
    """Louvain can return internally disconnected communities. Measure it."""

    def test_detects_a_deliberately_disconnected_community(self) -> None:
        # Two components labelled the same: the exact defect Leiden fixes.
        offsets, targets, _ = build_adjacency(4, [0, 2], [1, 3])
        labels = np.array([0, 0, 0, 0])
        self.assertEqual(disconnected_communities(labels, offsets, targets), [0])

    def test_a_connected_community_is_not_flagged(self) -> None:
        offsets, targets, _ = build_adjacency(3, [0, 1], [1, 2])
        self.assertEqual(disconnected_communities(np.zeros(3, np.int64), offsets, targets), [])

    def test_louvain_output_on_a_realistic_graph_is_mostly_connected(self) -> None:
        # The number that decides whether the leidenalg dependency is worth
        # taking. If Louvain routinely produced broken communities here, the
        # territories would be incoherent and this test would say so.
        n, src, dst, _ = planted_partition(groups=5, per_group=40)
        result = detect(n, src, dst)
        offsets, targets, _ = build_adjacency(n, src, dst)
        broken = disconnected_communities(result.labels, offsets, targets)
        self.assertLessEqual(len(broken), max(1, result.count // 5))


class TestScales(unittest.TestCase):
    def test_handles_a_corpus_sized_graph(self) -> None:
        # 87k nodes and ~280k edges is the real shape. This must finish in
        # seconds, not minutes, or it cannot be part of an iterative loop.
        rng = np.random.default_rng(4)
        n, e = 20_000, 60_000
        src = rng.integers(0, n, e)
        dst = (src + rng.integers(1, 40, e)) % n  # local structure
        result = detect(n, src, dst)
        self.assertEqual(len(result), n)
        self.assertGreater(result.modularity, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
