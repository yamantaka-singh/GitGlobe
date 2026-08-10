"""PageRank tests.

Two kinds of assertion here. The cheap kind checks invariants — sums to 1,
symmetric graphs give symmetric ranks. The expensive kind checks the properties
the *product* depends on: that a repository nothing depends on still gets a
usable rank, and that node sizes spread out instead of bunching.

The second kind is what catches the bugs that matter. A PageRank that leaks
dangling mass still passes "higher in-degree ranks higher" — it just quietly
returns numbers that mean nothing across runs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gitglobe.graph.pagerank import (  # noqa: E402
    banded_display_size,
    combine_edges,
    importance_order,
    pagerank,
    to_display_size,
)


def realistic_ranks(n=3000, seed=1):
    """PageRank shaped like a real corpus: a long tail pinned to the floor."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, n * 3)
    dst = (rng.pareto(1.0, n * 3) * 12).astype(int) % n
    keep = src != dst
    return pagerank(n, src[keep], dst[keep]).rank


class TestInvariants(unittest.TestCase):
    def test_rank_sums_to_one_even_with_dangling_nodes(self) -> None:
        # 0 -> 1, 2 has no out-edges. 2 is dangling; if its mass is dropped the
        # total shrinks a little every iteration.
        r = pagerank(3, np.array([0]), np.array([1]))
        self.assertAlmostEqual(r.rank.sum(), 1.0, places=9)
        self.assertTrue(r.converged)

    def test_a_graph_of_only_dangling_nodes_stays_uniform(self) -> None:
        r = pagerank(5, np.empty(0, np.int64), np.empty(0, np.int64))
        self.assertAlmostEqual(r.rank.sum(), 1.0, places=9)
        np.testing.assert_allclose(r.rank, 0.2, atol=1e-12)

    def test_no_node_is_ever_worse_than_the_teleport_floor(self) -> None:
        # An isolated node's rank must not be zero. It becomes a node size and a
        # sort key; zero is not a usable value for either.
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        r = pagerank(10, src, dst)  # nodes 3..9 are isolated
        self.assertTrue((r.rank > 0).all())
        self.assertGreaterEqual(r.rank[9], 0.15 / 10 * 0.99)

    def test_symmetric_graph_gives_symmetric_rank(self) -> None:
        src = np.array([0, 1, 1, 2, 2, 0])
        dst = np.array([1, 0, 2, 1, 0, 2])
        r = pagerank(3, src, dst)
        self.assertAlmostEqual(r.rank.std(), 0.0, places=9)

    def test_empty_graph(self) -> None:
        r = pagerank(0, np.empty(0, np.int64), np.empty(0, np.int64))
        self.assertEqual(len(r.rank), 0)


class TestRankingIsSensible(unittest.TestCase):
    def test_a_hub_outranks_its_dependents(self) -> None:
        # Nine repos all depending on repo 0 — the "everyone uses this" shape.
        src = np.arange(1, 10)
        dst = np.zeros(9, np.int64)
        r = pagerank(10, src, dst)
        self.assertEqual(r.top(1)[0], 0)
        self.assertGreater(r.rank[0], 5 * r.rank[1])

    def test_a_dependent_of_a_hub_outranks_an_equally_popular_leaf(self) -> None:
        # This is PageRank's whole point: WHO depends on you, not how many.
        # 1 has one dependent (the hub 0). 2 has one dependent (the nobody 3).
        src = np.array([0, 4, 5, 6, 7, 3])
        dst = np.array([1, 0, 0, 0, 0, 2])
        r = pagerank(8, src, dst)
        self.assertGreater(r.rank[1], r.rank[2])

    def test_weights_shift_rank(self) -> None:
        src = np.array([0, 0])
        dst = np.array([1, 2])
        even = pagerank(3, src, dst, np.array([1.0, 1.0]))
        self.assertAlmostEqual(even.rank[1], even.rank[2], places=9)

        skewed = pagerank(3, src, dst, np.array([9.0, 1.0]))
        self.assertGreater(skewed.rank[1], skewed.rank[2])
        self.assertAlmostEqual(skewed.rank.sum(), 1.0, places=9)

    def test_scaling_all_weights_changes_nothing(self) -> None:
        # Weights are row-normalised, so absolute magnitude must not matter.
        # If it does, mixing PPMI (~0-8) with dependency counts silently
        # hands one layer the whole ranking.
        src = np.array([0, 0, 1])
        dst = np.array([1, 2, 2])
        small = pagerank(3, src, dst, np.array([0.001, 0.002, 0.001]))
        large = pagerank(3, src, dst, np.array([1000.0, 2000.0, 1000.0]))
        np.testing.assert_allclose(small.rank, large.rank, atol=1e-9)

    def test_the_real_shape_converges(self) -> None:
        # Scale-free, mostly-dangling — GitHub's actual shape. 60% of nodes
        # depend on nothing.
        rng = np.random.default_rng(5)
        n = 5_000
        src = rng.integers(0, n, 20_000)
        dst = rng.integers(0, int(n * 0.4), 20_000)  # edges point into the core
        keep = src != dst
        r = pagerank(n, src[keep], dst[keep])
        self.assertTrue(r.converged, f"did not converge in {r.iterations} (delta {r.delta:.2e})")
        self.assertAlmostEqual(r.rank.sum(), 1.0, places=8)
        self.assertLess(r.iterations, 120)


class TestBadInput(unittest.TestCase):
    def test_out_of_range_endpoint_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            pagerank(3, np.array([0]), np.array([7]))

    def test_negative_weight_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            pagerank(2, np.array([0]), np.array([1]), np.array([-1.0]))

    def test_mismatched_edge_arrays_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            pagerank(3, np.array([0, 1]), np.array([1]))


class TestCombineEdges(unittest.TestCase):
    def test_layers_are_normalised_before_scaling(self) -> None:
        # Dependency weights are counts; used_with weights are PPMI. Raw
        # concatenation lets whichever layer has bigger numbers dominate.
        deps = (np.array([0, 1]), np.array([1, 2]), np.array([1.0, 3.0]), 1.0)
        used = (np.array([2]), np.array([0]), np.array([600.0]), 1.0)
        _, _, w = combine_edges([deps, used])
        # After mean-normalisation the single used_with edge is exactly 1.0,
        # not 600.
        self.assertAlmostEqual(w[2], 1.0, places=9)
        self.assertAlmostEqual(w[:2].mean(), 1.0, places=9)

    def test_scale_controls_influence(self) -> None:
        a = (np.array([0]), np.array([1]), np.array([1.0]), 1.0)
        b = (np.array([0]), np.array([2]), np.array([1.0]), 0.25)
        src, dst, w = combine_edges([a, b])
        r = pagerank(3, src, dst, w)
        self.assertGreater(r.rank[1], r.rank[2])

    def test_empty_and_all_empty_layers(self) -> None:
        empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), 1.0)
        for layers in ([], [empty], [empty, empty]):
            src, dst, w = combine_edges(layers)
            self.assertEqual((len(src), len(dst), len(w)), (0, 0, 0))


class TestDisplaySize(unittest.TestCase):
    def test_sizes_spread_across_the_full_range(self) -> None:
        # The bug this exists to prevent: a linear map from a power law puts
        # 98% of nodes in the bottom 2% of the size range, and the field
        # renders as uniform dust.
        rng = np.random.default_rng(2)
        rank = rng.pareto(1.16, 20_000) + 1
        rank /= rank.sum()
        size = to_display_size(rank)

        self.assertGreaterEqual(size.min(), 0.0)
        self.assertLessEqual(size.max(), 1.0)
        # Every decile should hold roughly a tenth of the nodes.
        counts, _ = np.histogram(size, bins=10, range=(0, 1))
        self.assertGreater(counts.min(), len(size) * 0.05)

    def test_order_is_preserved(self) -> None:
        rank = np.array([0.5, 0.1, 0.3, 0.05, 0.05])
        size = to_display_size(rank)
        self.assertEqual(np.argmax(size), 0)
        self.assertLess(size[1], size[0])
        self.assertGreater(size[2], size[1])

    def test_ties_get_identical_sizes(self) -> None:
        # Two repos with the same rank drawn at different radii is a visible
        # lie about the data.
        size = to_display_size(np.array([0.4, 0.2, 0.2, 0.2]))
        self.assertAlmostEqual(size[1], size[2], places=12)
        self.assertAlmostEqual(size[2], size[3], places=12)

    def test_degenerate_inputs(self) -> None:
        self.assertEqual(len(to_display_size(np.zeros(0))), 0)
        np.testing.assert_allclose(to_display_size(np.array([0.7])), [1.0])
        # All-equal ranks: every node the same size, nothing NaN.
        flat = to_display_size(np.full(50, 0.02))
        self.assertFalse(np.isnan(flat).any())
        self.assertAlmostEqual(flat.std(), 0.0, places=12)


class TestTheTailProblem(unittest.TestCase):
    """PageRank alone cannot order or size most of a real corpus."""

    def test_most_of_a_realistic_corpus_ties_on_the_teleport_floor(self) -> None:
        # Not an approximation — every node with no in-edges gets EXACTLY
        # (1-d)/n + leaked. This is the premise the tiebreak exists for; if it
        # ever stops being true, the tiebreak can go.
        rank = realistic_ranks()
        _, counts = np.unique(rank, return_counts=True)
        self.assertGreater(counts.max() / len(rank), 0.5)

    def test_without_a_tiebreak_the_tail_is_one_flat_size(self) -> None:
        size = to_display_size(np.sort(realistic_ranks())[::-1])
        tail = size[len(size) // 3 :]
        self.assertAlmostEqual(tail.std(), 0.0, places=12)

    def test_a_tiebreak_spreads_the_tail_out(self) -> None:
        rank = np.sort(realistic_ranks())[::-1]
        rng = np.random.default_rng(3)
        stars = np.log1p(rng.pareto(1.2, len(rank)) * 40)
        size = to_display_size(rank, stars)
        tail = size[len(size) // 3 :]
        self.assertGreater(tail.std(), 0.1)
        self.assertGreater(len(np.unique(tail)), len(tail) * 0.9)

    def test_the_tiebreak_never_overrides_pagerank(self) -> None:
        # A wildly popular repository that nothing depends on must still rank
        # below a quietly critical one. Stars break ties; they do not win them.
        rank = np.array([0.9, 0.1, 0.1])
        stars = np.array([0.0, 1000.0, 5.0])
        order = importance_order(rank, stars)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[1], 1)  # equal rank, more stars
        self.assertGreater(to_display_size(rank, stars)[0], to_display_size(rank, stars)[1])

    def test_identical_rows_still_get_identical_sizes(self) -> None:
        rank = np.array([0.5, 0.2, 0.2, 0.2])
        stars = np.array([1.0, 3.0, 3.0, 9.0])
        size = to_display_size(rank, stars)
        self.assertAlmostEqual(size[1], size[2], places=12)  # equal on both
        self.assertNotAlmostEqual(size[1], size[3], places=6)  # differ on stars

    def test_order_is_stable_across_runs(self) -> None:
        rank, stars = np.full(400, 0.1), np.zeros(400)
        np.testing.assert_array_equal(importance_order(rank, stars), np.arange(400))


class TestBandedDisplaySize(unittest.TestCase):
    def test_every_band_spans_its_own_range(self) -> None:
        # THE bug this replaced: a single global percentile puts the top 2% of
        # nodes in the top 2% of the radius range, so band 0 — the layer always
        # in focus — rendered every hub at the same size.
        rank = np.sort(realistic_ranks())[::-1]
        rng = np.random.default_rng(5)
        stars = np.log1p(rng.pareto(1.2, len(rank)) * 40)
        sizes = [60, 540, 2400]
        size = banded_display_size(rank, sizes, stars)

        start = 0
        for count in sizes:
            chunk = size[start : start + count]
            with self.subTest(band=start):
                self.assertGreater(chunk.max() - chunk.min(), 0.05)
            start += count

    def test_bands_do_not_overlap_and_higher_bands_are_bigger(self) -> None:
        rank = np.sort(realistic_ranks())[::-1]
        stars = np.log1p(np.arange(len(rank))[::-1].astype(float))
        sizes = [60, 540, 2400]
        size = banded_display_size(rank, sizes, stars)
        b0, b1, b2 = size[:60], size[60:600], size[600:]
        self.assertGreaterEqual(b0.min(), b1.max() - 1e-12)
        self.assertGreaterEqual(b1.min(), b2.max() - 1e-12)

    def test_output_stays_within_zero_and_one(self) -> None:
        rank = np.sort(realistic_ranks())[::-1]
        size = banded_display_size(rank, [60, 540, 2400])
        self.assertGreaterEqual(size.min(), 0.0)
        self.assertLessEqual(size.max(), 1.0)

    def test_empty_bands_are_tolerated(self) -> None:
        # A tiny corpus can round a band to zero points.
        size = banded_display_size(np.array([0.5, 0.3, 0.2]), [0, 0, 3])
        self.assertEqual(len(size), 3)
        self.assertFalse(np.isnan(size).any())

    def test_single_band_spans_the_floor_to_the_top(self) -> None:
        from gitglobe.graph.pagerank import MIN_DISPLAY_SIZE

        size = banded_display_size(np.array([0.5, 0.3, 0.2]), [3])
        self.assertAlmostEqual(size.max(), 1.0)
        # NOT zero. The point shader clamps at one pixel, so a size of 0.0 is
        # not "smallest visible" — it is "invisible", and 80% of the globe
        # lives in the bottom band.
        self.assertAlmostEqual(size.min(), MIN_DISPLAY_SIZE)

    def test_nothing_is_ever_sized_below_the_visibility_floor(self) -> None:
        from gitglobe.graph.pagerank import MIN_DISPLAY_SIZE

        rank = np.sort(realistic_ranks())[::-1]
        for bands in ([len(rank)], [60, 540, len(rank) - 600], [10, 20, 30, len(rank) - 60]):
            with self.subTest(bands=len(bands)):
                self.assertGreaterEqual(
                    banded_display_size(rank, bands).min(), MIN_DISPLAY_SIZE - 1e-9
                )

    def test_empty_input(self) -> None:
        self.assertEqual(len(banded_display_size(np.zeros(0), [])), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
