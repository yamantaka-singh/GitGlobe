"""Tests for the two measurement repairs: isotropy and scale-free comparison.

Both exist because a metric misled me. `cluster_purity` reported lift 0.06 and
fired a warning, when the structure was fine and the ruler was compressed into
35% of its range. Separately, lift fell 3.5x with group size, so it could not
compare partitions of different granularity — and it favours k-means, because
within-group similarity is exactly what k-means optimises.

The synthetic fixtures here CHECK THEIR OWN CONSTRUCTION before asserting
anything. Three earlier attempts at building anisotropic embeddings silently
produced isotropic ones (noise of norm sqrt(768) drowning a unit-norm cone),
and each time the conclusion drawn from them was wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gitglobe.embed import whiten  # noqa: E402
from gitglobe.graph.communities import build_adjacency  # noqa: E402
from gitglobe.graph.stability import MAX_SUBSTEP, stability  # noqa: E402
from gitglobe.project.cluster import cluster_purity  # noqa: E402


def coned_embeddings(baseline=0.6473, cluster=0.0652, D=256, K=40, N=6_000, seed=0):
    """Embeddings inside a cone, like a real LLM's.

    Noise is divided by sqrt(D) so its norm is ~1 and comparable to the unit
    cone and cluster vectors. Without that the noise has norm sqrt(D) and
    swamps everything, producing an ISOTROPIC fixture that silently fails to
    test the thing it was written for.
    """
    rng = np.random.default_rng(seed)
    cone = rng.normal(size=D)
    cone /= np.linalg.norm(cone)
    centres = rng.normal(size=(K, D))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    truth = rng.integers(0, K, N)
    noise = rng.normal(size=(N, D)) / np.sqrt(D)
    X = (
        np.sqrt(baseline) * cone
        + np.sqrt(cluster) * centres[truth]
        + np.sqrt(max(0.0, 1 - baseline - cluster)) * noise
    )
    return X / np.linalg.norm(X, axis=1, keepdims=True), truth


class TestFixtureIsWhatItClaims(unittest.TestCase):
    """If the fixture is not anisotropic, every test below proves nothing."""

    def test_the_cone_fixture_actually_has_a_cone(self) -> None:
        X, truth = coned_embeddings()
        measured = cluster_purity(truth, X.astype(np.float32))["baseline"]
        self.assertAlmostEqual(measured, 0.6473, delta=0.05)

    def test_the_fixture_also_has_real_cluster_structure(self) -> None:
        X, truth = coned_embeddings()
        rng = np.random.default_rng(1)
        real = cluster_purity(truth, X.astype(np.float32))["lift"]
        shuffled = cluster_purity(rng.permutation(truth), X.astype(np.float32))["lift"]
        self.assertGreater(real, shuffled + 0.02)


class TestWhitening(unittest.TestCase):
    def test_restores_the_full_dynamic_range(self) -> None:
        X, _ = coned_embeddings()
        w = whiten.fit(X)
        self.assertGreater(abs(w.baseline_before), 0.5)
        self.assertLess(abs(w.baseline_after), 0.05)

    def test_lift_improves_on_identical_structure(self) -> None:
        # THE point: the signal was always present, the ruler was compressed.
        X, truth = coned_embeddings()
        before = cluster_purity(truth, X.astype(np.float32))["lift"]
        after = cluster_purity(truth, whiten.fit(X).apply(X))["lift"]
        self.assertGreater(after, before * 2)

    def test_output_stays_unit_norm(self) -> None:
        X, _ = coned_embeddings()
        out = whiten.fit(X).apply(X)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_recommend_measures_rather_than_assumes(self) -> None:
        X, _ = coned_embeddings()
        k, table = whiten.recommend_components(X)
        self.assertEqual([c for c, _ in table], list(range(len(table))))
        self.assertLess(abs(dict(table)[k]), 0.05)

    def test_the_basis_is_reusable_across_batches(self) -> None:
        # Refitting per batch would put the "same" vector in different places
        # depending on which batch it arrived in.
        X, _ = coned_embeddings()
        w = whiten.fit(X)
        np.testing.assert_allclose(w.apply(X[:50]), w.apply(X)[:50], atol=1e-6)

    def test_wrong_dimension_is_refused(self) -> None:
        X, _ = coned_embeddings(D=64)
        w = whiten.fit(X)
        with self.assertRaises(ValueError):
            w.apply(np.zeros((5, 128)))

    def test_zero_rows_do_not_produce_nan(self) -> None:
        X, _ = coned_embeddings(D=32, N=500)
        w = whiten.fit(X)
        out = w.apply(np.vstack([X[:10], np.zeros((1, 32))]))
        self.assertFalse(np.isnan(out).any())


def hierarchical_graph(supers=4, subs=5, per=40, seed=0):
    rng = np.random.default_rng(seed)
    n = supers * subs * per
    sub = np.repeat(np.arange(supers * subs), per)
    sup = sub // subs
    s, d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.40 if sub[i] == sub[j] else (0.05 if sup[i] == sup[j] else 0.002)
            if rng.random() < p:
                s.append(i)
                d.append(j)
    return n, np.array(s), np.array(d), sub, sup


class TestMarkovStability(unittest.TestCase):
    def setUp(self) -> None:
        self.n, s, d, self.sub, self.sup = hierarchical_graph()
        self.offsets, self.targets, self.weights = build_adjacency(self.n, s, d)

    def score(self, labels, t):
        return stability(labels, self.offsets, self.targets, self.weights, times=(t,)).values[0]

    def test_finds_fine_structure_at_short_markov_time(self) -> None:
        self.assertGreater(self.score(self.sub, 0.25), self.score(self.sup, 0.25))

    def test_finds_coarse_structure_at_long_markov_time(self) -> None:
        # The scale is now an explicit parameter rather than an artefact of
        # group size — which is exactly what `lift` could not express.
        self.assertGreater(self.score(self.sup, 4.0), self.score(self.sub, 4.0))

    def test_beats_a_random_partition_of_the_same_size(self) -> None:
        rng = np.random.default_rng(3)
        random_labels = rng.integers(0, len(np.unique(self.sub)), self.n)
        self.assertGreater(self.score(self.sub, 1.0), self.score(random_labels, 1.0))

    def test_compares_across_granularities_on_one_axis(self) -> None:
        # `lift` fell 3.5x purely with group size, so a 1,200-group partition
        # and a 4-group one were not on the same scale. Here they are.
        singletons = np.arange(self.n)
        self.assertGreater(self.score(self.sup, 4.0), self.score(singletons, 4.0))
        self.assertGreater(self.score(self.sub, 0.25), self.score(singletons, 0.25))

    def test_one_giant_community_scores_zero(self) -> None:
        self.assertAlmostEqual(self.score(np.zeros(self.n, int), 1.0), 0.0, places=6)

    def test_stays_finite_at_large_markov_time(self) -> None:
        # A direct Taylor expansion diverges once t exceeds the truncation
        # order: at t=64 it returned 2.5e13. Sub-stepping keeps every expansion
        # inside the convergent regime.
        for t in (8.0, 32.0, 128.0):
            with self.subTest(t=t):
                value = self.score(self.sup, t)
                self.assertTrue(np.isfinite(value))
                self.assertLessEqual(abs(value), 1.0)

    def test_substep_ceiling_keeps_the_series_convergent(self) -> None:
        import math

        self.assertLess(MAX_SUBSTEP**12 / math.factorial(12), 1e-4)

    def test_isolated_nodes_do_not_break_the_walk(self) -> None:
        # 86% of the real corpus has no dependency edge.
        n, s, d, sub, _ = hierarchical_graph(supers=2, subs=2, per=20)
        offsets, targets, weights = build_adjacency(n + 200, s, d)
        labels = np.concatenate([sub, np.arange(200) + sub.max() + 1])
        value = stability(labels, offsets, targets, weights, times=(1.0,)).values[0]
        self.assertTrue(np.isfinite(value))

    def test_empty_graph(self) -> None:
        offsets, targets, weights = build_adjacency(0, np.zeros(0), np.zeros(0))
        self.assertEqual(stability(np.zeros(0, int), offsets, targets, weights).values, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
