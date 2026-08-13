"""Tests for spherical k-means and the domain assignment around it.

HDBSCAN is not installed in every environment and is not ours to test. The
spherical k-means underneath the domain assignment IS ours, and it is the piece
that decides whether the twelve territories are twelve contiguous regions or
twelve colours sprinkled across the whole globe.

An earlier version of the globe assigned domains by `i % 12`. Every domain ended
up everywhere, which broke fly-to-domain and made the territory rendering
meaningless — and nothing raised. These tests are the guard against a repeat.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gitglobe.project.cluster import (
    TAU,
    cluster_purity,
    spherical_kmeans,
    spherical_mean,
    to_angles,
    to_unit_vectors,
)


def blobs(centres_xyz, per_blob=300, spread=0.10, seed=0):
    """Tight clusters around given directions on the sphere."""
    rng = np.random.default_rng(seed)
    points, truth = [], []
    for i, c in enumerate(centres_xyz):
        c = np.asarray(c, dtype=np.float64)
        c /= np.linalg.norm(c)
        v = c + rng.normal(scale=spread, size=(per_blob, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        points.append(v)
        truth.append(np.full(per_blob, i))
    return np.vstack(points), np.concatenate(truth)


class TestSphericalMean(unittest.TestCase):
    def test_mean_of_a_tight_group_points_at_the_group(self) -> None:
        v, _ = blobs([[0, 0, 1]], per_blob=500, spread=0.05)
        mean = spherical_mean(v)
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1.0, places=12)
        self.assertGreater(float(mean @ np.array([0, 0, 1])), 0.99)

    def test_antipodal_pair_falls_back_instead_of_returning_noise(self) -> None:
        # The arithmetic mean here is the zero vector. Normalising it would
        # amplify floating-point dust into a confidently wrong direction.
        v = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        mean = spherical_mean(v)
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1.0, places=12)
        self.assertFalse(np.isnan(mean).any())

    def test_empty_input(self) -> None:
        self.assertEqual(len(spherical_mean(np.zeros((0, 3)))), 3)


class TestSphericalKMeans(unittest.TestCase):
    def test_well_separated_blobs_are_recovered_exactly(self) -> None:
        centres = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        v, truth = blobs(centres, per_blob=200, spread=0.08, seed=3)
        labels, found = spherical_kmeans(v, 6, seed=1)

        self.assertEqual(len(np.unique(labels)), 6)
        # Same partition, whatever the labels happen to be called.
        for t in range(6):
            self.assertEqual(len(np.unique(labels[truth == t])), 1)
        np.testing.assert_allclose(np.linalg.norm(found, axis=1), 1.0, atol=1e-9)

    def test_domains_are_spatially_contiguous(self) -> None:
        # THE property. Every point must be closer to its own domain's centre
        # than to any other — which is what makes a territory a territory
        # rather than a colour sprinkled over the globe.
        rng = np.random.default_rng(11)
        n = 4000
        theta = np.arccos(2 * rng.random(n) - 1)
        phi = rng.random(n) * TAU
        v = to_unit_vectors(theta, phi)

        labels, centres = spherical_kmeans(v, 12, seed=7)
        nearest = np.argmax(v @ centres.T, axis=1)
        self.assertTrue((nearest == labels).all())

    def test_farthest_point_seeding_beats_random_on_a_hard_case(self) -> None:
        # One crowded region and two sparse ones. Random seeding drops several
        # centres in the crowd and leaves the sparse regions merged.
        v, truth = blobs([[1, 0, 0]] * 1, per_blob=2000, spread=0.05, seed=1)
        sparse, sparse_truth = blobs([[-1, 0, 0], [0, 0, 1]], per_blob=60, spread=0.05, seed=2)
        allv = np.vstack([v, sparse])
        labels, _ = spherical_kmeans(allv, 3, seed=5)

        # The two sparse blobs must not be collapsed into one label.
        tail = labels[len(v):]
        self.assertNotEqual(tail[0], tail[-1])

    def test_no_empty_clusters(self) -> None:
        # An empty cluster stays empty forever unless it is reseeded, and an
        # unused domain means one of the twelve palette colours never appears.
        rng = np.random.default_rng(8)
        v = rng.normal(size=(500, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        labels, centres = spherical_kmeans(v, 12, seed=2)
        self.assertEqual(len(np.unique(labels)), 12)
        self.assertEqual(len(centres), 12)

    def test_k_larger_than_n_is_clamped(self) -> None:
        v, _ = blobs([[1, 0, 0], [0, 1, 0]], per_blob=2)
        labels, centres = spherical_kmeans(v, 12, seed=1)
        self.assertEqual(len(centres), 4)
        self.assertLessEqual(labels.max(), 3)

    def test_weights_pull_centres_towards_heavy_points(self) -> None:
        v = np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]])
        _, heavy = spherical_kmeans(v, 1, weights=np.array([100.0, 1.0, 1.0]), seed=0)
        _, even = spherical_kmeans(v, 1, weights=np.array([1.0, 1.0, 1.0]), seed=0)
        self.assertGreater(float(heavy[0] @ v[0]), float(even[0] @ v[0]))

    def test_deterministic_for_a_fixed_seed(self) -> None:
        v, _ = blobs([[1, 0, 0], [0, 1, 0], [0, 0, 1]], per_blob=100, seed=4)
        a, _ = spherical_kmeans(v, 3, seed=99)
        b, _ = spherical_kmeans(v, 3, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_empty_input(self) -> None:
        labels, centres = spherical_kmeans(np.zeros((0, 3)), 12)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(centres), 0)


class TestAngleConversion(unittest.TestCase):
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(12)
        theta = np.arccos(2 * rng.random(3000) - 1)
        phi = rng.random(3000) * TAU
        t2, p2 = to_angles(to_unit_vectors(theta, phi))
        np.testing.assert_allclose(t2, theta, atol=1e-12)
        np.testing.assert_allclose(p2, phi, atol=1e-10)

    def test_latitude_convention_matches_hdbscan(self) -> None:
        # cluster() feeds HDBSCAN (PI/2 - theta) as latitude. Getting the sign
        # backwards mirrors every cluster about the equator, which looks
        # entirely plausible and is entirely wrong.
        north = to_unit_vectors(np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(float(north[0][1]), 1.0, places=12)
        self.assertAlmostEqual(math.pi / 2 - 0.0, math.pi / 2, places=12)


class TestClusterPurity(unittest.TestCase):
    def test_tight_clusters_score_well_above_baseline(self) -> None:
        rng = np.random.default_rng(1)
        vectors = np.vstack([
            rng.normal(loc=c, scale=0.05, size=(200, 8))
            for c in (np.eye(8)[0] * 3, np.eye(8)[3] * 3, np.eye(8)[6] * 3)
        ])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = np.repeat([0, 1, 2], 200)
        stats = cluster_purity(ids, vectors)
        self.assertEqual(stats["clusters"], 3)
        self.assertGreater(stats["lift"], 0.5)

    def test_random_clusters_show_no_lift(self) -> None:
        # If clustering carried no signal, lift must be ~0. This is what tells
        # us clustering on the sphere is doing real work rather than nothing.
        rng = np.random.default_rng(2)
        vectors = rng.normal(size=(900, 8))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        stats = cluster_purity(rng.integers(0, 3, 900), vectors)
        self.assertLess(abs(stats["lift"]), 0.05)

    def test_the_metric_is_not_biased_by_cluster_size(self) -> None:
        # The bug this replaced: similarity-to-own-mean falls as ~1/sqrt(m*d),
        # so small clusters of PURE NOISE outscored a large corpus baseline by
        # 21x. Small random clusters must show no lift.
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(6000, 8))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for size in (60, 200, 1000):
            ids = np.full(6000, -1)
            ids[: size * 3] = np.repeat([0, 1, 2], size)
            with self.subTest(cluster_size=size):
                self.assertLess(abs(cluster_purity(ids, vectors)["lift"]), 0.05)

    def test_ratio_is_withheld_when_the_baseline_is_noise(self) -> None:
        # Dividing by a near-zero baseline produces an impressive number that
        # means nothing.
        rng = np.random.default_rng(4)
        vectors = rng.normal(size=(600, 8))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.assertIsNone(cluster_purity(rng.integers(0, 3, 600), vectors)["ratio"])

    def test_noise_labels_are_ignored(self) -> None:
        vectors = np.eye(3)
        self.assertEqual(cluster_purity(np.array([-1, -1, -1]), vectors)["clusters"], 0)

    def test_empty(self) -> None:
        self.assertEqual(cluster_purity(np.zeros(0), np.zeros((0, 8)))["clusters"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
