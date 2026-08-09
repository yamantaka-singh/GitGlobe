"""Tests for the parts of Phase 2 that run without umap-learn or a network.

UMAP itself is not tested here — testing someone else's optimiser is not our
job. What *is* our job is everything around it: folding its unbounded output
onto the sphere correctly, noticing when a projection has collapsed, and turning
its kNN graph into edges that mean something.

Those are exactly the places where a bug produces a globe that renders happily
and is wrong.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gitglobe.embed.vertex import (  # noqa: E402
    MAX_INPUT_CHARS,
    estimate,
    l2_normalise,
    pack,
    prepare_text,
    unpack,
    unpack_matrix,
)
from gitglobe.project.spherical import (  # noqa: E402
    TAU,
    ProjectionResult,
    assess_coverage,
    coverage,
    knn_to_edges,
    wrap_to_sphere,
)


class TestWrapToSphere(unittest.TestCase):
    def test_values_already_in_range_are_untouched(self) -> None:
        theta = np.array([0.0, 1.0, math.pi])
        phi = np.array([0.0, 3.0, TAU - 0.001])
        t, p = wrap_to_sphere(theta, phi)
        np.testing.assert_allclose(t, theta, atol=1e-12)
        np.testing.assert_allclose(p, phi, atol=1e-12)

    def test_going_past_the_south_pole_comes_down_the_far_side(self) -> None:
        # THE bug this function exists for. theta = PI + 0.1 is a real place:
        # 0.1 past the south pole, i.e. theta = PI - 0.1 with phi flipped by PI.
        # Clamping instead would pile every overshooting point onto the pole.
        t, p = wrap_to_sphere(np.array([math.pi + 0.1]), np.array([0.5]))
        self.assertAlmostEqual(t[0], math.pi - 0.1, places=12)
        self.assertAlmostEqual(p[0], 0.5 + math.pi, places=12)

    def test_going_past_the_north_pole(self) -> None:
        t, p = wrap_to_sphere(np.array([-0.1]), np.array([0.5]))
        self.assertAlmostEqual(t[0], 0.1, places=12)
        self.assertAlmostEqual(p[0], 0.5 + math.pi, places=12)

    def test_wrapping_preserves_the_actual_3d_position(self) -> None:
        # The real invariant: wrapping must be a relabelling, not a move.
        # Same point on the sphere before and after, to floating-point.
        rng = np.random.default_rng(4)
        theta = rng.uniform(-4 * math.pi, 4 * math.pi, 5000)
        phi = rng.uniform(-10, 10, 5000)

        def xyz(t, p):
            st = np.sin(t)
            return np.column_stack([st * np.cos(p), np.cos(t), st * np.sin(p)])

        before = xyz(theta, phi)
        after = xyz(*wrap_to_sphere(theta, phi))
        self.assertLess(np.abs(before - after).max(), 1e-12)

    def test_output_is_always_in_the_canonical_range(self) -> None:
        rng = np.random.default_rng(9)
        theta = rng.uniform(-50, 50, 20_000)
        phi = rng.uniform(-50, 50, 20_000)
        t, p = wrap_to_sphere(theta, phi)
        self.assertTrue(((t >= 0) & (t <= math.pi)).all())
        self.assertTrue(((p >= 0) & (p < TAU)).all())
        self.assertFalse(np.isnan(t).any() or np.isnan(p).any())

    def test_the_input_is_not_mutated(self) -> None:
        theta = np.array([5.0])
        phi = np.array([9.0])
        wrap_to_sphere(theta, phi)
        self.assertEqual(theta[0], 5.0)
        self.assertEqual(phi[0], 9.0)


class TestCoverage(unittest.TestCase):
    @staticmethod
    def uniform_sphere(n: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        return np.arccos(2 * rng.random(n) - 1), rng.random(n) * TAU

    def test_a_uniform_sphere_looks_healthy(self) -> None:
        stats = coverage(*self.uniform_sphere(50_000))
        self.assertEqual(assess_coverage(stats), [])
        self.assertGreater(stats["occupied_fraction"], 0.98)
        self.assertLess(stats["gini"], 0.3)

    def test_equal_area_binning_does_not_slander_the_poles(self) -> None:
        # With equal-ANGLE latitude bands, a uniform sphere looks empty near the
        # poles and the check reports a collapse that is not happening.
        stats = coverage(*self.uniform_sphere(20_000))
        self.assertEqual(assess_coverage(stats), [])

    def test_a_collapsed_projection_is_caught(self) -> None:
        # Everything in one small cap — the classic UMAP-didn't-converge shape.
        rng = np.random.default_rng(2)
        theta = rng.uniform(0, 0.25, 20_000)
        phi = rng.random(20_000) * TAU
        problems = assess_coverage(coverage(theta, phi))
        self.assertTrue(problems)
        self.assertTrue(any("collapsed" in p for p in problems))

    def test_a_pole_spike_is_caught(self) -> None:
        # What clamping instead of wrapping produces.
        t, p = self.uniform_sphere(20_000)
        t[:6000] = 0.0
        p[:6000] = 0.0
        problems = assess_coverage(coverage(t, p))
        self.assertTrue(any("one cell holds" in x for x in problems))

    def test_small_and_empty_inputs_do_not_crash(self) -> None:
        self.assertEqual(coverage(np.zeros(0), np.zeros(0))["points"], 0)
        self.assertEqual(assess_coverage(coverage(np.zeros(5), np.zeros(5))), [])


class TestKnnToEdges(unittest.TestCase):
    def test_only_mutual_pairs_survive(self) -> None:
        # 0 and 1 pick each other. 2 picks 0, but 0 does not pick 2 back —
        # the hub asymmetry that would otherwise connect a hub to everything.
        indices = np.array([[0, 1], [1, 0], [2, 0]])
        distances = np.array([[0.0, 0.1], [0.0, 0.1], [0.0, 0.2]])
        edges = knn_to_edges(indices, distances, np.array([10, 20, 30]), k=1)
        self.assertEqual(edges, [(10, 20, 0.9)])

    def test_distant_neighbours_are_dropped(self) -> None:
        # kNN always returns k results even when nothing is genuinely close.
        # Without a ceiling every isolated repo gets confidently-wrong edges.
        indices = np.array([[0, 1], [1, 0]])
        distances = np.array([[0.0, 0.9], [0.0, 0.9]])
        self.assertEqual(knn_to_edges(indices, distances, np.array([1, 2]), k=1), [])

    def test_weight_is_similarity_not_distance(self) -> None:
        indices = np.array([[0, 1], [1, 0]])
        distances = np.array([[0.0, 0.05], [0.0, 0.05]])
        edges = knn_to_edges(indices, distances, np.array([7, 8]), k=1)
        self.assertAlmostEqual(edges[0][2], 0.95, places=6)

    def test_each_pair_appears_once(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        indices = np.column_stack([np.arange(n), rng.integers(0, n, (n, 5))])
        distances = np.column_stack([np.zeros(n), rng.random((n, 5)) * 0.3])
        edges = knn_to_edges(indices, distances, np.arange(1, n + 1), k=5)
        pairs = {tuple(sorted((a, b))) for a, b, _ in edges}
        self.assertEqual(len(pairs), len(edges))
        self.assertFalse(any(a == b for a, b, _ in edges))

    def test_missing_knn_returns_nothing(self) -> None:
        self.assertEqual(knn_to_edges(None, None, np.array([1])), [])


class TestProjectionResult(unittest.TestCase):
    def test_xyz_is_on_the_unit_sphere_and_y_up(self) -> None:
        r = ProjectionResult(theta=np.array([0.0, math.pi, math.pi / 2]), phi=np.array([0.0, 0.0, 0.0]))
        xyz = r.to_xyz()
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(xyz[0], [0, 1, 0], atol=1e-12)   # north pole is +Y
        np.testing.assert_allclose(xyz[1], [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(xyz[2], [1, 0, 0], atol=1e-12)


class TestEmbedHelpers(unittest.TestCase):
    def test_long_text_is_cut_at_a_word_boundary(self) -> None:
        text = ("alpha beta gamma " * 2000).strip()
        out, truncated = prepare_text(text)
        self.assertTrue(truncated)
        self.assertLessEqual(len(out), MAX_INPUT_CHARS)
        self.assertFalse(out.endswith(" "))
        self.assertIn(out.split()[-1], {"alpha", "beta", "gamma"})

    def test_a_long_unbroken_token_still_gets_cut(self) -> None:
        # A minified bundle or a base64 blob has no spaces to cut at.
        out, truncated = prepare_text("x" * 50_000)
        self.assertTrue(truncated)
        self.assertEqual(len(out), MAX_INPUT_CHARS)

    def test_short_and_empty_text(self) -> None:
        self.assertEqual(prepare_text("  a web framework  "), ("a web framework", False))
        self.assertEqual(prepare_text(""), ("", False))
        self.assertEqual(prepare_text(None), ("", False))

    def test_normalisation_survives_a_zero_row(self) -> None:
        # A zero row divides by zero and turns the WHOLE matrix into NaN, which
        # UMAP reports thousands of rows later as something unrelated.
        m = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        out = l2_normalise(m)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-12)
        np.testing.assert_allclose(out[1], [0.0, 0.0], atol=1e-12)

    def test_pack_normalises_and_round_trips(self) -> None:
        # gemini-embedding-001 returns truncated vectors UN-normalised, so this
        # is the only place length gets fixed before it reaches UMAP.
        blob = pack(np.array([3.0, 4.0, 0.0], dtype=np.float32))
        vector = unpack(blob, 3)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        self.assertEqual(len(blob), 12)

    def test_unpack_refuses_the_wrong_dimension(self) -> None:
        # Catches a model or output_dimensionality change at load time rather
        # than as a confusing reshape error deep inside UMAP.
        with self.assertRaises(ValueError):
            unpack(pack(np.ones(768, np.float32)), 3072)

    def test_unpack_matrix_matches_row_by_row_unpacking(self) -> None:
        rng = np.random.default_rng(6)
        vectors = rng.normal(size=(50, 16)).astype(np.float32)
        blobs = [pack(v) for v in vectors]
        matrix = unpack_matrix(blobs, 16)
        self.assertEqual(matrix.shape, (50, 16))
        for i in range(50):
            np.testing.assert_allclose(matrix[i], unpack(blobs[i], 16), atol=0)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)

    def test_unpack_matrix_reports_which_row_is_wrong(self) -> None:
        blobs = [pack(np.ones(4, np.float32)), pack(np.ones(3, np.float32))]
        with self.assertRaises(ValueError) as ctx:
            unpack_matrix(blobs, 4)
        self.assertIn("row 1", str(ctx.exception))

    def test_unpack_matrix_on_empty(self) -> None:
        self.assertEqual(unpack_matrix([], 768).shape, (0, 768))

    def test_estimate_recommends_batch_prediction_only_when_it_is_worth_it(self) -> None:
        self.assertFalse(estimate(100_000, 1200)["recommend_batch_prediction"])
        self.assertTrue(estimate(1_000_000, 1200)["recommend_batch_prediction"])
        self.assertGreater(estimate(100_000, 1200)["est_usd_online"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
