"""Tests for the exit criterion itself.

The spot-check is the only thing standing between "the pipeline ran" and "the
map is correct", so it has to be trustworthy in both directions: it must pass a
good map and it must FAIL a bad one. A check that cannot fail is decoration.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gitglobe.checks.neighbours import (  # noqa: E402
    DEFAULT_EXPECTATIONS,
    Expectation,
    baseline_distance,
    great_circle,
    mean_pairwise,
    run_expectations,
    summarise,
    to_vector,
)

TAU = 2 * math.pi


def at(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """(theta, phi) from degrees, for readable fixtures."""
    return (math.radians(90 - lat_deg), math.radians(lon_deg % 360))


# A map that is right: ML near the north pole, frontend near the south.
GOOD_MAP = {
    "pytorch/pytorch": at(70, 10),
    "tensorflow/tensorflow": at(72, 20),
    "jax-ml/jax": at(68, 15),
    "facebook/react": at(-70, 200),
    "vuejs/core": at(-72, 210),
    "sveltejs/svelte": at(-68, 205),
}

# A map that is wrong: everything collapsed into one spot. This is what a failed
# UMAP run actually looks like, and the checks must catch it.
COLLAPSED_MAP = {name: at(0, 0) for name in GOOD_MAP}


class TestGreatCircle(unittest.TestCase):
    def test_known_angles(self) -> None:
        x = np.array([1.0, 0, 0])
        y = np.array([0, 1.0, 0])
        self.assertAlmostEqual(great_circle(x, x), 0.0, places=12)
        self.assertAlmostEqual(great_circle(x, y), math.pi / 2, places=12)
        self.assertAlmostEqual(great_circle(x, -x), math.pi, places=9)

    def test_accurate_at_small_angles(self) -> None:
        # arccos(dot) loses precision exactly where this check cares most.
        # atan2(|cross|, dot) does not.
        for angle in (1e-4, 1e-6, 1e-8):
            a = np.array([1.0, 0.0, 0.0])
            b = np.array([math.cos(angle), math.sin(angle), 0.0])
            with self.subTest(angle=angle):
                self.assertAlmostEqual(great_circle(a, b) / angle, 1.0, places=5)

    def test_to_vector_is_unit_and_y_up(self) -> None:
        for theta, phi in (at(90, 0), at(-90, 0), at(0, 0), at(0, 123)):
            self.assertAlmostEqual(float(np.linalg.norm(to_vector(theta, phi))), 1.0, places=12)
        np.testing.assert_allclose(to_vector(*at(90, 0)), [0, 1, 0], atol=1e-12)


class TestExpectations(unittest.TestCase):
    def test_a_good_map_passes(self) -> None:
        outcomes = run_expectations(GOOD_MAP)
        failed = [o for o in outcomes if o.status == "fail"]
        self.assertEqual(failed, [], f"good map failed: {[o.name for o in failed]}")

    def test_a_collapsed_map_fails_the_separation_check(self) -> None:
        # The important direction. Every "near" check passes trivially when
        # everything is in one place — only a separation check catches it.
        outcomes = {o.name: o for o in run_expectations(COLLAPSED_MAP)}
        self.assertEqual(outcomes["frontend is not machine learning"].status, "fail")

    def test_a_scattered_group_fails_the_near_check(self) -> None:
        scattered = dict(GOOD_MAP)
        scattered["jax-ml/jax"] = at(-80, 300)  # opposite side of the globe
        outcomes = {o.name: o for o in run_expectations(scattered)}
        self.assertEqual(outcomes["deep learning frameworks"].status, "fail")

    def test_missing_repositories_skip_rather_than_fail(self) -> None:
        # A 5k proof run contains almost none of these. Failing would teach
        # people to ignore the check.
        outcomes = run_expectations({"pytorch/pytorch": at(70, 10)})
        self.assertTrue(all(o.status == "skipped" for o in outcomes))
        self.assertTrue(all(o.missing for o in outcomes))

    def test_two_of_three_present_is_still_checked(self) -> None:
        partial = {k: v for k, v in GOOD_MAP.items() if k != "jax-ml/jax"}
        outcomes = {o.name: o for o in run_expectations(partial)}
        self.assertEqual(outcomes["deep learning frameworks"].status, "pass")
        self.assertEqual(outcomes["deep learning frameworks"].missing, ["jax-ml/jax"])

    def test_lookup_is_case_insensitive(self) -> None:
        upper = {k.upper(): v for k, v in GOOD_MAP.items()}
        outcomes = {o.name: o for o in run_expectations(upper)}
        self.assertEqual(outcomes["deep learning frameworks"].status, "pass")

    def test_empty_positions(self) -> None:
        outcomes = run_expectations({})
        self.assertTrue(all(o.status == "skipped" for o in outcomes))

    def test_custom_expectation_both_directions(self) -> None:
        positions = {"a/a": at(80, 0), "b/b": at(78, 5), "c/c": at(-80, 180)}
        near_ok = run_expectations(positions, [Expectation("near", near=["a/a", "b/b"], max_near=0.2)])
        self.assertEqual(near_ok[0].status, "pass")
        far_ok = run_expectations(
            positions, [Expectation("far", far=(["a/a"], ["c/c"]), min_far=2.0)]
        )
        self.assertEqual(far_ok[0].status, "pass")


class TestBaseline(unittest.TestCase):
    def test_a_uniform_sphere_gives_pi_over_two(self) -> None:
        rng = np.random.default_rng(1)
        n = 20_000
        theta = np.arccos(2 * rng.random(n) - 1)
        phi = rng.random(n) * TAU
        self.assertAlmostEqual(baseline_distance(theta, phi), math.pi / 2, delta=0.03)

    def test_a_collapsed_projection_gives_a_small_baseline(self) -> None:
        # This is the anchor: without it, every "near" check can pass because
        # the whole map is small, and nobody would notice.
        rng = np.random.default_rng(2)
        theta = rng.uniform(0, 0.2, 5000)
        phi = rng.random(5000) * TAU
        self.assertLess(baseline_distance(theta, phi), 0.3)

    def test_degenerate_inputs(self) -> None:
        self.assertEqual(baseline_distance(np.zeros(0), np.zeros(0)), 0.0)
        self.assertEqual(baseline_distance(np.zeros(1), np.zeros(1)), 0.0)


class TestSummary(unittest.TestCase):
    def test_a_low_baseline_is_called_out(self) -> None:
        text, ok = summarise(run_expectations(GOOD_MAP), baseline=0.4)
        self.assertIn("collapsed", text)
        self.assertTrue(ok)  # the checks passed; the warning is separate

    def test_failures_explain_themselves(self) -> None:
        text, ok = summarise(run_expectations(COLLAPSED_MAP))
        self.assertFalse(ok)
        # The reasoning must be in the output — "FAIL" alone is not actionable.
        self.assertIn("popularity", text)

    def test_a_healthy_baseline_is_not_flagged(self) -> None:
        text, _ = summarise(run_expectations(GOOD_MAP), baseline=1.55)
        self.assertNotIn("WARNING", text)


class TestExpectationSet(unittest.TestCase):
    def test_every_expectation_is_well_formed(self) -> None:
        for exp in DEFAULT_EXPECTATIONS:
            with self.subTest(exp.name):
                self.assertTrue(exp.why, "an expectation without a reason cannot be judged")
                self.assertTrue(exp.near or exp.far)
                if exp.near:
                    self.assertGreaterEqual(len(exp.near), 2)
                    self.assertLess(exp.max_near, math.pi)
                if exp.far:
                    self.assertTrue(exp.far[0] and exp.far[1])
                    self.assertGreater(exp.min_far, 0)

    def test_names_look_like_github_paths(self) -> None:
        # A typo here means a permanent silent skip.
        for exp in DEFAULT_EXPECTATIONS:
            for name in (exp.near or []) + (list(exp.far[0]) + list(exp.far[1]) if exp.far else []):
                with self.subTest(name):
                    self.assertEqual(name.count("/"), 1)
                    self.assertEqual(name, name.lower())

    def test_both_kinds_of_check_are_present(self) -> None:
        # Only "near" checks would pass on a fully collapsed map.
        self.assertTrue(any(e.near for e in DEFAULT_EXPECTATIONS))
        self.assertTrue(any(e.far for e in DEFAULT_EXPECTATIONS))


class TestMeanPairwise(unittest.TestCase):
    def test_fewer_than_two_points(self) -> None:
        self.assertEqual(mean_pairwise([]), 0.0)
        self.assertEqual(mean_pairwise([np.array([1.0, 0, 0])]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
