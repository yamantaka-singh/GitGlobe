"""Tests for ranking against all of GitHub rather than against our own corpus.

The bug this file exists to prevent is a rank that *looks* absolute and is not.
An in-corpus percentile presented as "top 2% of GitHub" is wrong by orders of
magnitude and wrong in the flattering direction, which is exactly the kind of
error nobody reports because the number seems plausible.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.rank.calibrate import (  # noqa: E402
    GlobalRank,
    RepoSignals,
    composite_score,
)
from gitglobe.rank.global_scale import (  # noqa: E402
    STAR_LADDER,
    StarScale,
    Weights,
    dependents_percentile,
    monotonic_repair,
)


def realistic_scale() -> StarScale:
    """Roughly GitHub-shaped: a steep power law over 420M repositories.

    Not a claim about the real numbers — `gitglobe calibrate` measures those.
    This is a fixture with the right SHAPE so the maths can be checked.
    """
    thresholds = [0, 1, 10, 100, 1_000, 10_000, 100_000]
    counts = [420_000_000, 30_000_000, 3_000_000, 300_000, 28_000, 2_200, 60]
    scale = StarScale(thresholds=thresholds, counts=counts, total_repos=420_000_000)
    scale.validate()
    return scale


class TestStarScale(unittest.TestCase):
    def test_measured_rungs_are_returned_exactly(self) -> None:
        scale = realistic_scale()
        for threshold, count in zip(scale.thresholds, scale.counts):
            with self.subTest(threshold):
                self.assertAlmostEqual(scale.repos_at_least(threshold), count, delta=1)

    def test_interpolates_monotonically_between_rungs(self) -> None:
        scale = realistic_scale()
        previous = float("inf")
        for stars in range(0, 5_000, 37):
            current = scale.repos_at_least(stars)
            with self.subTest(stars=stars):
                self.assertLessEqual(current, previous + 1e-6)
            previous = current

    def test_does_not_extrapolate_past_the_top_rung(self) -> None:
        # Inventing precision about the handful of repos above the last measured
        # threshold would produce confident nonsense at exactly the point people
        # look hardest.
        scale = realistic_scale()
        self.assertEqual(scale.repos_at_least(500_000), scale.repos_at_least(10**9))

    def test_percentile_is_bounded_and_ordered(self) -> None:
        scale = realistic_scale()
        self.assertGreaterEqual(scale.percentile(0), 0.0)
        self.assertLessEqual(scale.percentile(10**7), 1.0)
        self.assertLess(scale.percentile(10), scale.percentile(10_000))

    def test_a_corpus_percentile_would_have_been_wildly_wrong(self) -> None:
        # THE point of this module. Our corpus starts around 50 stars, so its
        # median repo looks mid-table locally and is globally exceptional.
        scale = realistic_scale()
        global_pct = scale.percentile(200)
        self.assertGreater(global_pct, 0.99)
        # A 200-star repo is roughly the 50th percentile of OUR corpus but the
        # 99th+ of GitHub. Reporting the former as the latter is off by ~50x.
        self.assertGreater(global_pct / 0.50, 1.9)

    def test_rank_and_description_read_sensibly(self) -> None:
        scale = realistic_scale()
        self.assertGreater(scale.rank_of(10), scale.rank_of(10_000))
        text = scale.describe(50_000)
        self.assertIn("#", text)
        self.assertIn("420M", text)

    def test_round_trips_through_json(self) -> None:
        scale = realistic_scale()
        restored = StarScale.from_dict(scale.to_dict())
        self.assertEqual(restored.thresholds, scale.thresholds)
        self.assertAlmostEqual(restored.percentile(500), scale.percentile(500))


class TestScaleRejectsBadMeasurements(unittest.TestCase):
    def test_a_rising_survival_function_is_refused(self) -> None:
        # Counts can only fall as the threshold rises. A rise means a rung was
        # rate-limited or truncated — and baking that into the scale would
        # distort every rank derived from it, silently.
        scale = StarScale(thresholds=[0, 10, 100], counts=[100, 500, 10])
        with self.assertRaises(ValueError):
            scale.validate()

    def test_too_few_rungs_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            StarScale(thresholds=[0, 10], counts=[100, 10]).validate()

    def test_unsorted_thresholds_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            StarScale(thresholds=[10, 0, 100], counts=[10, 5, 1]).validate()

    def test_monotonic_repair_drops_the_bad_rung_not_the_scale(self) -> None:
        t, c = monotonic_repair([0, 10, 100, 1000], [1000, 500, 900, 50])
        self.assertEqual(t, [0, 10, 1000])
        self.assertEqual(c, [1000, 500, 50])

    def test_repair_drops_zero_and_ceiling_counts(self) -> None:
        t, c = monotonic_repair([0, 10, 100], [10**10, 500, 0])
        self.assertEqual(t, [10])
        self.assertEqual(c, [500])

    def test_the_default_ladder_is_usable(self) -> None:
        self.assertEqual(STAR_LADDER, sorted(STAR_LADDER))
        self.assertGreater(len(STAR_LADDER), 20)
        # Dense at the bottom, where almost every repository actually lives.
        self.assertGreaterEqual(sum(1 for s in STAR_LADDER if s <= 100), 10)


class TestDependentsPercentile(unittest.TestCase):
    def test_absolute_and_monotone(self) -> None:
        self.assertEqual(dependents_percentile(0), 0.0)
        values = [dependents_percentile(n) for n in (1, 10, 100, 1_000, 10_000)]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(max(values), 1.0)

    def test_one_dependent_beats_none_by_a_lot(self) -> None:
        # Being depended on at all is the single biggest qualitative jump.
        self.assertGreater(dependents_percentile(1) - dependents_percentile(0), 0.2)

    def test_saturates_rather_than_running_away(self) -> None:
        self.assertLessEqual(dependents_percentile(10**7), 1.0)


class TestCompositeScore(unittest.TestCase):
    def test_score_is_bounded(self) -> None:
        scale = realistic_scale()
        for signals in (
            RepoSignals(),
            RepoSignals(stars=10**6, dependents=10**6, pagerank_ratio=10**4,
                        criticality=1.0, stars_90d=10**5),
        ):
            result = composite_score(signals, scale)
            with self.subTest(signals):
                self.assertGreaterEqual(result.score, 0.0)
                self.assertLessEqual(result.score, 100.0)

    def test_dependents_outweigh_stars(self) -> None:
        # The product's central claim: a quietly load-bearing library should
        # outrank a popular tutorial. If stars dominated, this whole project
        # would be GitHub's own ranking with extra steps.
        scale = realistic_scale()
        tutorial = RepoSignals(stars=40_000, dependents=0, pagerank_ratio=1.0)
        library = RepoSignals(stars=800, dependents=5_000, pagerank_ratio=60.0)
        self.assertGreater(
            composite_score(library, scale).score,
            composite_score(tutorial, scale).score,
        )

    def test_stars_still_break_ties_among_otherwise_equal_repos(self) -> None:
        scale = realistic_scale()
        low = RepoSignals(stars=100, dependents=50)
        high = RepoSignals(stars=10_000, dependents=50)
        self.assertGreater(
            composite_score(high, scale).score, composite_score(low, scale).score
        )

    def test_pagerank_is_a_ratio_so_corpus_size_does_not_change_the_score(self) -> None:
        # A raw PageRank value shrinks as 1/n purely because the corpus grew.
        # Expressing it as a multiple of the mean keeps the score stable as the
        # globe scales from 87k to 1M.
        scale = realistic_scale()
        signals = RepoSignals(stars=500, dependents=20, pagerank_ratio=50.0)
        self.assertEqual(
            composite_score(signals, scale).score,
            composite_score(signals, scale).score,
        )
        weaker = RepoSignals(stars=500, dependents=20, pagerank_ratio=1.0)
        self.assertGreater(
            composite_score(signals, scale).score, composite_score(weaker, scale).score
        )

    def test_components_are_reported_for_inspection(self) -> None:
        # A single blended number is unarguable-with. The parts must be visible.
        result = composite_score(
            RepoSignals(stars=1_000, dependents=100, criticality=0.5), realistic_scale()
        )
        self.assertEqual(
            sorted(result.components),
            ["criticality", "dependents", "pagerank", "stars", "velocity"],
        )

    def test_zero_weights_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            composite_score(
                RepoSignals(stars=10),
                realistic_scale(),
                Weights(stars=0, dependents=0, pagerank=0, criticality=0, velocity=0),
            )

    def test_star_rank_accompanies_the_score(self) -> None:
        result = composite_score(RepoSignals(stars=10_000), realistic_scale())
        self.assertIsInstance(result, GlobalRank)
        self.assertLess(result.star_rank, 10_000)
        self.assertGreater(result.star_percentile, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
