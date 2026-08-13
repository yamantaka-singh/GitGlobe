"""Tests for the student regressor.

The load-bearing test is `test_reports_failure_on_pure_noise`. Twice in this
project a metric has flattered something that had learned nothing, and both
times changes were made on the strength of it before anyone checked. So the
fixtures verify their own construction, and the suite asserts what the model
must FAIL to do as carefully as what it must do.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gitglobe.brain import student as st
from gitglobe.brain.rubric import DIMENSION_KEYS


def learnable(n=1200, d=24, noise=6.0, seed=0):
    """Non-linear and interacting on purpose: a linear fixture would be passed
    by a model that only ever splits on one feature, hiding a broken search."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (
        30.0 * (X[:, 0] > 0.4)
        + 25.0 * np.abs(X[:, 1])
        + 18.0 * (X[:, 2] * X[:, 3] > 0.5)
        + 50.0
        + rng.normal(scale=noise, size=n)
    )
    return X, np.clip(y, 0, 100)


def pure_noise(n=1200, d=24, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.uniform(0, 100, size=n)


def split(n, fraction=0.25, seed=3):
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * fraction)
    return np.sort(order[cut:]), np.sort(order[:cut])


def names(d):
    return [f"f{i}" for i in range(d)]


def fit_one(X, y, key="maintenance", seed=11):
    train, test = split(len(X))
    return st.fit(X, {key: y}, names(X.shape[1]), train_idx=train, test_idx=test, seed=seed)[key]


class TestFixtures(unittest.TestCase):
    """If the fixtures are wrong, every test below proves nothing."""

    def test_learnable_has_signal(self) -> None:
        _, y = learnable()
        self.assertGreater(y.std(), 15.0)

    def test_noise_has_none(self) -> None:
        X, y = pure_noise()
        for j in range(X.shape[1]):
            self.assertLess(abs(np.corrcoef(X[:, j], y)[0, 1]), 0.12)


class TestLearning(unittest.TestCase):
    def test_beats_the_baseline_on_real_signal(self) -> None:
        model = fit_one(*learnable())
        self.assertTrue(model.holdout["beats_baseline"])
        self.assertGreater(model.holdout["r2"], 0.55)

    def test_reports_failure_on_pure_noise(self) -> None:
        # THE test. A model that cannot tell signal from noise is worse than
        # useless here — its scores are shown as judgements about real repos.
        model = fit_one(*pure_noise())
        self.assertLess(model.holdout["r2"], 0.05)
        self.assertFalse(model.holdout["beats_baseline"])

    def test_beats_baseline_requires_more_than_a_hair(self) -> None:
        # Regression test for a real defect: the first `score` used
        # `rmse < baseline_rmse` and on noise returned 28.29 vs 28.34 — a WIN.
        model = fit_one(*pure_noise(seed=9))
        gap = model.holdout["baseline_rmse"] - model.holdout["rmse"]
        self.assertLess(gap, model.holdout["margin"])

    def test_captures_an_interaction_not_just_a_main_effect(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(size=(1500, 8))
        # XOR: neither feature alone carries any signal at all.
        y = 70.0 * ((X[:, 0] > 0) ^ (X[:, 1] > 0)) + 15.0
        self.assertGreater(fit_one(X, y).holdout["r2"], 0.75)

    def test_early_stopping_keeps_fewer_trees_than_the_ceiling(self) -> None:
        model = fit_one(*learnable(noise=25.0))
        self.assertLess(len(model.trees), st.MAX_TREES)
        self.assertGreater(len(model.trees), 0)

    def test_predictions_stay_on_the_rubric_scale(self) -> None:
        X, y = learnable()
        out = fit_one(X, y).predict(X)
        self.assertGreaterEqual(out.min(), st.SCORE_MIN)
        self.assertLessEqual(out.max(), st.SCORE_MAX)


class TestBinning(unittest.TestCase):
    def test_quantile_bins_spread_a_skewed_column(self) -> None:
        # Exponential: equal-width binning would put nearly all rows in bin 0.
        X = np.random.default_rng(0).exponential(scale=3.0, size=(2000, 1)) ** 3
        bins, _ = st.bin_features(X)
        self.assertGreater(len(np.unique(bins[:, 0])), 30)

    def test_a_constant_column_collapses_to_one_bin(self) -> None:
        X = np.hstack([np.ones((200, 1)), np.random.default_rng(0).normal(size=(200, 1))])
        bins, edges = st.bin_features(X)
        self.assertEqual(len(np.unique(bins[:, 0])), 1)
        self.assertLessEqual(len(edges[0]), 1)

    def test_edges_are_reusable_across_batches(self) -> None:
        # Re-deriving quantiles per batch would put the same repo in a
        # different bin depending on which batch it arrived in.
        X, _ = learnable()
        bins, edges = st.bin_features(X)
        np.testing.assert_array_equal(st.apply_bins(X[:40], edges), bins[:40])

    def test_wrong_width_is_refused(self) -> None:
        _, edges = st.bin_features(learnable(d=6)[0])
        with self.assertRaises(ValueError):
            st.apply_bins(np.zeros((3, 5)), edges)


class TestTreeShape(unittest.TestCase):
    def test_every_row_lands_in_a_real_leaf(self) -> None:
        X, y = learnable(n=400)
        bins, _ = st.bin_features(X)
        tree = st.grow_tree(bins, y - y.mean())
        leaves = tree.leaf_of(bins)
        self.assertEqual(len(tree.value), 1 << st.MAX_DEPTH)
        self.assertGreaterEqual(leaves.min(), 0)
        self.assertLess(leaves.max(), 1 << st.MAX_DEPTH)

    def test_a_node_with_no_admissible_split_sends_rows_left(self) -> None:
        rng = np.random.default_rng(0)
        bins, _ = st.bin_features(rng.normal(size=(10, 4)))
        tree = st.grow_tree(bins, rng.normal(size=10))
        self.assertTrue((tree.feature == st.Tree.NO_SPLIT).all())
        np.testing.assert_array_equal(tree.leaf_of(bins), np.zeros(10, dtype=np.int64))


class TestBlindfold(unittest.TestCase):
    def test_popularity_features_are_refused(self) -> None:
        for bad in ("stars", "log_stars", "forks", "pagerank", "stars_90d", "trending"):
            with self.subTest(feature=bad), self.assertRaises(ValueError):
                st.assert_no_popularity_features(["days_since_push", bad])

    def test_innocent_features_pass(self) -> None:
        st.assert_no_popularity_features(
            ["days_since_push", "license_permissive", "in_degree", "emb_07", "readme_len"]
        )

    def test_fit_refuses_a_popularity_column(self) -> None:
        X, y = learnable(d=4)
        train, test = split(len(X))
        with self.assertRaises(ValueError):
            st.fit(X, {"maintenance": y}, ["a", "stars", "c", "d"],
                   train_idx=train, test_idx=test)


class TestContract(unittest.TestCase):
    def test_name_count_must_match_columns(self) -> None:
        X, y = learnable(d=6)
        train, test = split(len(X))
        with self.assertRaises(ValueError):
            st.fit(X, {"maintenance": y}, names(3), train_idx=train, test_idx=test)

    def test_label_length_must_match_rows(self) -> None:
        X, y = learnable(d=6)
        train, test = split(len(X))
        with self.assertRaises(ValueError):
            st.fit(X, {"maintenance": y[:-5]}, names(6), train_idx=train, test_idx=test)

    def test_empty_split_is_refused(self) -> None:
        X, y = learnable(d=6)
        with self.assertRaises(ValueError):
            st.fit(X, {"maintenance": y}, names(6),
                   train_idx=np.arange(10), test_idx=np.array([], dtype=np.int64))

    def test_missing_dimension_is_skipped_not_fatal(self) -> None:
        # A teacher run that failed on one dimension should still yield the
        # others rather than losing the whole run.
        X, y = learnable(d=6)
        train, test = split(len(X))
        out = st.fit(X, {DIMENSION_KEYS[0]: y}, names(6), train_idx=train, test_idx=test)
        self.assertEqual(list(out), [DIMENSION_KEYS[0]])

    def test_predict_refuses_the_wrong_feature_count(self) -> None:
        model = fit_one(*learnable(d=6))
        with self.assertRaises(ValueError):
            model.predict(np.zeros((3, 5)))


class TestComposite(unittest.TestCase):
    def test_averages_the_dimensions(self) -> None:
        X, y = learnable(d=8)
        train, test = split(len(X))
        students = st.fit(X, {DIMENSION_KEYS[0]: y, DIMENSION_KEYS[1]: 100.0 - y},
                          names(8), train_idx=train, test_idx=test)
        out = st.composite(students, X)
        self.assertEqual(len(out), len(X))
        # Two dimensions that are exact mirrors must average near the midpoint.
        self.assertLess(abs(float(out.mean()) - 50.0), 6.0)

    def test_no_students_is_an_error_not_a_silent_zero(self) -> None:
        with self.assertRaises(ValueError):
            st.composite({}, np.zeros((3, 4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
