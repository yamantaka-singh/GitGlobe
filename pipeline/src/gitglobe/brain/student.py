"""Student regressor — the distillation half of the teacher-student pair.

The teacher (an LLM, `teacher.py`) scores a stratified sample against the six
rubric dimensions. That costs a network round trip and real money per repo. The
student learns to reproduce those scores from cheap tabular features and labels
the rest of the corpus for free. Teacher supplies judgement, student supplies
scale — and scale is the point once new repositories start arriving.

**The student must never see stars.** `rubric.FORBIDDEN_IN_PROMPT` keeps
popularity out of the teacher's prompt so the labels are not a proxy for it.
That care is wasted if the student then reads `stars` as a feature and
rediscovers the correlation. `assert_no_popularity_features` enforces it on the
feature names rather than on trust.

Pure numpy, matching `graph/communities.py` (Louvain) and `graph/stability.py`.
Boosted trees are the right family for ~4k rows x ~60 heterogeneous columns, and
a from-scratch implementation avoids a dependency that is only ever used here.

Histogram splitting: each feature is binned into at most 64 buckets once, so
split search is one `bincount` per feature per level instead of a sort per node.
Level-wise growth into a complete tree: Power of 10 rule 1 forbids unbounded
recursion, and a complete tree has none — the loop bound is `max_depth`.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field

import numpy as np

from .rubric import DIMENSION_KEYS, FORBIDDEN_IN_PROMPT

log = logging.getLogger(__name__)

N_BINS = 64
MAX_DEPTH = 3
LEARNING_RATE = 0.06
MAX_TREES = 400
PATIENCE = 25
MIN_SAMPLES_LEAF = 12
SUBSAMPLE = 0.8
SCORE_MIN = 0.0
SCORE_MAX = 100.0


def assert_no_popularity_features(names: list[str]) -> None:
    """Refuse to train on a popularity column.

    The teacher is deliberately blindfolded to stars. If the student is not, the
    blindfold bought nothing — the labels stay clean and the model learns the
    correlation from the feature side instead.
    """
    banned = set(FORBIDDEN_IN_PROMPT)
    offenders = [n for n in names if banned & set(n.lower().split("_"))]
    offenders += [n for n in names if n.lower() in banned and n not in offenders]
    if offenders:
        raise ValueError(
            f"popularity features defeat the teacher's blindfold: {sorted(set(offenders))}. "
            "Drop them, or the student is a slow way to recompute stars."
        )


def blindfold(values: np.ndarray, names: list[str]) -> tuple:
    """Drop the popularity columns. Returns `(values, names)` the student may see.

    `build_features` produces `log_stars`, `log_forks`, `stars_per_day`,
    `log_pagerank` and `criticality` on purpose — the globe sizes nodes with
    them and the global rank scores with them. The student is the one consumer
    that must not have them, so the filter belongs here at its boundary rather
    than in the feature builder, which would take them away from everyone.

    Uses the same rule as `assert_no_popularity_features`, so the two cannot
    disagree about what counts as popularity: this drops exactly what that
    would have raised on, and the assert still runs afterwards as the check
    that this worked.
    """
    banned = set(FORBIDDEN_IN_PROMPT)
    keep = [
        i for i, n in enumerate(names)
        if not (banned & set(n.lower().split("_"))) and n.lower() not in banned
    ]
    if not keep:
        raise ValueError("every feature is a popularity feature; nothing to train on")
    dropped = [n for i, n in enumerate(names) if i not in set(keep)]
    if dropped:
        log.info("Blindfold dropped %d popularity features: %s",
                 len(dropped), ", ".join(sorted(dropped)))
    return np.asarray(values)[:, keep], [names[i] for i in keep]


def bin_features(values: np.ndarray, n_bins: int = N_BINS) -> tuple:
    """Quantile-bin each column. Returns (uint8 bins, per-column edges).

    Quantiles not equal width: the columns include heavily skewed counts, and
    equal-width bins would put 90% of rows in bin zero and learn nothing.
    """
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")
    if not 2 <= n_bins <= 256:
        raise ValueError(f"n_bins must be in [2, 256], got {n_bins}")

    n, d = values.shape
    bins = np.zeros((n, d), dtype=np.uint8)
    edges = []
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]

    for j in range(d):
        # A constant column yields no cuts; searchsorted on an empty array
        # returns all zeros, which is the correct answer: one bin.
        cut = np.unique(np.quantile(values[:, j], quantiles))
        edges.append(cut)
        bins[:, j] = np.searchsorted(cut, values[:, j], side="right").astype(np.uint8)

    return bins, edges


def apply_bins(values: np.ndarray, edges: list) -> np.ndarray:
    """Bin new rows with the edges learned at fit time.

    Re-deriving quantiles per batch would put the same repository in different
    bins depending on which batch it arrived in — the same failure the PCA basis
    is careful to avoid.
    """
    if values.shape[1] != len(edges):
        raise ValueError(f"expected {len(edges)} columns, got {values.shape[1]}")
    bins = np.zeros(values.shape, dtype=np.uint8)
    for j, cut in enumerate(edges):
        bins[:, j] = np.searchsorted(cut, values[:, j], side="right").astype(np.uint8)
    return bins


@dataclass
class Tree:
    """A complete binary tree in heap order, grown to exactly `depth`.

    Internal nodes number 2**depth - 1, leaves 2**depth. A node with
    `feature == NO_SPLIT` sends every row left, which keeps the shape regular
    and the arithmetic vectorised.
    """

    NO_SPLIT = -1

    depth: int
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray

    def leaf_of(self, bins: np.ndarray) -> np.ndarray:
        """Which leaf each row lands in. Bounded loop, no recursion."""
        position = np.zeros(len(bins), dtype=np.int64)
        for level in range(self.depth):
            node = (1 << level) - 1 + position
            feat = self.feature[node]
            split = self.threshold[node]
            live = feat != self.NO_SPLIT
            go_right = np.zeros(len(bins), dtype=np.int64)
            if live.any():
                rows = np.where(live)[0]
                go_right[rows] = (bins[rows, feat[rows]] > split[rows]).astype(np.int64)
            position = position * 2 + go_right
        return position

    def predict(self, bins: np.ndarray) -> np.ndarray:
        return self.value[self.leaf_of(bins)]


def _best_splits(bins, residual, node_of, n_nodes, min_samples) -> tuple:
    """Best (feature, bin) per node at one level, by squared-error reduction.

    For a squared loss the gain depends only on per-side sums and counts, which
    come from a prefix sum over the histogram — so each feature costs two
    `bincount` calls regardless of how many nodes are at this level.
    """
    d = bins.shape[1]
    best_gain = np.full(n_nodes, -np.inf)
    best_feature = np.full(n_nodes, Tree.NO_SPLIT, dtype=np.int64)
    best_threshold = np.zeros(n_nodes, dtype=np.int64)

    width = N_BINS
    total_sum = np.bincount(node_of, weights=residual, minlength=n_nodes)
    total_count = np.bincount(node_of, minlength=n_nodes).astype(np.float64)

    for j in range(d):
        index = node_of * width + bins[:, j]
        grad = np.bincount(index, weights=residual, minlength=n_nodes * width)
        count = np.bincount(index, minlength=n_nodes * width).astype(np.float64)
        grad = grad.reshape(n_nodes, width).cumsum(axis=1)
        count = count.reshape(n_nodes, width).cumsum(axis=1)

        right_sum = total_sum[:, None] - grad
        right_count = total_count[:, None] - count
        ok = (count >= min_samples) & (right_count >= min_samples)
        gain = np.where(
            ok,
            grad**2 / np.maximum(count, 1.0) + right_sum**2 / np.maximum(right_count, 1.0),
            -np.inf,
        )
        candidate = gain.argmax(axis=1)
        value = gain[np.arange(n_nodes), candidate]
        better = value > best_gain
        best_gain = np.where(better, value, best_gain)
        best_feature = np.where(better, j, best_feature)
        best_threshold = np.where(better, candidate, best_threshold)

    return best_feature, best_threshold, best_gain


def grow_tree(bins, residual, *, depth=MAX_DEPTH, min_samples=MIN_SAMPLES_LEAF) -> Tree:
    """One regression tree, grown level by level to exactly `depth`."""
    n_internal = (1 << depth) - 1
    feature = np.full(n_internal, Tree.NO_SPLIT, dtype=np.int64)
    threshold = np.zeros(n_internal, dtype=np.int64)
    node_of = np.zeros(len(bins), dtype=np.int64)

    for level in range(depth):
        n_nodes = 1 << level
        offset = n_nodes - 1
        feat, thresh, gain = _best_splits(bins, residual, node_of, n_nodes, min_samples)
        usable = np.isfinite(gain)
        feature[offset:offset + n_nodes] = np.where(usable, feat, Tree.NO_SPLIT)
        threshold[offset:offset + n_nodes] = thresh

        rows = np.where(usable[node_of])[0]
        go_right = np.zeros(len(bins), dtype=np.int64)
        if len(rows):
            node = node_of[rows]
            go_right[rows] = (bins[rows, feat[node]] > thresh[node]).astype(np.int64)
        node_of = node_of * 2 + go_right

    n_leaves = 1 << depth
    leaf_sum = np.bincount(node_of, weights=residual, minlength=n_leaves)
    leaf_count = np.bincount(node_of, minlength=n_leaves).astype(np.float64)
    return Tree(depth, feature, threshold, leaf_sum / np.maximum(leaf_count, 1.0))


@dataclass
class Student:
    """A fitted per-dimension ensemble plus the binning it was fitted with."""

    dimension: str
    base: float
    trees: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    learning_rate: float = LEARNING_RATE
    holdout: dict = field(default_factory=dict)

    def predict(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"student was fitted on {len(self.feature_names)} features, "
                f"got {values.shape[1]}"
            )
        bins = apply_bins(np.asarray(values, dtype=np.float64), self.edges)
        out = np.full(len(bins), self.base, dtype=np.float64)
        for tree in self.trees:
            out += self.learning_rate * tree.predict(bins)
        # The teacher's scale is 0-100; nothing outside it means anything.
        return np.clip(out, SCORE_MIN, SCORE_MAX)


def _fit_one(bins, y, val_bins, val_y, *, seed: int) -> tuple:
    """Boosting loop with early stopping. Returns (base, trees, history)."""
    rng = np.random.default_rng(seed)
    base = float(y.mean())
    prediction = np.full(len(y), base)
    val_prediction = np.full(len(val_y), base)

    trees: list = []
    best_rmse = float(np.sqrt(np.mean((val_y - val_prediction) ** 2)))
    best_n = 0
    history = [best_rmse]
    take = max(1, int(len(y) * SUBSAMPLE))

    for _ in range(MAX_TREES):
        pick = rng.choice(len(y), take, replace=False)
        tree = grow_tree(bins[pick], (y - prediction)[pick])
        prediction += LEARNING_RATE * tree.predict(bins)
        val_prediction += LEARNING_RATE * tree.predict(val_bins)
        trees.append(tree)

        rmse = float(np.sqrt(np.mean((val_y - val_prediction) ** 2)))
        history.append(rmse)
        if rmse < best_rmse - 1e-9:
            best_rmse, best_n = rmse, len(trees)
        elif len(trees) - best_n >= PATIENCE:
            break

    # Trees after the best held-out point are measured overfitting, not a
    # judgement call.
    return base, trees[:best_n], history


def _fit_dimension_worker(key: str, bins_train: np.ndarray, y_train: np.ndarray, bins_test: np.ndarray, y_test: np.ndarray, values_test: np.ndarray, edges: list, feature_names: list, seed: int) -> tuple:
    base, trees, history = _fit_one(bins_train, y_train, bins_test, y_test, seed=seed)
    student = Student(key, base, trees, edges, list(feature_names))
    student.holdout = score(student, values_test, y_test)
    student.holdout["trees"] = len(trees)
    student.holdout["rmse_curve"] = [round(h, 3) for h in history[:6]]
    return key, student, len(trees), student.holdout


def fit(values, labels: dict, feature_names: list, *, train_idx, test_idx, seed: int = 11) -> dict:
    """Train one student per rubric dimension. Returns dimension -> Student."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] != len(feature_names):
        raise ValueError(f"{values.shape[1]} columns but {len(feature_names)} feature names")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train and test splits must both be non-empty")
    assert_no_popularity_features(feature_names)

    bins, edges = bin_features(values)
    students = {}

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for key in DIMENSION_KEYS:
            if key not in labels:
                log.warning("No teacher labels for dimension %r — skipping", key)
                continue
            y = np.asarray(labels[key], dtype=np.float64)
            if len(y) != len(values):
                raise ValueError(f"{key}: {len(y)} labels for {len(values)} rows")

            futures.append(
                executor.submit(
                    _fit_dimension_worker,
                    key,
                    bins[train_idx],
                    y[train_idx],
                    bins[test_idx],
                    y[test_idx],
                    values[test_idx],
                    edges,
                    list(feature_names),
                    seed,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            key, student, n_trees, holdout = future.result()
            students[key] = student
            log.info(
                "%s: %d trees, held-out RMSE %.2f, R2 %.3f (baseline RMSE %.2f)",
                key, n_trees, holdout["rmse"], holdout["r2"],
                holdout["baseline_rmse"],
            )

    return students


def score(student: Student, values: np.ndarray, truth: np.ndarray) -> dict:
    """Held-out quality, always against the honest baseline."""
    truth = np.asarray(truth, dtype=np.float64)
    residual = truth - student.predict(values)
    baseline = truth - student.base

    rmse = float(np.sqrt(np.mean(residual**2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))

    # `rmse < baseline_rmse` is NOT good enough, and measuring proved it: on
    # pure noise this scored 28.29 against a baseline of 28.34 and the naive
    # comparison called it a win. Any difference at all passes a strict
    # inequality, so on a model that learned nothing the flag is a coin flip.
    # The improvement must clear its own sampling noise, ~rmse/sqrt(2n).
    n = max(len(truth), 1)
    margin = baseline_rmse / np.sqrt(2.0 * n)
    return {
        "n": int(len(truth)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(residual))),
        "baseline_rmse": baseline_rmse,
        "margin": float(margin),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "beats_baseline": bool(baseline_rmse - rmse > margin),
    }


def composite(students: dict, values: np.ndarray) -> np.ndarray:
    """One 0-100 score per repository, a flat mean across dimensions.

    Flat deliberately: weighting the dimensions is a product judgement with no
    evidence behind it yet, and a weighted sum invented here would look
    principled while being arbitrary.
    """
    if not students:
        raise ValueError("no fitted students")
    return np.stack([s.predict(values) for s in students.values()]).mean(axis=0)
