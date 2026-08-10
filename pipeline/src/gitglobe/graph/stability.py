"""Markov stability — a partition quality measure that is honest about scale.

`cluster_purity` cannot referee between partitions of different granularity.
Measured on one fixed dataset, varying only k:

    median group size    4     10     25     50    166    333
    lift              0.330  0.240  0.191  0.162  0.117  0.095

Lift falls 3.5x purely with size, so "0.06 at median 208" and "0.20 at median 2"
are not comparable numbers. Worse, lift measures within-group similarity, which
is exactly what k-means optimises — on that same data a k-means partition scored
0.162 against the TRUE generating partition's 0.003. The metric structurally
favours geometric methods over graph ones and cannot arbitrate between them.

**Markov stability (Delvenne, Yaliraki & Barahona, 2010) replaces distance with
escape time.** Release a random walker on the graph; a good community is one it
stays inside. Formally, for a partition H at Markov time t:

    r(t) = sum_c [ h_c' diag(pi) P^t h_c  -  (pi' h_c)^2 ]

where P is the transition matrix and pi its stationary distribution.

The property that matters here: **t is the scale**, made explicit and
continuous, instead of hidden inside a group-size artefact. At t=1 this reduces
to Newman-Girvan modularity; as t grows, larger communities become the better
description. So two partitions are compared *at matched t*, and a partition that
stays stable across a wide range of t is genuinely robust rather than an
artefact of one setting.

This is the referee. `cluster_purity` remains useful for one thing only —
telling you whether groups are semantically alike at a FIXED granularity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import math

import numpy as np

log = logging.getLogger(__name__)

#: Markov times to evaluate. Log-spaced because the interesting structure spans
#: orders of magnitude: t=1 is modularity's scale, t>>1 finds continents.
DEFAULT_TIMES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

#: Terms kept in the Taylor series for each sub-step.
WALK_STEPS = 12

#: Largest Markov time handled in one series expansion. The terms t^k/k! only
#: shrink once k > t, so a direct expansion at t=64 with twelve terms is
#: dominated by its growing terms — it returned a stability of 2.5e13, which is
#: obvious nonsense at t=64 and merely "large" at t=16, where it would have
#: passed unnoticed. Splitting t into sub-steps of at most this keeps every
#: expansion inside the convergent regime.
MAX_SUBSTEP = 2.0


@dataclass
class StabilityCurve:
    times: list = field(default_factory=list)
    values: list = field(default_factory=list)
    communities: int = 0

    def best_time(self) -> float:
        return float(self.times[int(np.argmax(self.values))]) if self.values else 0.0

    def summary(self) -> str:
        pairs = ", ".join(f"t={t:g}:{v:.3f}" for t, v in zip(self.times, self.values))
        return f"{self.communities} groups | {pairs}"


def transition_matrix(offsets: np.ndarray, targets: np.ndarray, weights: np.ndarray):
    """Row-stochastic P and stationary distribution pi for an undirected graph.

    For a connected undirected graph the stationary distribution is degree
    proportional, which is exact and avoids a power iteration. Dangling nodes —
    and 86% of this corpus has no dependency edge — get zero stationary mass
    rather than a division by zero.
    """
    n = len(offsets) - 1
    degree = np.zeros(n)
    np.add.at(degree, np.repeat(np.arange(n), np.diff(offsets)), weights)
    total = degree.sum()
    if total <= 0:
        return None, None, degree

    # Transition probabilities: weight / row degree, zero where the row is empty.
    row = np.repeat(np.arange(n), np.diff(offsets))
    probs = np.zeros(len(weights))
    live = degree[row] > 0
    probs[live] = weights[live] / degree[row][live]
    return probs, degree / total, degree


def _walk(vectors: np.ndarray, offsets, targets, probs, steps: int, t: float):
    """exp(t(P - I)) @ vectors, by sub-stepped Taylor series.

    Continuous time is the right formulation: it makes t a real number rather
    than an integer count of hops, so scale sweeps smoothly.

    **Sub-stepping is not optional.** A truncated series is only valid while the
    terms t^k/k! are decreasing, which needs k > t. Applying it directly at
    t=64 with 12 terms leaves a partial sum dominated by the growing terms, and
    the measured stability came back as 2.5e13 — obvious nonsense at a glance,
    and quietly wrong at t=16 where it merely looked large. So t is split into
    chunks of at most 1.0 and the operator applied repeatedly:

        exp(t A) = [exp((t/m) A)]^m

    which is the standard scaling-and-squaring identity, and each factor is well
    inside the series' convergent regime.
    """
    n = len(offsets) - 1
    counts = np.diff(offsets)
    nonempty = counts > 0
    starts = np.clip(offsets[:-1], 0, max(len(targets) - 1, 0))

    # dt <= MAX_SUBSTEP keeps every series term shrinking; 2^12/12! is 9e-6, so
    # twelve terms are ample. `np.add.at` was the obvious way to sum edge
    # contributions per row and is unusably slow — it took this past a two
    # minute timeout. CSR rows are contiguous by construction, so `reduceat`
    # does the same segmented sum at C speed.
    chunks = max(1, int(np.ceil(t / MAX_SUBSTEP)))
    dt = t / chunks
    result = vectors

    for _ in range(chunks):
        term = result
        total = result
        for k in range(1, steps + 1):
            weighted = probs[:, None] * term[targets]
            if len(targets):
                summed = np.add.reduceat(weighted, starts, axis=0)
                # reduceat returns the element AT the index for empty segments
                # rather than zero, which would invent weight for isolated nodes.
                summed[~nonempty] = 0.0
            else:
                summed = np.zeros((n, term.shape[1]))
            term = summed - term
            scale = (dt**k) / math.factorial(k)
            total = total + term * scale
            if np.abs(term).max() * scale < 1e-14:
                break
        result = total
    return result


def stability(
    labels: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    times: tuple = DEFAULT_TIMES,
    *,
    block: int = 64,
) -> StabilityCurve:
    """Markov stability of a partition across a range of Markov times."""
    probs, pi, degree = transition_matrix(offsets, targets, weights)
    curve = StabilityCurve(communities=int(len(np.unique(labels))))
    if probs is None:
        return curve

    unique = np.unique(labels)
    n = len(offsets) - 1

    for t in times:
        total = 0.0
        # Communities in blocks: the indicator matrix is n x k, and k can be
        # tens of thousands. One block at a time keeps peak memory bounded
        # regardless of how finely the graph was partitioned.
        for start in range(0, len(unique), block):
            chunk = unique[start:start + block]
            indicator = np.zeros((n, len(chunk)))
            for j, c in enumerate(chunk):
                indicator[labels == c, j] = 1.0

            walked = _walk(indicator, offsets, targets, probs, WALK_STEPS, t)
            # h' diag(pi) P^t h  -  (pi' h)^2
            autocovariance = (indicator * pi[:, None] * walked).sum(axis=0)
            mass = (pi[:, None] * indicator).sum(axis=0)
            total += float((autocovariance - mass**2).sum())
        curve.times.append(float(t))
        curve.values.append(total)
    return curve


def compare(partitions: dict, offsets, targets, weights, times=DEFAULT_TIMES) -> dict:
    """Score several partitions on the same graph, at the same Markov times.

    This is the comparison `cluster_purity` could not make. Every partition is
    judged by the same walker on the same graph, so a 400-group partition and a
    17,000-group one are finally on one scale.
    """
    return {
        name: stability(labels, offsets, targets, weights, times)
        for name, labels in partitions.items()
    }
