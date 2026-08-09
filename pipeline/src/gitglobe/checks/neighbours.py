"""The Phase 2 exit criterion: does the map mean anything?

Every automated check so far proves the pipeline is *self-consistent* — bytes
round-trip, PageRank sums to 1, the CSR is symmetric. None of them can tell you
whether `pytorch` landed next to `tensorflow` or next to a Minecraft mod.

That question has only one honest form: name the things that must be near each
other and the things that must not, before looking, and then look. The
expectations below are written from domain knowledge, not from the output. If
you find yourself editing them because the map disagreed, that is the map
telling you something and the expectations are not the place to fix it.

Distances are great-circle on the unit sphere, in radians. The whole sphere is
PI across, so:

    < 0.25   the same neighbourhood
    < 0.60   the same region
    > 1.40   opposite sides of the map
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Expectation:
    """One claim about the map, checkable and falsifiable."""

    name: str
    #: These should sit close together.
    near: list[str] = field(default_factory=list)
    #: These two groups should sit far apart.
    far: tuple[list[str], list[str]] | None = None
    max_near: float = 0.60
    min_far: float = 1.00
    why: str = ""


#: Chosen so that each one fails for a different, diagnosable reason.
DEFAULT_EXPECTATIONS = [
    Expectation(
        name="deep learning frameworks",
        near=["pytorch/pytorch", "tensorflow/tensorflow", "jax-ml/jax"],
        max_near=0.50,
        why="Direct competitors solving the same problem. If these are not "
            "close, the embeddings are not capturing what software does.",
    ),
    Expectation(
        name="frontend frameworks",
        near=["facebook/react", "vuejs/core", "sveltejs/svelte"],
        max_near=0.50,
        why="Same. Chosen from a different domain so a pass on one and a "
            "failure on the other localises the problem.",
    ),
    Expectation(
        name="frontend is not machine learning",
        far=(["facebook/react", "vuejs/core"], ["pytorch/pytorch", "tensorflow/tensorflow"]),
        min_far=1.00,
        why="The strongest signal available. Both groups are popular, "
            "JavaScript-and-Python heavy, and heavily starred — so if the "
            "embedding is really encoding popularity or language rather than "
            "capability, these collapse together and this is what catches it.",
    ),
    Expectation(
        name="container orchestration",
        near=["kubernetes/kubernetes", "docker/compose", "hashicorp/nomad"],
        max_near=0.65,
        why="A looser group: infrastructure that is related by role rather "
            "than by being alternatives.",
    ),
    Expectation(
        name="databases",
        near=["postgres/postgres", "redis/redis", "mongodb/mongo"],
        max_near=0.70,
        why="Different data models, same job. Tests that the map groups by "
            "purpose and not by implementation language — these are C, C and "
            "C++ respectively, so language alone would also group them; the "
            "frontend/ML check is what rules that explanation out.",
    ),
    Expectation(
        name="build tools are not web frameworks",
        far=(["webpack/webpack", "vitejs/vite"], ["pytorch/pytorch"]),
        min_far=0.80,
        why="A weaker separation than frontend-vs-ML, deliberately: it should "
            "pass with less margin, and a suspiciously large margin here "
            "suggests the map is over-separated and clusters are islands.",
    ),
]


def great_circle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two unit vectors, in radians.

    `arccos` of the dot product loses precision badly for small angles — which
    is exactly the range this check cares about. `arctan2` of the cross and dot
    stays accurate all the way down.
    """
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b))))


def to_vector(theta: float, phi: float) -> np.ndarray:
    st = math.sin(theta)
    return np.array([st * math.cos(phi), math.cos(theta), st * math.sin(phi)])


def mean_pairwise(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    return float(np.mean([
        great_circle(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]))


def mean_between(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    if not a or not b:
        return 0.0
    return float(np.mean([great_circle(x, y) for x in a for y in b]))


@dataclass
class CheckOutcome:
    name: str
    status: str  # "pass" | "fail" | "skipped"
    measured: float | None = None
    threshold: float | None = None
    missing: list[str] = field(default_factory=list)
    why: str = ""
    #: True for separation checks, where the threshold is a floor not a ceiling.
    far_style: bool = False

    def line(self) -> str:
        if self.status == "skipped":
            return f"  skip  {self.name} — not in corpus: {', '.join(self.missing)}"
        mark = "ok  " if self.status == "pass" else "FAIL"
        comparator = ">" if self.far_style else "<"
        return (
            f"  {mark}  {self.name} — {self.measured:.3f} rad "
            f"(need {comparator} {self.threshold:.3f})"
        )


def run_expectations(
    positions: dict[str, tuple[float, float]],
    expectations: list[Expectation] | None = None,
) -> list[CheckOutcome]:
    """Evaluate every expectation against a name -> (theta, phi) map.

    Missing repositories are SKIPPED, not failed. A 5,000-row proof run will not
    contain most of these, and a check that fails because the data is small
    teaches you to ignore it.
    """
    expectations = expectations or DEFAULT_EXPECTATIONS
    lookup = {k.lower(): to_vector(*v) for k, v in positions.items()}
    outcomes = []

    for exp in expectations:
        if exp.far:
            group_a, group_b = exp.far
            wanted = group_a + group_b
        else:
            wanted = exp.near

        missing = [n for n in wanted if n.lower() not in lookup]
        if len(wanted) - len(missing) < 2:
            outcomes.append(CheckOutcome(exp.name, "skipped", missing=missing, why=exp.why))
            continue

        if exp.far:
            a = [lookup[n.lower()] for n in exp.far[0] if n.lower() in lookup]
            b = [lookup[n.lower()] for n in exp.far[1] if n.lower() in lookup]
            if not a or not b:
                outcomes.append(CheckOutcome(exp.name, "skipped", missing=missing, why=exp.why))
                continue
            measured = mean_between(a, b)
            outcomes.append(CheckOutcome(
                exp.name, "pass" if measured >= exp.min_far else "fail",
                measured, exp.min_far, missing, exp.why, far_style=True,
            ))
        else:
            vectors = [lookup[n.lower()] for n in exp.near if n.lower() in lookup]
            measured = mean_pairwise(vectors)
            outcomes.append(CheckOutcome(
                exp.name, "pass" if measured <= exp.max_near else "fail",
                measured, exp.max_near, missing, exp.why,
            ))
    return outcomes


def baseline_distance(theta: np.ndarray, phi: np.ndarray, samples: int = 5000, seed: int = 0) -> float:
    """Mean distance between random pairs — what "unrelated" looks like here.

    Without it the thresholds are unanchored. On a uniformly covered sphere this
    is PI/2 ~ 1.571; a value far below that means the projection has collapsed
    and every "near" check will pass for the wrong reason.
    """
    n = len(theta)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, samples), rng.integers(0, n, samples)
    keep = i != j
    st_i, st_j = np.sin(theta[i[keep]]), np.sin(theta[j[keep]])
    dot = (
        st_i * np.cos(phi[i[keep]]) * st_j * np.cos(phi[j[keep]])
        + np.cos(theta[i[keep]]) * np.cos(theta[j[keep]])
        + st_i * np.sin(phi[i[keep]]) * st_j * np.sin(phi[j[keep]])
    )
    return float(np.mean(np.arccos(np.clip(dot, -1, 1))))


def summarise(outcomes: list[CheckOutcome], baseline: float | None = None) -> tuple[str, bool]:
    lines = [o.line() for o in outcomes]
    passed = sum(o.status == "pass" for o in outcomes)
    failed = sum(o.status == "fail" for o in outcomes)
    skipped = sum(o.status == "skipped" for o in outcomes)

    if baseline is not None:
        lines.append("")
        lines.append(f"  baseline distance between random repositories: {baseline:.3f} rad")
        if baseline < 1.0:
            lines.append(
                "  WARNING: that is far below PI/2 (1.571). The projection has "
                "collapsed onto part of the sphere, and the 'near' checks above "
                "may be passing because everything is near everything."
            )
    lines.append("")
    lines.append(f"  {passed} passed, {failed} failed, {skipped} skipped")

    if failed:
        lines.append("")
        for o in outcomes:
            if o.status == "fail":
                lines.append(f"  {o.name}: {o.why}")
    return "\n".join(lines), failed == 0
