"""Features the student learns from — all computable for 1M repos, no API calls.

The teacher reads a README and forms a judgement. That costs a model call per
repository, which is fine for four thousand and impossible for a million. The
student's job is to predict the teacher's judgement from things we already hold.

So every feature here comes out of Postgres. Nothing fetches, nothing scrapes,
nothing calls a model. If a feature cannot be computed for every row in one
query, it does not belong in this file.

**Three groups, and the reason each earns its place:**

* *Activity* — recency, velocity, ratios. Predicts `maintenance` almost on its
  own: a repository pushed to last week behaves differently from one last
  touched in 2019, and no amount of README polish hides that.
* *Shape* — README length, cleaner reduction, topic count, licence. Predicts
  `specificity` and `onboarding_ease`. An awesome-list has a distinctive
  signature: enormous raw README, enormous cleaner reduction (it is nearly all
  links), many topics, no licence.
* *Position* — PageRank, degree, cluster typicality, embedding components. This
  is the group popularity cannot fake. In-degree from the dependency graph is
  the single strongest evidence for `canonicity`: being depended upon is a
  different fact from being starred.

**On stars.** The teacher never sees them; the student may. That asymmetry is
the whole design. If the target were contaminated by popularity, the student
would score well by recomputing stars and we would have learned nothing —
`popularity_leakage` in `student.py` is what checks that it did not happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

#: Languages get their own column; everything else falls into `other`. Beyond
#: roughly this many, the tail columns are almost always zero and cost more in
#: overfitting than they return.
TOP_LANGUAGES = [
    "python", "javascript", "typescript", "java", "go", "c++", "c", "rust",
    "ruby", "php", "c#", "shell", "swift", "kotlin", "scala", "html", "css",
    "jupyter notebook", "dart", "r", "lua", "perl", "haskell", "elixir", "zig",
]

#: Licences that let you actually use the thing without legal review. Grouping
#: is deliberate: the model needs "can I ship this" not thirty sparse columns.
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "0bsd"}
COPYLEFT = {"gpl-3.0", "gpl-2.0", "agpl-3.0", "lgpl-3.0", "lgpl-2.1", "mpl-2.0", "epl-2.0"}

#: Embedding components kept. The raw 768 would dominate every tree split by
#: sheer count while carrying mostly noise; the leading components hold the
#: broad semantic axes, which is what the trees can actually use.
EMBEDDING_COMPONENTS = 48


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, cap: float = 1e6) -> np.ndarray:
    """Elementwise ratio that never returns NaN or inf.

    Ratios here divide by counts that are legitimately zero — a repository with
    no stars, an account created today. XGBoost tolerates NaN, but it treats it
    as a missing-value branch, which silently turns "this repo has no stars"
    into "we do not know this repo's stars". Those are different facts.
    """
    denominator = np.asarray(denominator, dtype=np.float64)
    numerator = np.asarray(numerator, dtype=np.float64)
    out = np.zeros_like(numerator, dtype=np.float64)
    ok = denominator > 0
    np.divide(numerator, denominator, out=out, where=ok)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=cap, neginf=-cap), -cap, cap)


def _days_since(values, now: datetime) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            # -1, not NaN: "never pushed" is information, and a distinct value
            # lets a tree split on it. NaN would route it to the missing branch
            # alongside genuinely unknown rows.
            out[i] = -1.0
            continue
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                out[i] = -1.0
                continue
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        out[i] = max(0.0, (now - v).total_seconds() / 86400.0)
    return out


@dataclass
class FeatureMatrix:
    values: np.ndarray            # (n, d) float32
    names: list[str]
    repo_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.repo_ids)

    @property
    def shape(self) -> tuple:
        return self.values.shape

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def validate(self) -> None:
        if self.values.shape != (len(self.repo_ids), len(self.names)):
            raise ValueError(
                f"matrix is {self.values.shape}, expected "
                f"({len(self.repo_ids)}, {len(self.names)})"
            )
        if not np.isfinite(self.values).all():
            bad = [self.names[j] for j in np.where(~np.isfinite(self.values).all(axis=0))[0]]
            raise ValueError(f"non-finite values in: {bad}")
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate feature names")


@dataclass
class GraphFeatures:
    """Per-repo graph signals, keyed by repo id. Missing means zero."""

    rank: dict = field(default_factory=dict)
    in_degree: dict = field(default_factory=dict)
    out_degree: dict = field(default_factory=dict)
    similar_degree: dict = field(default_factory=dict)


def build_features(
    rows: list[dict],
    *,
    graph: GraphFeatures | None = None,
    embeddings: np.ndarray | None = None,
    now: datetime | None = None,
) -> FeatureMatrix:
    """One row per repository, one column per signal.

    `rows` are dicts straight from `Database.world_rows`-shaped queries.
    `embeddings` is an (n, k) matrix already reduced to components and aligned
    with `rows` — reduction happens in `reduce_embeddings` so the same basis can
    be reused between training and inference.
    """
    now = now or datetime.now(timezone.utc)
    graph = graph or GraphFeatures()
    n = len(rows)
    if n == 0:
        raise ValueError("no rows")

    repo_ids = np.array([r["id"] for r in rows])
    get = lambda k, d=0: np.array([r.get(k) if r.get(k) is not None else d for r in rows])  # noqa: E731

    stars = get("stars").astype(np.float64)
    forks = get("forks").astype(np.float64)
    issues = get("open_issues").astype(np.float64)
    velocity = get("stars_90d").astype(np.float64)
    age = np.maximum(_days_since([r.get("created_at") for r in rows], now), 1.0)
    since_push = _days_since([r.get("pushed_at") for r in rows], now)

    columns: dict[str, np.ndarray] = {
        # --- activity ---------------------------------------------------
        "log_stars": np.log1p(np.maximum(stars, 0)),
        "log_forks": np.log1p(np.maximum(forks, 0)),
        "log_issues": np.log1p(np.maximum(issues, 0)),
        "log_velocity": np.log1p(np.maximum(velocity, 0)),
        "criticality": get("criticality", 0.0).astype(np.float64),
        "age_days": age,
        "days_since_push": since_push,
        # A project pushed to last week behaves differently from one dormant
        # since 2019, and this ratio says so independently of absolute age.
        "push_recency_ratio": _safe_ratio(since_push, age),
        "stars_per_day": _safe_ratio(stars, age),
        # What share of all-time stars arrived recently. Separates "alive" from
        # "was popular once", which raw star counts cannot.
        "velocity_share": _safe_ratio(velocity, np.maximum(stars, 1)),
        "fork_ratio": _safe_ratio(forks, np.maximum(stars, 1)),
        "issues_per_star": _safe_ratio(issues, np.maximum(stars, 1)),
        # --- shape ------------------------------------------------------
        "readme_chars": np.array([len(r.get("clean_text") or "") for r in rows], np.float64),
        "raw_readme_chars": np.array([len(r.get("readme_raw") or "") for r in rows], np.float64),
        # Near 1.0 means the README was almost entirely boilerplate or links —
        # the signature of an awesome-list or a badge-heavy template.
        "clean_reduction": get("clean_reduction", 0.0).astype(np.float64),
        "dropped_sections": np.array(
            [len(r.get("dropped_sections") or []) for r in rows], np.float64
        ),
        "topic_count": np.array([len(r.get("topics") or []) for r in rows], np.float64),
        "has_description": np.array(
            [1.0 if (r.get("description") or "").strip() else 0.0 for r in rows]
        ),
        "desc_chars": np.array([len(r.get("description") or "") for r in rows], np.float64),
        "has_license": np.array([1.0 if r.get("license") else 0.0 for r in rows]),
        "license_permissive": np.array(
            [1.0 if str(r.get("license") or "").lower() in PERMISSIVE else 0.0 for r in rows]
        ),
        "license_copyleft": np.array(
            [1.0 if str(r.get("license") or "").lower() in COPYLEFT else 0.0 for r in rows]
        ),
        "is_fork": np.array([1.0 if r.get("is_fork") else 0.0 for r in rows]),
        "is_archived": np.array([1.0 if r.get("is_archived") else 0.0 for r in rows]),
        "low_signal": np.array([1.0 if r.get("low_signal") else 0.0 for r in rows]),
        "non_english": np.array([1.0 if r.get("non_english") else 0.0 for r in rows]),
        "name_depth": np.array(
            [str(r.get("full_name") or "").count("-") for r in rows], np.float64
        ),
        # --- position -----------------------------------------------------
        # Being depended upon is a different fact from being starred, and it is
        # the strongest evidence available for canonicity.
        "log_pagerank": np.log1p(
            np.array([graph.rank.get(int(r["id"]), 0.0) for r in rows]) * 1e6
        ),
        "in_degree": np.array([graph.in_degree.get(int(r["id"]), 0) for r in rows], np.float64),
        "out_degree": np.array([graph.out_degree.get(int(r["id"]), 0) for r in rows], np.float64),
        "similar_degree": np.array(
            [graph.similar_degree.get(int(r["id"]), 0) for r in rows], np.float64
        ),
        "is_clustered": np.array(
            [1.0 if (r.get("cluster_id") is not None and r["cluster_id"] >= 0) else 0.0
             for r in rows]
        ),
    }
    columns["dependents_per_star"] = _safe_ratio(
        columns["in_degree"], np.maximum(stars, 1)
    )

    language = [str(r.get("language") or "").lower() for r in rows]
    for lang in TOP_LANGUAGES:
        columns[f"lang_{lang.replace(' ', '_').replace('+', 'p').replace('#', 'sharp')}"] = (
            np.array([1.0 if x == lang else 0.0 for x in language])
        )
    columns["lang_other"] = np.array(
        [1.0 if (x and x not in TOP_LANGUAGES) else 0.0 for x in language]
    )
    columns["lang_missing"] = np.array([1.0 if not x else 0.0 for x in language])

    names = list(columns.keys())
    matrix = np.column_stack([columns[k] for k in names]).astype(np.float32)

    if embeddings is not None:
        if len(embeddings) != n:
            raise ValueError(f"embeddings has {len(embeddings)} rows, expected {n}")
        matrix = np.hstack([matrix, np.asarray(embeddings, np.float32)])
        names += [f"emb_{i}" for i in range(embeddings.shape[1])]

    features = FeatureMatrix(values=matrix, names=names, repo_ids=repo_ids)
    features.validate()
    return features


def reduce_embeddings(
    vectors: np.ndarray,
    components: int = EMBEDDING_COMPONENTS,
    *,
    basis: tuple | None = None,
) -> tuple[np.ndarray, tuple]:
    """Project embeddings onto their leading principal components.

    Returns `(reduced, basis)`. **Pass the training basis back in at inference
    time.** Recomputing PCA on a different set of rows produces different axes,
    so component 3 would mean one thing during training and another during
    scoring — the model would be reading a column that silently changed meaning.

    Mean-centring here does double duty: it is required by PCA, and it removes
    the dominant shared direction that Gemini embeddings carry, which otherwise
    consumes the first component with a constant.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")

    if basis is not None:
        mean, axes = basis
        # Rule 7. A basis fitted on a different embedding dimension produces a
        # shape error here — but a basis fitted on the SAME dimension from a
        # different corpus does not, and that is the dangerous one: it projects
        # onto axes that mean something else, silently.
        if len(mean) != vectors.shape[1]:
            raise ValueError(
                f"basis was fitted on {len(mean)} dimensions, "
                f"input has {vectors.shape[1]}"
            )
        return ((vectors - mean) @ axes.T).astype(np.float32), basis

    mean = vectors.mean(axis=0)
    centred = vectors - mean
    k = min(components, min(centred.shape))
    # Randomised SVD would be faster at 1M rows; on a sample this size the exact
    # one is simpler and the difference is not worth a dependency.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = vt[:k]
    return (centred @ axes.T).astype(np.float32), (mean, axes)


def describe(features: FeatureMatrix, top: int = 12) -> str:
    """Human-readable sanity check. Read this before training on it."""
    lines = [f"{len(features):,} rows x {features.shape[1]} features"]
    variance = features.values.var(axis=0)
    dead = [features.names[j] for j in np.where(variance == 0)[0]]
    if dead:
        # A constant column cannot inform a split. Usually it means a field is
        # never populated — which is worth knowing before it silently does
        # nothing for the whole training run.
        lines.append(f"  {len(dead)} constant columns (no signal): {dead[:top]}")
    order = np.argsort(-variance)[:top]
    lines.append("  highest-variance columns:")
    for j in order:
        col = features.values[:, j]
        lines.append(
            f"    {features.names[j]:<24} mean={col.mean():>10.3f}  sd={col.std():>10.3f}"
        )
    return "\n".join(lines)
