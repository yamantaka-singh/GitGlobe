"""Apply the global scale to a whole corpus.

`calibrate.composite_score` ranks ONE repository. This turns a corpus into
signals and applies it to all of them — the part that was missing, which is why
`calibrate.py` and its tests existed for a long time without anything ever
calling them or writing a score anywhere.

Pure: rows in, ranks out, no database and no network, so the arithmetic that
decides what a user sees is testable without either.

Two signals have to be derived rather than read:

**`pagerank_ratio`** is PageRank as a multiple of the corpus mean. Raw PageRank
sums to 1, so every value shrinks as 1/n purely because the corpus grew — and a
score built on it would drift downward for every repository on every ingest,
which is indistinguishable from every repository getting worse. That matters
more now that new repositories are arriving continuously.

**`dependents`** is in-degree over `depends_on` edges only. Not total degree:
`similar_to` is manufactured by our own k-NN step, so counting it would let the
corpus vote on its own importance. A dependent is a fact about the world; a
nearest neighbour is a fact about our embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .calibrate import GlobalRank, RepoSignals, composite_score
from .global_scale import StarScale, Weights

log = logging.getLogger(__name__)

#: Edge kind for a real dependency, matching migration 002.
DEPENDS_ON = 0


@dataclass
class CorpusRanking:
    ranks: dict                  # repo_id -> GlobalRank
    pagerank_mean: float
    dependents_total: int
    scored: int

    def summary(self) -> str:
        if not self.ranks:
            return "no repositories ranked"
        scores = sorted((r.score for r in self.ranks.values()), reverse=True)
        return (
            f"{self.scored:,} ranked · best {scores[0]:.1f} · "
            f"median {scores[len(scores) // 2]:.1f} · worst {scores[-1]:.1f} · "
            f"{self.dependents_total:,} dependency edges"
        )


def dependent_counts(edges, *, kind: int = DEPENDS_ON) -> dict:
    """In-degree over dependency edges only, keyed by repo id.

    Only `dst` is counted: A depends on B makes B depended upon. Counting `src`
    too would reward a repository for having a long requirements file.
    """
    counts: dict = {}
    for row in edges:
        if len(row) < 4:
            raise ValueError(f"expected (src, dst, weight, kind) rows, got {len(row)} fields")
        if int(row[3]) != kind:
            continue
        dst = int(row[1])
        counts[dst] = counts.get(dst, 0) + 1
    return counts


def signals_for(row: dict, *, pagerank_mean: float, dependents: int) -> RepoSignals:
    """One repository's absolute signals.

    Missing values become their neutral element rather than raising: this runs
    over the whole corpus, and a null `criticality` — most of it, since OSSF
    only publishes scores for a subset — is normal, not exceptional.
    """
    if pagerank_mean <= 0:
        raise ValueError(f"pagerank_mean must be positive, got {pagerank_mean}")

    rank = float(row.get("rank") or 0.0)
    return RepoSignals(
        stars=float(row.get("stars") or 0),
        dependents=float(dependents),
        pagerank_ratio=(rank / pagerank_mean) if rank > 0 else 1.0,
        criticality=float(row.get("criticality") or 0.0),
        stars_90d=float(row.get("stars_90d") or 0),
    )


def rank_corpus(rows: list, edges, scale: StarScale, *, weights: Weights | None = None):
    """Score every repository in the corpus against the global scale."""
    if not rows:
        return CorpusRanking({}, 0.0, 0, 0)
    scale.validate()

    positive = [v for v in (float(r.get("rank") or 0.0) for r in rows) if v > 0]
    # If the rank stage has not run, every ratio is 1.0 and PageRank simply
    # contributes nothing — better than dividing by zero, and better than
    # refusing to score when four of five signals are available.
    pagerank_mean = (sum(positive) / len(positive)) if positive else 1.0
    if not positive:
        log.warning("No PageRank values — global scores will omit that component")

    dependents = dependent_counts(edges)
    out = {
        int(r["id"]): composite_score(
            signals_for(r, pagerank_mean=pagerank_mean,
                        dependents=dependents.get(int(r["id"]), 0)),
            scale, weights,
        )
        for r in rows
    }
    return CorpusRanking(out, pagerank_mean, sum(dependents.values()), len(out))


def leaderboard(ranking: CorpusRanking, rows: list, *, top: int = 10) -> list:
    """The highest-scoring repositories, which is the only view that shows
    whether the composite is sane.

    `disagreement` cannot answer this. It ranks by distance moved, so a tiny
    package with many dependents always tops it — it starts near-last by stars,
    so it has the most room to move. That table looks identical whether the
    score is good or broken.
    """
    name_of = {int(r["id"]): r.get("full_name", "?") for r in rows}
    best = sorted(ranking.ranks.items(), key=lambda kv: -kv[1].score)[:top]
    return [
        {"full_name": name_of.get(rid, "?"), "score": rank.score,
         "components": rank.components}
        for rid, rank in best
    ]


def disagreement(ranking: CorpusRanking, rows: list, *, top: int = 10) -> list:
    """Where the global score most disagrees with a pure star ranking.

    The honest test of whether the composite earned its keep: a score that never
    disagrees with stars IS stars.
    """
    star_position = {int(r["id"]): i
                     for i, r in enumerate(sorted(rows, key=lambda r: -(r.get("stars") or 0)))}
    score_position = {rid: i for i, (rid, _) in
                      enumerate(sorted(ranking.ranks.items(), key=lambda kv: -kv[1].score))}
    name_of = {int(r["id"]): r.get("full_name", "?") for r in rows}

    moves = sorted(
        ((star_position[rid] - score_position[rid], rid)
         for rid in score_position if rid in star_position),
        key=lambda m: -abs(m[0]),
    )
    return [
        {"full_name": name_of.get(rid, "?"), "places": delta,
         "score": ranking.ranks[rid].score, "star_rank": ranking.ranks[rid].star_rank}
        for delta, rid in moves[:top]
    ]
