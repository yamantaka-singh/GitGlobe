"""Tests for applying the global scale across a whole corpus.

The property that matters most is `test_scores_do_not_drift_when_the_corpus_grows`.
Raw PageRank sums to 1, so every value shrinks as 1/n as repositories are added.
A score built on the raw number would fall for every repository on every ingest
— which looks exactly like every repository getting worse, and would be very
hard to notice because it moves everything together. New repos arriving
continuously makes that a live risk, not a theoretical one.
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from gitglobe.rank.corpus import (  # noqa: E402
    dependent_counts,
    disagreement,
    leaderboard,
    rank_corpus,
    signals_for,
)
from gitglobe.rank.global_scale import StarScale  # noqa: E402

#: Every column `signals_for` reads off a row. Kept here rather than derived so
#: that adding a signal forces a deliberate edit in both places.
SIGNAL_COLUMNS = ("stars", "rank", "criticality", "stars_90d")


def scale():
    """A plausible measured survival function, descending as it must."""
    return StarScale(
        thresholds=[0, 1, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000],
        counts=[420_000_000, 90_000_000, 6_000_000, 900_000, 420_000,
                58_000, 26_000, 3_400, 1_300, 90],
        total_repos=420_000_000,
        measured_at="2026-08-10",
    )


def corpus(n=6, rank_scale=1.0):
    return [{
        "id": i + 1,
        "full_name": f"org/repo{i}",
        "stars": 10 ** (i % 5),
        "stars_90d": 5 * i,
        "criticality": 0.1 * (i % 5),
        # Sums to rank_scale, mimicking PageRank's normalisation.
        "rank": rank_scale * (i + 1) / sum(range(1, n + 1)),
    } for i in range(n)]


def edges(pairs, kind=0):
    return [(s, d, 1.0, kind) for s, d in pairs]


class TestDependentCounts(unittest.TestCase):
    def test_counts_only_the_depended_upon_end(self) -> None:
        # A depends on B and C. B and C gain a dependent; A gains nothing.
        self.assertEqual(dependent_counts(edges([(1, 2), (1, 3)])), {2: 1, 3: 1})

    def test_ignores_similarity_edges(self) -> None:
        # similar_to is manufactured by our own k-NN step. Counting it would let
        # the corpus vote on its own importance.
        mixed = edges([(1, 2)], kind=0) + edges([(3, 2), (4, 2)], kind=1)
        self.assertEqual(dependent_counts(mixed), {2: 1})

    def test_malformed_rows_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            dependent_counts([(1, 2, 1.0)])

    def test_empty(self) -> None:
        self.assertEqual(dependent_counts([]), {})


class TestQueryCoversTheSignals(unittest.TestCase):
    """The query that feeds the score must select everything the score reads.

    This is not hypothetical. `world_rows` once omitted `criticality` and
    `stars_90d`; `signals_for` turns an absent key into its neutral value by
    design, so 84,434 backfilled criticality scores were discarded on read and
    `calibrate` returned byte-identical output. No exception, no warning, and
    the symptom looked like bad source data rather than a missing column.

    Checking the SQL text is crude, but it is the only thing that couples the
    two ends without a live database, and the failure it prevents is silent.

    Read from the file rather than by importing `Database`, which would drag
    asyncpg into a module that is otherwise pure arithmetic — and match only
    the SELECT list, because the first version of this test searched the whole
    function including its docstring, where every column name appears in prose.
    It passed against a query with the columns deleted.
    """

    def select_list(self) -> str:
        source = (SRC / "gitglobe" / "db.py").read_text(encoding="utf-8")
        match = re.search(r"async def world_rows.*?SELECT(.*?)FROM repo",
                          source, re.DOTALL)
        assert match, "world_rows no longer contains a SELECT ... FROM repo"
        return match.group(1)

    def test_every_signal_column_is_selected(self) -> None:
        columns = self.select_list()
        for column in SIGNAL_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(
                    column, columns,
                    f"signals_for reads {column!r} but world_rows does not select it; "
                    f"it will silently score as its neutral value for every repository",
                )

    def test_the_guard_can_actually_fail(self) -> None:
        # A check that cannot fail is decoration. Prove it notices a deletion.
        damaged = self.select_list().replace("r.criticality", "")
        self.assertNotIn("criticality", damaged)

    def test_signals_for_reads_exactly_the_documented_columns(self) -> None:
        # If someone adds a signal to signals_for without adding it here, the
        # test above cannot know to check for it. This catches that drift.
        source = inspect.getsource(signals_for)
        for column in SIGNAL_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(f'"{column}"', source)


class TestSignals(unittest.TestCase):
    def test_nulls_become_neutral_not_an_exception(self) -> None:
        s = signals_for({"id": 1}, pagerank_mean=0.01, dependents=0)
        self.assertEqual((s.stars, s.criticality, s.pagerank_ratio), (0.0, 0.0, 1.0))

    def test_ratio_is_relative_to_the_mean(self) -> None:
        s = signals_for({"id": 1, "rank": 0.05}, pagerank_mean=0.01, dependents=0)
        self.assertAlmostEqual(s.pagerank_ratio, 5.0)

    def test_zero_mean_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            signals_for({"id": 1}, pagerank_mean=0.0, dependents=0)


class TestRankCorpus(unittest.TestCase):
    def test_every_repository_gets_a_score_in_range(self) -> None:
        result = rank_corpus(corpus(), edges([(1, 2)]), scale())
        self.assertEqual(result.scored, 6)
        for rank in result.ranks.values():
            self.assertGreaterEqual(rank.score, 0.0)
            self.assertLessEqual(rank.score, 100.0)

    def test_scores_do_not_drift_when_the_corpus_grows(self) -> None:
        # THE test. Same repositories, same relative importance, PageRank
        # renormalised as if the corpus had grown tenfold. Scores must not move.
        a = rank_corpus(corpus(rank_scale=1.0), edges([(1, 2)]), scale())
        b = rank_corpus(corpus(rank_scale=0.1), edges([(1, 2)]), scale())
        for repo_id in a.ranks:
            self.assertAlmostEqual(
                a.ranks[repo_id].score, b.ranks[repo_id].score, places=4,
                msg=f"repo {repo_id} drifted purely because the corpus grew",
            )

    def test_more_dependents_raises_the_score(self) -> None:
        rows = corpus()
        bare = rank_corpus(rows, [], scale())
        linked = rank_corpus(rows, edges([(i, 2) for i in range(3, 6)]), scale())
        self.assertGreater(linked.ranks[2].score, bare.ranks[2].score)

    def test_runs_without_pagerank(self) -> None:
        rows = [dict(r, rank=0.0) for r in corpus()]
        self.assertEqual(rank_corpus(rows, edges([(1, 2)]), scale()).scored, len(rows))

    def test_empty_corpus_is_not_an_error(self) -> None:
        result = rank_corpus([], [], scale())
        self.assertEqual(result.scored, 0)
        self.assertIn("no repositories", result.summary())

    def test_a_broken_scale_is_refused(self) -> None:
        # A survival function that rises means a rung was rate-limited, and
        # every percentile computed from it would be quietly wrong.
        bad = StarScale(thresholds=[0, 10, 100], counts=[100, 500, 10])
        with self.assertRaises(ValueError):
            rank_corpus(corpus(), [], bad)

    def test_stars_alone_do_not_decide_the_order(self) -> None:
        # The whole point. The least-starred repo with many dependents and high
        # criticality must outrank a more-starred repo with neither.
        rows = [
            {"id": 1, "full_name": "a", "stars": 50_000, "criticality": 0.0, "rank": 0.001},
            {"id": 2, "full_name": "b", "stars": 40, "criticality": 0.95, "rank": 0.02},
        ]
        result = rank_corpus(rows, edges([(i, 2) for i in range(10, 400)]), scale())
        self.assertGreater(result.ranks[2].score, result.ranks[1].score)


class TestDisagreement(unittest.TestCase):
    def test_reports_the_biggest_movers(self) -> None:
        rows = [
            {"id": 1, "full_name": "popular/thing", "stars": 50_000, "rank": 0.001},
            {"id": 2, "full_name": "quiet/library", "stars": 30, "criticality": 0.9, "rank": 0.02},
        ]
        result = rank_corpus(rows, edges([(i, 2) for i in range(10, 400)]), scale())
        moves = disagreement(result, rows, top=2)
        self.assertTrue(any(m["full_name"] == "quiet/library" and m["places"] > 0 for m in moves))

    def test_no_disagreement_when_score_tracks_stars(self) -> None:
        # If the composite ever collapses onto stars, every delta is zero and
        # this is how we would find out.
        rows = corpus()
        moves = disagreement(rank_corpus(rows, [], scale()), rows, top=len(rows))
        self.assertTrue(any(m["places"] != 0 for m in moves))


class TestLeaderboard(unittest.TestCase):
    def test_orders_by_score_descending(self) -> None:
        rows = corpus()
        board = leaderboard(rank_corpus(rows, [], scale()), rows, top=len(rows))
        scores = [e["score"] for e in board]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_carries_components_so_a_bad_top_can_be_diagnosed(self) -> None:
        # Seeing that a leaf package is #1 is only half the answer; the point
        # is being able to read WHICH signal put it there without a second run.
        rows = corpus()
        board = leaderboard(rank_corpus(rows, [], scale()), rows, top=1)
        self.assertIn("dependents", board[0]["components"])

    def test_answers_a_question_disagreement_cannot(self) -> None:
        # A high-dependent, low-star package tops the movers table by
        # construction — it starts near-last by stars, so it has the most room
        # to move. That says nothing about whether it is genuinely the best.
        # Filler matters: with only two rows the deltas are +1 and -1, a tie,
        # and the table cannot express "moved further" at all.
        rows = [
            {"id": 1, "full_name": "big/framework", "stars": 200_000, "rank": 0.02,
             "criticality": 0.95},
            {"id": 2, "full_name": "tiny/leaf", "stars": 40, "rank": 0.001},
            *({"id": 100 + i, "full_name": f"mid/repo{i}", "stars": 5_000 - i * 100,
               "rank": 0.001} for i in range(20)),
        ]
        result = rank_corpus(rows, edges([(i, 2) for i in range(1000, 1300)]), scale())
        top_mover = disagreement(result, rows, top=1)[0]["full_name"]
        self.assertEqual(top_mover, "tiny/leaf")
        self.assertEqual(leaderboard(result, rows, top=1)[0]["full_name"], "big/framework")

    def test_empty_ranking(self) -> None:
        self.assertEqual(leaderboard(rank_corpus([], [], scale()), []), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
