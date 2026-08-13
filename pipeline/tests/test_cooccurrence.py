"""Tests for the co-occurrence relatedness layer.

The maths is where this goes subtly wrong, and the failure mode is not a crash —
it is a graph that looks plausible and connects the wrong things. So the tests
are built around scenarios with a known right answer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.graph.cooccurrence import (
    basket_weight,
    co_occurrence,
    mutual_top_k,
    pairs_from_basket,
    ppmi,
    top_k_per_item,
)


def score_of(pairs, a: str, b: str) -> float:
    key = tuple(sorted((a, b)))
    for p in pairs:
        if tuple(sorted((p.a, p.b))) == key:
            return p.ppmi
    return 0.0


class TestPopularityIsDividedOut(unittest.TestCase):
    """The reason PMI exists rather than raw counts.

    `react` co-occurs with everything, so raw co-occurrence ranks it first for
    every query. PMI asks whether two things appear together more than their
    individual popularity predicts.
    """

    def setUp(self) -> None:
        # `react` is in every basket. `langchain` and `chromadb` appear together
        # in a small, consistent subset — the genuine association.
        baskets = []
        for _ in range(60):
            baskets.append(["react", "webpack", "eslint"])
        for _ in range(12):
            baskets.append(["react", "langchain", "chromadb"])
        self.pairs = ppmi(*co_occurrence(baskets), min_pair_count=2)

    def test_genuine_association_beats_the_popular_one(self) -> None:
        genuine = score_of(self.pairs, "langchain", "chromadb")
        popular = score_of(self.pairs, "react", "langchain")
        self.assertGreater(genuine, popular,
                           "PMI failed to divide out popularity — raw counts would do this")

    def test_the_ubiquitous_item_scores_low_against_everything(self) -> None:
        for other in ("langchain", "chromadb", "webpack"):
            with self.subTest(other=other):
                self.assertLess(score_of(self.pairs, "react", other),
                                score_of(self.pairs, "langchain", "chromadb"))


class TestBasketWeighting(unittest.TestCase):
    def test_small_baskets_count_for_more(self) -> None:
        # Eight stars is a statement about each one. Three hundred is browsing.
        self.assertGreater(basket_weight(8), basket_weight(300))

    def test_weighting_decays_gently(self) -> None:
        # 1/n would make anything but a tiny basket worthless, and tiny baskets
        # are the noisiest. 1/log keeps large baskets contributing.
        self.assertGreater(basket_weight(100) / basket_weight(10), 0.4)

    def test_oversized_baskets_are_dropped_entirely(self) -> None:
        # A 5,000-star account would contribute 12.5M pairs by itself.
        huge = [f"repo{i}" for i in range(5000)]
        pair_counts, _, total = co_occurrence([huge], max_basket=400)
        self.assertEqual(len(pair_counts), 0)
        self.assertEqual(total, 0.0)

    def test_single_item_baskets_are_ignored(self) -> None:
        pair_counts, _, _ = co_occurrence([["solo"], ["also-solo"]])
        self.assertEqual(len(pair_counts), 0)


class TestPairGeneration(unittest.TestCase):
    def test_pairs_are_unordered_and_deduplicated(self) -> None:
        pairs = list(pairs_from_basket(["b", "a", "b", "c"]))
        self.assertEqual(pairs, [("a", "b"), ("a", "c"), ("b", "c")])

    def test_a_repeated_star_does_not_inflate_a_pair(self) -> None:
        # GH Archive can contain the same (user, repo) more than once.
        single, _, _ = co_occurrence([["a", "b"]])
        doubled, _, _ = co_occurrence([["a", "b", "a", "b"]])
        self.assertEqual(single[("a", "b")], doubled[("a", "b")])


class TestNoiseSuppression(unittest.TestCase):
    def test_a_pair_seen_once_is_discarded(self) -> None:
        """Two obscure repos sharing one user produce a spectacular PMI.

        Rarity inflates PMI — that is its known weakness. A minimum count is
        what stops the top of the results being dominated by coincidences.
        """
        baskets = [["obscure-a", "obscure-b"]] + [["x", "y"] for _ in range(50)]
        pairs = ppmi(*co_occurrence(baskets), min_pair_count=3)
        self.assertEqual(score_of(pairs, "obscure-a", "obscure-b"), 0.0)

    def test_smoothing_damps_the_rare_item_bias(self) -> None:
        baskets = [["common1", "common2"] for _ in range(40)]
        baskets += [["rare1", "rare2"] for _ in range(4)]
        unsmoothed = ppmi(*co_occurrence(baskets), min_pair_count=3, smoothing=1.0)
        smoothed = ppmi(*co_occurrence(baskets), min_pair_count=3, smoothing=0.75)
        self.assertLess(score_of(smoothed, "rare1", "rare2"),
                        score_of(unsmoothed, "rare1", "rare2"))

    def test_negative_pmi_is_not_returned(self) -> None:
        # "Co-occur less than chance" is real information, but not an arc.
        baskets = [["a", "b"] for _ in range(20)] + [["a", "c"] for _ in range(20)]
        baskets += [["b", "d"] for _ in range(20)] + [["c", "d"] for _ in range(20)]
        self.assertTrue(all(p.ppmi > 0 for p in ppmi(*co_occurrence(baskets), min_pair_count=2)))


class TestTopK(unittest.TestCase):
    def test_every_node_gets_the_same_budget(self) -> None:
        """A global threshold gives hubs a hairball and the tail nothing."""
        baskets = [["hub", f"leaf{i}"] * 1 for i in range(40) for _ in range(5)]
        pairs = ppmi(*co_occurrence(baskets), min_pair_count=2)
        top = top_k_per_item(pairs, k=5)
        self.assertLessEqual(len(top.get("hub", [])), 5)

    def test_mutual_filtering_removes_the_star_graph(self) -> None:
        """Asymmetric edges are popularity leaking back in.

        Every small React component library counts `react` among its strongest
        associations; `react` counts none of them. Requiring mutuality removes
        that whole class of edge, and it is what turns a star into a structure.
        """
        baskets = []
        for i in range(30):
            baskets += [["react", f"tiny-lib-{i}"] for _ in range(4)]
        # One genuine mutual relationship, isolated from react.
        baskets += [["vite", "vitest"] for _ in range(20)]

        pairs = ppmi(*co_occurrence(baskets), min_pair_count=3)
        mutual = mutual_top_k(pairs, k=3)
        names = {tuple(sorted((p.a, p.b))) for p in mutual}

        self.assertIn(("vite", "vitest"), names)
        react_edges = [n for n in names if "react" in n]
        self.assertLessEqual(len(react_edges), 3,
                             f"react kept {len(react_edges)} edges; mutual filtering failed")


class TestDegenerateInput(unittest.TestCase):
    """This runs over millions of GH Archive rows. Nothing may raise."""

    def test_empty_input(self) -> None:
        self.assertEqual(ppmi(*co_occurrence([])), [])

    def test_all_baskets_filtered_out(self) -> None:
        self.assertEqual(ppmi(*co_occurrence([["only-one"]])), [])

    def test_zero_total_weight_is_handled(self) -> None:
        from collections import Counter
        self.assertEqual(ppmi(Counter(), Counter(), 0.0), [])

    def test_results_are_deterministic(self) -> None:
        baskets = [["a", "b", "c"] for _ in range(10)] + [["b", "c", "d"] for _ in range(8)]
        first = [(p.a, p.b, round(p.ppmi, 9)) for p in ppmi(*co_occurrence(baskets), min_pair_count=2)]
        second = [(p.a, p.b, round(p.ppmi, 9)) for p in ppmi(*co_occurrence(baskets), min_pair_count=2)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
