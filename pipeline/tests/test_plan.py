"""Search-plan tests.

Pure logic, so these run without httpx, a token, or a network — which is the
point of keeping the planner separate from the client.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.ingest.plan import (  # noqa: E402
    EXPECTED_YIELD,
    LANGUAGE_SHARDS,
    SEARCH_RESULT_CAP,
    SearchPlan,
    parse_shard,
    plan_for_target,
    star_shards,
)


class TestStarShards(unittest.TestCase):
    def test_bands_are_contiguous_with_no_gaps_or_overlaps(self) -> None:
        """A gap silently drops every repo in it; an overlap double-counts.

        Neither is visible in the output — you just quietly get the wrong
        corpus — which is exactly why it is asserted here.
        """
        shards = star_shards()
        bounded = [parse_shard(s) for s in shards[:-1]]
        for (lo, hi), (next_lo, _) in zip(bounded, bounded[1:]):
            self.assertLess(lo, hi, f"band {lo}..{hi} is inverted")
            self.assertEqual(hi + 1, next_lo, f"gap or overlap at {hi} -> {next_lo}")

    def test_top_band_is_open_ended(self) -> None:
        low, high = parse_shard(star_shards()[-1])
        self.assertIsNone(high)
        self.assertEqual(low, 400_000)

    def test_the_open_band_continues_from_the_last_bounded_one(self) -> None:
        shards = star_shards()
        _, last_high = parse_shard(shards[-2])
        open_low, _ = parse_shard(shards[-1])
        self.assertEqual(last_high + 1, open_low)

    def test_stars_alone_cannot_reach_100k(self) -> None:
        """Documents the ceiling that forced the second axis.

        Geometric star bands top out around 30 shards. If this ever starts
        passing, the band maths changed and `plan_for_target` should be
        revisited — it may no longer need the language axis at all.
        """
        self.assertFalse(SearchPlan().covers(100_000))

    def test_bands_widen_with_star_count(self) -> None:
        # Repository counts fall off roughly as a power law, so equal-width
        # bands would put millions of repos in the bottom one and three in the
        # top. Geometric widening is what keeps each under the cap.
        #
        # Monotonicity is checked with a tolerance: integer truncation in the
        # band maths makes an occasional band one star narrower than the last,
        # which is harmless and not worth complicating the generator for.
        widths = [hi - lo for lo, hi in (parse_shard(s) for s in star_shards()[:-1])]
        self.assertLess(widths[0], widths[-1])
        self.assertGreater(widths[-1] / max(1, widths[0]), 100)
        regressions = sum(1 for a, b in zip(widths, widths[1:]) if b < a)
        self.assertLessEqual(regressions, 1, f"widths not monotonic: {widths}")

    def test_rejects_nonsense_parameters(self) -> None:
        with self.assertRaises(ValueError):
            star_shards(low=100, high=50)
        with self.assertRaises(ValueError):
            star_shards(step_factor=1.0)

    def test_parse_shard_rejects_junk(self) -> None:
        with self.assertRaises(ValueError):
            parse_shard("stars:lots")


class TestPlanForTarget(unittest.TestCase):
    def test_small_targets_skip_the_language_axis(self) -> None:
        # Sharding by language multiplies the query count by 26, and each query
        # is a round trip. A 5,000-repo proof run should not pay for that.
        plan = plan_for_target(5_000)
        self.assertEqual(plan.languages, ())
        self.assertTrue(plan.covers(5_000))

    def test_100k_gets_the_language_axis(self) -> None:
        plan = plan_for_target(100_000)
        self.assertEqual(plan.languages, LANGUAGE_SHARDS)
        self.assertTrue(plan.covers(100_000),
                        f"only {plan.reachable:,} reachable")

    def test_language_sharded_plans_still_catch_unlabelled_repos(self) -> None:
        """Every `language:` query excludes repos with no detected language.

        Those are a large and genuinely distinct group — docs, dotfiles,
        awesome-lists, datasets — so each star band also gets an unfiltered
        query. Without it they would be invisible to the entire ingest.
        """
        plan = plan_for_target(100_000)
        unfiltered = [q for q in plan.queries() if "language:" not in q]
        self.assertEqual(len(unfiltered), len(plan.star_bands))

    def test_impossible_targets_fail_loudly(self) -> None:
        # Better to refuse than to quietly return a third of what was asked for.
        with self.assertRaises(ValueError) as ctx:
            plan_for_target(5_000_000)
        self.assertIn("third sharding axis", str(ctx.exception))

    def test_reachable_uses_expected_yield_not_the_raw_cap(self) -> None:
        plan = SearchPlan(star_bands=["stars:1..2"])
        self.assertEqual(plan.reachable, EXPECTED_YIELD)
        self.assertLess(EXPECTED_YIELD, SEARCH_RESULT_CAP)


class TestSearchPlan(unittest.TestCase):
    def test_queries_run_most_important_first(self) -> None:
        # A run cut short must still hold the repos that matter.
        first = SearchPlan().queries()[0]
        self.assertIn(">=400000", first.replace(" ", ""))

    def test_every_query_excludes_archived_repos(self) -> None:
        for query in SearchPlan().queries():
            self.assertIn("archived:false", query)
            self.assertIn("is:public", query)

    def test_one_query_per_band_without_the_language_axis(self) -> None:
        plan = SearchPlan()
        self.assertEqual(len(plan.queries()), len(plan.star_bands))

    def test_cap_is_what_we_think_it_is(self) -> None:
        self.assertEqual(SEARCH_RESULT_CAP, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
