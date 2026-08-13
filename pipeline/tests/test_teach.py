"""Tests for what the teaching stage decides before it spends money.

The LLM call is not the risky part — it retries, checkpoints every 200 rows,
and reports its own failures. The risky part is the row selection, because both
ways of getting it wrong are silent and both cost money:

**Redrawing on resume.** Sampling from the unrated remainder instead of
subtracting afterwards means every restart draws a fresh stratified sample from
a shrinking pool. An interrupted 4,000-row run then rates far more than 4,000
rows, and the strata drift further from the corpus each time. Nothing in the
output would show it.

**Leaking popularity.** The rows carry `stars` because the sampler stratifies
on it. If that field ever reaches the rendered prompt, the student learns to
predict popularity and the entire rubric is decoration.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.brain.rubric import (
    assert_no_popularity,
    build_teacher_prompt,
)
from gitglobe.brain.sampling import plan_teaching


def rows(n: int = 600) -> list[dict]:
    """A population with enough spread for the sampler to find strata in."""
    return [
        {
            "id": i, "full_name": f"o/r{i}", "description": "d", "language": "Go",
            "topics": [], "license": "MIT", "clean_text": "readme " * 50,
            "content_hash": f"h{i}",
            "stars": (i * 37) % 40_000,
            "domain": i % 12,
            "days_since_push": float((i * 13) % 900),
        }
        for i in range(n)
    ]


class TestPlanTeaching(unittest.TestCase):
    def test_picks_rows_when_nothing_is_rated(self) -> None:
        todo, sample = plan_teaching(rows(), set(), total=100)
        self.assertGreater(len(todo), 0)
        self.assertIsNotNone(sample)

    def test_respects_the_budget_when_the_budget_clears_the_floor(self) -> None:
        todo, sample = plan_teaching(rows(), set(), total=400)
        self.assertGreater(len(sample.strata), 0)
        self.assertLessEqual(len(todo), 400)

    def test_the_per_stratum_floor_can_exceed_a_small_budget(self) -> None:
        # Documented, not a defect: every non-empty cell gets at least `floor`,
        # because a cell the teacher never sees is a region the student guesses
        # about. So `total` is a target with a hard lower bound of
        # floor x non-empty strata, and a small budget cannot go below it.
        #
        # Safe only because the cost estimate is computed from the returned
        # list, never from `total` — if that ever inverts, the dry run starts
        # understating the bill and this test is the reminder why.
        todo, sample = plan_teaching(rows(), set(), total=100)
        self.assertGreater(len(todo), 100)
        # Exact, not a bound: a cell smaller than `floor` contributes its own
        # size, so floor x cells is an upper bound and asserting it as a lower
        # one fails. What must hold is that the rows drawn equal the rows
        # allocated — that is what makes the coverage report trustworthy.
        self.assertEqual(len(todo), sum(sample.allocation.values()))

    def test_production_budget_is_not_swamped_by_the_floor(self) -> None:
        # The real run uses --total 4000. If the floor ever dominated there,
        # the flag would stop meaning anything.
        _, sample = plan_teaching(rows(), set(), total=4_000)
        filled = sum(1 for v in sample.allocation.values() if v > 0)
        self.assertLess(filled * 3, 4_000)

    def test_already_rated_rows_are_subtracted(self) -> None:
        population = rows()
        todo, _ = plan_teaching(population, {r["id"] for r in population}, total=100)
        self.assertEqual(todo, [], "re-rated rows that already have labels")

    def test_resuming_never_grows_the_work(self) -> None:
        # THE property. Sample first, subtract second. If this inverted, a
        # partially-rated corpus would produce MORE work than a fresh one.
        population = rows()
        fresh, _ = plan_teaching(population, set(), total=100)
        partial = {r["id"] for r in population[:300]}
        resumed, _ = plan_teaching(population, partial, total=100)
        self.assertLessEqual(len(resumed), len(fresh))

    def test_resuming_returns_a_subset_of_the_original_sample(self) -> None:
        # Stronger than the count check: the rows left to do must be the SAME
        # rows, not a fresh draw that happens to be no larger.
        population = rows()
        fresh, _ = plan_teaching(population, set(), total=100)
        first_ids = {r["id"] for r in fresh}
        done = set(list(first_ids)[:10])
        resumed, _ = plan_teaching(population, done, total=100)
        self.assertTrue({r["id"] for r in resumed} <= first_ids)
        self.assertEqual(len(resumed), len(fresh) - len(done))

    def test_the_seed_makes_the_sample_reproducible(self) -> None:
        population = rows()
        a, _ = plan_teaching(population, set(), total=80, seed=7)
        b, _ = plan_teaching(population, set(), total=80, seed=7)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        population = rows()
        a, _ = plan_teaching(population, set(), total=80, seed=7)
        b, _ = plan_teaching(population, set(), total=80, seed=99)
        self.assertNotEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_an_empty_corpus_is_not_an_exception(self) -> None:
        todo, sample = plan_teaching([], set(), total=100)
        self.assertEqual(todo, [])
        self.assertIsNone(sample)

    def test_the_sample_is_not_just_the_most_popular(self) -> None:
        # Rating only the top of the corpus would teach the student that
        # popularity is the target, which is the one thing the rubric forbids.
        population = rows()
        todo, _ = plan_teaching(population, set(), total=100)
        top_by_stars = {r["id"] for r in sorted(
            population, key=lambda r: -r["stars"])[:100]}
        overlap = len({r["id"] for r in todo} & top_by_stars) / max(len(todo), 1)
        self.assertLess(overlap, 0.6, "sample is dominated by the most-starred rows")


class TestEstimate(unittest.TestCase):
    """The two numbers a user decides on before spending anything.

    Both were provider-blind and both were wrong for NIM: $3.19 quoted for a
    free tier, and 2.8 minutes quoted for a run that takes hours. An ETA off by
    two orders of magnitude is not cosmetic — it makes a healthy run look hung.
    """

    def config(self, provider: str):
        from gitglobe.brain.teacher import TeacherConfig
        return TeacherConfig(provider=provider, project="p")

    def test_the_free_tier_is_not_quoted_a_price(self) -> None:
        from gitglobe.brain.teacher import estimate
        result = estimate(4_000, 4_800, self.config("nim"))
        self.assertEqual(result["est_usd"], 0.0)
        self.assertFalse(result["billed"])

    def test_a_billed_provider_still_gets_a_price(self) -> None:
        from gitglobe.brain.teacher import estimate
        result = estimate(4_000, 4_800, self.config("vertex"))
        self.assertGreater(result["est_usd"], 0.0)
        self.assertTrue(result["billed"])

    def test_duration_respects_the_rate_limit(self) -> None:
        from gitglobe.brain.teacher import estimate
        config = self.config("nim")
        result = estimate(4_000, 4_800, config)
        self.assertAlmostEqual(result["est_minutes"], 4_000 / config.requests_per_minute,
                               places=1)
        # The specific regression: the old formula returned 2.8 for this input.
        self.assertGreater(result["est_minutes"], 60)

    def test_no_config_does_not_crash(self) -> None:
        # `estimate` is also useful ad hoc, and defaulting must not raise.
        from gitglobe.brain.teacher import estimate
        result = estimate(100, 3_000)
        self.assertGreater(result["est_minutes"], 0)


class TestBlindfoldSurvivesTheRowShape(unittest.TestCase):
    def test_the_rendered_prompt_carries_no_popularity_field(self) -> None:
        # rubric enforces this on the rendered text, but only for the row shape
        # actually passed. These rows carry `stars` for the sampler, so this is
        # the check that it does not reach the prompt.
        row = rows(1)[0]
        self.assertIn("stars", row, "fixture must carry stars for the sampler")
        assert_no_popularity(build_teacher_prompt(row))

    def test_the_guard_would_catch_a_leak(self) -> None:
        # A blindfold that cannot detect a leak is decoration.
        with self.assertRaises(Exception):
            assert_no_popularity("<repository><stars>40000</stars></repository>")


class TestQuerySuppliesEveryFeature(unittest.TestCase):
    """`brain_rows` must select every column `build_features` reads.

    Not hypothetical, and not once. `world_rows` omitted `criticality`, which
    silently discarded 84,434 backfilled values. Then `brain_rows` turned out to
    omit `forks`, `open_issues` and `clean_reduction` — so those three features,
    plus `fork_ratio` and `issues_per_star` derived from them, were constant
    zero for every repository. Five of 55 features doing nothing, with no error
    anywhere, because `row.get(k)` returns a neutral value for an absent key.

    Reads the source text rather than importing `Database`, which would pull
    asyncpg into a module that is otherwise pure arithmetic.
    """

    SRC = Path(__file__).resolve().parents[1] / "src" / "gitglobe"

    def select_list(self) -> str:
        """The SELECT list with SQL comments stripped.

        Stripping matters: the comment beside those columns names them in
        prose, so a guard that greps the raw text passes even when the columns
        are deleted. That is the third time in this codebase a check has been
        satisfied by a comment rather than by code — see also the manifest
        serialisation test and the world_rows guard.
        """
        source = (self.SRC / "db.py").read_text(encoding="utf-8")
        match = re.search(r"async def brain_rows.*?SELECT(.*?)FROM repo",
                          source, re.DOTALL)
        assert match, "brain_rows no longer contains a SELECT ... FROM repo"
        return re.sub(r"--[^\n]*", "", match.group(1))

    def feature_keys(self) -> set:
        source = (self.SRC / "brain" / "features.py").read_text(encoding="utf-8")
        return set(re.findall(r'\.get\("([a-z_]+)"', source))

    def test_no_feature_reads_a_column_the_query_omits(self) -> None:
        columns = self.select_list()
        missing = sorted(k for k in self.feature_keys() if k not in columns)
        self.assertEqual(
            missing, [],
            f"build_features reads {missing} but brain_rows does not select "
            f"them — each will be a constant-zero feature with no error raised",
        )

    def test_the_guard_can_actually_fail(self) -> None:
        # A check that cannot fail is decoration.
        damaged = self.select_list().replace("r.forks", "")
        self.assertNotIn("forks", damaged)


class TestBlindfoldFilter(unittest.TestCase):
    """Dropping popularity columns before the student ever sees them.

    `build_features` emits log_stars, log_forks, stars_per_day, log_pagerank
    and criticality on purpose — the globe sizes nodes with them and the global
    rank scores with them. Only the student must be denied them, so the filter
    belongs at the student's boundary, not in the feature builder.
    """

    NAMES = ["log_stars", "log_forks", "stars_per_day", "log_pagerank",
             "criticality", "readme_chars", "has_license", "in_degree"]

    def matrix(self):
        import numpy as np
        return np.arange(len(self.NAMES) * 5, dtype=float).reshape(5, len(self.NAMES))

    def test_drops_exactly_what_the_assert_would_reject(self) -> None:
        from gitglobe.brain.student import (
            assert_no_popularity_features,
            blindfold,
        )
        _, kept = blindfold(self.matrix(), self.NAMES)
        assert_no_popularity_features(kept)  # must not raise
        self.assertEqual(kept, ["readme_chars", "has_license", "in_degree"])

    def test_keeps_the_columns_aligned_with_their_names(self) -> None:
        # The silent failure: filter names but slice the wrong columns, and the
        # student trains on readme length while calling it license presence.
        from gitglobe.brain.student import blindfold
        values = self.matrix()
        kept_values, kept_names = blindfold(values, self.NAMES)
        for i, name in enumerate(kept_names):
            with self.subTest(name=name):
                original = values[:, self.NAMES.index(name)]
                self.assertTrue((kept_values[:, i] == original).all())

    def test_all_popularity_is_refused_rather_than_returning_nothing(self) -> None:
        from gitglobe.brain.student import blindfold
        import numpy as np
        with self.assertRaises(ValueError):
            blindfold(np.zeros((3, 2)), ["log_stars", "log_forks"])

    def test_a_clean_matrix_is_untouched(self) -> None:
        from gitglobe.brain.student import blindfold
        import numpy as np
        names = ["readme_chars", "has_license"]
        values = np.ones((3, 2))
        kept_values, kept_names = blindfold(values, names)
        self.assertEqual(kept_names, names)
        self.assertEqual(kept_values.shape, values.shape)


class TestNothingIsStoredWithoutEvidence(unittest.TestCase):
    """The gate `stage_learn` puts in front of 87,227 writes.

    A distilled model is only worth storing if it beats predicting the mean.
    Without this check a student that learned nothing still emits a confident
    number for every repository — all of them near-identical, which on a globe
    reads as "we measured this" rather than "we gave up".
    """

    def fit_on(self, signal: bool):
        import numpy as np

        from gitglobe.brain.student import fit

        rng = np.random.default_rng(3)
        n, d = 900, 6
        values = rng.normal(size=(n, d))
        names = [f"feature_{i}" for i in range(d)]
        target = (values[:, 0] * 18 + 50) if signal else rng.normal(50, 20, n)
        labels = {"maintenance": np.clip(target, 0, 100)}
        idx = np.arange(n)
        return fit(values, labels, names,
                   train_idx=idx[:700], test_idx=idx[700:], seed=5)

    def test_real_signal_is_kept(self) -> None:
        students = self.fit_on(signal=True)
        self.assertTrue(students["maintenance"].holdout["beats_baseline"])

    def test_pure_noise_is_rejected(self) -> None:
        # The specific trap: on noise the student still scores marginally
        # better than the baseline by chance, so a bare `rmse < baseline`
        # comparison is a coin flip. It must clear its own sampling noise.
        students = self.fit_on(signal=False)
        self.assertFalse(students["maintenance"].holdout["beats_baseline"])

    def test_the_stage_keeps_only_dimensions_that_pass(self) -> None:
        # Mirrors the filter in stage_learn. If this ever inverts, the run
        # stores predictions it has no evidence for.
        good = self.fit_on(signal=True)
        bad = self.fit_on(signal=False)
        both = {"good": good["maintenance"], "bad": bad["maintenance"]}
        honest = {k: s for k, s in both.items() if s.holdout["beats_baseline"]}
        self.assertEqual(sorted(honest), ["good"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
