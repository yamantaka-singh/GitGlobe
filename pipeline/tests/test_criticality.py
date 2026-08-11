"""Tests for the criticality backfill.

The failure mode this guards against is not a crash. It is attaching the wrong
score to the right repository — a GitLab path matched against a GitHub name, a
column that moved, a case mismatch that silently drops half the corpus. All of
those produce a plausible-looking globe and no exception, which is exactly the
class of bug this project keeps finding the hard way.
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.rank.criticality import (  # noqa: E402
    DUMP_MEASURED_AT,
    DUMP_URL,
    match_corpus,
    parse_criticality,
    repo_from_url,
)

HEADER = "repo.url,repo.language,default_score"


def dump(*rows: str):
    return io.StringIO("\n".join([HEADER, *rows]))


class TestRepoFromUrl(unittest.TestCase):
    def test_plain_github_url(self) -> None:
        self.assertEqual(
            repo_from_url("https://github.com/kubernetes/kubernetes"),
            "kubernetes/kubernetes",
        )

    def test_trailing_slash_and_whitespace(self) -> None:
        self.assertEqual(repo_from_url("  https://github.com/a/b/  "), "a/b")

    def test_non_github_forge_is_refused(self) -> None:
        # The dump says it intends to add other forges. A GitLab `a/b` matched
        # against a GitHub `a/b` would attach one project's score to another.
        self.assertIsNone(repo_from_url("https://gitlab.com/a/b"))
        self.assertIsNone(repo_from_url("https://bitbucket.org/a/b"))

    def test_deeper_paths_are_refused(self) -> None:
        # Not a repository root — a tree or blob URL names a file, not a repo.
        self.assertIsNone(repo_from_url("https://github.com/a/b/tree/main"))

    def test_incomplete_paths_are_refused(self) -> None:
        for url in ("https://github.com/onlyowner", "https://github.com/", "", "  "):
            with self.subTest(url=url):
                self.assertIsNone(repo_from_url(url))


class TestParse(unittest.TestCase):
    def test_reads_url_and_score(self) -> None:
        scores = parse_criticality(dump(
            "https://github.com/a/B,Go,0.5",
            "https://github.com/c/d,Rust,0.25",
        ))
        self.assertEqual(scores, {"a/b": 0.5, "c/d": 0.25})

    def test_keys_are_lowercased_for_joining(self) -> None:
        scores = parse_criticality(dump("https://github.com/FaceBook/ReAct,JS,0.9"))
        self.assertIn("facebook/react", scores)

    def test_a_missing_column_is_fatal_not_empty(self) -> None:
        # Returning {} here would look like "no matches" and quietly leave
        # criticality NULL for the whole corpus.
        with self.assertRaises(ValueError):
            parse_criticality(io.StringIO("url,score\nhttps://github.com/a/b,0.5\n"))

    def test_a_score_outside_zero_to_one_is_fatal(self) -> None:
        # If the column ever moves, the neighbouring values are counts in the
        # thousands. Clamping those to 1.0 would mark everything maximally
        # critical and look entirely plausible on the globe.
        with self.assertRaises(ValueError):
            parse_criticality(dump("https://github.com/a/b,Go,79583"))

    def test_unparseable_rows_are_skipped_not_fatal(self) -> None:
        scores = parse_criticality(dump(
            "https://github.com/a/b,Go,0.5",
            "https://gitlab.com/x/y,Go,0.7",
            "https://github.com/c/d,Go,",
            "https://github.com/e/f,Go,notanumber",
        ))
        self.assertEqual(scores, {"a/b": 0.5})

    def test_empty_dump(self) -> None:
        self.assertEqual(parse_criticality(dump()), {})


class TestMatchCorpus(unittest.TestCase):
    def test_joins_case_insensitively_onto_stored_spelling(self) -> None:
        scores = {"facebook/react": 0.9, "torvalds/linux": 0.99}
        case_map = {"facebook/react": "facebook/React", "torvalds/linux": "torvalds/linux"}
        self.assertEqual(
            match_corpus(scores, case_map),
            {"facebook/React": 0.9, "torvalds/linux": 0.99},
        )

    def test_unheld_repositories_are_dropped(self) -> None:
        # The dump scores far more projects than we ingest. Those are not
        # errors, and writing them would fail the UPDATE for no reason.
        matched = match_corpus({"a/b": 0.5, "c/d": 0.5}, {"a/b": "a/b"})
        self.assertEqual(matched, {"a/b": 0.5})

    def test_no_overlap_is_empty_not_an_exception(self) -> None:
        self.assertEqual(match_corpus({"a/b": 0.5}, {"c/d": "c/d"}), {})


class TestProvenance(unittest.TestCase):
    def test_the_dump_date_appears_in_the_url(self) -> None:
        # The date is the whole reason this constant is a full URL rather than
        # assembled from parts. If they drift apart, the printed provenance is
        # a lie and nobody would notice.
        self.assertIn(DUMP_MEASURED_AT.replace("-", "."), DUMP_URL)

    def test_the_dump_is_served_over_https(self) -> None:
        self.assertTrue(DUMP_URL.startswith("https://"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
