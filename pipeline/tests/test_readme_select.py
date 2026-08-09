"""Tests for README selection.

Every case here is taken from a repository that actually went wrong in the first
100k ingest, with the payload shaped exactly as GitHub's GraphQL returns it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.ingest.readme_select import (  # noqa: E402
    SYMLINK_MODE,
    looks_like_symlink_content,
    readme_names_in_tree,
    select_readme,
)


def node(*, blobs: dict[str, str] | None = None, entries: list[dict] | None = None) -> dict:
    payload: dict = {}
    for alias, text in (blobs or {}).items():
        payload[alias] = {"text": text}
    if entries is not None:
        payload["root_tree"] = {"entries": entries}
    return payload


def blob_entry(name: str, mode: int = 0o100644) -> dict:
    return {"name": name, "type": "blob", "mode": mode}


class TestSymlinkRejection(unittest.TestCase):
    """The bug that silently poisoned the first run.

    Git stores a symlink as a blob whose content IS the target path. certbot's
    README.rst is a symlink, so GraphQL returned the 18-character string
    "certbot/README.rst" as its README — short, non-empty, and passing every
    downstream emptiness check.
    """

    def test_certbot_symlink_is_rejected(self) -> None:
        result = select_readme(node(
            blobs={"readme_rst": "certbot/README.rst"},
            entries=[blob_entry("README.rst", mode=SYMLINK_MODE)],
        ))
        self.assertEqual(result.text, "")
        self.assertNotIn("certbot/README.rst", result.text)

    def test_symlink_detected_without_a_tree(self) -> None:
        # Belt and braces: if the tree is missing we can still spot a blob whose
        # entire content is a bare path.
        result = select_readme(node(blobs={"readme_rst": "certbot/README.rst"}))
        self.assertEqual(result.text, "")

    def test_real_readme_beside_a_symlink_stub_still_wins(self) -> None:
        # A repo can carry a stub README.md symlinking to the real README.rst.
        # First-match-wins picks the stub; longest-non-symlink-wins does not.
        result = select_readme(node(
            blobs={
                "readme_md": "docs/README.rst",
                "readme_rst": "# Real Project\n\nDoes a genuinely useful thing.\n",
            },
            entries=[blob_entry("README.md", mode=SYMLINK_MODE), blob_entry("README.rst")],
        ))
        self.assertIn("genuinely useful thing", result.text)
        self.assertEqual(result.path, "README.rst")

    def test_prose_that_happens_to_be_short_is_not_mistaken_for_a_symlink(self) -> None:
        for text in ("# tool\n\nA thing.", "Fast JSON parser", "hello world"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_symlink_content(text))

    def test_path_shaped_content_is_flagged(self) -> None:
        for text in ("certbot/README.rst", "../docs/README.md", "packages/core/README.md"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_symlink_content(text))


class TestSymlinksAreFollowed(unittest.TestCase):
    """A symlink's content is its target path — which is what to fetch next.

    Monorepos rely on this constantly: zod, vuetify, unocss and certbot all
    point their root README at a package subdirectory. Rejecting the symlink
    loses a good repository; following it recovers one.
    """

    def test_symlink_target_becomes_a_backfill_path(self) -> None:
        result = select_readme(node(
            blobs={"readme_md": "packages/zod/README.md"},
            entries=[blob_entry("README.md", mode=SYMLINK_MODE)],
        ))
        self.assertTrue(result.needs_backfill)
        self.assertEqual(result.path, "packages/zod/README.md")
        self.assertEqual(result.reason, "symlink")
        # Still never returned as text.
        self.assertEqual(result.text, "")

    def test_the_real_repositories_this_was_built_for(self) -> None:
        for repo, target in [
            ("colinhacks/zod", "packages/zod/README.md"),
            ("vuetifyjs/vuetify", "packages/vuetify/README.md"),
            ("unocss/unocss", "packages-presets/unocss/README.md"),
            ("certbot/certbot", "certbot/README.rst"),
            ("jxnblk/mdx-deck", "packages/mdx-deck/README.md"),
        ]:
            with self.subTest(repo=repo):
                result = select_readme(node(blobs={"readme_md": target}))
                self.assertTrue(result.needs_backfill, f"{repo}: symlink not followed")
                self.assertEqual(result.path, target)

    def test_real_content_wins_over_a_symlink_elsewhere(self) -> None:
        result = select_readme(node(blobs={
            "readme_md": "packages/core/README.md",
            "readme_rst": "# Real\n\nActual capability prose here.\n",
        }))
        self.assertIn("Actual capability prose", result.text)
        self.assertFalse(result.needs_backfill)


class TestQueryCost(unittest.TestCase):
    """The inline candidate list is a performance budget, not a wish list.

    Each entry is one blob read per repository per page. Thirteen candidates at
    50 repos a page meant 650 file reads in one GraphQL query, and GitHub
    answered with a wall of 502s. The root tree names the file exactly, so long
    guess-lists are paying twice for the same answer.
    """

    def test_inline_candidates_stay_small(self) -> None:
        from gitglobe.ingest.readme_select import INLINE_CANDIDATES
        self.assertLessEqual(
            len(INLINE_CANDIDATES), 4,
            f"{len(INLINE_CANDIDATES)} candidates x 50 repos = "
            f"{len(INLINE_CANDIDATES) * 50} blob reads per query. This is what caused the 502s.",
        )


class TestLongestWins(unittest.TestCase):
    def test_picks_the_longest_candidate_not_the_first(self) -> None:
        result = select_readme(node(blobs={
            "readme_md": "# stub\n\nSee the docs.\n",
            "readme_rst": "# Full\n\n" + ("Real capability prose. " * 40),
        }))
        self.assertIn("Real capability prose", result.text)

    def test_empty_blobs_are_skipped(self) -> None:
        result = select_readme(node(blobs={"readme_md": "", "readme_lower": "# t\n\nContent here.\n"}))
        self.assertIn("Content here", result.text)

    def test_null_blob_objects_do_not_crash(self) -> None:
        payload = {"readme_md": None, "readme_rst": {"text": None}}
        self.assertEqual(select_readme(payload).text, "")


class TestBackfillDiscovery(unittest.TestCase):
    """The zero-character repos — nestjs/nest and friends.

    Six guessed filenames still missed them. The root tree removes the guessing.
    """

    def test_tree_reveals_a_readme_we_did_not_guess(self) -> None:
        result = select_readme(node(entries=[blob_entry("Readme.md"), blob_entry("src")]))
        self.assertTrue(result.needs_backfill)
        self.assertEqual(result.path, "Readme.md")

    def test_recognises_the_common_spellings(self) -> None:
        names = ["README.md", "readme.md", "Readme.MD", "README", "README.rst",
                 "README.txt", "README.adoc", "README.org"]
        found = readme_names_in_tree(node(entries=[blob_entry(n) for n in names]))
        self.assertEqual(len(found), len(names))

    def test_does_not_mistake_other_files_for_readmes(self) -> None:
        entries = [blob_entry(n) for n in
                   ("READMEME.md", "readme-dev.md", "CONTRIBUTING.md", "read.me", "LICENSE")]
        self.assertEqual(readme_names_in_tree(node(entries=entries)), [])

    def test_a_symlink_with_no_readable_target_is_not_offered_for_backfill(self) -> None:
        # The tree says README.md is a symlink but carries no blob text, so
        # there is no target path to follow. Honest failure beats guessing.
        result = select_readme(node(entries=[blob_entry("README.md", mode=SYMLINK_MODE)]))
        self.assertFalse(result.needs_backfill)
        self.assertEqual(result.text, "")

    def test_no_readme_at_all_reports_cleanly(self) -> None:
        result = select_readme(node(entries=[blob_entry("main.go")]))
        self.assertEqual(result.text, "")
        self.assertFalse(result.needs_backfill)
        self.assertEqual(result.reason, "no readme found")

    def test_directories_named_readme_are_ignored(self) -> None:
        entries = [{"name": "README", "type": "tree", "mode": 0o040000}]
        self.assertEqual(readme_names_in_tree(node(entries=entries)), [])


class TestMalformedPayloads(unittest.TestCase):
    """Ingest touches 100k unreviewed repositories. Nothing may raise."""

    def test_survives_missing_and_null_fields(self) -> None:
        for payload in ({}, {"root_tree": None}, {"root_tree": {"entries": None}},
                        {"root_tree": {"entries": [None]}}, {"readme_md": {}}):
            with self.subTest(payload=payload):
                self.assertIsInstance(select_readme(payload).text, str)

    def test_mode_as_string_is_still_a_symlink(self) -> None:
        # GraphQL returns mode as an Int, but a proxy or a cached fixture can
        # hand it back as a string. Treating "40960" as a normal file would
        # reintroduce the exact bug this module exists to prevent.
        result = select_readme(node(
            blobs={"readme_md": "docs/README.md"},
            entries=[{"name": "README.md", "type": "blob", "mode": str(SYMLINK_MODE)}],
        ))
        self.assertEqual(result.text, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
