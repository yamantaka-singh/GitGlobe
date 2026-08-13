"""Cleaner tests, driven by frozen READMEs from real repositories.

Stdlib `unittest` on purpose: this suite must run before anyone has a database,
a GitHub token, or a virtualenv. `python -m unittest discover tests` is enough.

The assertions are deliberately about *meaning*, not string equality. Asserting
on exact output would make the suite break every time a stop-heading is added,
which trains you to update the expectation instead of thinking. Asserting that
the tagline survives and the licence prose does not stays true across changes
and actually catches regressions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.clean.readme import (
    MIN_SIGNAL_CHARS,
    clean_readme,
    normalise_heading,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / f"{name}.md").read_text(encoding="utf-8")


class TestCapabilitySurvives(unittest.TestCase):
    """The sentence that says what the project does must always come through."""

    def test_react_keeps_its_tagline(self) -> None:
        out = clean_readme(fixture("react"), name="react", language="JavaScript")
        self.assertIn("JavaScript library for building user interfaces", out.text)
        self.assertIn("Declarative", out.text)
        self.assertIn("Component-Based", out.text)

    def test_requests_keeps_its_tagline_and_feature_list(self) -> None:
        out = clean_readme(fixture("requests"), name="requests", language="Python")
        self.assertIn("simple, yet elegant, HTTP library", out.text)
        self.assertIn("Connection Pooling", out.text)
        self.assertIn("SOCKS Proxy Support", out.text)

    def test_flask_keeps_its_description(self) -> None:
        out = clean_readme(fixture("flask"), name="flask", language="Python")
        self.assertIn("lightweight", out.text)
        self.assertIn("web application framework", out.text)

    def test_ollama_keeps_its_one_line_pitch(self) -> None:
        out = clean_readme(fixture("ollama"), name="ollama", language="Go")
        self.assertIn("Start building with open models", out.text)

    def test_setext_headings_are_understood(self) -> None:
        out = clean_readme(fixture("setext"), name="awesome-parser")
        self.assertIn("streaming parser for structured logs", out.text)
        self.assertNotIn("Please read the guidelines", out.text)


class TestBoilerplateRemoved(unittest.TestCase):
    """Boilerplate is the whole reason this module exists.

    Every repo with a CI badge and an MIT licence embeds to nearly the same
    vector. If these assertions fail, Phase 2's clusters are meaningless.
    """

    def test_no_badge_urls_anywhere(self) -> None:
        for name in ("react", "requests", "flask", "ollama"):
            with self.subTest(fixture=name):
                out = clean_readme(fixture(name))
                for marker in ("shields.io", "img.src", "pepy.tech", "readthedocs.org/projects"):
                    self.assertNotIn(marker, out.text)

    def test_no_urls_survive(self) -> None:
        for name in ("react", "requests", "flask", "ollama"):
            with self.subTest(fixture=name):
                out = clean_readme(fixture(name))
                self.assertNotIn("http://", out.text)
                self.assertNotIn("https://", out.text)

    def test_contributing_and_licence_sections_are_dropped(self) -> None:
        out = clean_readme(fixture("react"))
        self.assertNotIn("Code of Conduct", out.text)
        self.assertNotIn("MIT licensed", out.text)
        self.assertNotIn("good first issues", out.text.lower())

    def test_dropping_a_section_takes_its_children(self) -> None:
        # React's Code of Conduct and Good First Issues are h3 under
        # "## Contributing". Dropping the parent must take them too.
        out = clean_readme(fixture("react"))
        self.assertIn("contributing", out.dropped_sections)
        self.assertNotIn("Facebook has adopted", out.text)

    def test_install_sections_are_dropped(self) -> None:
        out = clean_readme(fixture("requests"))
        self.assertNotIn("pip install requests", out.text)
        self.assertNotIn("Cloning the repository", out.text)

    def test_html_wrappers_are_stripped_but_content_kept(self) -> None:
        # Flask and Ollama both open with a centred div/p containing a logo.
        # Dropping the whole element would take the tagline with it.
        out = clean_readme(fixture("ollama"))
        self.assertNotIn("<p align", out.text)
        self.assertNotIn("<img", out.text)
        self.assertIn("Ollama", out.text)

    def test_link_reference_definitions_are_dropped(self) -> None:
        out = clean_readme(fixture("flask"))
        self.assertNotIn("wsgi.readthedocs.io", out.text)
        self.assertNotIn("palletsprojects.com", out.text)
        # ...but the link *text* stays, because it is part of the sentence.
        self.assertIn("WSGI", out.text)

    def test_the_link_dump_at_the_end_of_ollama_goes(self) -> None:
        # "Community Integrations" is 200 lines of third-party links. It is the
        # single worst thing that can reach an embedding: it describes other
        # projects, so the repo lands in their neighbourhood instead of its own.
        out = clean_readme(fixture("ollama"))
        self.assertNotIn("Open WebUI", out.text)
        self.assertNotIn("LibreChat", out.text)


class TestMarkupResidue(unittest.TestCase):
    """Markup that survives cleaning becomes noise tokens in the embedding.

    Every one of these was found by reading the output, not by the suite —
    which is the argument for always looking at what a text pipeline produces
    rather than trusting that green tests mean good text.
    """

    def test_html_entities_are_decoded(self) -> None:
        out = clean_readme(fixture("react"), name="react")
        self.assertNotIn("&middot;", out.text)
        self.assertNotIn("&nbsp;", out.text)
        self.assertNotIn("&amp;", out.text)

    def test_shortcut_reference_links_lose_their_brackets(self) -> None:
        # Flask writes `[WSGI]` with the target defined at the bottom. The
        # definition is stripped, so the brackets are left dangling.
        out = clean_readme(fixture("flask"), name="flask")
        self.assertIn("WSGI", out.text)
        self.assertNotIn("[WSGI]", out.text)
        self.assertNotIn("[Werkzeug]", out.text)

    def test_emphasis_markers_are_stripped(self) -> None:
        out = clean_readme("# t\n\n**Declarative:** it is *fast* and __simple__.\n")
        self.assertIn("Declarative:", out.text)
        self.assertNotIn("**", out.text)
        self.assertNotIn("__", out.text)

    def test_heading_hashes_are_stripped(self) -> None:
        out = clean_readme(fixture("react"), name="react")
        self.assertNotIn("#", out.text)

    def test_inline_code_backticks_are_stripped(self) -> None:
        out = clean_readme("# t\n\nCall `parse()` on a `Buffer` to begin.\n")
        self.assertIn("parse()", out.text)
        self.assertNotIn("`", out.text)


class TestCodeBlocks(unittest.TestCase):
    def test_short_snippets_are_kept(self) -> None:
        # `ollama run gemma4` is a clearer statement of purpose than a paragraph.
        out = clean_readme(fixture("ollama"))
        self.assertIn("ollama run gemma4", out.text)

    def test_long_blocks_are_dropped(self) -> None:
        out = clean_readme(fixture("react"))
        self.assertNotIn("createRoot(document.getElementById", out.text)

    def test_unterminated_fence_does_not_swallow_the_file(self) -> None:
        raw = "# tool\n\nDoes a useful thing.\n\n```python\nprint(1)\n"
        out = clean_readme(raw, name="tool")
        self.assertIn("Does a useful thing", out.text)


class TestSignalQuality(unittest.TestCase):
    def test_reduction_is_substantial_on_real_readmes(self) -> None:
        # If we are not removing most of a real README, we are not doing the job.
        for name, floor in (("react", 0.45), ("requests", 0.40), ("ollama", 0.55)):
            with self.subTest(fixture=name):
                out = clean_readme(fixture(name))
                self.assertGreater(
                    out.reduction, floor,
                    f"{name}: only removed {out.reduction:.0%}, expected > {floor:.0%}",
                )

    def test_thin_readmes_are_flagged_low_signal(self) -> None:
        self.assertTrue(clean_readme(fixture("minimal"), name="tiny-lib").low_signal)
        self.assertTrue(clean_readme(fixture("empty"), name="nothing").low_signal)

    def test_a_good_description_rescues_a_thin_readme(self) -> None:
        out = clean_readme(
            fixture("minimal"),
            name="tiny-lib",
            description=(
                "A zero-dependency streaming CSV parser for Node with backpressure "
                "support and RFC 4180 compliance, built for multi-gigabyte files."
            ),
        )
        self.assertFalse(out.low_signal)

    def test_substantial_readmes_are_not_flagged(self) -> None:
        for name in ("react", "requests", "flask", "ollama"):
            with self.subTest(fixture=name):
                self.assertFalse(clean_readme(fixture(name)).low_signal)

    def test_non_english_is_detected_not_discarded(self) -> None:
        out = clean_readme(fixture("nonenglish"), name="shujv")
        self.assertTrue(out.non_english)
        # Flagged, not dropped — Phase 2 can decide what to do about it.
        self.assertGreater(len(out.text), 0)

    def test_empty_input_does_not_raise(self) -> None:
        out = clean_readme("", name="ghost")
        self.assertEqual(out.text, "")
        self.assertEqual(out.reduction, 0.0)
        self.assertIn("ghost", out.embedding_input)


class TestEmbeddingInput(unittest.TestCase):
    def test_composes_name_description_language_and_topics(self) -> None:
        out = clean_readme(
            fixture("requests"),
            name="requests",
            description="Python HTTP for Humans.",
            language="Python",
            topics=["http", "client", "python"],
        )
        self.assertIn("requests — Python HTTP for Humans.", out.embedding_input)
        self.assertIn("Language: Python.", out.embedding_input)
        self.assertIn("Topics: http, client, python.", out.embedding_input)
        self.assertIn("simple, yet elegant, HTTP library", out.embedding_input)

    def test_missing_metadata_leaves_no_dangling_punctuation(self) -> None:
        out = clean_readme("Some prose about the thing.", name="solo")
        self.assertNotIn("—", out.embedding_input.split("\n")[0].replace("solo", ""))
        self.assertFalse(out.embedding_input.startswith("—"))

    def test_truncation_respects_the_limit(self) -> None:
        raw = "# big\n\n" + ("This is a sentence about capability. " * 2000)
        out = clean_readme(raw, name="big", max_chars=1200)
        self.assertLessEqual(len(out.text), 1200)
        self.assertTrue(out.text.endswith((".", "y")))


class TestHeadingNormalisation(unittest.TestCase):
    def test_handles_real_world_heading_mess(self) -> None:
        cases = {
            "Installation": "installation",
            "📦 Installation": "installation",
            "Installation:": "installation",
            "Getting Started {#start}": "getting started",
            "[Contributing](CONTRIBUTING.md)": "contributing",
            "🚀  Quick   Start": "quick start",
            "License / Licence": "license licence",
        }
        for raw, expected in cases.items():
            with self.subTest(heading=raw):
                self.assertEqual(normalise_heading(raw), expected)


class TestNoCrashesOnHostileInput(unittest.TestCase):
    """Ingest touches 100k unreviewed files. It must not die on any of them."""

    def test_survives_pathological_inputs(self) -> None:
        hostile = [
            "#" * 500,
            "```" * 200,
            "[" * 1000 + "]" * 1000,
            "<div>" * 300,
            "\x00\x01\x02 binary-ish content",
            "# h\n" + "|a|b|\n" * 500,
            "\n" * 10000,
            "![](" + "x" * 5000 + ")",
        ]
        for i, raw in enumerate(hostile):
            with self.subTest(case=i):
                out = clean_readme(raw, name=f"case{i}")
                self.assertIsInstance(out.text, str)

    def test_min_signal_constant_is_sane(self) -> None:
        self.assertGreater(MIN_SIGNAL_CHARS, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
