"""README -> capability text.

The highest-leverage code in the project, and the part most pipelines skip.

A raw README is roughly: badge row, logo, title, tagline, table of contents,
installation, usage, API reference, contributing, license, sponsors. Only the
tagline, a little prose, and the feature list describe what the project *does*.
Everything else is boilerplate — and boilerplate is actively harmful, because
every repository with a CI badge and an MIT license embeds to nearly the same
vector. Skip this step and Phase 2's clusters are meaningless.

Deliberately zero-dependency. It is pure text in, text out, which means it can
be tested exhaustively against frozen fixtures of real READMEs, and those tests
run anywhere without a database, a network, or a virtualenv.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Section headings whose content is process, not capability.
#
# Tuned against the fixtures. Two rules kept this list honest:
#   - "usage" and "example" are NOT here. They usually carry the clearest
#     statement of what a tool is for, once the long code blocks are stripped.
#   - "documentation" IS here. In practice it is a bare list of links.
# --------------------------------------------------------------------------- #
STOP_HEADINGS: frozenset[str] = frozenset(
    {
        "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
        "author", "authors", "backers", "badges", "build", "building",
        "building from source", "changelog", "change log", "citation", "cite",
        "citing", "code of conduct", "community", "contact", "contents",
        "contribute", "contributing", "contribution", "contributions",
        "contributors", "credits", "dependencies", "development", "developing",
        "donate", "donating", "donation", "donations", "documentation", "docs",
        "download", "downloads", "faq", "funding", "getting help", "help",
        "install", "installation", "installing", "license", "licence",
        "licensing", "links", "maintainers", "notes", "prerequisites",
        "cloning", "community integrations", "ecosystem", "integrations",
        "projects using", "related", "related projects", "release notes", "releases",
        "showcase", "used by", "who uses",
        "requirements", "resources", "roadmap", "security", "setup",
        "sponsor", "sponsors", "sponsorship", "star history", "stargazers",
        "support", "supported platforms", "table of contents", "testing",
        "tests", "thanks", "toc", "troubleshooting",
    }
)

# Image/badge hosts that never carry meaning.
_BADGE_HOSTS = (
    "img.shields.io", "badge.fury.io", "travis-ci", "circleci.com",
    "codecov.io", "coveralls.io", "appveyor.com", "badgen.net",
    "pepy.tech", "readthedocs.org", "snyk.io", "codeclimate.com",
    "opencollective.com", "herokucdn.com", "gitpod.io", "deepsource.io",
    "bestpractices.coreinfrastructure.org", "isitmaintained.com",
    "githubusercontent.com/.*badge", "sonarcloud.io", "codefactor.io",
)

_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_LINKED_IMAGE = re.compile(r"\[\s*!\[[^\]]*\]\([^)]*\)\s*\]\([^)]*\)")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_LINK_REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_RE_INLINE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_REF_LINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
# `[WSGI]` with its definition elsewhere. Once the definitions are stripped the
# brackets are left dangling around ordinary words.
_RE_SHORTCUT_LINK = re.compile(r"\[([^\]\[]+)\](?![\(\[:])")
_RE_EMPHASIS = re.compile(r"(\*\*\*|\*\*|___|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`+([^`]*)`+")
_RE_LEADING_HASH = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_RE_BARE_URL = re.compile(r"https?://\S+")
_RE_FENCE = re.compile(r"^\s*(```+|~~~+)")
_RE_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RE_SETEXT_UNDERLINE = re.compile(r"^\s*(=+|-{2,})\s*$")
_RE_HEADING_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")
_RE_EMOJI_SHORTCODE = re.compile(r":[a-z0-9_+-]+:")
_RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_RE_HR = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")

# Non-Latin scripts. Used only to detect *predominantly* non-English READMEs,
# which embed poorly against an English-dominated corpus.
_RE_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_RE_CYRILLIC_ARABIC = re.compile(r"[Ѐ-ӿ؀-ۿ]")

MAX_CODE_BLOCK_LINES = 5
MAX_CHARS = 6000
MIN_SIGNAL_CHARS = 120


@dataclass
class CleanResult:
    """The cleaned text plus everything needed to judge whether to trust it."""

    text: str
    #: Composed embedding input — name, description, language, topics, prose.
    embedding_input: str
    low_signal: bool
    original_chars: int
    clean_chars: int
    dropped_sections: list[str] = field(default_factory=list)
    #: Fraction of the original that survived. Very high values are suspicious:
    #: they usually mean the README had no structure for us to strip.
    reduction: float = 0.0
    non_english: bool = False


def _strip_badges_and_images(text: str) -> str:
    text = _RE_LINKED_IMAGE.sub("", text)
    text = _RE_IMAGE.sub("", text)
    # Any surviving link pointing at a known badge host.
    for host in _BADGE_HOSTS:
        text = re.sub(r"\[[^\]]*\]\([^)]*" + host + r"[^)]*\)", "", text)
    return text


def _strip_html(text: str) -> str:
    """Drop tags but keep the text inside them.

    READMEs routinely wrap their entire opening in `<div align="center">` or
    `<p align="center">`. Dropping the whole element would take the tagline —
    which is usually the single best sentence in the file — with it.
    """
    text = _RE_HTML_COMMENT.sub("", text)
    # Elements whose *content* is also noise.
    text = re.sub(r"<(script|style|svg)\b.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return _RE_HTML_TAG.sub("", text)


def _split_code_fences(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """Partition into (is_code, lines) runs. Fence markers are not returned."""
    blocks: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    in_code = False
    fence = ""

    for line in lines:
        match = _RE_FENCE.match(line)
        if match and not in_code:
            if current:
                blocks.append((False, current))
            current, in_code, fence = [], True, match.group(1)[0]
            continue
        if match and in_code and match.group(1)[0] == fence:
            blocks.append((True, current))
            current, in_code = [], False
            continue
        current.append(line)

    if current:
        blocks.append((in_code, current))
    return blocks


def _heading_of(lines: list[str], index: int) -> tuple[int, str] | None:
    """ATX (`## Title`) or setext (`Title` over `-----`) heading at `index`."""
    atx = _RE_ATX_HEADING.match(lines[index])
    if atx:
        return len(atx.group(1)), atx.group(2)
    # A run of dashes matches BOTH the setext-underline and horizontal-rule
    # patterns. CommonMark resolves the ambiguity in favour of the heading when
    # the line above is non-blank, which is exactly the check below — excluding
    # HR-shaped lines here is what made `Contributing\n------------` invisible.
    if (
        index + 1 < len(lines)
        and lines[index].strip()
        and not _RE_ATX_HEADING.match(lines[index])
        and _RE_SETEXT_UNDERLINE.match(lines[index + 1])
    ):
        return (1 if lines[index + 1].strip().startswith("=") else 2), lines[index].strip()
    return None


def normalise_heading(title: str) -> str:
    """Reduce a heading to a comparable key.

    Handles the real-world mess: `## 📦 Installation`, `### Install [↑](#toc)`,
    `## Getting Started {#start}`, `Installation:`.
    """
    title = _RE_HEADING_ANCHOR.sub("", title)
    title = _RE_INLINE_LINK.sub(r"\1", title)
    title = _RE_REF_LINK.sub(r"\1", title)
    title = _RE_EMOJI_SHORTCODE.sub("", title)
    # Strip anything that is not a letter, digit or space — emoji, punctuation,
    # decorative arrows, trailing colons.
    title = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in title)
    return _RE_MULTISPACE.sub(" ", title).strip().lower()


def _is_stop_heading(key: str) -> bool:
    """Exact match, or the heading *opens* with a stop phrase.

    Real headings are rarely bare: "Installing Requests and Supported Versions",
    "Cloning the repository", "Community Integrations". Matching only exact
    strings misses nearly all of them.

    Prefix matching is on whole words (`phrase + " "`), which is what keeps
    "Supported Features" from being eaten by the "support" entry.
    """
    if key in STOP_HEADINGS:
        return True
    return any(key.startswith(phrase + " ") for phrase in STOP_HEADINGS)


_RE_LIST_ITEM = re.compile(r"^\s*[-*+]\s+")
_RE_ANY_LINK = re.compile(r"\[[^\]]*\]\(|\[[^\]]*\]\[")


def _is_link_dump(body: list[str]) -> bool:
    """True for sections that are mostly a list of links to other projects.

    This is the single worst thing that can reach an embedding. A page of
    "integrations" describes *other* people's projects, so the repository lands
    in their neighbourhood instead of its own. Heading names never catch all of
    these, so the shape of the content decides.
    """
    items = [ln for ln in body if _RE_LIST_ITEM.match(ln)]
    if len(items) < 6:
        return False
    linked = sum(1 for ln in items if _RE_ANY_LINK.search(ln))
    return linked / len(items) >= 0.7


def _drop_stop_sections(text: str) -> tuple[str, list[str]]:
    """Remove sections whose heading is process rather than capability.

    A section runs until the next heading at the same level *or shallower*, so
    dropping `## Contributing` also drops its `### Code of Conduct` child.
    """
    lines = text.split("\n")
    keep: list[str] = []
    dropped: list[str] = []
    skip_until_level: int | None = None
    i = 0

    while i < len(lines):
        heading = _heading_of(lines, i)
        if heading is not None:
            level, title = heading
            key = normalise_heading(title)
            consumed = 2 if _RE_ATX_HEADING.match(lines[i]) is None else 1

            if skip_until_level is not None and level <= skip_until_level:
                skip_until_level = None

            if skip_until_level is None and _is_stop_heading(key):
                skip_until_level = level
                dropped.append(key)
                i += consumed
                continue

            if skip_until_level is None:
                keep.extend(lines[i : i + consumed])
            i += consumed
            continue

        if skip_until_level is None:
            keep.append(lines[i])
        i += 1

    return "\n".join(keep), dropped


def _drop_link_dumps(text: str) -> tuple[str, list[str]]:
    """Second pass: drop sections that are mostly links, whatever they are called.

    Heading names never catch all of these — "Ecosystem", "Built With",
    "Powered By", "Alternatives", "Awesome X" — so the shape of the content
    decides. Runs after the stop-heading pass so it only sees survivors.
    """
    lines = text.split("\n")
    sections: list[tuple[int, int, str, int]] = []  # start, body_start, key, level
    for i in range(len(lines)):
        heading = _heading_of(lines, i)
        if heading is None:
            continue
        level, title = heading
        consumed = 1 if _RE_ATX_HEADING.match(lines[i]) else 2
        sections.append((i, i + consumed, normalise_heading(title), level))

    drop_ranges: list[tuple[int, int]] = []
    dropped: list[str] = []
    for idx, (start, body_start, key, level) in enumerate(sections):
        end = len(lines)
        for later_start, _, _, later_level in sections[idx + 1 :]:
            if later_level <= level:
                end = later_start
                break
        if _is_link_dump(lines[body_start:end]):
            drop_ranges.append((start, end))
            dropped.append(key)

    if not drop_ranges:
        return text, []

    keep = [ln for i, ln in enumerate(lines)
            if not any(lo <= i < hi for lo, hi in drop_ranges)]
    return "\n".join(keep), dropped


def _truncate(text: str, limit: int) -> str:
    """Cut at the nearest sensible boundary, never mid-word.

    Tries paragraph, then sentence, then word. Text that stops mid-word reads
    as corrupted, and an embedding of a truncated token is noise.
    """
    if len(text) <= limit:
        return text
    floor = int(limit * 0.6)

    para = text.rfind("\n\n", floor, limit)
    if para != -1:
        return text[:para].rstrip()

    sentence = max(text.rfind(c, floor, limit) for c in (". ", ".\n", "! ", "? "))
    if sentence != -1:
        return text[: sentence + 1].rstrip()

    word = text.rfind(" ", floor, limit)
    return text[: word if word != -1 else limit].rstrip()


def _strip_long_code(text: str) -> str:
    """Drop long fenced blocks; keep short ones.

    A three-line snippet often *is* the clearest statement of what a library
    does — `requests.get(url)` says more than a paragraph. A sixty-line config
    dump says nothing and drowns everything around it.
    """
    out: list[str] = []
    for is_code, block in _split_code_fences(text.split("\n")):
        if not is_code:
            out.extend(block)
            continue
        meaningful = [ln for ln in block if ln.strip()]
        if len(meaningful) <= MAX_CODE_BLOCK_LINES:
            out.extend(block)
    return "\n".join(out)


def _flatten_links(text: str) -> str:
    text = _RE_LINK_REF_DEF.sub("", text)
    text = _RE_INLINE_LINK.sub(r"\1", text)
    text = _RE_REF_LINK.sub(r"\1", text)
    text = _RE_SHORTCUT_LINK.sub(r"\1", text)
    return _RE_BARE_URL.sub("", text)


def _strip_markup(text: str) -> str:
    """Remove markdown syntax that carries no meaning once flattened.

    `**Declarative:**` and `Declarative:` mean the same thing to a reader and
    tokenise differently for an embedding model. The asterisks are pure noise.
    """
    text = html.unescape(text)
    text = _RE_LEADING_HASH.sub("", text)
    # Twice: `***bold italic***` needs two rounds to unwrap.
    text = _RE_EMPHASIS.sub(r"\2", text)
    text = _RE_EMPHASIS.sub(r"\2", text)
    return _RE_INLINE_CODE.sub(r"\1", text)


def _tidy(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        if _RE_HR.match(line) or _RE_TABLE_ROW.match(line):
            continue
        line = _RE_EMOJI_SHORTCODE.sub("", line)
        line = _RE_MULTISPACE.sub(" ", line).rstrip()
        # Drop lines with no letters at all — leftover punctuation, separators,
        # and emoji-only decoration.
        if line.strip() and not any(ch.isalpha() for ch in line):
            continue
        out.append(line)
    return _RE_MULTINEWLINE.sub("\n\n", "\n".join(out)).strip()


def is_predominantly_non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 40:
        return False
    foreign = len(_RE_CJK.findall(text)) + len(_RE_CYRILLIC_ARABIC.findall(text))
    return foreign / len(letters) > 0.25


def clean_readme(
    raw: str,
    *,
    name: str = "",
    description: str = "",
    language: str = "",
    topics: list[str] | None = None,
    max_chars: int = MAX_CHARS,
) -> CleanResult:
    """Reduce a raw README to the part that describes capability."""
    original_chars = len(raw or "")

    text = _strip_html(_strip_badges_and_images(raw or ""))
    text, dropped = _drop_stop_sections(text)
    text, link_dumps = _drop_link_dumps(text)
    dropped.extend(link_dumps)
    text = _strip_long_code(text)
    text = _flatten_links(text)
    text = _strip_markup(text)
    text = _tidy(text)

    text = _truncate(text, max_chars)

    topics = topics or []
    non_english = is_predominantly_non_english(text)
    # A repo whose cleaned text is a couple of words tells us nothing, and
    # letting it into the layout fit smears the clusters. Still indexed, just
    # flagged, dimmed in the render, and excluded from fitting.
    low_signal = len(text) < MIN_SIGNAL_CHARS and len(description) < MIN_SIGNAL_CHARS

    parts = [f"{name} — {description}".strip(" —")]
    meta = []
    if language:
        meta.append(f"Language: {language}.")
    if topics:
        meta.append(f"Topics: {', '.join(topics[:12])}.")
    if meta:
        parts.append(" ".join(meta))
    if text:
        parts.append(text)

    return CleanResult(
        text=text,
        embedding_input="\n".join(p for p in parts if p).strip(),
        low_signal=low_signal,
        original_chars=original_chars,
        clean_chars=len(text),
        dropped_sections=dropped,
        reduction=1 - (len(text) / original_chars) if original_chars else 0.0,
        non_english=non_english,
    )
