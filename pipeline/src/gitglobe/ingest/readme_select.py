"""Choosing the right README blob from a GraphQL response.

Pure logic, no I/O, so it can be tested against the exact payloads that caused
trouble in production.

## Why this is not trivial

Two failures showed up in the first 100k run, both silent:

**Symlinks.** Git stores a symlink as a blob whose *content is the target path*.
`certbot/README.rst` is a symlink, so GraphQL returned the eighteen-character
string `certbot/README.rst` as its "README". That is not merely useless — it
would embed a repository on the strength of its own filename, and it passes
every non-empty check. The tree entry's file mode (`120000`) is the only
reliable way to tell.

**Filename roulette.** READMEs live under a dozen spellings and cases, and
guessing six of them still missed `nestjs/nest`. Listing the root tree costs one
extra field and removes the guessing entirely.

The two interact: a repository can have a real `README.rst` *and* a stub
`README.md` symlinking to it, so first-match-wins picks the stub.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Git file mode for a symbolic link.
SYMLINK_MODE = 0o120000  # 40960

#: Candidate names fetched inline. DELIBERATELY SHORT.
#:
#: Each entry costs one blob read per repository per page. At 50 repos a page,
#: thirteen candidates meant 650 file reads in a single GraphQL query, and
#: GitHub answered with a wall of 502s. Three candidates is 150 reads and covers
#: the large majority of repositories outright.
#:
#: Everything else is handled by the root tree, which names the file exactly and
#: costs one cheap field — that is the whole reason it was added. Guessing
#: thirteen filenames *and* listing the tree was paying twice for the same
#: answer, and the second payment was what broke the run.
INLINE_CANDIDATES: dict[str, str] = {
    "readme_md": "README.md",
    "readme_lower": "readme.md",
    "readme_rst": "README.rst",
}

_README_NAME = re.compile(r"^readme(\.(md|markdown|rst|txt|adoc|org))?$", re.IGNORECASE)

#: A symlink's content is a path: one line, no blank lines, plausibly a filename.
#: Belt-and-braces for repositories whose tree we could not read.
_LOOKS_LIKE_PATH = re.compile(r"^[\w./@+-]{1,200}$")


@dataclass
class ReadmeChoice:
    text: str = ""
    #: Which path the text came from, for debugging and backfill.
    path: str = ""
    #: True when `path` names a README we could not fetch inline. The caller
    #: should schedule a targeted second fetch for it.
    needs_backfill: bool = False
    #: Why the inline attempt failed, if it did.
    reason: str = ""


def tree_entries(node: dict[str, Any]) -> list[dict[str, Any]]:
    tree = node.get("root_tree") or {}
    return [e for e in (tree.get("entries") or []) if e]


def symlink_names(node: dict[str, Any]) -> set[str]:
    """Root-level names that are symlinks, by file mode."""
    return {
        e["name"]
        for e in tree_entries(node)
        if e.get("mode") == SYMLINK_MODE or e.get("mode") == str(SYMLINK_MODE)
    }


def readme_names_in_tree(node: dict[str, Any]) -> list[str]:
    """Every root-level file that looks like a README, symlinks excluded."""
    links = symlink_names(node)
    return [
        e["name"]
        for e in tree_entries(node)
        if e.get("type") == "blob" and _README_NAME.match(e["name"] or "") and e["name"] not in links
    ]


def looks_like_symlink_content(text: str) -> bool:
    """Fallback detection when the tree is unavailable.

    A symlink blob is a single line naming a path. Real README content
    essentially always has a newline; a one-line file that is a bare path and
    nothing else is not a README anyone wrote.
    """
    stripped = text.strip()
    if not stripped or "\n" in stripped or " " in stripped:
        return False
    return bool(_LOOKS_LIKE_PATH.match(stripped)) and ("/" in stripped or "." in stripped)


def select_readme(node: dict[str, Any]) -> ReadmeChoice:
    """Pick the best README blob, or say where to find the real one.

    Three outcomes, in order of preference:
      1. usable inline text
      2. a path to fetch (`needs_backfill`) — from a symlink target or the tree
      3. nothing
    """
    links = symlink_names(node)

    best = ReadmeChoice()
    symlink_target = ""

    for alias, filename in INLINE_CANDIDATES.items():
        text = (node.get(alias) or {}).get("text") or ""
        if not text:
            continue

        if filename in links or looks_like_symlink_content(text):
            # A symlink's content IS its target path, which is exactly the
            # information needed to fetch the real file. Monorepos rely on this
            # constantly — zod, vuetify, unocss and certbot all point their root
            # README at a package subdirectory. Rejecting the symlink loses a
            # good repository; following it recovers one.
            if not symlink_target and looks_like_symlink_content(text):
                symlink_target = text.strip()
            continue

        # Longest wins. A repo can carry a two-line stub README.md beside a real
        # README.rst, and first-match-wins picks the stub.
        if len(text) > len(best.text):
            best = ReadmeChoice(text=text, path=filename)

    if best.text:
        return best

    if symlink_target:
        return ReadmeChoice(needs_backfill=True, path=symlink_target, reason="symlink")

    # Nothing inline and no symlink. The tree names the file exactly, which is
    # what makes trimming INLINE_CANDIDATES safe.
    for name in readme_names_in_tree(node):
        return ReadmeChoice(needs_backfill=True, path=name, reason="not an inline candidate")

    return ReadmeChoice(reason="no readme found")
