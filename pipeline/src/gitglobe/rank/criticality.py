"""Backfill OpenSSF criticality scores from the published bulk dump.

`criticality` has been NULL for every row since Phase 1, which means the 0.10 of
weight it carries in `Weights` has been contributing nothing while still sitting
in the denominator — the reason no repository could score above 80 out of 100.

**Why a bulk file and not the API.** The score is derived from ten signals that
each cost several GitHub API calls to compute; running the upstream tool over
87k repositories would take days. OpenSSF publishes the finished scores for
every repository it has ever scored as a single CSV, free and unauthenticated,
so the backfill is a download and a join.

**What the score actually contains**, because it is not the independent axis it
looks like. Two inputs carry double weight: `contributor_count` and
`dependents_count`. The second of those is close to the dependents signal that
already has 0.35 of the weight here, so criticality partially double-counts it
rather than balancing it. What it adds that nothing else in the composite has:
contributor and organisation spread, commit frequency, release cadence, and
`updated_since` at weight -1, which actively penalises stagnation. That last
term is why a widely-depended-upon but abandoned utility scores below an
actively maintained framework, and it is the closest thing to a velocity signal
available without crawling 87k repositories.

**Staleness is the real caveat.** Dumps ran roughly fortnightly until
2025-07-25 and then stopped. The date is part of the constant below and is
printed on every run on purpose: this project already refuses to quote an
unmeasured star scale as fact, and a criticality score is no different.
"""

from __future__ import annotations

import csv
import io
import logging

log = logging.getLogger(__name__)

#: The most recent published dump. Encoded as a full URL rather than assembled
#: from parts so that the date is impossible to miss when reading or grepping.
#: Check for a newer prefix at
#: https://storage.googleapis.com/storage/v1/b/ossf-criticality-score/o?delimiter=/
DUMP_URL = (
    "https://storage.googleapis.com/ossf-criticality-score/"
    "2025.07.25/010355/all.csv"
)
DUMP_MEASURED_AT = "2025-07-25"

#: Columns as published by the v2 collector.
URL_COLUMN = "repo.url"
SCORE_COLUMN = "default_score"

#: A score is defined on [0, 1]. Anything outside it means the column moved or
#: the file is not what we think it is, and silently clamping would bake a
#: broken signal into every rank derived from it.
SCORE_MIN = 0.0
SCORE_MAX = 1.0


def repo_from_url(url: str) -> str | None:
    """`https://github.com/owner/name` -> `owner/name`, or None if not GitHub.

    The dump is GitHub-only today but says it intends to add other forges, and
    a GitLab path silently matched against a GitHub full_name would attach one
    project's score to a different project entirely.
    """
    text = url.strip().rstrip("/")
    marker = "github.com/"
    index = text.find(marker)
    if index < 0:
        return None
    path = text[index + len(marker):]
    parts = path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1]}"


def parse_criticality(handle) -> dict[str, float]:
    """Read the dump into `owner/name` (lowercased) -> score.

    Pure and stream-shaped so the parsing can be tested against a few rows
    without downloading 119 MB.
    """
    reader = csv.DictReader(handle)
    fields = reader.fieldnames or []
    if URL_COLUMN not in fields or SCORE_COLUMN not in fields:
        raise ValueError(
            f"dump is missing {URL_COLUMN!r} or {SCORE_COLUMN!r}; got {fields[:8]}"
        )

    scores: dict[str, float] = {}
    skipped = 0
    for row in reader:
        name = repo_from_url(row.get(URL_COLUMN) or "")
        if name is None:
            skipped += 1
            continue
        try:
            score = float(row.get(SCORE_COLUMN) or "")
        except ValueError:
            skipped += 1
            continue
        if not SCORE_MIN <= score <= SCORE_MAX:
            raise ValueError(f"{name}: score {score} outside [0, 1]; wrong column?")
        scores[name.lower()] = score
    if skipped:
        log.info("Skipped %d unparseable or non-GitHub rows", skipped)
    return scores


async def fetch_criticality(url: str = DUMP_URL) -> dict[str, float]:
    """Download and parse the dump. ~119 MB, no authentication."""
    import httpx

    log.info("Downloading OpenSSF criticality dump measured %s", DUMP_MEASURED_AT)
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        text = response.text
    log.info("Read %.0f MB", len(text) / 1_000_000)
    return parse_criticality(io.StringIO(text))


def match_corpus(scores: dict[str, float], case_map: dict[str, str]) -> dict[str, float]:
    """Join dump scores onto the names this corpus actually stores.

    `case_map` is lowercase -> stored spelling. GitHub treats owner and repo
    names case-insensitively while the dump and our rows may disagree on case,
    so matching on the raw string loses real rows for no reason.
    """
    matched = {
        case_map[lower]: score
        for lower, score in scores.items()
        if lower in case_map
    }
    log.info(
        "Matched %d of %d corpus repositories against %d scored projects",
        len(matched), len(case_map), len(scores),
    )
    return matched
