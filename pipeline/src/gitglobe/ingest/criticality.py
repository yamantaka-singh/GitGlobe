"""OSSF Criticality Score.

A better importance signal than stars, because it weighs dependents, release
cadence and contributor spread rather than popularity. A build tool nobody
stars but everything depends on scores high; a viral tutorial repo does not.

Published as a public dataset rather than an API, so this is a bulk download
and join, not a per-repo lookup.
"""

from __future__ import annotations

import csv
import io
import logging

import httpx

log = logging.getLogger(__name__)

# The OSSF publishes periodic CSV snapshots. Pinned by convention rather than
# discovered, so a run is reproducible.
CRITICALITY_CSV_URL = (
    "https://commondatastorage.googleapis.com/ossf-criticality-score/"
    "2025.07.25/010355/all.csv"
)


async def fetch_criticality_scores(
    full_names: list[str], *, url: str = CRITICALITY_CSV_URL, timeout: float = 120.0
) -> dict[str, float]:
    """Scores for the repos we hold, keyed by `owner/name`.

    Missing scores are not an error. Coverage is partial by design — the OSSF
    scores a large but bounded set — and `criticality` stays NULL, which
    ADR-008's blended size formula already handles.
    """
    if not full_names:
        return {}

    wanted = {name.lower() for name in full_names}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            body = response.text
    except httpx.HTTPError as exc:
        # Non-fatal. Losing this signal degrades node sizing slightly; failing
        # the whole run over it would be wrong.
        log.warning("Criticality scores unavailable (%s) — continuing without", exc)
        return {}

    scores: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(body)):
        repo_url = (row.get("repo.url") or row.get("url") or "").rstrip("/")
        if "github.com/" not in repo_url:
            continue
        name = repo_url.split("github.com/", 1)[1]
        if name.lower() not in wanted:
            continue
        raw = row.get("default_score") or row.get("criticality_score") or ""
        try:
            scores[name] = float(raw)
        except ValueError:
            continue

    log.info("Criticality scores for %d of %d repos", len(scores), len(full_names))
    return scores
