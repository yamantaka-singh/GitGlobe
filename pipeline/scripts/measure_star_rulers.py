"""Compare candidate star rulers against the corpus we actually hold.

The whole reason the star component contributes nothing today is that it was
calibrated against GitHub's distribution while being applied to a corpus that
is nothing like it: ingest sharded star-descending and completed 353 of 864
shards, so this corpus is far more top-heavy than GitHub-above-66. Picking a
replacement ruler by reasoning about power laws in the abstract would repeat
exactly that mistake.

So: read the real star counts, score them under every candidate, and print two
things that decide the question.

**Spread** — how much of the 0-1 range the middle of the corpus occupies. A
ruler that leaves 80% of repos inside one tenth of the range has not fixed the
compression, it has moved it.

**Titan separation** — the mean gap between consecutive repos in the top 100.
This is the product requirement stated plainly: react and kubernetes must not
look like a 5k-star project, and must not look like each other either. A ruler
can score well on spread and still flatten the top, which is exactly what a
percentile does.

Run it, read the table, then edit `Weights`/`global_scale.py` once. This script
is a decision tool and is deliberately not production code.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitglobe.db import Database
from gitglobe.rank.global_scale import StarScale
from gitglobe.settings import Settings

#: Where the piecewise ruler hands over from log to linear, and how much of the
#: output range it reserves for everything above that. 0.60 means the top 40% of
#: visual weight belongs to repositories above the knee.
PIECEWISE_KNEE = 10_000
PIECEWISE_SPLIT = 0.60


def _norm(value: float, lo: float, hi: float) -> float:
    return 0.0 if hi <= lo else min(1.0, max(0.0, (value - lo) / (hi - lo)))


def conditional(stars, scale: StarScale, floor: float, top: float):
    """P(X < s | X >= corpus floor), straight off the measured curve."""
    base = scale.repos_at_least(floor)
    return [1.0 - scale.repos_at_least(s) / base for s in stars]


def log_rank(stars, scale: StarScale, floor: float, top: float):
    """Even spacing per order of magnitude of global rank."""
    base = math.log(max(scale.repos_at_least(floor), 2.0))
    return [1.0 - math.log(max(scale.repos_at_least(s), 1.0)) / base for s in stars]


def root(power: float):
    """Fractional power: gentler on the tail than log, softer than linear."""
    def ruler(stars, scale, floor, top):
        lo, hi = floor ** (1 / power), top ** (1 / power)
        return [_norm(s ** (1 / power), lo, hi) for s in stars]
    return ruler


def piecewise(stars, scale: StarScale, floor: float, top: float):
    """Log below the knee, linear above it — visual control over the top band."""
    lo, knee = math.log(max(floor, 1.0)), math.log(PIECEWISE_KNEE)
    out = []
    for s in stars:
        if s <= PIECEWISE_KNEE:
            out.append(_norm(math.log(max(s, 1.0)), lo, knee) * PIECEWISE_SPLIT)
        else:
            frac = _norm(s, PIECEWISE_KNEE, top)
            out.append(PIECEWISE_SPLIT + frac * (1.0 - PIECEWISE_SPLIT))
    return out


RULERS = {
    "conditional": conditional,
    "log-rank": log_rank,
    "cube-root": root(3),
    "fourth-root": root(4),
    "piecewise": piecewise,
}


def report(name: str, scores: list[float]) -> str:
    ordered = sorted(scores)
    n = len(ordered)
    p10, p50, p90 = (ordered[int(n * q)] for q in (0.10, 0.50, 0.90))
    # Mean gap between neighbours in the top 100 — flat here means the titans
    # are indistinguishable no matter how good the overall spread looks.
    top100 = ordered[-100:]
    gaps = [b - a for a, b in zip(top100, top100[1:])]
    titan = statistics.mean(gaps) if gaps else 0.0
    # Occupancy: how many of ten equal bands contain at least 1% of the corpus.
    bands = [0] * 10
    for s in scores:
        bands[min(9, int(s * 10))] += 1
    used = sum(1 for b in bands if b >= n * 0.01)
    return (f"{name:<12} p10 {p10:5.3f}  p50 {p50:5.3f}  p90 {p90:5.3f}  "
            f"spread(p10-p90) {p90 - p10:5.3f}  titan-gap {titan:.5f}  "
            f"bands-used {used}/10")


async def main() -> None:
    settings = Settings.from_env()
    db = await Database.connect(settings.database_url)
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT stars FROM repo WHERE stars IS NOT NULL ORDER BY stars"
            )
        raw = await db.latest_star_scale()
    finally:
        await db.close()

    stars = [float(r["stars"]) for r in rows]
    assert stars, "no repositories with stars; run ingest first"
    scale = StarScale.from_dict(raw) if raw else None
    assert scale is not None, "no star scale stored; run `gitglobe calibrate --remeasure`"

    floor, top = stars[0], stars[-1]
    print(f"{len(stars):,} repositories · stars {floor:,.0f} to {top:,.0f} · "
          f"median {stars[len(stars) // 2]:,.0f}\n")
    for name, ruler in RULERS.items():
        print(report(name, ruler(stars, scale, floor, top)))
    print("\nWant: spread near 1.0, titan-gap as large as possible, bands-used 10/10.")


if __name__ == "__main__":
    asyncio.run(main())
