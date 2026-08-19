"""One-off: rate the top-star repos the stratified `teach` sample never touched.

`gitglobe teach` deliberately does NOT sample by star count — `stage_teach`'s
docstring is explicit that a top-N sample would teach the student popularity is
the target, which `rubric.FORBIDDEN_IN_PROMPT` exists to prevent. That is
correct for training data and wrong for display coverage: a visitor who opens
`torvalds/linux` should see a summary, and the stratified sample only had a
random chance of including any specific popular repo.

So this is not `gitglobe teach` with a different flag — it is a separate,
deliberately top-N sample, writing to `source=2` (`SOURCE_TEACHER_DISPLAY` in
`api/main.py`) instead of `source=0`, so `gitglobe learn` never trains on it.
Mixing the two into one CLI path risked exactly the bug this script exists to
avoid: someone reaching for `--min-stars` on `teach` two years from now and
quietly poisoning the student's training distribution. Two call sites, one
obviously-labelled purpose each, is the safer shape.

Run: `python coverage_fill.py [--min-stars 10000] [--dry-run]`
Needs the same .env as `gitglobe teach` (NVIDIA_API_KEY or NVIDIA_API_KEYS).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

sys.path.insert(0, "src")

from gitglobe.brain.teacher import Teacher, TeacherConfig, estimate
from gitglobe.db import Database
from gitglobe.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("coverage_fill")

#: See the module docstring on `REPO_SELECT` in `api/main.py` — this must
#: match, or a repo could get scored under a value `learn`/the API don't
#: expect.
SOURCE_TEACHER_DISPLAY = 2


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-stars", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=0, help="0 = provider default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env(require_github=False)
    db = await Database.connect(settings.database_url)
    try:
        await db.migrate()

        # `rows_for_teacher` already applies the two filters that matter here
        # too — not low_signal, has clean_text — so this only adds the star
        # floor and excludes repos already scored under either non-training
        # source, source=0 (a popular repo the stratified sample happened to
        # hit) or source=2 (a previous coverage-fill or on-demand run).
        all_rows = await db.rows_for_teacher()
        already = set(await db.scores(source=Database.TEACHER)) | set(
            await db.scores(source=SOURCE_TEACHER_DISPLAY)
        )
        todo = [r for r in all_rows if r["stars"] >= args.min_stars and r["id"] not in already]
        todo.sort(key=lambda r: r["stars"], reverse=True)

        if not todo:
            log.info("Nothing to do: every repo >= %d stars already has a summary.", args.min_stars)
            return 0

        mean_chars = sum(len(r["clean_text"] or "") for r in todo) / len(todo)
        config = TeacherConfig(**({"concurrency": args.concurrency} if args.concurrency else {}))
        cost = estimate(len(todo), mean_chars, config)
        log.info(
            "%d repos >= %d stars, no summary yet. Model %s, ~%.1f min at ~%.0f RPM (%s).",
            len(todo), args.min_stars, config.model,
            cost["est_minutes"], cost["rate_limit_rpm"],
            "free" if not cost["billed"] else f"${cost['est_usd']}",
        )

        if args.dry_run:
            log.info("Dry run — top 10: %s", [r["full_name"] for r in todo[:10]])
            return 0

        hashes = {r["id"]: r["content_hash"] for r in todo}

        async def save(batch: dict) -> None:
            written = await db.store_scores(
                batch, source=SOURCE_TEACHER_DISPLAY, model=config.model, hashes=hashes
            )
            log.info("Checkpointed %d display summaries (source=%d)", written, SOURCE_TEACHER_DISPLAY)

        async with Teacher(config) as teacher:
            await teacher.rate_many(todo, on_batch=save)

        stats = teacher.stats
        log.info(
            "Done: %d scored, %d failed, %d unparseable, $%.2f.",
            stats.scored, stats.failed, stats.unparseable, stats.cost(),
        )
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
