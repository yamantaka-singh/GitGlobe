"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _cmd_ingest(args: argparse.Namespace) -> int:
    from .flow import ingest_flow

    result = await ingest_flow(target=args.target, skip_bigquery=args.skip_bigquery)
    print(f"\nIngested {result.repos:,} repos, {result.packages:,} packages, {result.edges:,} edges")
    return 0


async def _cmd_report(args: argparse.Namespace) -> int:
    from .db import Database
    from .settings import Settings

    db = await Database.connect(Settings.from_env().database_url)
    try:
        report = await db.quality_report()
        repos = report["repos"] or 1

        print("\nGitGlobe ingest — quality report")
        print("=" * 58)
        for key, value in report.items():
            print(f"  {key:<20} {value:>14,}" if isinstance(value, int) else f"  {key:<20} {value:>14}")

        low_pct = 100 * (report["low_signal"] or 0) / repos
        print("\nExit criterion")
        print("-" * 58)
        checks = [
            ("repos ingested", report["repos"], report["repos"] >= args.expect),
            ("with clean_text", report["with_clean_text"], report["with_clean_text"] >= repos * 0.9),
            ("low-signal share", f"{low_pct:.1f}%", low_pct < 25),
            ("mean reduction", report["mean_reduction"], float(report["mean_reduction"] or 0) > 0.35),
        ]
        ok = True
        for label, value, passed in checks:
            print(f"  {'ok  ' if passed else 'FAIL'} {label:<22} {value}")
            ok &= passed

        print("\nThe criterion also requires reading 20 cleaned READMEs by eye.")
        print("Run `gitglobe sample` — green numbers are necessary, not sufficient.")
        return 0 if ok else 1
    finally:
        await db.close()


async def _cmd_sample(args: argparse.Namespace) -> int:
    from .db import Database
    from .settings import Settings

    db = await Database.connect(Settings.from_env().database_url)
    try:
        for row in await db.sample_clean_text(args.count):
            print("\n" + "=" * 74)
            print(f"{row['full_name']}   ({row['clean_reduction']:.0%} removed)")
            print(f"  desc: {row['description'] or '—'}")
            print("-" * 74)
            print(row["excerpt"])
        print(
            "\n\nLooking for: capability prose. Any install commands, licence text,\n"
            "badge residue or lists of OTHER projects means the cleaner needs work —\n"
            "and Phase 2's clusters inherit whatever you accept here."
        )
        return 0
    finally:
        await db.close()


async def _cmd_doctor(args: argparse.Namespace) -> int:
    """One command that says whether a run actually worked.

    Written because the report alone cannot distinguish "ingested 40k because
    the plan ran out" from "ingested 40k because it crashed at 2am" — and those
    need completely different responses.
    """
    import json as _json

    from .db import Database
    from .ingest.plan import plan_for_target
    from .settings import Settings

    out: dict = {}
    try:
        settings = Settings.from_env()
        out["env"] = {
            "tokens": len(settings.github_tokens),
            "gcp_project": settings.gcp_project or None,
            "database": settings.database_url.rsplit("@", 1)[-1],
        }
    except RuntimeError as exc:
        print(f"env: {exc}")
        return 1

    db = await Database.connect(settings.database_url)
    try:
        out["report"] = {k: (float(v) if hasattr(v, "as_integer_ratio") or str(type(v)).endswith("Decimal'>") else v)
                         for k, v in (await db.quality_report()).items()}
        shards = await db.shard_progress()
        done = [s for s in shards if s["completed"]]
        stalled = [s for s in shards if not s["completed"] and s["has_cursor"]]

        plan = plan_for_target(args.expect)
        out["shards"] = {
            "planned": len(plan.queries()),
            "attempted": len(shards),
            "completed": len(done),
            "stalled_midway": len(stalled),
            "rows_per_completed_shard": {
                "min": min((s["rows_seen"] for s in done), default=0),
                "max": max((s["rows_seen"] for s in done), default=0),
                "mean": round(sum(s["rows_seen"] for s in done) / len(done), 1) if done else 0,
            },
            "empty_shards": sum(1 for s in done if s["rows_seen"] == 0),
            "stalled_examples": [s["source"] for s in stalled[:3]],
        }
        out["distribution"] = await db.distribution()
        out["readme_health"] = await db.suspect_readmes()
    finally:
        await db.close()

    def _plain(obj):
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_plain(v) for v in obj]
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if type(obj).__name__ == "Decimal":
            return float(obj)
        return obj

    print(_json.dumps(_plain(out), indent=2))

    # A verdict, not just numbers.
    health = out["readme_health"]
    if health["symlink_like"]:
        print(f"\n!! {health['symlink_like']} rows have a symlink path where their README "
              f"should be.\n   Examples: "
              + ", ".join(f"{e['full_name']} -> {e['readme_raw']!r}" for e in health["examples"][:3])
              + "\n   Fix: `gitglobe reset` then re-run ingest. Upserts overwrite in place.")

    repos = out["report"]["repos"]
    sh = out["shards"]
    print("\n" + "=" * 60)
    if sh["stalled_midway"]:
        print(f"INTERRUPTED — {sh['stalled_midway']} shard(s) stopped mid-page.")
        print("  Re-run the same command; it resumes from the checkpoint.")
    elif sh["completed"] >= sh["planned"] and repos < args.expect:
        print(f"PLAN EXHAUSTED — every shard finished but only {repos:,} repos exist.")
        print("  The star bands hold fewer repos than the ceiling assumed.")
        print("  Widen the plan (lower `low`, or add the language axis).")
    elif repos >= args.expect:
        print(f"COMPLETE — {repos:,} repos ingested.")
    else:
        print(f"IN PROGRESS — {repos:,} of {args.expect:,}; "
              f"{sh['completed']}/{sh['planned']} shards done.")
    return 0


async def _cmd_reset(args: argparse.Namespace) -> int:
    """Clear checkpoints so a re-run re-fetches everything."""
    from .db import Database
    from .settings import Settings

    if not args.yes:
        print("This clears every shard checkpoint, so the next `ingest` re-fetches\n"
              "from scratch. Repo rows are kept and overwritten in place.\n"
              "Re-run with --yes to confirm.")
        return 1

    db = await Database.connect(Settings.from_env().database_url)
    try:
        cleared = await db.reset_checkpoints()
        print(f"Cleared {cleared} shard checkpoints. Re-run `gitglobe ingest`.")
        return 0
    finally:
        await db.close()


def _cmd_clean(args: argparse.Namespace) -> int:
    """Clean a README from stdin or a file. No database, no network."""
    from .clean.readme import clean_readme

    raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    result = clean_readme(raw, name=args.name)
    print(f"# {result.original_chars} -> {result.clean_chars} chars "
          f"({result.reduction:.0%} removed), low_signal={result.low_signal}")
    print(f"# dropped: {', '.join(result.dropped_sections) or '—'}\n")
    print(result.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gitglobe", description="GitGlobe ingest pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Fetch, clean and store repositories")
    p.add_argument("--target", type=int, default=5_000,
                   help="how many repos to ingest (default: 5000, the proof run)")
    p.add_argument("--skip-bigquery", action="store_true",
                   help="GitHub only; no velocity, criticality or dependency edges")
    p.set_defaults(func=_cmd_ingest, is_async=True)

    p = sub.add_parser("report", help="Quality report and exit-criterion check")
    p.add_argument("--expect", type=int, default=5_000)
    p.set_defaults(func=_cmd_report, is_async=True)

    p = sub.add_parser("sample", help="Print cleaned READMEs for manual review")
    p.add_argument("--count", type=int, default=20)
    p.set_defaults(func=_cmd_sample, is_async=True)

    p = sub.add_parser("doctor", help="Full diagnostic: env, shards, distribution, verdict")
    p.add_argument("--expect", type=int, default=100_000)
    p.set_defaults(func=_cmd_doctor, is_async=True)

    p = sub.add_parser("reset", help="Clear shard checkpoints to force a re-fetch")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=_cmd_reset, is_async=True)

    p = sub.add_parser("clean", help="Clean one README (offline)")
    p.add_argument("path", nargs="?", default="-")
    p.add_argument("--name", default="")
    p.set_defaults(func=_cmd_clean, is_async=False)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return asyncio.run(args.func(args)) if args.is_async else args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
