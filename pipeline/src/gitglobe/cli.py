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

    def _plain(obj, depth: int = 0):
        # Power of 10 rule 1 bans recursion; rule 2 wants provable bounds. A
        # hand-rolled stack here would be slower to read than the recursion, so
        # the compromise is an explicit depth ceiling. Report shapes nest three
        # or four deep, and anything past eight is a cycle or a bug — better a
        # clear error than a RecursionError sixty frames down.
        if depth > 8:
            return f"<nesting deeper than 8: {type(obj).__name__}>"
        if isinstance(obj, dict):
            return {k: _plain(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_plain(v, depth + 1) for v in obj]
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


# --------------------------------------------------------------------- phase 2


async def _with_db(fn):
    """Open a pool, run, close. Phase 2 needs no GitHub token."""
    from .db import Database
    from .settings import Settings

    settings = Settings.from_env(require_github=False)
    db = await Database.connect(settings.database_url)
    try:
        await db.migrate()
        return await fn(db, settings)
    finally:
        await db.close()


async def _cmd_embed(args: argparse.Namespace) -> int:
    from .phase2 import stage_embed

    async def run(db, settings):
        if not settings.gcp_project:
            print("Set GCP_PROJECT (or run `gcloud config set project <id>`).", file=sys.stderr)
            return 1
        result = await stage_embed(db, settings, limit=args.limit, dry_run=args.dry_run)
        print(f"\nEmbedded {result.rows:,} rows. {result.detail}")
        return 0

    return await _with_db(run)


async def _cmd_project(args: argparse.Namespace) -> int:
    from .phase2 import stage_project
    from .project.spherical import ProjectionParams

    async def run(db, _settings):
        params = ProjectionParams(
            n_neighbors=args.neighbors, min_dist=args.min_dist, seed=args.seed
        )
        result = await stage_project(db, params=params, similar_k=args.similar_k)
        print(f"\nProjected {result.rows:,} repositories.")
        for problem in result.detail["problems"]:
            print(f"  WARNING: {problem}")
        print(f"  coverage: {result.detail['coverage']}")
        print(f"  similar_to edges: {result.detail['similar_edges']:,}")
        print("\nNext: gitglobe cluster")
        return 1 if result.detail["problems"] else 0

    return await _with_db(run)


def _print_diagnosis(report: dict) -> None:
    """Measurements of the real graph, formatted for reading."""
    import json

    print("\nGraph diagnosis")
    print("=" * 62)
    print(f"  nodes {report['nodes']:,}")
    print("\n  edge weights by kind (raw, before any scaling):")
    for name, stats in report["kinds"].items():
        print(f"    {name:<12} {stats['edges']:>8,} edges   "
              f"min {stats['weight_min']:>8} median {stats['weight_median']:>8} "
              f"p99 {stats['weight_p99']:>9} max {stats['weight_max']:>10}")
    print(f"\n  degree: {json.dumps(report['degree'])}")
    big = report["largest_community"]
    print(f"\n  largest community: {big['members']:,} members "
          f"({big['share_of_corpus']:.0%} of the corpus)")
    print(f"    internal edges  {json.dumps(big['internal_edges_by_kind'])}")
    print(f"    internal weight {json.dumps(big['internal_weight_by_kind'])}")
    print("\n  Whichever kind dominates the internal WEIGHT is what is holding")
    print("  that community together.")


def _print_sweep(edges: str, rows: list) -> None:
    print(f"\nedges: {edges}")
    print(f"\n{'res':>6} {'communities':>12} {'modularity':>11} "
          f"{'median':>7} {'largest':>9} {'<10':>7}")
    print("-" * 58)
    for row in rows:
        print(f"{row['resolution']:>6.2f} {row['communities']:>12,} "
              f"{row['modularity']:>11.4f} {row['median']:>7} "
              f"{row['largest']:>9,} {row['under_10']:>6.0%}")
    print("\n  HIGHER resolution splits, lower merges. If `largest` stays a big")
    print("  share at every value, resolution is not the problem — compare")
    print("  --edges deps against --edges similar to find where structure lives.")
    print("\n  Do NOT compare modularity across rows: Q = internal/2m -")
    print("  resolution x sum(...)^2, so lowering resolution inflates it")
    print("  mechanically. It is only comparable within one resolution.")


async def _cmd_cluster(args: argparse.Namespace) -> int:
    from .phase2 import stage_cluster

    async def run(db, _settings):
        if args.diagnose:
            from .phase2 import diagnose_graph

            _print_diagnosis(await diagnose_graph(db))
            return 0

        if args.sweep:
            from .phase2 import sweep_resolution

            _print_sweep(args.edges, await sweep_resolution(db, edges=args.edges))
            return 0

        result = await stage_cluster(
            db, min_cluster_size=args.min_cluster_size, seed=args.seed,
            method=args.method, resolution=args.resolution, regions=args.regions,
        )
        print(f"\n{result.detail['summary']}")
        print(f"  purity: {result.detail['purity']}")
        sizes = result.detail.get("sizes") or {}
        if sizes:
            ordered = sorted(sizes.values(), reverse=True)
            median = ordered[len(ordered) // 2]
            print(f"  median community: {median} members "
                  f"({sum(1 for s in ordered if s < 10) / len(ordered):.0%} under 10)")
            if median < 5:
                print("  -> mostly pairs, not colonies. Try --resolution 0.4")
        print("\nNext: gitglobe rank")
        return 0

    return await _with_db(run)


async def _cmd_edges(args: argparse.Namespace) -> int:
    from .phase2 import stage_edges

    async def run(db, settings):
        if not settings.gcp_project:
            print("Set GCP_PROJECT — both edge sources are BigQuery public datasets.",
                  file=sys.stderr)
            return 1
        result = await stage_edges(
            db, settings, months=args.months,
            skip_dependencies=args.skip_dependencies,
            skip_costar=args.skip_costar, top_k=args.top_k,
            probe_only=args.probe, max_scan_gb=args.max_scan_gb,
            reuse_packages=not args.rebuild_packages,
        )
        if args.probe:
            tables = result.detail.get("tables") or []
            print(f"\ndeps_dev_v1 tables by size — {len(tables)}\n" + "-" * 70)
            for t in tables:
                size = t.get("size_bytes")
                rows = t.get("row_count")
                if size is None:
                    print(f"  {t.get('table_id')}")
                    continue
                print(f"  {t.get('table_id'):<44} {size / 1024**4:>8.2f} TiB "
                      f"{(rows or 0):>15,} rows")
            print("\n  A full scan costs about $6.25 per TiB. Anything over ~1 TiB")
            print("  needs a reason; over 10 TiB is almost certainly the wrong table.")

            for table, fields in result.detail["schemas"].items():
                print(f"\n{table} — {len(fields)} fields\n" + "-" * 70)
                for f in fields:
                    print(f"  {f.get('field_path','?'):<44} {f.get('data_type','')}")

            sample = result.detail["probe"]
            print(f"\nPackageVersionToProjectLatest sample — {len(sample)} rows\n" + "-" * 70)
            for row in sample[:5]:
                print(f"  System={row.get('System')!r:<12} "
                      f"ProjectType={row.get('ProjectType')!r:<10} "
                      f"ProjectName={row.get('ProjectName')!r}")
            print("\n  Field paths above are exact. Any UNNEST or nested reference in the")
            print("  edge query must match one of them character for character.")
            return 0

        counts = result.detail["counts"]
        print("\nEdges")
        print("-" * 46)
        for key, value in counts.items():
            print(f"  {key:<18} {value:>12,}")
        if counts["rankable_edges"] == 0:
            print("\nStill zero. Nothing to rank — check the BigQuery output above.")
            return 1
        print("\nNext: gitglobe rank && gitglobe build")
        return 0

    return await _with_db(run)


async def _cmd_rank(args: argparse.Namespace) -> int:
    from .phase2 import stage_rank

    async def run(db, _settings):
        result = await stage_rank(db)
        print(f"\nRanked {result.rows:,} repositories over {result.detail['edges']:,} edges "
              f"in {result.detail['iterations']} iterations.")
        if not result.detail["converged"]:
            print("  WARNING: PageRank did not converge — ranks are provisional.")
        print("\nNext: gitglobe build")
        return 0

    return await _with_db(run)


async def _cmd_build(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .phase2 import stage_build

    async def run(db, _settings):
        out = Path(args.out).resolve()
        result = await stage_build(db, out, seed=args.seed)
        print(f"\nWrote {result.rows:,} nodes, {result.detail['bytes'] / 1e6:.1f} MB to {out}")
        print("\nNext:")
        print("  cd ../web && npm run verify     # 47 integrity checks")
        print("  gitglobe spotcheck              # does the map mean anything")
        return 0

    return await _with_db(run)


async def _cmd_spotcheck(args: argparse.Namespace) -> int:
    """The Phase 2 exit criterion.

    Everything else verifies that the pipeline is self-consistent. This is the
    only check that asks whether the result is *correct* — whether repositories
    that belong together ended up together.
    """
    import numpy as np

    from .checks.neighbours import (
        DEFAULT_EXPECTATIONS,
        baseline_distance,
        run_expectations,
        summarise,
    )

    async def run(db, _settings):
        wanted = sorted({
            name
            for exp in DEFAULT_EXPECTATIONS
            for name in (exp.near or []) + (list(exp.far[0]) + list(exp.far[1]) if exp.far else [])
        })
        found = await db.find_repos(wanted)
        positions = {
            row["full_name"]: (float(row["theta"]), float(row["phi"]))
            for row in found.values()
            if row["theta"] is not None
        }

        rows = await db.world_rows()
        baseline = baseline_distance(
            np.array([r["theta"] for r in rows]), np.array([r["phi"] for r in rows])
        ) if rows else None

        print("\nGitGlobe — does the map mean anything?")
        print("=" * 62)
        text, ok = summarise(run_expectations(positions), baseline)
        print(text)

        if not positions:
            print("\n  None of the reference repositories are in this corpus.")
            print("  On a 5k proof run that is expected — the check needs the full ingest.")
            return 0
        print("\n" + ("PASS" if ok else "FAIL — the map does not group things correctly."))
        return 0 if ok else 1

    return await _with_db(run)


async def _cmd_status(args: argparse.Namespace) -> int:
    async def run(db, _settings):
        report = await db.phase2_report()
        edges = await db.edge_counts()

        print("\nGitGlobe — Phase 2 status")
        print("=" * 46)
        for key, value in report.items():
            print(f"  {key:<18} {value:>12,}")
        print("\nGraph")
        print("-" * 46)
        for key, value in edges.items():
            print(f"  {key:<18} {value:>12,}")

        embeddable = report["embeddable"] or 1
        print("\nNext step")
        print("-" * 46)
        if report["stale"]:
            print(f"  gitglobe embed        ({report['stale']:,} rows need embedding)")
        elif report["embedded"] < embeddable * 0.9:
            print("  gitglobe embed        (most rows are not embedded)")
        elif not report["projected"]:
            print("  gitglobe project")
        elif not report["with_domain"]:
            print("  gitglobe cluster")
        elif edges["rankable_edges"] == 0:
            print("  gitglobe edges        <-- NO EDGES. PageRank is meaningless")
            print("                            until this runs; ordering is stars only.")
        else:
            print("  gitglobe rank && gitglobe build")
        return 0

    return await _with_db(run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gitglobe", description="GitGlobe pipeline")
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

    # ---- phase 2 -----------------------------------------------------------

    p = sub.add_parser("embed", help="Embed capability text with Vertex AI (costs money)")
    p.add_argument("--limit", type=int, help="stop after N rows — use this to sanity-check cost")
    p.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    p.set_defaults(func=_cmd_embed, is_async=True)

    p = sub.add_parser("project", help="UMAP embeddings onto the sphere (CPU, slow)")
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--similar-k", type=int, default=8,
                   help="neighbours per repo in the similar_to layer. Mutual "
                        "filtering discards roughly half, so k=8 yields ~2 edges "
                        "per node — sparse enough to shatter the graph into "
                        "components that no clustering can merge. Try 16.")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_project, is_async=True)

    p = sub.add_parser("cluster", help="Semantic regions and the twelve domains")
    p.add_argument("--method", choices=("regions", "communities", "spatial"),
                   default="regions",
                   help="regions = partition the sphere by similarity (default); "
                        "communities = Louvain over the edge graph (what is "
                        "CONNECTED); spatial = HDBSCAN (leaves 51%% as noise)")
    p.add_argument("--regions", type=int, default=400,
                   help="how many semantic neighbourhoods to carve (regions method)")
    p.add_argument("--min-cluster-size", type=int, default=60,
                   help="spatial method only")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Louvain resolution. Below 1.0 gives FEWER, LARGER "
                        "communities; above gives more, smaller ones. At 1.0 on "
                        "an 87k corpus the median community holds 2 members, "
                        "which is a pair rather than a colony — try 0.4.")
    p.add_argument("--sweep", action="store_true",
                   help="try six resolutions and report, writing nothing")
    p.add_argument("--diagnose", action="store_true",
                   help="measure the real graph: weights, degrees, and what the "
                        "largest community is actually made of")
    p.add_argument("--edges", choices=("all", "deps", "similar"), default="all",
                   help="which edge kinds the clusterer sees. Use with --sweep "
                        "to find where structure actually lives: `deps` is hard "
                        "evidence, `similar` is the kNN mesh.")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_cluster, is_async=True)

    p = sub.add_parser("edges", help="Build the relatedness graph (deps.dev + GH Archive)")
    p.add_argument("--months", type=int, default=12,
                   help="co-star window (default: 12; 3 months is far too thin)")
    p.add_argument("--skip-dependencies", action="store_true")
    p.add_argument("--skip-costar", action="store_true")
    p.add_argument("--top-k", type=int, default=12, help="mutual top-k per repo for used_with")
    p.add_argument("--probe", action="store_true",
                   help="print 20 raw deps.dev rows and stop — costs ~nothing, "
                        "verifies the schema before a terabyte-scale scan")
    p.add_argument("--rebuild-packages", action="store_true",
                   help="re-derive the package->repo map (1.4 TiB, ~$9). It is "
                        "cached in Postgres after the first run; only needed "
                        "when the repo set changes materially.")
    p.add_argument("--max-scan-gb", type=float, default=200.0,
                   help="per-query scan ceiling (default: 200). deps.dev package "
                        "tables need ~1400; BigQuery bills per column read, so no "
                        "filter reduces this. ~$6.25/TiB, first TiB free monthly.")
    p.set_defaults(func=_cmd_edges, is_async=True)

    p = sub.add_parser("rank", help="PageRank over depends_on and used_with")
    p.set_defaults(func=_cmd_rank, is_async=True)

    p = sub.add_parser("build", help="Write tiles, graph and manifest for the web app")
    p.add_argument("--out", default="../web/public/tiles")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_build, is_async=True)

    p = sub.add_parser("spotcheck", help="Does the map mean anything? The Phase 2 exit criterion")
    p.set_defaults(func=_cmd_spotcheck, is_async=True)

    p = sub.add_parser("status", help="What is embedded, projected, clustered")
    p.set_defaults(func=_cmd_status, is_async=True)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return asyncio.run(args.func(args)) if args.is_async else args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
