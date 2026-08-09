"""Phase 2: embed, project, cluster, rank, build.

Five stages, each independently re-runnable and each leaving the database in a
usable state:

    1. embed     capability text -> 768-d vectors     (Vertex AI, costs money)
    2. project   vectors -> positions on S²           (UMAP, costs hours)
    3. cluster   positions -> clusters and domains    (HDBSCAN, costs minutes)
    4. rank      edges -> PageRank                    (numpy, costs seconds)
    5. build     everything -> tiles the browser loads

The order is by cost, descending, and that is deliberate. Stage 1 is the only
one that spends money and it caches on `content_hash`, so a re-run costs nothing
for unchanged rows. Stages 4 and 5 are cheap enough to re-run freely while
tuning, which is where most of the iterating actually happens.

Deliberately NOT a single command. Each stage has failure modes worth looking at
before continuing: a projection that collapsed, clusters that came out as one
blob, a PageRank that did not converge. Chaining them hides all of that behind
one long-running process that either works or does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .db import Database
from .embed.vertex import (
    DEFAULT_DIM,
    EmbedConfig,
    VertexEmbedder,
    estimate,
    pack,
    unpack_matrix,
)
from .graph.pagerank import combine_edges, pagerank
from .project.cluster import N_DOMAINS, cluster, cluster_purity
from .project.spherical import (
    ProjectionParams,
    assess_coverage,
    coverage,
    knn_to_edges,
    project,
)
from .settings import Settings
from .tiles.build import WorldInput, build_world, cluster_manifest_entries

log = logging.getLogger(__name__)

#: Palette slot names. Order is the Uint8 `domain` field in the tile, so
#: reordering this repaints the globe.
DOMAIN_NAMES = [
    "machine learning", "web frontend", "data and storage", "infrastructure",
    "languages and compilers", "systems and embedded", "data engineering",
    "security", "graphics and games", "mobile", "automation and tooling",
    "science and numerics",
]


@dataclass
class StageResult:
    stage: str
    rows: int
    detail: dict


async def stage_embed(
    db: Database,
    settings: Settings,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> StageResult:
    """Embed everything whose vector is missing or stale."""
    filled = await db.backfill_content_hashes()
    if filled:
        log.info("Computed content_hash for %d pre-existing rows", filled)

    pending = await db.rows_needing_embedding(limit)
    if not pending:
        log.info("Nothing to embed — every row is current.")
        return StageResult("embed", 0, {"skipped": True})

    mean_chars = float(np.mean([len(r["embedding_input"] or "") for r in pending]))
    forecast = estimate(len(pending), mean_chars)
    log.info(
        "%d rows to embed, ~%d tokens, ~$%.2f, ~%.1f h online",
        forecast["rows"], forecast["est_tokens"],
        forecast["est_usd_online"], forecast["est_hours_online"],
    )
    if forecast["recommend_batch_prediction"]:
        log.warning(
            "At this size Vertex batch prediction is roughly half the price and "
            "immune to per-minute quota. The online path will still work."
        )
    if dry_run:
        return StageResult("embed", 0, forecast)

    config = EmbedConfig(project=settings.gcp_project, dimensions=DEFAULT_DIM)
    written = 0

    async with VertexEmbedder(config) as embedder:
        async def persist(batch: dict) -> None:
            nonlocal written
            written += await db.store_embeddings(
                {rid: pack(v) for rid, v in batch.items()}, config.dimensions
            )
            log.info("Stored %d/%d", written, len(pending))

        await embedder.embed_many(
            [(r["id"], r["embedding_input"]) for r in pending], on_batch=persist
        )
        log.info(embedder.stats.summary())
        if embedder.stats.failures:
            log.warning("Failures by reason: %s", embedder.stats.failures)

    return StageResult("embed", written, {"pending": len(pending), "dim": config.dimensions})


async def stage_project(
    db: Database,
    *,
    params: ProjectionParams | None = None,
    similar_k: int = 8,
) -> StageResult:
    """Vectors to positions on the sphere, plus the `similar_to` edge layer."""
    rows = await db.embedded_rows()
    if len(rows) < 100:
        raise RuntimeError(f"Only {len(rows)} embedded rows — run `gitglobe embed` first.")

    dim = int(rows[0]["embedding_dim"] or DEFAULT_DIM)
    vectors = unpack_matrix([r["embedding"] for r in rows], dim)
    repo_ids = np.array([r["id"] for r in rows])
    params = params or ProjectionParams()

    run_id = await db.start_projection_run(
        n_points=len(rows), embed_model="gemini-embedding-001",
        embed_dim=dim, umap_params=params.to_dict(), seed=params.seed,
    )

    result = project(vectors, params)

    stats = coverage(result.theta, result.phi)
    problems = assess_coverage(stats)
    for problem in problems:
        log.warning("Coverage: %s", problem)
    log.info("Coverage: %s", stats)

    await db.store_projection(
        {int(rid): (float(t), float(p))
         for rid, t, p in zip(repo_ids, result.theta, result.phi)}
    )

    similar = knn_to_edges(result.knn_indices, result.knn_distances, repo_ids, k=similar_k)
    await db.replace_similar_edges(similar)
    log.info("Stored %d similar_to edges", len(similar))

    await db.finish_projection_run(run_id, notes="; ".join(problems) if problems else "clean")
    return StageResult("project", len(rows), {"coverage": stats, "problems": problems,
                                              "similar_edges": len(similar)})


async def stage_cluster(db: Database, *, min_cluster_size: int = 60, seed: int = 42) -> StageResult:
    """Positions to clusters and the twelve domains the palette indexes."""
    rows = await db.world_rows()
    if not rows:
        raise RuntimeError("Nothing projected — run `gitglobe project` first.")

    theta = np.array([r["theta"] for r in rows], dtype=np.float64)
    phi = np.array([r["phi"] for r in rows], dtype=np.float64)
    result = cluster(theta, phi, min_cluster_size=min_cluster_size,
                     n_domains=N_DOMAINS, seed=seed)
    log.info("Clustering: %s", result.summary())

    await db.store_projection(
        {int(r["id"]): (float(r["theta"]), float(r["phi"])) for r in rows},
        {int(r["id"]): (int(c), int(d))
         for r, c, d in zip(rows, result.cluster_id, result.domain)},
    )

    entries = cluster_manifest_entries(result.cluster_id, theta, phi, result.domain)
    await db.store_clusters(entries)

    # Honest accounting for clustering on the sphere rather than in 768-d.
    embedded = await db.embedded_rows()
    purity = {}
    if embedded:
        dim = int(embedded[0]["embedding_dim"] or DEFAULT_DIM)
        by_id = {int(r["id"]): i for i, r in enumerate(rows)}
        keep = [r for r in embedded if int(r["id"]) in by_id]
        vectors = unpack_matrix([r["embedding"] for r in keep], dim)
        ids = np.array([result.cluster_id[by_id[int(r["id"])]] for r in keep])
        purity = cluster_purity(ids, vectors)
        log.info("Cluster purity: %s", purity)
        if purity.get("lift", 0) < 0.05:
            log.warning(
                "Clusters barely beat the corpus baseline (lift %.3f). The "
                "territories will look arbitrary. Check the cleaner output "
                "before blaming UMAP.", purity.get("lift", 0),
            )

    return StageResult("cluster", len(rows),
                       {"summary": result.summary(), "purity": purity,
                        "clusters": len(result.cluster_sizes)})


async def stage_rank(db: Database, *, used_with_scale: float = 0.7) -> StageResult:
    """PageRank over the union of `depends_on` and `used_with`."""
    rows = await db.world_rows()
    index_of = {int(r["id"]): i for i, r in enumerate(rows)}
    raw = await db.relatedness_edges()

    layers = []
    for kind, scale in ((0, 1.0), (2, used_with_scale)):
        edges = [(index_of[s], index_of[d], w) for s, d, w, k in raw
                 if k == kind and s in index_of and d in index_of]
        if edges:
            src, dst, weight = (np.array(x) for x in zip(*edges))
            layers.append((src, dst, weight, scale))
            log.info("kind=%d: %d edges", kind, len(edges))

    src, dst, weight = combine_edges(layers)
    result = pagerank(len(rows), src, dst, weight)
    log.info(
        "PageRank: %d iterations, delta %.2e, converged=%s",
        result.iterations, result.delta, result.converged,
    )
    if not result.converged:
        log.warning("PageRank did not converge — ranks are provisional.")

    await db.store_ranks({int(r["id"]): float(v) for r, v in zip(rows, result.rank)})
    return StageResult("rank", len(rows), {
        "edges": int(len(src)), "iterations": result.iterations,
        "converged": result.converged,
    })


async def stage_build(db: Database, out_dir: Path, *, seed: int = 42) -> StageResult:
    """Everything in Postgres to the files the browser fetches."""
    rows = await db.world_rows()
    if not rows:
        raise RuntimeError("Nothing to build — run project and cluster first.")

    world = WorldInput(
        repo_id=np.array([r["id"] for r in rows]),
        full_name=np.array([r["full_name"] for r in rows]),
        theta=np.array([r["theta"] for r in rows], dtype=np.float64),
        phi=np.array([r["phi"] for r in rows], dtype=np.float64),
        rank=np.array([r["rank"] or 0.0 for r in rows], dtype=np.float64),
        domain=np.array([r["domain"] if r["domain"] is not None else 0 for r in rows], np.uint8),
        cluster_id=np.array([r["cluster_id"] if r["cluster_id"] is not None else -1 for r in rows],
                            np.int32),
        low_signal=np.array([bool(r["low_signal"]) for r in rows]),
        is_archived=np.array([bool(r["is_archived"]) for r in rows]),
        is_fork=np.array([bool(r["is_fork"]) for r in rows]),
        stars=np.array([r["stars"] or 0 for r in rows]),
    )

    index_of = {int(r["id"]): i for i, r in enumerate(rows)}
    raw = await db.relatedness_edges()
    usable = [(s, d, w) for s, d, w, _ in raw if s in index_of and d in index_of]
    if usable:
        src, dst, weight = (np.array(x) for x in zip(*usable))
    else:
        src = dst = np.zeros(0, np.int64)
        weight = np.zeros(0)

    # Recomputed rather than read back, so the manifest's convergence numbers
    # describe the ranks actually written into this graph.
    ranks = pagerank(
        len(rows),
        np.array([index_of[s] for s in src], np.int64) if len(src) else np.zeros(0, np.int64),
        np.array([index_of[d] for d in dst], np.int64) if len(dst) else np.zeros(0, np.int64),
        weight,
    )

    result = build_world(
        world,
        edges=(src, dst, weight),
        pagerank_result=ranks,
        out_dir=out_dir,
        domains=DOMAIN_NAMES,
        clusters=cluster_manifest_entries(
            world.cluster_id, world.theta, world.phi, world.domain
        ),
        seed=seed,
    )
    log.info("Wrote %.1f MB to %s", result.bytes_written / 1e6, out_dir)
    return StageResult("build", len(rows), {
        "bytes": result.bytes_written,
        "files": [p.name for p in result.files],
    })
