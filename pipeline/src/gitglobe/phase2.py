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
from .project.cluster import ClusterResult, N_DOMAINS, cluster, cluster_purity
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

#: Semantic neighbourhoods carved out of the sphere. Matches the manifest cap,
#: because a region that is never published is a region nobody can reach.
DEFAULT_REGIONS = 400


def _measure_purity(cluster_id: np.ndarray, vectors: np.ndarray) -> dict:
    """Semantic coherence, measured on ISOTROPIC vectors.

    Raw LLM embeddings sit inside a narrow cone: random pairs in this corpus
    score 0.6473, so every real distinction competes for the remaining 0.35 of
    the range. On synthetic data built to that exact baseline, whitening moved
    an UNCHANGED partition's lift from 0.064 to 0.183. The structure was never
    the problem — the ruler was compressed to a third of its length.
    """
    from .embed.whiten import fit as fit_whitener

    whitener = fit_whitener(vectors)
    purity = cluster_purity(cluster_id, whitener.apply(vectors))
    purity["isotropy"] = whitener.to_dict()
    return purity


def _warn_if_incoherent(purity: dict) -> None:
    """Flag weak territories, with a threshold that scales by granularity.

    `lift` falls steeply as groups grow — measured 0.330 at median size 4 down
    to 0.095 at 333, on fixed data with fixed structure. A constant threshold
    therefore fires on large-group partitions that are perfectly healthy and
    stays silent on small-group ones that are trivially tight. The 0.05 that
    shipped before was calibrated against HDBSCAN's granularity and applied to
    everything.
    """
    lift = purity.get("lift", 0.0)
    median = max(int(purity.get("median_size", 1)), 1)
    floor = 0.30 / (1.0 + np.log10(median))
    if lift < floor:
        log.warning(
            "Lift %.3f at median group size %d is below the %.3f expected at "
            "that granularity. Territories may read as arbitrary — check the "
            "cleaner output before blaming UMAP.", lift, median, floor,
        )


async def stage_cluster(
    db: Database,
    *,
    min_cluster_size: int = 60,
    seed: int = 42,
    method: str = "regions",
    resolution: float = 1.0,
    regions: int = DEFAULT_REGIONS,
) -> StageResult:
    """Positions to regions and the twelve domains the palette indexes.

    Three methods, in the order they were tried and what each measured:

    * `regions` (default) — spherical k-means on positions, at two scales.
      Every repository assigned, territories contiguous by construction, no
      noise class. Colour then carries SIMILARITY, which is what the position
      axis already encodes and what the spotcheck verifies.
    * `communities` — Louvain over the edge graph. Measured lift 0.2033 with
      17,685 groups (median size 2) using all edges, or 0.0436 with 44 groups
      using dependencies alone. Dependency communities are close to
      semantically orthogonal: what depends on what is not what resembles what.
    * `spatial` — HDBSCAN on the sphere. Left 51% of the corpus as noise.

    Dependencies are not discarded; they are drawn as ARCS. Asking colour to
    carry them too gave two visual channels one relationship, and they
    disagreed — which is what five successive clustering fixes were actually
    fighting.
    """
    rows = await db.world_rows()
    if not rows:
        raise RuntimeError("Nothing projected — run `gitglobe project` first.")

    theta = np.array([r["theta"] for r in rows], dtype=np.float64)
    phi = np.array([r["phi"] for r in rows], dtype=np.float64)

    if method == "regions":
        result, _ = _region_clusters(theta, phi, seed=seed, regions=regions)
        log.info("Region params: %s", result.params)
    elif method == "communities":
        result = await _community_clusters(
            db, rows, theta, phi, seed=seed, resolution=resolution
        )
    else:
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
        ids = np.array([result.cluster_id[by_id[int(r["id"])]] for r in keep])
        purity = _measure_purity(
            ids, unpack_matrix([r["embedding"] for r in keep], dim)
        )
        log.info("Cluster purity: %s", purity)
        _warn_if_incoherent(purity)

    return StageResult("cluster", len(rows),
                       {"summary": result.summary(), "purity": purity,
                        "clusters": len(result.cluster_sizes),
                        "sizes": result.cluster_sizes})


async def stage_edges(
    db: Database,
    settings: Settings,
    *,
    months: int = 12,
    skip_dependencies: bool = False,
    skip_costar: bool = False,
    top_k: int = 12,
    probe_only: bool = False,
    max_scan_gb: float = 200.0,
    reuse_packages: bool = True,
) -> StageResult:
    """Build the relatedness graph — the part `--skip-bigquery` leaves empty.

    Without this, `depends_on` and `used_with` are both absent, PageRank returns
    the uniform 1/n vector, and node ordering falls back entirely to stars. The
    globe renders, every consistency check passes, and the network the product
    is *about* does not exist.

    Two independent sources, either skippable:

    * **deps.dev** gives `depends_on`. Precise, directed, and blind to any
      repository that does not publish a package — which is most of GitHub.
    * **GH Archive co-stars** gives `used_with`. Behavioural, so it reaches
      exactly the repositories deps.dev cannot see: awesome-lists, dotfiles,
      notebooks, models, anything without a package identity.
    """
    from .graph.cooccurrence import co_occurrence, mutual_top_k, ppmi

    if not settings.gcp_project:
        raise RuntimeError("Set GCP_PROJECT — both edge sources are BigQuery public datasets.")

    from .ingest.bigquery import BigQueryConfig, BigQueryExtractor, trailing_months

    # The ceiling belongs on the command line, not in source. deps.dev's
    # package tables have a genuine terabyte floor — billing is per column read,
    # so no filter reduces it — and editing a constant to get past that is how
    # a cost guard quietly stops guarding.
    extractor = BigQueryExtractor(
        BigQueryConfig(
            project=settings.gcp_project,
            maximum_bytes_billed=int(max_scan_gb * 1024**3),
        )
    )
    detail: dict = {}

    async with db.pool.acquire() as conn:
        names = [r["full_name"] for r in await conn.fetch("SELECT full_name FROM repo")]
    log.info("Building edges over %d repositories", len(names))

    if probe_only:
        try:
            tables = extractor.list_dataset_tables()
        except Exception as exc:  # noqa: BLE001 - a probe must never be fatal
            tables = [{"table_id": f"ERROR: {exc}"}]
        # The per-ecosystem `*Requirements` tables are two to four ORDERS OF
        # MAGNITUDE smaller than `Dependencies` (95 TiB) or
        # `DependencyGraphEdges` (292 TiB): npm 0.97, PyPI 0.06, Cargo 0.02,
        # RubyGems 0.01 TiB. They hold declared requirements rather than the
        # resolved graph, which for a repo-level map is arguably the better
        # signal anyway — direct, intentional dependencies rather than the
        # transitive closure.
        schemas = {}
        for table in (
            "bigquery-public-data.deps_dev_v1.PackageVersionToProjectLatest",
            "bigquery-public-data.deps_dev_v1.NPMRequirementsLatest",
            "bigquery-public-data.deps_dev_v1.PyPIRequirementsLatest",
            "bigquery-public-data.deps_dev_v1.CargoRequirementsLatest",
            "bigquery-public-data.deps_dev_v1.RubyGemsRequirementsLatest",
        ):
            try:
                schemas[table.rsplit(".", 1)[1]] = extractor.describe_table(table)
            except Exception as exc:  # noqa: BLE001 - a probe must not be fatal
                schemas[table.rsplit(".", 1)[1]] = [{"field_path": f"ERROR: {exc}"}]
        return StageResult("edges", 0, {
            "probe": extractor.probe_package_to_repo(),
            "schemas": schemas,
            "tables": tables,
        })

    if not skip_dependencies:
        # The package map costs 1.4 TiB to build, so it is built ONCE and cached
        # in Postgres. `--reuse-packages` skips straight to the edge query,
        # which is the whole point of persisting it.
        package_map = await db.package_repo_map() if reuse_packages else []
        packages = len(package_map)

        if not package_map:
            mappings = extractor.package_to_repo(names)
            packages = await db.upsert_packages(
                (m["ecosystem"], m["package_name"], m["full_name"])
                for m in mappings if m.get("full_name")
            )
            package_map = await db.package_repo_map()
            log.info("Mapped %d packages to repositories", packages)
        else:
            log.info("Reusing %d cached package mappings (no 1.4 TiB rescan)", packages)

        # Both small tables are uploaded, so the only large scan is the four
        # per-ecosystem requirements views — about 1 TiB, against 76 TiB for the
        # resolved-graph tables the cost ceiling refused.
        raw_edges = extractor.dependency_repo_edges_from_map(names, package_map)

        # deps.dev lowercases nothing consistently and GitHub's API preserves
        # owner casing, so the join has to go through a case map. Matching raw
        # strings drops every repo whose owner capitalises differently, and it
        # does so as missing edges rather than an error.
        case = await db.repo_name_case_map()
        resolved, unmatched = [], 0
        for e in raw_edges:
            src = case.get(str(e["src_repo"]).lower())
            dst = case.get(str(e["dst_repo"]).lower())
            if src and dst:
                resolved.append((src, dst, float(e["weight"])))
            else:
                unmatched += 1

        edges = await db.upsert_repo_edges(resolved)
        detail["packages"] = packages
        detail["dependency_edges"] = edges
        detail["unmatched_endpoints"] = unmatched
        log.info(
            "Wrote %d dependency edges (%d of %d dropped — endpoint not in corpus)",
            edges, unmatched, len(raw_edges),
        )
        if raw_edges and edges == 0:
            log.error(
                "deps.dev returned %d edges and NONE matched a repository. "
                "That is a name-format mismatch, not missing data.", len(raw_edges),
            )

    if not skip_costar:
        events = extractor.co_star_events(names, months=trailing_months(months))
        stored = await db.upsert_star_events(
            (e["full_name"], e["actor"], e.get("starred_at")) for e in events
        )
        log.info("Stored %d star events", stored)

        baskets = await db.star_baskets()
        pair_counts, item_counts, total = co_occurrence(baskets)
        scored = ppmi(pair_counts, item_counts, total)
        related = mutual_top_k(scored, k=top_k)
        written = await db.replace_used_with_edges(
            (p.a, p.b, p.ppmi, p.count) for p in related
        )

        # Report the funnel, not just the outcome. "136 edges" tells you nothing
        # about WHERE the signal was lost; these four numbers tell you whether
        # the problem is too few events, baskets too thin to form pairs, the
        # PPMI count threshold, or the mutual-top-k filter.
        mean_basket = sum(len(b) for b in baskets) / max(len(baskets), 1)
        log.info(
            "Co-star funnel: %d baskets (mean %.1f repos) -> %d distinct pairs "
            "-> %d above PPMI threshold -> %d mutual -> %d edges",
            len(baskets), mean_basket, len(pair_counts), len(scored), len(related), written,
        )
        if mean_basket < 4:
            log.warning(
                "Baskets average %.1f repositories. Pairs grow as the SQUARE of "
                "basket size, so thin baskets produce almost no co-occurrence "
                "regardless of how many events there are. Widen --months.",
                mean_basket,
            )
        detail["baskets"] = len(baskets)
        detail["mean_basket"] = round(mean_basket, 2)
        detail["distinct_pairs"] = len(pair_counts)
        detail["used_with_edges"] = written

    counts = await db.edge_counts()
    detail["counts"] = counts
    if counts["rankable_edges"] == 0:
        log.warning(
            "Still zero rankable edges. Both sources returned nothing — check the "
            "BigQuery job logs above rather than continuing to rank."
        )
    return StageResult("edges", counts["rankable_edges"], detail)


def _region_clusters(theta, phi, *, seed: int, regions: int = DEFAULT_REGIONS):
    """Partition the sphere into semantic regions. No graph, no noise class.

    **A map needs regions, not clusters.** Cartography does not discover
    continents by clustering; it partitions. Five attempts at finding
    communities in a graph each fixed the previous symptom and broke something
    else, and the reason was architectural: `colour` was being asked to carry
    the dependency relationship that `arcs` already carry. Two channels, one
    fact, and they disagreed.

    Position is already semantic — UMAP optimised for exactly that, and the
    spotcheck verifies it (pytorch/tensorflow/jax within 0.170 rad, frontend
    2.220 rad from ML). Partitioning that space inherits the property instead
    of trying to rediscover it through a worse lens.

    One algorithm, two scales, so the two levels can never contradict:
      * `regions` neighbourhoods over the points  -> cluster_id
      * 12 domains over THOSE centroids           -> the palette index

    Every repository is assigned, every territory is contiguous by
    construction, and the result is deterministic. Compare the measured
    alternatives: HDBSCAN left 51% of the corpus as noise; Louvain over the
    edge graph produced 17,685 groups with a median size of 2, or 44 groups at
    a purity lift of 0.0436.
    """
    from .project.cluster import spherical_kmeans, to_unit_vectors

    vectors = to_unit_vectors(theta, phi)
    n = len(vectors)
    k = min(regions, max(1, n // 20))
    log.info("Partitioning %d repositories into %d semantic regions", n, k)

    labels, centroids = spherical_kmeans(vectors, k, seed=seed)
    labels = labels.astype(np.int32)

    # Tightness: mean cosine of a region's members to its own centre. This is
    # what "densest neighbourhood" means once the partition exists, and it
    # decides which regions are coherent enough to deserve a name.
    sizes = np.bincount(labels, minlength=len(centroids))
    tightness = np.zeros(len(centroids))
    for i in range(len(centroids)):
        members = labels == i
        if members.any():
            tightness[i] = float(np.mean(vectors[members] @ centroids[i]))

    present = np.where(sizes > 0)[0]
    domain_of, centres = spherical_kmeans(
        centroids[present], N_DOMAINS,
        weights=sizes[present].astype(np.float64), seed=seed,
    )
    lookup = {int(c): int(d) for c, d in zip(present, domain_of)}
    domain = np.array([lookup[int(c)] for c in labels], dtype=np.uint8)

    return ClusterResult(
        cluster_id=labels,
        domain=domain,
        centres=centres,
        cluster_sizes={int(i): int(s) for i, s in enumerate(sizes) if s},
        noise_count=0,
        params={
            "method": "regions",
            "regions": int(len(present)),
            "median_size": int(np.median(sizes[present])),
            "tightness_mean": round(float(tightness[present].mean()), 4),
            "tightness_best": round(float(tightness[present].max()), 4),
            "seed": seed,
        },
    ), tightness


async def _community_clusters(db, rows, theta, phi, *, seed: int, resolution: float = 1.0):
    """Communities from DEPENDENCIES, then twelve spatially coherent domains.

    **`similar_to` is deliberately excluded**, and the measurement that decided
    it is worth keeping. With it included, the largest community held 64,911
    members — 74% of the corpus — and its internals were:

        depends_on:      29 edges
        similar_to: 246,852 edges

    A kNN graph over a UMAP embedding is a mesh on a smooth manifold. Every node
    has k neighbours by construction and there is no dense-inside,
    sparse-between contrast anywhere, which is the only thing modularity can
    detect. Given it, Louvain either melts the corpus into one blob or slices
    the continuum into arbitrary tiles. Neither is a colony.

    Dependencies carry genuine structure but reach only 12,588 repositories.
    The other 74,639 take their community from POSITION, which is the weaker
    claim — "this sits among these" rather than "this is connected to these" —
    and is recorded separately so the difference stays visible.
    """
    from .graph.communities import build_adjacency, detect
    from .project.cluster import to_unit_vectors

    src, dst, weight = await _clustering_edges(db, rows, kinds=(0, 2))
    n = len(rows)
    log.info(
        "Louvain over %d dependency edges spanning %d repositories (resolution %.2f)",
        len(src), n, resolution,
    )
    found = detect(
        n, np.array(src), np.array(dst), np.array(weight),
        resolution=resolution, seed=seed,
    )
    offsets, targets, _ = build_adjacency(n, np.array(src), np.array(dst))
    broken = _report_communities(found, offsets, targets, n, len(src))

    vectors = to_unit_vectors(theta, phi)
    found, adopted = _adopt_isolated(found, vectors, offsets)
    if adopted:
        log.info(
            "%d repositories have no dependency edge; each adopted the community "
            "of its nearest connected neighbour on the sphere", adopted,
        )

    domain, centres = _domains_from_communities(found, vectors, seed)

    return ClusterResult(
        cluster_id=found.labels.astype(np.int32),
        domain=domain,
        centres=centres,
        cluster_sizes=found.sizes,
        noise_count=0,  # Louvain labels everything; nothing is called noise.
        params={"method": "louvain-deps", "modularity": round(found.modularity, 4),
                "levels": found.levels, "disconnected": len(broken),
                "resolution": resolution, "seed": seed,
                "adopted_by_position": adopted},
    )


def _report_graph_shape(offsets, targets, n: int, edges: int, communities: int) -> None:
    """Say whether tuning can help at all, before anyone spends time tuning.

    Modularity optimisation can only group nodes that are CONNECTED — no
    resolution setting merges two components, because there is no edge across
    which to measure a gain. When the component count approaches the community
    count, the communities ARE the components and the parameter is irrelevant.
    """
    from .graph.communities import connected_components

    components, largest, isolated = connected_components(offsets, targets)
    log.info(
        "Graph shape: %d connected components, largest holds %d (%.1f%%), "
        "%d nodes with no edges at all",
        components, largest, 100 * largest / max(n, 1), isolated,
    )
    if components > communities * 0.5:
        log.warning(
            "%d components vs %d communities — the communities ARE largely the "
            "components, so no `--resolution` value can merge them. The graph is "
            "too sparse at %.1f edges per node. `similar_to` is filtered to "
            "MUTUAL top-k, which discards roughly half the kNN graph; re-run "
            "`gitglobe project --similar-k 16` (about two minutes) to densify it.",
            components, communities, edges / max(n, 1),
        )


#: Resolutions to try. Spans BOTH directions on purpose.
#:
#: Higher resolution splits; lower merges. An earlier version of this swept only
#: downward from 1.0, which is exactly wrong for a graph that is already
#: collapsed — every value offered made the blob bigger. Modularity also cannot
#: resolve communities smaller than about sqrt(2m) nodes, a well-known limit, and
#: raising the resolution is the standard way past it.
SWEEP_RESOLUTIONS = (20.0, 10.0, 5.0, 2.0, 1.0, 0.5)

#: Which edge kinds to feed the clusterer, for isolating where structure lives.
EDGE_SOURCES = {
    "all": None,
    "deps": (0, 2),   # depends_on + used_with: hard evidence only
    "similar": (1,),  # the kNN mesh alone
}


async def sweep_resolution(
    db,
    values: tuple = SWEEP_RESOLUTIONS,
    *,
    edges: str = "all",
) -> list[dict]:
    """Try several resolutions and report, writing nothing.

    Louvain runs in seconds, so guessing a value, writing it, looking at the
    globe and guessing again is a slower loop than trying six. Nothing is
    persisted — this only says which value to pass.

    `edges` isolates where community structure actually comes from. A kNN mesh
    is lattice-like: every node has k neighbours by construction, so it connects
    everything without creating the dense-inside/sparse-between contrast that
    modularity looks for. If `deps` shows structure and `all` does not, the mesh
    is drowning it and the fix is weighting, not resolution.
    """
    from .graph.communities import detect

    rows = await db.world_rows()
    src, dst, weight = await _clustering_edges(db, rows, kinds=EDGE_SOURCES[edges])
    n = len(rows)
    src_a, dst_a, w_a = np.array(src), np.array(dst), np.array(weight)
    log.info("Sweeping over %d edges (%s) spanning %d repositories", len(src), edges, n)

    out = []
    for resolution in values:
        found = detect(n, src_a, dst_a, w_a, resolution=resolution, seed=42)
        sizes = sorted(found.sizes.values(), reverse=True)
        median = sizes[len(sizes) // 2] if sizes else 0
        out.append({
            "resolution": resolution,
            "communities": found.count,
            "modularity": round(found.modularity, 4),
            "median": median,
            "largest": sizes[0] if sizes else 0,
            "under_10": round(sum(1 for s in sizes if s < 10) / max(len(sizes), 1), 3),
        })
        log.info("resolution %.2f -> %s", resolution, out[-1])
    return out


def _weight_stats(by_kind: dict) -> dict:
    """Weight distribution per edge kind.

    The spread matters more than the mean: if one kind's p99 is three orders of
    magnitude above another's median, the two are not commensurable and any
    algorithm summing them is really only seeing one of them.
    """
    names = {0: "depends_on", 1: "similar_to", 2: "used_with"}
    out = {}
    for k, weights in sorted(by_kind.items()):
        w = np.array(weights)
        out[names.get(k, str(k))] = {
            "edges": len(w),
            "weight_min": round(float(w.min()), 3),
            "weight_median": round(float(np.median(w)), 3),
            "weight_p99": round(float(np.percentile(w, 99)), 3),
            "weight_max": round(float(w.max()), 3),
        }
    return out


async def diagnose_graph(db) -> dict:
    """Measure the real clustering graph. No modelling, no guessing.

    Written after two synthetic hypotheses — dependency hubs bridging
    neighbourhoods, and incommensurable edge weights — both failed to reproduce
    the collapse. When a model repeatedly fails to reproduce the symptom, the
    model is wrong and the thing to do is instrument the real system.

    The number that matters most is the composition of the LARGEST community:
    what is actually holding 55% of the corpus together, and via which edges.
    """
    from .graph.communities import build_adjacency, detect

    rows = await db.world_rows()
    n = len(rows)
    index_of = {int(r["id"]): i for i, r in enumerate(rows)}

    async with db.pool.acquire() as conn:
        raw = await conn.fetch("SELECT src, dst, weight, kind FROM edge")

    by_kind: dict = {}
    src, dst, weight, kind = [], [], [], []
    for row in raw:
        a, b = index_of.get(int(row["src"])), index_of.get(int(row["dst"]))
        if a is None or b is None:
            continue
        k = int(row["kind"])
        w = float(row["weight"] or 1.0)
        by_kind.setdefault(k, []).append(w)
        src.append(a)
        dst.append(b)
        kind.append(k)
        weight.append(w * (0.4 if k == 1 else 1.0))

    report: dict = {"nodes": n, "kinds": _weight_stats(by_kind)}
    names = {0: "depends_on", 1: "similar_to", 2: "used_with"}
    src_a, dst_a = np.array(src), np.array(dst)
    kind_a, w_a = np.array(kind), np.array(weight)
    found = detect(n, src_a, dst_a, w_a, resolution=5.0, seed=42)

    # What is the biggest community actually made of?
    sizes = found.sizes
    biggest = max(sizes, key=sizes.get)
    member = found.labels == biggest
    internal = member[src_a] & member[dst_a]
    report["largest_community"] = {
        "members": int(member.sum()),
        "share_of_corpus": round(float(member.sum()) / max(n, 1), 3),
        "internal_edges_by_kind": {
            names.get(k, str(k)): int(((kind_a == k) & internal).sum())
            for k in sorted(by_kind)
        },
        "internal_weight_by_kind": {
            names.get(k, str(k)): round(float(w_a[(kind_a == k) & internal].sum()), 1)
            for k in sorted(by_kind)
        },
    }

    degree = np.bincount(np.concatenate([src_a, dst_a]), minlength=n)
    report["degree"] = {
        "mean": round(float(degree.mean()), 2),
        "median": int(np.median(degree)),
        "p99": int(np.percentile(degree, 99)),
        "max": int(degree.max()),
        "zero": int((degree == 0).sum()),
    }
    return report


async def _clustering_edges(db, rows, kinds=None) -> tuple[list, list, list]:
    """Every edge kind, remapped to row indices and weighted by evidence.

    Dependencies are hard evidence: a maintainer wrote the dependency down.
    kNN similarity is softer — it is an inference from a projection — so it is
    down-weighted to 0.4. That is enough for it to group the 86% of the corpus
    with no package identity, without overwhelming the structure the dependency
    graph actually knows about.
    """
    index_of = {int(r["id"]): i for i, r in enumerate(rows)}
    async with db.pool.acquire() as conn:
        raw = await conn.fetch("SELECT src, dst, weight, kind FROM edge")

    src, dst, weight = [], [], []
    for row in raw:
        if kinds is not None and row["kind"] not in kinds:
            continue
        a, b = index_of.get(int(row["src"])), index_of.get(int(row["dst"]))
        if a is None or b is None:
            continue
        src.append(a)
        dst.append(b)
        weight.append(float(row["weight"] or 1.0) * (0.4 if row["kind"] == 1 else 1.0))
    return src, dst, weight


def _report_communities(found, offsets, targets, n: int, edges: int) -> list:
    """Log what Louvain found and how trustworthy it is. Returns broken ones."""
    from .graph.communities import disconnected_communities

    broken = disconnected_communities(found.labels, offsets, targets)
    log.info(
        "%s; %d internally disconnected (%.1f%%) — the defect Leiden fixes",
        found.summary(), len(broken), 100 * len(broken) / max(found.count, 1),
    )
    _report_graph_shape(offsets, targets, n, edges, found.count)
    return broken


def _domains_from_communities(found, vectors: np.ndarray, seed: int):
    """Twelve palette slots, assigned over community CENTRES.

    Weighted by community size and computed on centroids rather than on every
    point: k-means over all 87,000 nodes would let one crowded region claim
    several of the twelve colours purely for being crowded, leaving sparse
    regions of the map sharing one.
    """
    from .project.cluster import spherical_kmeans

    real = np.array(sorted(found.sizes))
    centroids = np.array([_unit_mean(vectors[found.labels == c]) for c in real])
    weights = np.array([found.sizes[int(c)] for c in real], dtype=np.float64)
    community_domain, centres = spherical_kmeans(
        centroids, N_DOMAINS, weights=weights, seed=seed
    )
    domain_of = {int(c): int(d) for c, d in zip(real, community_domain)}
    return np.array([domain_of[int(c)] for c in found.labels], dtype=np.uint8), centres


def _adopt_isolated(found, vectors: np.ndarray, offsets: np.ndarray):
    """Give every unconnected repository the community of its nearest neighbour.

    86% of the corpus publishes no package, so the dependency graph cannot see
    it. Leaving those repositories as singleton communities would be *honest*
    but useless — 74,000 one-member territories carry no information and cannot
    be coloured or labelled.

    Position is the fallback, and it is a weaker claim than a discovered
    community: it says "this sits among these", not "this is connected to
    these". Recording it separately in `params` keeps that distinction visible
    rather than letting geometry masquerade as connectivity.
    """
    connected = np.diff(offsets) > 0
    if connected.all() or not connected.any():
        return found, 0

    labels = found.labels.copy()
    anchors = np.where(connected)[0]
    orphans = np.where(~connected)[0]

    # Nearest connected repository by cosine on the unit sphere, in blocks so
    # a 75k x 12k similarity matrix never exists all at once.
    block = 4096
    for start in range(0, len(orphans), block):
        chunk = orphans[start:start + block]
        nearest = np.argmax(vectors[chunk] @ vectors[anchors].T, axis=1)
        labels[chunk] = labels[anchors[nearest]]

    counts = np.bincount(labels)
    return (
        type(found)(
            labels=labels,
            modularity=found.modularity,
            levels=found.levels,
            sizes={int(i): int(c) for i, c in enumerate(counts) if c},
        ),
        len(orphans),
    )


def _unit_mean(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-9 else vectors[0]


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

    if len(src) == 0:
        # PageRank over an empty graph is the uniform vector. It sums to 1, every
        # rank is positive, and it converges in one iteration — so nothing
        # downstream complains while the ranking carries no information at all.
        # Say so here, where it is fixable, rather than letting it surface as
        # `max/min = 1x` in a verifier three commands later.
        log.error(
            "ZERO edges. PageRank will return 1/n for every repository and node "
            "ordering will fall back entirely to stars. Run `gitglobe edges` "
            "first — this is what `--skip-bigquery` leaves undone."
        )

    result = pagerank(len(rows), src, dst, weight)
    log.info(
        "PageRank: %d iterations, delta %.2e, converged=%s",
        result.iterations, result.delta, result.converged,
    )
    if not result.converged:
        log.warning("PageRank did not converge — ranks are provisional.")
    elif len(src) and result.rank.max() / max(result.rank.min(), 1e-30) < 10:
        log.warning(
            "Ranks span only %.1fx. A real dependency graph is power-law and "
            "should span hundreds. The edge set is probably too sparse to rank.",
            result.rank.max() / max(result.rank.min(), 1e-30),
        )

    await db.store_ranks({int(r["id"]): float(v) for r, v in zip(rows, result.rank)})
    return StageResult("rank", len(rows), {
        "edges": int(len(src)), "iterations": result.iterations,
        "converged": result.converged,
    })


async def stage_calibrate(db: Database, *, tokens: list | None = None,
                          remeasure: bool = False) -> StageResult:
    """Rank the corpus against all of GitHub, not just against itself.

    `stage_rank` produces PageRank, which answers "how important is this among
    the repositories we happen to have". This answers "how does this stand among
    all ~420 million public repositories" — the question a user actually has, and
    the one a corpus-relative number cannot address as the corpus keeps growing.

    The star scale is a MEASUREMENT from the GitHub search API, so it is stored
    as a run and reused. `--remeasure` takes a fresh one; without it the most
    recent stored scale is used, which keeps re-runs free and scores comparable.
    """
    from .rank.calibrate import measure_star_scale
    from .rank.corpus import disagreement, leaderboard, rank_corpus
    from .rank.global_scale import StarScale

    stored = await db.latest_star_scale()
    if remeasure or stored is None:
        if not tokens:
            raise ValueError(
                "measuring the star scale needs a GitHub token (GITHUB_TOKEN). "
                "Run once with --remeasure and a token; later runs reuse it."
            )
        log.info("Measuring the global star distribution from the GitHub search API")
        scale = await measure_star_scale(tokens)
        scale_id = await db.store_star_scale(
            scale.thresholds, scale.counts, total_repos=scale.total_repos,
            notes="gitglobe calibrate --remeasure",
        )
    else:
        scale = StarScale(
            thresholds=list(stored["thresholds"]), counts=list(stored["counts"]),
            total_repos=int(stored["total_repos"]), measured_at=str(stored["measured_at"]),
        )
        scale_id = int(stored["id"])
        log.info("Reusing star scale #%d measured %s", scale_id, scale.measured_at)

    rows = await db.world_rows()
    ranking = rank_corpus(rows, await db.relatedness_edges(), scale)
    written = await db.store_global_ranks(ranking.ranks, scale_id)

    log.info("Global rank: %s", ranking.summary())
    return StageResult("calibrate", written, {
        "scale_id": scale_id,
        "measured_at": str(scale.measured_at),
        "summary": ranking.summary(),
        "top": leaderboard(ranking, rows, top=10),
        "movers": disagreement(ranking, rows, top=8),
    })


async def stage_teach(db: Database, *, total: int = 4_000, provider: str = "nim",
                      project: str = "", model: str = "", seed: int = 42,
                      concurrency: int = 0, dry_run: bool = False) -> StageResult:
    """Have an LLM rate a stratified sample, so the student has something to learn.

    Everything downstream of the brain has existed for a while — the rubric, the
    prompt guard, the sampler, the student and its 24 tests — and none of it has
    ever run, because nothing produced labels. This is that missing step.

    **Stratified, not top-N.** Rating the 4,000 most popular repositories would
    teach the student that popularity is the target, which is precisely what
    `rubric.FORBIDDEN_IN_PROMPT` exists to prevent. The sampler spreads the
    budget across star band x domain x recency so the student sees quiet, good
    software too.

    **Resumable.** The sample is drawn from the full population with a fixed
    seed, then rows already rated are removed. An interrupted run therefore
    resumes into the same sample rather than drawing a fresh one from what is
    left, and `rate_many` checkpoints every 200 rows regardless.
    """
    from .brain.sampling import plan_teaching
    from .brain.teacher import Teacher, TeacherConfig, estimate

    rows = await db.rows_for_teacher()
    if not rows:
        return StageResult("teach", 0, {"note": "no rows with clean text; run ingest"})

    already = await db.scores(source=Database.TEACHER)
    todo, sample = plan_teaching(rows, set(already), total=total, seed=seed)
    mean_chars = sum(len(r["clean_text"] or "") for r in todo) / max(len(todo), 1)
    # Built before the dry-run branch so the estimate reflects the provider that
    # would actually run. Estimating with defaults quoted Vertex pricing and
    # Vertex throughput for a NIM run: $3.19 for something free, 2.8 minutes for
    # something that takes hours.
    config = TeacherConfig(provider=provider, project=project, model=model or "",
                           **({"concurrency": concurrency} if concurrency else {}))
    cost = estimate(len(todo), mean_chars, config)

    if dry_run or not todo:
        return StageResult("teach", 0, {
            "sample": sample.summary(), "already": len(already),
            "todo": len(todo), "estimate": cost, "dry_run": True,
        })

    run_id = await db.start_brain_run(teacher_model=config.model, teacher_n=len(todo))
    hashes = {r["id"]: r["content_hash"] for r in todo}

    async def save(batch: dict) -> None:
        # Persist as we go. At 40 requests/minute a 4,000-row pass is over an
        # hour, and losing it to a dropped connection would be unforgivable.
        written = await db.store_scores(batch, source=Database.TEACHER,
                                        model=config.model, hashes=hashes)
        log.info("Checkpointed %d teacher scores", written)

    async with Teacher(config) as teacher:
        await teacher.rate_many(todo, on_batch=save)

    stats = teacher.stats
    await db.finish_brain_run(run_id, metrics={
        "scored": stats.scored, "unparseable": stats.unparseable,
        "failed": stats.failed, "usd": round(stats.cost(), 2),
        "flags": stats.flags, "failures": stats.failures,
    }, notes="gitglobe teach")

    return StageResult("teach", stats.scored, {
        "sample": sample.summary(), "already": len(already),
        "summary": stats.summary(), "run_id": run_id,
        "flags": stats.flags, "failures": stats.failures,
    })


def _train_students(values, names: list, repo_ids, rows: list, labels: dict,
                    labelled: list, *, seed: int) -> dict:
    """Fit one student per rubric dimension on the labelled subset.

    Split out because rule 4 refused `stage_learn` at 89 lines, and this is the
    half with independent meaning: everything here is pure given the feature
    matrix, so the training can be reasoned about without the database.
    """
    from datetime import datetime, timezone

    import numpy as np

    from .brain.rubric import DIMENSION_KEYS
    from .brain.sampling import stratify, train_test_split
    from .brain.student import fit

    by_dimension = {
        key: np.array([
            labels[int(repo_ids[i])].get(key) or 0.0 for i in labelled
        ], dtype=np.float64)
        for key in DIMENSION_KEYS
    }

    # Stratified on the real cells, not a placeholder. A plain random split
    # leaves the held-out set with a different mix of star band x domain x
    # recency from the training set by chance, and the held-out RMSE then
    # partly measures that difference — flattering or damning the student at
    # random between runs, which is exactly what `beats_baseline` must not be.
    now = datetime.now(timezone.utc)
    picked = [rows[i] for i in labelled]
    strata = stratify(
        np.array([r.get("stars") or 0 for r in picked], dtype=np.float64),
        np.array([r.get("domain") or 0 for r in picked], dtype=np.int64),
        np.array([
            (now - r["pushed_at"]).days if r.get("pushed_at") else 3650
            for r in picked
        ], dtype=np.float64),
    )
    train_local, test_local = train_test_split(
        np.arange(len(labelled)), strata, seed=seed
    )
    return fit(values[labelled], by_dimension, names,
               train_idx=train_local, test_idx=test_local, seed=seed)


def _column(rows: list, key: str):
    """Optional numeric column as float64, missing values as NaN."""
    import numpy as np

    return np.array(
        [float(r[key]) if r.get(key) is not None else np.nan for r in rows],
        dtype=np.float64,
    )


async def _store_predictions(db: Database, honest: dict, values, names: list, repo_ids,
                             *, metrics: dict, labelled: int, seed: int) -> tuple:
    """Predict for the whole corpus and persist, recording the run.

    Only the dimensions in `honest` are written. A dimension that failed its
    baseline check is absent from the row rather than stored as a null or a
    default — "we could not judge this" and "this scored average" are different
    claims, and the second one is a lie the globe would render confidently.
    """
    predictions: dict = {int(rid): {} for rid in repo_ids}
    for key, student in honest.items():
        for rid, value in zip(repo_ids, student.predict(values)):
            predictions[int(rid)][key] = float(value)

    run_id = await db.start_brain_run(teacher_n=labelled, feature_names=list(names))
    written = await db.store_scores(predictions, source=Database.STUDENT,
                                    model=f"student-gbm-seed{seed}")
    await db.finish_brain_run(run_id, metrics=metrics, student_n=written,
                              notes="gitglobe learn")
    return written, run_id


async def stage_learn(db: Database, *, seed: int = 11, min_labels: int = 200) -> StageResult:
    """Train the student on the teacher's labels, then score the whole corpus.

    The teacher reads READMEs and costs money per repository; the student reads
    cheap structural features and costs nothing, so it is what actually scores
    all 87k. Distillation only works if the student is measurably better than
    guessing the mean, which is why nothing is stored until `beats_baseline`
    has been checked per dimension — a model that learned nothing would
    otherwise write 87,227 confident numbers that are all the same.

    The popularity blindfold is enforced twice: `rubric` keeps stars out of the
    teacher's prompt, and `assert_no_popularity_features` keeps them out of the
    student's inputs. Either alone is insufficient — a student given `stars`
    would rediscover popularity no matter how clean the labels are.
    """
    import numpy as np

    from .brain.features import GraphFeatures, build_features
    from .brain.student import blindfold, composite

    labels = await db.scores(source=Database.TEACHER)
    if len(labels) < min_labels:
        return StageResult("learn", 0, {
            "note": f"only {len(labels)} teacher labels (need {min_labels}); "
                    f"run `gitglobe teach` first",
        })

    rows = await db.brain_rows()
    rank, in_deg, out_deg, similar = await db.graph_features()
    matrix = build_features(rows, graph=GraphFeatures(rank, in_deg, out_deg, similar))
    # Drop popularity columns ONCE, and use the result for both training and
    # prediction. `build_features` emits log_stars, log_forks, stars_per_day,
    # log_pagerank and criticality because the globe and the global rank need
    # them; the student is the one consumer that must not see them, or the
    # teacher's blindfold bought nothing. Both `fit` and `Student.predict`
    # refuse a mismatched matrix, so filtering in one place and not the other
    # fails loudly rather than silently scoring on the wrong columns.
    safe, safe_names = blindfold(matrix.values, matrix.names)

    # Positions of the labelled rows within the full matrix. The student trains
    # on these and predicts for everything.
    position = {int(rid): i for i, rid in enumerate(matrix.repo_ids)}
    labelled = [position[r] for r in labels if r in position]
    if len(labelled) < min_labels:
        return StageResult("learn", 0, {
            "note": f"{len(labels)} labels but only {len(labelled)} match the corpus",
        })

    students = _train_students(safe, safe_names, matrix.repo_ids, rows, labels,
                               labelled, seed=seed)
    honest = {k: s for k, s in students.items() if s.holdout["beats_baseline"]}
    metrics = {k: s.holdout for k, s in students.items()}
    if not honest:
        return StageResult("learn", 0, {
            "note": "no dimension beat its baseline by more than sampling noise; "
                    "nothing stored",
            "metrics": metrics, "trained": len(students),
        })

    written, run_id = await _store_predictions(
        db, honest, safe, safe_names, matrix.repo_ids,
        metrics=metrics, labelled=len(labelled), seed=seed,
    )
    overall = composite(honest, safe)
    return StageResult("learn", written, {
        "labels": len(labelled), "features": len(safe_names),
        "dropped_features": len(matrix.names) - len(safe_names),
        "trained": len(students), "kept": sorted(honest),
        "dropped": sorted(set(students) - set(honest)),
        "metrics": metrics, "run_id": run_id,
        "composite": {"best": float(overall.max()), "median": float(np.median(overall)),
                      "worst": float(overall.min())},
    })


def _world_input(rows: list) -> "WorldInput":
    """Postgres rows to the tile writer's input. Split out for rule 4."""
    return WorldInput(
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
        # NaN, not 0.0, when a repository has no score: the writer turns NaN
        # into JSON null so the panel can say "not scored" instead of showing
        # a confident zero for something nothing ever judged.
        global_score=_column(rows, "global_score"),
        star_rank=_column(rows, "star_rank"),
        brain_score=_column(rows, "brain_score"),
    )


async def stage_build(db: Database, out_dir: Path, *, seed: int = 42) -> StageResult:
    """Everything in Postgres to the files the browser fetches."""
    rows = await db.world_rows()
    if not rows:
        raise RuntimeError("Nothing to build — run project and cluster first.")

    world = _world_input(rows)

    index_of = {int(r["id"]): i for i, r in enumerate(rows)}
    raw = await db.relatedness_edges()
    usable = [(s, d, w) for s, d, w, _ in raw if s in index_of and d in index_of]
    if not usable:
        log.error(
            "Building a world with NO EDGES. The globe will render, but there are "
            "no arcs, no backbone, and every node has identical rank. "
            "`gitglobe edges` is the missing step."
        )
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
