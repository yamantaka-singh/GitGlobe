"""BigQuery extractors — GH Archive and deps.dev.

Two public datasets, two very different jobs:

* **GH Archive** (`githubarchive.month.*`) is the event firehose since 2011. We
  want star *velocity*, not totals: a repository that gained 3,000 stars in the
  last quarter is alive in a way that a 2015 project sitting on 40,000 is not,
  and the GitHub API only exposes the total.

* **deps.dev** (`bigquery-public-data.deps_dev_v1`) is Google's resolved
  dependency graph across npm, PyPI, Go, Maven, Cargo and NuGet. Resolved is the
  operative word — it accounts for version ranges and transitive pins, which
  parsing manifests yourself does not.

Every query here is cost-bounded and says so. BigQuery bills on bytes scanned,
and `SELECT *` against GH Archive is how people discover that the hard way:
the full history is well over 20 TB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# GH Archive
# --------------------------------------------------------------------------- #

# Partitioned by month, so the FROM clause is generated to touch only the months
# we need. `_TABLE_SUFFIX` filtering is what keeps this from scanning years.
STAR_VELOCITY_SQL = """
SELECT
  repo.name              AS full_name,
  COUNT(*)               AS stars_90d
FROM `githubarchive.month.{suffix_pattern}`
WHERE _TABLE_SUFFIX BETWEEN '{start_month}' AND '{end_month}'
  AND type = 'WatchEvent'
  AND repo.name IN UNNEST(@repo_names)
GROUP BY full_name
"""

# Co-starring. The behavioural signal, and the one that covers repositories the
# dependency graph cannot see at all — awesome-lists, dotfiles, model repos,
# notebooks, anything without a package identity.
#
# Bounded on both sides deliberately:
#   - actors with more than `max_stars` are dropped. Someone who starred 5,000
#     repositories is browsing, not choosing, and would contribute 12.5M pairs.
#   - only repositories we actually hold are counted, so the join stays small.
CO_STAR_SQL = """
WITH events AS (
  SELECT
    actor.login AS actor,
    repo.name   AS full_name,
    DATE(created_at) AS starred_at
  FROM `githubarchive.month.*`
  WHERE _TABLE_SUFFIX BETWEEN '{start_month}' AND '{end_month}'
    AND type = 'WatchEvent'
    AND repo.name IN UNNEST(@repo_names)
),
deduped AS (
  SELECT actor, full_name, MIN(starred_at) AS starred_at
  FROM events GROUP BY actor, full_name
),
sized AS (
  SELECT actor, COUNT(*) AS n FROM deduped GROUP BY actor
)
SELECT d.actor, d.full_name, d.starred_at
FROM deduped d
JOIN sized s USING (actor)
WHERE s.n BETWEEN @min_stars AND @max_stars
"""

# --------------------------------------------------------------------------- #
# deps.dev
# --------------------------------------------------------------------------- #

# Package -> source repository. Needed because deps.dev edges are between
# packages, while GitGlobe's nodes are repositories.
#
# **Use the `...Latest` views, never `SELECT MAX(SnapshotAt)` as a subquery.**
# deps.dev keeps every historical snapshot in the base tables. A correlated
# subquery for the newest one cannot be resolved before the scan, so BigQuery
# reads the entire history to answer it: measured at 1,378 GB on
# PackageVersionToProject, against ~10 GB for the equivalent Latest view. The
# views exist precisely for this and are documented at
# https://docs.deps.dev/bigquery/v1/.
#
# `ProjectName` is already `owner/repo` — there is NO `github.com/` prefix. The
# host lives in `ProjectType`. Two earlier versions of this query tried to strip
# a prefix that was never there; both returned zero rows from perfectly valid
# SQL, which is indistinguishable from "deps.dev has no data for you". The first
# cost 34 minutes and 1.4 TB to discover.
#
# `probe_package_to_repo` answers this in two seconds for nothing. Verify the
# shape of someone else's data before writing a filter against it.
PACKAGE_TO_REPO_SQL = """
SELECT DISTINCT
  LOWER(System)      AS ecosystem,
  Name               AS package_name,
  LOWER(ProjectName) AS full_name
FROM `bigquery-public-data.deps_dev_v1.PackageVersionToProjectLatest`
WHERE ProjectType = 'GITHUB'
  AND ProjectName IS NOT NULL
"""

# Direct dependencies only. Transitive edges explode the graph by roughly two
# orders of magnitude and add nothing a viewer can read — the arc layer draws at
# most a couple of thousand edges regardless.
# `DependenciesLatest` is ALREADY FLATTENED: one row per (package-version,
# dependency) pair, with `Dependency` a single STRUCT<System, Name, Version> —
# not a repeated field. There is nothing to UNNEST, and depth lives in
# `MinimumDepth`, not `Distance`.
#
# Confirmed against INFORMATION_SCHEMA.COLUMN_FIELD_PATHS, which costs nothing
# and reports nested field paths exactly. Guessing this shape cost five wrong
# attempts and about forty minutes of query time before anyone thought to look.
DEPENDENCY_EDGES_SQL = """
SELECT
  LOWER(d.System)        AS ecosystem,
  d.Name                 AS src_package,
  d.Dependency.Name      AS dst_package,
  COUNT(*)               AS weight
FROM `bigquery-public-data.deps_dev_v1.DependenciesLatest` d
WHERE d.MinimumDepth = 1
  AND d.Name IN UNNEST(@package_names)
  AND d.Dependency.Name IN UNNEST(@package_names)
GROUP BY ecosystem, src_package, dst_package
"""

#: BigQuery's jobs.insert request body limit is 10 MB. Query parameters are
#: serialised into it, so a large array silently becomes a 413 with an HTML
#: error page rather than anything actionable.
MAX_PARAMETER_BYTES = 8 * 1024 * 1024

# Repo -> repo edges from the PER-ECOSYSTEM requirements tables.
#
# `Dependencies` is 95 TiB and `DependencyGraphEdges` is 292 TiB — both hold the
# fully resolved transitive graph for every package version ever published. The
# per-ecosystem `*Requirements` tables hold the same *direct* relationships for
# a thousandth of the bytes: npm 0.97, PyPI 0.06, Cargo 0.02, RubyGems 0.01 TiB.
#
# Declared requirements rather than a resolved graph is arguably the better
# signal for a repo-level map anyway: it is what a maintainer chose to depend
# on, not the closure that choice implies.
#
# **Every ecosystem names its fields differently.** npm has
# `Dependencies.Name`, PyPI has `Dependencies.ProjectName`, RubyGems calls the
# array `RuntimeDependencies`, Cargo carries a `Kind`. All four were read from
# INFORMATION_SCHEMA rather than assumed; guessing any one of them fails the
# whole UNION.
#
# Dev, peer, and optional dependencies are excluded on purpose. They would wire
# every JavaScript repository to eslint and jest — true, universal, and
# therefore carrying no information about what anything actually is.
DEPENDENCY_REPO_EDGES_SQL = """
WITH ours AS (
  SELECT LOWER(full_name) AS repo FROM `{names_table}`
),
pkg AS (
  SELECT LOWER(ecosystem) AS system, LOWER(name) AS name, LOWER(repo) AS repo
  FROM `{package_table}`
),
reqs AS (
  SELECT 'npm' AS system, LOWER(r.Name) AS name, LOWER(dep.Name) AS dep_name
  FROM `bigquery-public-data.deps_dev_v1.NPMRequirementsLatest` r,
       UNNEST(r.Dependencies) AS dep
  UNION ALL
  -- PyPI names the field ProjectName, not Name.
  SELECT 'pypi', LOWER(r.Name), LOWER(dep.ProjectName)
  FROM `bigquery-public-data.deps_dev_v1.PyPIRequirementsLatest` r,
       UNNEST(r.Dependencies) AS dep
  UNION ALL
  -- Cargo carries an Optional flag; a BOOL is unambiguous where the Kind
  -- string's values have not been verified.
  SELECT 'cargo', LOWER(r.Name), LOWER(dep.Name)
  FROM `bigquery-public-data.deps_dev_v1.CargoRequirementsLatest` r,
       UNNEST(r.Dependencies) AS dep
  WHERE dep.Optional IS NOT TRUE
  UNION ALL
  -- RubyGems splits runtime from dev at the schema level.
  SELECT 'rubygems', LOWER(r.Name), LOWER(dep.Name)
  FROM `bigquery-public-data.deps_dev_v1.RubyGemsRequirementsLatest` r,
       UNNEST(r.RuntimeDependencies) AS dep
)
SELECT
  s.repo   AS src_repo,
  t.repo   AS dst_repo,
  -- One row per (version, dependency), so this counts how many published
  -- versions carry the dependency. A requirement sustained across two hundred
  -- releases is load-bearing; one that appears once is an experiment.
  COUNT(*) AS weight
FROM reqs
JOIN pkg s   ON s.system = reqs.system AND s.name = reqs.name
JOIN pkg t   ON t.system = reqs.system AND t.name = reqs.dep_name
JOIN ours os ON os.repo = s.repo
JOIN ours ot ON ot.repo = t.repo
WHERE s.repo <> t.repo
GROUP BY src_repo, dst_repo
"""


@dataclass
class BigQueryConfig:
    project: str
    location: str = "US"
    #: Hard ceiling per query. BigQuery bills on bytes scanned, and a careless
    #: GH Archive query can scan tens of terabytes. This makes a mistake fail
    #: loudly and instantly instead of quietly costing money.
    maximum_bytes_billed: int = 2000 * 1024**3  # 2000 GB
    dry_run_first: bool = True


class BigQueryExtractor:
    def __init__(self, config: BigQueryConfig) -> None:
        self.config = config
        # Imported lazily so the cleaner tests, the CLI, and anything else that
        # does not touch BigQuery never need the dependency installed.
        from google.cloud import bigquery

        self._bq = bigquery
        self.client = bigquery.Client(project=config.project, location=config.location)

    def _job_config(self, params: list, *, dry_run: bool = False):
        return self._bq.QueryJobConfig(
            query_parameters=params,
            maximum_bytes_billed=self.config.maximum_bytes_billed,
            dry_run=dry_run,
            use_query_cache=not dry_run,
        )

    @staticmethod
    def _parameter_bytes(params: list) -> int:
        total = 0
        for p in params:
            values = getattr(p, "values", None)
            if values is not None:
                total += sum(len(str(v)) + 4 for v in values)
        return total

    def _run(self, sql: str, params: list) -> list[dict]:
        # Check before sending. Exceeding the request limit returns a 413 with
        # an HTML error page — a wall of markup with no mention of parameter
        # size, arriving after however long the previous stage took.
        size = self._parameter_bytes(params)
        if size > MAX_PARAMETER_BYTES:
            raise RuntimeError(
                f"Query parameters are ~{size / 1024**2:.1f} MB, over the "
                f"{MAX_PARAMETER_BYTES / 1024**2:.0f} MB request limit. Upload the "
                "list to a table and JOIN against it instead of passing it as a "
                "parameter — see `_upload_names`."
            )

        scanned = self._check_cost(sql, params) if self.config.dry_run_first else 0.0

        job = self.client.query(sql, job_config=self._job_config(params))
        rows = [dict(row) for row in job.result()]
        # Zero rows from a valid query is the quietest possible failure: it
        # looks identical to "this data does not exist". Say it out loud.
        if not rows:
            log.warning(
                "Query returned ZERO rows after scanning %.2f GB. The SQL is "
                "valid, so this is a filter or schema mismatch, not missing data.",
                scanned,
            )
        else:
            log.info("Query returned %d rows", len(rows))
        return rows

    def _check_cost(self, sql: str, params: list) -> float:
        """Dry-run and refuse anything over the ceiling. Returns GB scanned.

        Split out of `_run` because a cost gate and a query runner are separate
        concerns — and because adding the table-size guidance below pushed
        `_run` past the 60-line rule, which is exactly the signal that rule
        exists to give.
        """
        probe = self.client.query(sql, job_config=self._job_config(params, dry_run=True))
        gb = probe.total_bytes_processed / 1024**3
        ceiling = self.config.maximum_bytes_billed / 1024**3

        # Log the ceiling beside the estimate. Printing only the estimate makes
        # a large number look like a failure that was allowed through, when it
        # may simply have been under a ceiling someone raised — not something
        # you can tell from the log alone.
        log.info("Query will scan %.2f GB (ceiling %.0f GB)", gb, ceiling)

        # Thresholds from measured runs, not guesses:
        #   ~0.3 TiB  per-ecosystem requirements join  — the good path
        #   ~1.4 TiB  PackageVersionToProject          — build once, then cache
        #   74 TiB    resolved dependency graph        — the wrong table
        usd = max(0.0, (gb / 1024) - 1.0) * 6.25
        if gb > 2048:
            log.warning(
                "%.0f GB (~$%.2f). Large enough to suspect the WRONG TABLE "
                "rather than a tunable query — `Dependencies` and "
                "`DependencyGraphEdges` hold the resolved transitive graph at "
                "95 and 292 TiB. `gitglobe edges --probe` lists every table "
                "with its size, for free.",
                gb, usd,
            )
        elif gb > 500:
            log.info(
                "%.0f GB (~$%.2f after the free monthly TiB). Expected when "
                "building the package map; it is cached afterwards.",
                gb, usd,
            )

        if probe.total_bytes_processed > self.config.maximum_bytes_billed:
            raise RuntimeError(
                f"Query would scan {gb:.1f} GB, over the {ceiling:.0f} GB "
                "ceiling. Raise --max-scan-gb only after checking, with "
                "`gitglobe edges --probe`, that this is the right table."
            )
        return gb

    def star_velocity(self, full_names: list[str], *, months: list[str]) -> dict[str, int]:
        """Stars gained per repo over the given `YYYYMM` months.

        `full_names` is passed as a parameter rather than interpolated, both to
        avoid injection and because BigQuery can then prune partitions against
        it.
        """
        if not full_names or not months:
            return {}

        sql = STAR_VELOCITY_SQL.format(
            suffix_pattern="*",
            start_month=min(months),
            end_month=max(months),
        )
        params = [
            self._bq.ArrayQueryParameter("repo_names", "STRING", full_names),
        ]
        return {row["full_name"]: row["stars_90d"] for row in self._run(sql, params)}

    def co_star_events(
        self,
        full_names: list[str],
        *,
        months: list[str],
        min_stars: int = 2,
        max_stars: int = 400,
    ) -> list[dict]:
        """Star events for the repositories we hold, from bounded actors.

        Returns raw `(actor, full_name, starred_at)` rather than pre-aggregated
        pairs on purpose. Pair generation and PPMI are cheap and get retuned
        often; the BigQuery scan is the expensive half and should happen once.
        """
        if not full_names or not months:
            return []
        sql = CO_STAR_SQL.format(start_month=min(months), end_month=max(months))
        params = [
            self._bq.ArrayQueryParameter("repo_names", "STRING", full_names),
            self._bq.ScalarQueryParameter("min_stars", "INT64", min_stars),
            self._bq.ScalarQueryParameter("max_stars", "INT64", max_stars),
        ]
        return self._run(sql, params)

    def probe_package_to_repo(self, limit: int = 20) -> list[dict]:
        """Read a handful of raw rows to see what the columns actually contain.

        A bare `SELECT ... LIMIT n` with no ORDER BY or aggregation lets
        BigQuery stop after the first block, so this costs near nothing against
        a table whose full scan is over a terabyte.

        This exists because the first attempt spent 34 minutes and 1.4 TB to
        return zero rows, and zero rows from valid SQL is indistinguishable from
        "no such data". Guessing `ProjectType = 'GITHUB'` was the likely culprit
        — but it was a guess, and the way to stop guessing is to look. Verifying
        someone else's schema should cost nothing, and here it does.
        """
        return self._run_raw(
            "SELECT System, Name, Version, ProjectType, ProjectName "
            "FROM `bigquery-public-data.deps_dev_v1.PackageVersionToProjectLatest` "
            f"LIMIT {int(limit)}",
            [],
        )

    def list_dataset_tables(self, dataset: str = "bigquery-public-data.deps_dev_v1") -> list[dict]:
        """Every table in a dataset with its row count and size, for free.

        `__TABLES__` is metadata: no table data is read, so this costs nothing
        and answers the only question worth asking before a large scan — is
        there a smaller table that holds the same fact?

        `DependenciesLatest` dry-ran at 74 TiB (~$458). That is not a query to
        tune; it is the wrong table. Listing sizes first turns "which table"
        from a guess into a lookup.
        """
        try:
            return self._run_raw(
                f"SELECT table_id, row_count, size_bytes FROM `{dataset}.__TABLES__` "
                "ORDER BY size_bytes DESC",
                [],
            )
        except Exception as exc:  # noqa: BLE001 - fall back, never fatal
            log.warning("__TABLES__ unavailable (%s); trying INFORMATION_SCHEMA", exc)
            return self._run_raw(
                f"SELECT table_name AS table_id, NULL AS row_count, NULL AS size_bytes "
                f"FROM `{dataset}.INFORMATION_SCHEMA.TABLES` ORDER BY table_name",
                [],
            )

    def describe_table(self, table: str) -> list[dict]:
        """Exact column names and types, including nested struct fields.

        `INFORMATION_SCHEMA.COLUMN_FIELD_PATHS` is metadata — it costs nothing
        and reads no table data — and it reports `field_path` for nested
        fields, which is the part that cannot be guessed. A repeated STRUCT
        called `Dependencies` and a flat column called `Dependency` produce
        completely different SQL, and getting it wrong returns
        `Name X not found inside d` after the previous stage has already run
        for eight minutes.

        I guessed at this dataset's shape five times and was wrong five times.
        Metadata is free. Read it.
        """
        dataset, name = table.rsplit(".", 1)
        sql = f"""
            SELECT field_path, data_type
            FROM `{dataset}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
            WHERE table_name = @table
            ORDER BY field_path
        """
        return self._run_raw(
            sql, [self._bq.ScalarQueryParameter("table", "STRING", name)]
        )

    def _run_raw(self, sql: str, params: list) -> list[dict]:
        """Run without the dry-run gate — for probes that are cheap by shape."""
        job = self.client.query(sql, job_config=self._bq.QueryJobConfig(query_parameters=params))
        rows = [dict(r) for r in job.result()]
        log.info("Probe returned %d rows, scanned %.3f GB",
                 len(rows), (job.total_bytes_processed or 0) / 1024**3)
        return rows

    def package_to_repo(self, full_names: list[str] | None = None) -> list[dict]:
        """Package -> GitHub repo mappings, optionally only for repos we hold.

        Passing `full_names` does NOT reduce the bytes BigQuery scans — billing
        is on columns read, not rows returned — but it cuts the result from tens
        of millions of rows to the handful we can use, which is the difference
        between a manageable transfer and an out-of-memory client.
        """
        if full_names:
            return self._run(
                PACKAGE_TO_REPO_SQL + "  AND LOWER(ProjectName) IN UNNEST(@names)",
                [self._bq.ArrayQueryParameter("names", "STRING", [n.lower() for n in full_names])],
            )
        return self._run(PACKAGE_TO_REPO_SQL, [])

    def _upload_names(self, full_names: list[str], dataset: str = "gitglobe_tmp") -> str:
        """Put the repo list in BigQuery as a table, not in the request body.

        A load job streams rows and has no 10 MB ceiling; a query parameter is
        serialised into the POST body and does. Same data, two transports, and
        only one of them scales past a few tens of thousands of rows.

        The dataset carries a 6-hour default table expiry so these clean
        themselves up — an abandoned run should not leave litter in the project.
        """
        client = self.client
        dataset_id = f"{self.config.project}.{dataset}"
        ds = self._bq.Dataset(dataset_id)
        ds.location = self.config.location
        ds.default_table_expiration_ms = 6 * 60 * 60 * 1000
        client.create_dataset(ds, exists_ok=True)

        table_id = f"{dataset_id}.repo_names"
        job_config = self._bq.LoadJobConfig(
            schema=[self._bq.SchemaField("full_name", "STRING")],
            write_disposition="WRITE_TRUNCATE",
        )
        rows = [{"full_name": n.lower()} for n in full_names]
        client.load_table_from_json(rows, table_id, job_config=job_config).result()
        log.info("Uploaded %d repo names to %s", len(rows), table_id)
        return table_id

    def _upload_package_map(
        self, mappings: list[tuple[str, str, str]], dataset: str = "gitglobe_tmp"
    ) -> str:
        """Send the package -> repo map back to BigQuery as a table.

        We already scanned 1.4 TiB to build this and it is sitting in Postgres.
        Re-deriving it inside the edge query would pay for it a second time; a
        load job of 2.3M short rows is a few seconds and costs nothing to scan.

        Uploading data you already have is usually a smell. Here the alternative
        is a terabyte, so it is the cheap option by three orders of magnitude.
        """
        client = self.client
        dataset_id = f"{self.config.project}.{dataset}"
        ds = self._bq.Dataset(dataset_id)
        ds.location = self.config.location
        ds.default_table_expiration_ms = 6 * 60 * 60 * 1000
        client.create_dataset(ds, exists_ok=True)

        table_id = f"{dataset_id}.package_map"
        job_config = self._bq.LoadJobConfig(
            schema=[
                self._bq.SchemaField("ecosystem", "STRING"),
                self._bq.SchemaField("name", "STRING"),
                self._bq.SchemaField("repo", "STRING"),
            ],
            write_disposition="WRITE_TRUNCATE",
        )
        rows = [
            {"ecosystem": e.lower(), "name": n.lower(), "repo": r.lower()}
            for e, n, r in mappings
        ]
        client.load_table_from_json(rows, table_id, job_config=job_config).result()
        log.info("Uploaded %d package->repo mappings to %s", len(rows), table_id)
        return table_id

    def dependency_repo_edges_from_map(
        self, full_names: list[str], package_map: list[tuple[str, str, str]]
    ) -> list[dict]:
        """Repo -> repo edges, scanning only the per-ecosystem requirements.

        Both small tables are uploaded, so the only large scan is the four
        `*Requirements` views — roughly 1 TiB total against 76 TiB for the
        resolved-graph route that the cost ceiling refused.
        """
        if not full_names or not package_map:
            raise ValueError(
                f"need both a repo list ({len(full_names)}) and a package map "
                f"({len(package_map)}); run the package mapping stage first"
            )
        names_table = self._upload_names(full_names)
        package_table = self._upload_package_map(package_map)
        return self._run(
            DEPENDENCY_REPO_EDGES_SQL.format(
                names_table=names_table, package_table=package_table
            ),
            [],
        )

    def dependency_repo_edges(self, full_names: list[str]) -> list[dict]:
        """Repo -> repo dependency edges for the repositories we hold.

        Resolves packages to repositories inside BigQuery rather than shipping
        the mapping back and forth. Returns `(src_repo, dst_repo, weight)` with
        `full_name` on both ends, so no package join is needed on our side.
        """
        if not full_names:
            return []
        table = self._upload_names(full_names)
        return self._run(DEPENDENCY_REPO_EDGES_SQL.format(names_table=table), [])

    def dependency_edges(self, package_names: list[str]) -> list[dict]:
        """Direct dependency edges among the given packages.

        Restricted to packages we actually hold, on both ends. An edge to a
        repository that is not in the globe points at nothing.
        """
        if not package_names:
            return []
        params = [self._bq.ArrayQueryParameter("package_names", "STRING", package_names)]
        return self._run(DEPENDENCY_EDGES_SQL, params)


def trailing_months(count: int = 3, *, today=None) -> list[str]:
    """The last `count` months as `YYYYMM`, most recent first."""
    from datetime import date

    today = today or date.today()
    out: list[str] = []
    year, month = today.year, today.month
    for _ in range(count):
        out.append(f"{year:04d}{month:02d}")
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return out
