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
PACKAGE_TO_REPO_SQL = """
SELECT DISTINCT
  LOWER(System)                                   AS ecosystem,
  Name                                            AS package_name,
  REGEXP_EXTRACT(
    ProjectName, r'^github\\.com/(.+)$')          AS full_name
FROM `bigquery-public-data.deps_dev_v1.PackageVersionToProject`
WHERE SnapshotAt = (
        SELECT MAX(SnapshotAt)
        FROM `bigquery-public-data.deps_dev_v1.PackageVersionToProject`)
  AND ProjectType = 'GITHUB'
  AND ProjectName IS NOT NULL
"""

# Direct dependencies only. Transitive edges explode the graph by roughly two
# orders of magnitude and add nothing a viewer can read — the arc layer draws at
# most a couple of thousand edges regardless.
DEPENDENCY_EDGES_SQL = """
WITH latest AS (
  SELECT MAX(SnapshotAt) AS snap
  FROM `bigquery-public-data.deps_dev_v1.Dependencies`
)
SELECT
  LOWER(d.System)        AS ecosystem,
  d.Name                 AS src_package,
  dep.Name               AS dst_package,
  COUNT(*)               AS weight
FROM `bigquery-public-data.deps_dev_v1.Dependencies` d,
     UNNEST(d.Dependencies) AS dep,
     latest
WHERE d.SnapshotAt = latest.snap
  AND dep.Distance = 1
  AND d.Name IN UNNEST(@package_names)
  AND dep.Name IN UNNEST(@package_names)
GROUP BY ecosystem, src_package, dst_package
"""


@dataclass
class BigQueryConfig:
    project: str
    location: str = "US"
    #: Hard ceiling per query. BigQuery bills on bytes scanned, and a careless
    #: GH Archive query can scan tens of terabytes. This makes a mistake fail
    #: loudly and instantly instead of quietly costing money.
    maximum_bytes_billed: int = 200 * 1024**3  # 200 GB
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

    def _run(self, sql: str, params: list) -> list[dict]:
        if self.config.dry_run_first:
            # A dry run costs nothing and reports exactly what the real one will
            # scan. Worth it every time on datasets this size.
            probe = self.client.query(sql, job_config=self._job_config(params, dry_run=True))
            gb = probe.total_bytes_processed / 1024**3
            log.info("Query will scan %.2f GB", gb)
            if probe.total_bytes_processed > self.config.maximum_bytes_billed:
                raise RuntimeError(
                    f"Query would scan {gb:.1f} GB, over the "
                    f"{self.config.maximum_bytes_billed / 1024**3:.0f} GB ceiling. "
                    "Narrow the date range or the repo list."
                )

        job = self.client.query(sql, job_config=self._job_config(params))
        return [dict(row) for row in job.result()]

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

    def package_to_repo(self) -> list[dict]:
        """Every package -> GitHub repo mapping deps.dev knows about."""
        return self._run(PACKAGE_TO_REPO_SQL, [])

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
