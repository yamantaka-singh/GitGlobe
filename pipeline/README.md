# GitGlobe pipeline

**Phase 1** turns GitHub into a table of repositories with clean, embeddable
**capability text**, and ends at Postgres.
**Phase 2** turns that table into the globe: embed, project onto a sphere,
cluster, rank, and write the binary tiles the browser loads.

Jump to [Phase 2](#phase-2-embed-project-build).

---

## Run it

```bash
cd pipeline
cp .env.example .env          # add your GITHUB_TOKEN
docker compose up -d          # Postgres on :5433
uv venv && uv pip install -e ".[dev]"

gitglobe ingest --target 5000 # the proof run, ~10 minutes
gitglobe doctor               # did it work, and if not why
gitglobe report               # exit-criterion check
gitglobe sample               # read 20 cleaned READMEs — do not skip this
```

Then the real run:

```bash
gitglobe ingest --target 100000
```

A GitHub token needs only `public_repo`. Set `GITHUB_TOKENS` to a comma-separated
pool if you have several — the 5,000 points/hour limit is per token, so N tokens
ingest roughly N times faster.

`--skip-bigquery` runs GitHub only, with no star velocity, criticality, or
dependency edges. Useful before your GCP project is set up.

---

## The cleaner is the whole phase

Everything else here is plumbing. This is the part that decides whether Phase 2
produces a meaningful map or an expensive blob.

A raw README is roughly: badge row, logo, title, tagline, table of contents,
installation, usage, API reference, contributing, license, sponsors. Only the
tagline, some prose, and the feature list describe what the project *does*.

The rest is not merely useless, it is **actively harmful**. Every repository
with a CI badge and an MIT licence embeds to nearly the same vector. Leave that
in and the dominant axis of your embedding space becomes "has badges", not "what
this software is for".

Worst of all is the link dump — "Community Integrations", "Ecosystem", "Built
With". A page listing other people's projects makes a repository land in *their*
neighbourhood instead of its own. Those sections are detected by shape (a
section that is mostly links) as well as by heading, because the headings are
never consistent.

What survives, on the real fixtures:

| Repository | Before | After | Removed |
|---|---:|---:|---:|
| react | 2,580 | 804 | 69% |
| requests | 1,944 | 766 | 61% |
| flask | 1,305 | 490 | 62% |
| ollama | 1,701 | 371 | 78% |

```bash
# Clean one README with no database and no network:
gitglobe clean path/to/README.md --name mytool
curl -s https://raw.githubusercontent.com/psf/requests/main/README.md | gitglobe clean -
```

### Tests

49 tests, stdlib `unittest`, no dependencies:

```bash
python -m unittest discover -s tests
```

They run before you have a database, a token, or a virtualenv — which is the
point. The fixtures are frozen from real repositories, and the assertions are
about *meaning*, not string equality: the tagline must survive, the licence
prose must not. Asserting exact output would break on every stop-word addition
and train you to update the expectation instead of thinking.

Several of these tests exist because the output was read by eye and found
wanting — `&middot;` entities, dangling `[WSGI]` brackets from shortcut links,
`**emphasis**` markers surviving into the embedding. Green tests on a text
pipeline are necessary, not sufficient. **Always read the output.**

---

## The 1,000-result wall

GitHub's search API returns at most 1,000 results for any query, however you
paginate. 100,000 repositories therefore means ~100 different questions.

The obvious axis is stars — monotonic, cheap to filter, power-law distributed,
so bands can widen geometrically. It is not enough. Geometric bands from 50 to
400,000 stars give **32 shards, a ceiling of 25,600**. Narrowing them does not
help: below a few hundred stars the population is in the millions, so a band
blows the cap however thin you slice it.

So the plan is two-dimensional, **stars × language**. Language is near-partitioning,
about 25 values cover most of GitHub, and it is orthogonal to stars — which is
what a second axis has to be. That gives 832 queries and a ceiling of ~665,000.

`plan_for_target()` adds axes only when needed, so the 5k proof run issues 32
queries rather than 832. Past ~665k it raises rather than quietly returning a
third of what you asked for; a third axis (creation-date ranges) would be next.

Every `language:` query excludes repositories with no detected language, so each
star band also gets an unfiltered query. Those repos — docs, dotfiles,
awesome-lists, datasets — are a large and genuinely distinct group that would
otherwise be invisible to the entire ingest.

---

## Data sources

| Source | Provides | Notes |
|---|---|---|
| **GitHub GraphQL v4** | Metadata + READMEs | 100 repos per query with READMEs inline. REST would need 101 calls for the same 100 rows. |
| **GH Archive** (BigQuery) | Star velocity | Stars gained in 90 days. A repo up 3,000 this quarter is alive in a way a 2015 project on 40,000 is not, and the API only exposes the total. |
| **deps.dev** (BigQuery) | Dependency edges | Google's *resolved* graph across npm, PyPI, Go, Maven, Cargo, NuGet — it accounts for version ranges, which parsing manifests does not. |
| **OSSF Criticality** | Importance score | Weighs dependents and maintenance, not popularity. A build tool nobody stars but everything depends on scores high. |

Every BigQuery query is capped at 200 GB scanned and dry-runs first. BigQuery
bills on bytes read, and the full GH Archive history is well over 20 TB — a
careless `SELECT *` is an expensive way to learn that.

---

## Resumability

Every page writes its cursor to `ingest_state` **after** the rows are committed,
never before. Checkpointing first is how a crash silently loses a page.

A completed shard is skipped on re-run, so `gitglobe ingest` is safe to run
repeatedly and can be used to top up a partial dataset rather than restarting
it. At GitHub's rate limits, a forced restart costs real hours.

Stage order matters for the same reason: the expensive rate-limited stage runs
first, and the BigQuery enrichments only update rows that already exist. A run
interrupted after stage 1 still leaves a usable database.

---

## Diagnosing a run

```bash
gitglobe doctor --expect 100000
```

`report` counts rows. `doctor` explains them — which matters because a run that
returns 40,000 repos has two completely different causes with completely
different fixes:

- **INTERRUPTED** — shards stopped mid-page with a live cursor. Re-run the same
  command; it resumes from the checkpoint.
- **PLAN EXHAUSTED** — every shard finished and 40,000 is genuinely all there
  was. The star bands hold fewer repositories than the ceiling assumed. Widen
  the plan; the ceiling is an upper bound, not a promise.
- **IN PROGRESS** — neither; some shards have not been attempted.

It also prints the star and language distribution, and the eight *least*-cleaned
repositories — the ones most likely to be carrying boilerplate into Phase 2.

### 502s and query cost

GitHub returns `502 Bad Gateway` for GraphQL queries it finds too expensive.
Nothing about the message says so, which makes it easy to treat as a transient
network fault and retry the identical query forever.

The cost is `page_size x readme_candidates` file reads per query. Thirteen
candidates at 50 repos a page is **650 blob reads**, and that reliably 502s.
Three candidates is 150 and does not.

Two things keep it there:

- **The root tree replaces guessing.** It names the README exactly for one cheap
  field, so a long list of guessed filenames is paying twice for the same
  answer — and the second payment is what broke the run. `INLINE_CANDIDATES` is
  a performance budget, and a test enforces it.
- **The page size shrinks on 502** (down to 5) and recovers after 20 clean
  responses. Backing off without changing the query just fails more slowly.

`GITHUB_PAGE_SIZE` (default 50) sets the ceiling.

### One bad band does not kill the run

A band that fails after its retries is **skipped and logged**, not raised. Left
unhandled it aborts the task, Prefect restarts the whole loop, the loop reaches
the same band, and it fails identically — forever. Nothing is checkpointed for a
skipped band, so a later run picks it up cleanly.

### README backfill

Repositories arrive needing a second fetch for two reasons, both recoverable:

- **Symlinked README.** A symlink blob's content *is* its target path, so it
  says exactly where the real file lives. Monorepos rely on this — zod,
  vuetify, unocss, mdx-deck and certbot all point their root README at a package
  subdirectory. Rejecting the symlink loses a good repository and quietly biases
  the corpus against monorepos; following it recovers one.
- **A filename outside the three inline candidates**, which the root tree names.

The backfill runs per band rather than once at the end, so an interrupted run
keeps what it recovered.

## Exit criterion

```bash
gitglobe report
```

- 100k rows with `clean_text`
- fewer than 25% flagged `low_signal`
- mean reduction above 35%
- **20 cleaned READMEs reviewed by eye** (`gitglobe sample`)

That last one is the real gate. The numbers can be green while the text is
wrong, and everything downstream inherits whatever you accept here. If clusters
look wrong in Phase 2, the bug is almost certainly in the cleaner, not in UMAP.

---

## Phase 2: embed, project, build

```bash
gitglobe status                 # what's done, what's next
gitglobe embed --dry-run        # cost estimate before spending credits
gitglobe embed                  # Vertex AI, cached on content_hash
gitglobe project                # UMAP onto the sphere (CPU, slow)
gitglobe cluster                # HDBSCAN + the twelve domains
gitglobe rank                   # PageRank over depends_on ∪ used_with
gitglobe build                  # writes ../web/public/tiles

cd ../web && npm run verify     # 47 integrity checks on the output
gitglobe spotcheck              # does the map actually mean anything
```

Five separate commands, not one. Each stage has failure modes worth looking at
before continuing — a projection that collapsed, clusters that came out as one
blob, a PageRank that never converged. Chaining them hides all of that behind
one long process that either works or does not.

Ordered by cost, descending. `embed` is the only stage that spends money, and it
caches on `content_hash`, so a re-run costs nothing for unchanged rows. `rank`
and `build` take seconds and are meant to be re-run freely while tuning.

### Three decisions that shape everything

**One text per request.** `gemini-embedding-001` accepts a single instance per
call, so 100,000 repositories is 100,000 round-trips. Throughput is entirely a
function of concurrency. Past ~300k rows, use Vertex batch prediction instead —
`gitglobe embed --dry-run` says when.

**768 dimensions, not 3072.** The model is Matryoshka-trained, so a 768-prefix
is a real embedding rather than a lossy compression — 0.26% quality loss for a
quarter of the storage. It matters because UMAP holds the whole matrix in RAM:
1M × 3072 float32 is 12 GB and will not fit on a laptop; 1M × 768 is 3 GB.

**Always L2-normalise.** `gemini-embedding-001` returns truncated vectors
*un-normalised* — only `gemini-embedding-2` normalises for you. Skip it and
every cosine distance in UMAP is computed against vectors of differing length.
That does not raise, does not look wrong, and produces a map whose clusters are
partly an artifact of text length.

### The sphere is the point

UMAP runs with `output_metric="haversine"`, so the loss function is spherical
and neighbour relationships are preserved in the geometry actually rendered.
Projecting to 3D and normalising afterwards optimises for a space nobody looks
at and then discards the dimension it spent its effort on.

Every 2D layout has a boundary, and a boundary is a lie: it says "nothing beyond
here", when what is beyond is the other side of the map.

UMAP's haversine output is *unbounded* — θ runs past the poles. Clamping piles
points onto the pole; `wrap_to_sphere` reflects them properly, because θ = π+0.1
means 0.1 past the south pole, which is a real place.

### Clustering happens on the sphere, not in 768-d

This looks like the weaker choice and is deliberate. The colour field has to be
spatially contiguous — a territory speckled with three other colours is not a
territory. Cluster in 768-d and you get semantically pure groups scattered
wherever the projection was imperfect; cluster on the sphere and contiguity is
guaranteed. UMAP already did the semantic work.

The cost is real: where UMAP misplaced a repository, this inherits the mistake.
`cluster_purity` measures exactly that, so it stays a known quantity rather than
a surprise. Its headline is **lift** — within-cluster similarity minus the
corpus baseline. Near zero means the territories are decoration.

### Node id is rank position

Sort by PageRank descending; that ordinal is the graph's node index, and
`ordinal + 1` is the tile's `repoId` (0 means "nothing picked"). One ordering
does four jobs: band membership is a contiguous slice, tile→graph is `repoId-1`
rather than a million-entry map, the picking id space is dense, and "the top N
repositories" is a slice rather than a sort.

**PageRank alone cannot sort the tail.** Every repository with no in-edges lands
on *exactly* the teleport floor — around 80% of a real corpus sharing one
identical float. Sorting by rank alone leaves that 80% arbitrary and sizes it
all at one radius: a flat field over most of the globe. Stars break the ties.
They never override rank; a wildly popular repository that nothing depends on
still ranks below a quietly critical one.

Sizing is also **per band**, not global. A single global percentile puts the top
2% of nodes in the top 2% of the radius range, so band 0 — the layer always in
focus — rendered every hub identically, hiding a thousandfold rank range.

### Two verifiers, and why they are not in Python

`npm run verify` is 47 checks written against the synthetic generator. Every one
applies to real output too, so the Python test suite *runs that verifier* on a
Python-built world rather than reimplementing it. Two definitions of "correct"
would be free to drift.

Same for the binary format: `tests/test_tile_format.py` encodes a tile in Python
and decodes it with `web/src/tile/format.ts` under Node. A format mismatch there
does not raise — the globe renders garbage, silently.

### Exit criterion

```bash
gitglobe spotcheck
```

Everything else proves the pipeline is self-consistent. This is the only check
that asks whether the result is *correct*: `pytorch`/`tensorflow`/`jax` close
together, `react`/`vue`/`svelte` close together, and the two groups far apart.

That last one is the strongest signal available. Both groups are popular,
heavily starred, and split across Python and JavaScript — so if the embedding is
really encoding popularity or language rather than capability, they collapse
together and this is what catches it.

The expectations were written from domain knowledge before looking at any
output. If you find yourself editing them because the map disagreed, that is the
map telling you something, and the expectations are not the place to fix it.

Missing repositories **skip** rather than fail — a 5k proof run contains almost
none of them, and a check that fails because the data is small teaches you to
ignore it.

---

## Layout

```
migrations/
  001_init.sql              repo, package, edge, ingest_state, ingest_run
  002_relatedness.sql       three edge kinds, star events, PageRank
  003_embeddings.sql        vectors (BYTEA, not pgvector), theta/phi, clusters
src/gitglobe/
  clean/readme.py           THE cleaner — zero dependencies, 33 tests
  ingest/
    plan.py                 search sharding — pure logic, 16 tests
    github.py               GraphQL client, token pool, backoff
    bigquery.py             GH Archive + deps.dev, cost-capped
    criticality.py          OSSF scores
  embed/vertex.py           Vertex AI, batched and resumable
  project/
    spherical.py            UMAP on S², coverage checks, kNN -> similar_to
    cluster.py              HDBSCAN + spherical k-means -> 12 domains
  graph/
    cooccurrence.py         PPMI over star baskets
    pagerank.py             power iteration, sizing, the tail problem
  tiles/
    format.py               binary writers — byte-exact with TypeScript
    build.py                the world writer
  checks/neighbours.py      the exit criterion
  db.py                     upserts, checkpoints, reports
  phase2.py                 the five stages
  cli.py                    ingest/report/sample/doctor/clean + embed/project/
                            cluster/rank/build/spotcheck/status
tests/                      215 tests, stdlib unittest
```
