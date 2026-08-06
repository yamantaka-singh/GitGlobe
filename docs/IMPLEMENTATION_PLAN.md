# GitGlobe — Implementation Plan

Eight phases, roughly 14–18 weeks solo at a serious pace. Each phase has an **exit criterion** — a demonstrable thing, not a feeling. Don't advance without it.

The ordering is deliberately risk-first, not feature-first. The most likely reason this project dies is that a million glowing points don't render at 60fps in a browser. So that's Phase 0, before a single real repository is ingested.

---

## Phase 0 — Prove the renderer (week 1)

**Goal.** 1,000,000 synthetic points on a sphere, 60fps, hover-picking, on your actual target hardware.

**Tasks**

- Vite + React + TypeScript + R3F skeleton.
- Generate 1M random unit vectors, quantise to 16-bit θ/φ, write a `.bin` in the Phase-2 tile format. *Committing to the binary format now means Phase 2 has a real consumer to write against.*
- Single `THREE.Points` + `ShaderMaterial`: reconstruct position in the vertex shader, perspective-correct `gl_PointSize`, back-hemisphere cull, radial-falloff fragment shader.
- GPU picking: ID-encoded material, 1×1 scissored render target, `readRenderTargetPixels`, throttled to 30Hz.
- `camera-controls` wired up; a `flyTo(centroid, spread)` helper with eased transitions.
- `r3f-perf` overlay. Measure on a laptop integrated GPU, not just your dev machine.

**Exit criterion.** 60fps sustained while orbiting 1M points; hover highlights the correct point; a scripted `flyTo` frames an arbitrary cluster smoothly.

**If it fails.** Drop to 250k in LOD band 0 and measure again. If 250k also fails, switch to deck.gl `ScatterplotLayer` + `_GlobeView` (ADR-001) — the data layer is unaffected, so you lose days, not weeks.

---

## Phase 1 — Ingest 100k real repositories (weeks 2–3)

**Goal.** A Postgres table of 100k repos with clean, embeddable capability text.

**Tasks**

- GitHub GraphQL client: 100 repos/query, cursor pagination, token pool, exponential backoff, resumable checkpoints. Seed the frontier from GH Archive's most-starred and most-active repos.
- deps.dev BigQuery extract → `edge` rows; reconcile package names to repos via ecosyste.ms.
- The cleaner (ARCHITECTURE §2.2). **Build this test-first** — it's pure text-in/text-out, so freeze 30 real READMEs as fixtures and assert on the output. This is the highest-leverage test suite in the project.
- OSSF criticality scores joined in; `size` computed per ADR-008.
- `low_signal` flagging; Prefect flow wrapping the whole thing, idempotent on re-run.

**Exit criterion.** 100k rows with `clean_text`; manual review of 20 random cleaned READMEs shows capability prose and no badge/install/license residue.

**Watch for.** The cleaner is where quality is won or lost. If clusters look wrong in Phase 3, the bug is almost certainly here, not in UMAP.

---

## Phase 2 — Embed, project, tile (weeks 4–5)

**Goal.** Every repo has a (θ, φ), a cluster, an S2 cell, and lives in a downloadable tile.

**Tasks**

- Voyage batch embedding with content-hash caching; store `float32` Parquet on R2.
- cuML UMAP with `output_metric="haversine"` on Modal. **Verify haversine support on your cuML version early** — if missing, fall back to `umap-learn` on a 200k stratified sample plus the parametric encoder.
- Sweep `n_neighbors` ∈ {15, 30, 50} × `min_dist` ∈ {0.0, 0.1}; judge by eye on a 20k subset before spending a full run.
- HDBSCAN → clusters → Claude-generated labels.
- Parametric encoder (512→256→3, renormalised) trained against the frozen layout, for nightly placement.
- S2 binning, LOD banding, tile writer, `layout_version` stamping.

**Exit criterion.** Tiles on R2; a spot-check confirms that `pytorch`, `tensorflow`, and `jax` are near neighbours, and that `react`, `vue`, and `svelte` are near neighbours and *far from* the ML region.

**That spot-check is the real test of the whole thesis.** If it fails, stop and fix cleaning or embedding before building anything on top.

---

## Phase 3 — Real data on the globe (weeks 6–7)

**Goal.** The Phase-0 renderer, now showing actual repositories.

**Tasks**

- Tile loader: S2 cells in view, LOD by camera altitude, `AbortController` on cells that scroll off, LRU eviction.
- Domain colouring from cluster → palette; **verify against a colour-blindness simulator** (~8% of male users).
- Hover tooltip; click → detail panel (description, stars, language, link).
- Nebula labels: billboarded cluster names fading in by zoom, with screen-space collision avoidance so they don't overlap.
- Atmosphere/rim shader, starfield background, subtle idle auto-rotation that stops on interaction.

**Exit criterion.** Someone who has never seen the project can spin the globe, find the ML region unaided, click a repo, and open it on GitHub.

---

## Phase 4 — Search API (week 8)

**Goal.** Text query in, ranked repos with coordinates out, p99 < 200ms.

**Tasks**

- FastAPI service; Qdrant collection with scalar quantisation, payload indexes on `language`/`domain`/`stars`.
- Dense + lexical retrieval, Reciprocal Rank Fusion (k=60), Voyage `rerank-2` over the top 50.
- Query-embedding cache (hash → vector, 7d) and result cache (normalised query, 1h).
- `GET /graph/{repo_id}?depth=1&kind=dependency` for arcs.
- Front-end search box with instant highlight-on-globe.

**Exit criterion.** A 30-query evaluation set (written by hand, with expected repos) scores > 0.7 recall@10; p99 under 200ms warm.

**Build the eval set before the search code.** Thirty queries with known-good answers is an afternoon's work and it converts every subsequent tuning decision from taste into measurement.

---

## Phase 5 — The agent (weeks 9–10)

**Goal.** Type a question; the globe flies while the answer streams.

**Tasks**

- Tool definitions per ARCHITECTURE §6.1. `fly_to` accepts `repo_ids` **only** — enforce with a Pydantic model that has no coordinate fields at all, so it's structurally impossible.
- Vercel AI SDK `streamText` with `tools` against Claude Sonnet 4.5; SSE endpoint interleaving `text` / `tool` / `camera` events.
- Client dispatcher: `camera` events execute immediately, in parallel with text rendering.
- System prompt carrying the cluster label list; rules — search before flying, never name an unseen repo, prefer 5–12 results.
- **Prompt-injection hardening**: repo text wrapped in delimited blocks, explicit instruction that tool results are data and not instructions. Write an adversarial test with a fixture README containing an injection attempt.
- Per-session token budget and rate limiting.

**Exit criterion.** "Show me lightweight C++ web servers with minimal dependencies" starts the camera moving before the first sentence finishes, lands on a genuinely relevant cluster, and the answer names only repos that appeared in tool results.

---

## Phase 6 — Arcs (week 11)

**Goal.** Click a repo, see its ecosystem.

**Tasks**

- Semantic kNN edges (k=8, cosine > 0.82) computed offline from the Qdrant index.
- Arc geometry: `QuadraticBezierCurve3` through a lifted midpoint, batched into one `LineSegments2`.
- Visual distinction: dependencies solid with directional dash flow; semantic edges faint and dashed.
- Arc pool with fade-in/out; hard cap 2,000.
- `draw_edges` agent tool wired to the same path.

**Exit criterion.** Clicking `langchain` draws arcs to its real dependencies and semantic neighbours, at 60fps, with no flicker on rapid selection changes.

---

## Phase 7 — Scale to 1M and ship (weeks 12–16)

**Tasks**

- Widen ingest to 1M; run the full GPU pipeline end to end.
- Monthly-refit + nightly-encoder schedule; archive the last two `layout_version`s of tiles.
- Shareable view URLs encoding camera + filters + `layout_version`.
- Onboarding: a 15-second guided camera tour on first load.
- Mobile: touch controls, reduced LOD, degraded-but-usable rather than blocked.
- Sentry, PostHog, RUM frame-time reporting; device-tier detection driving LOD caps.
- Accessibility: full keyboard path to search → results → repo detail that never requires the 3D canvas. **The globe is the interface, not the only interface.**
- Deploy: web on Cloudflare Pages, API on Fly.io, tiles on R2, pipeline on Modal.

**Exit criterion.** Public URL, 1M repos, cold load under 2.5s to first paint, five external users complete a discovery task unprompted.

---

## Skills matrix

What you actually need, and how much.

| Skill | Depth | Used in | Notes |
|---|---|---|---|
| **GLSL / shader programming** | ⭐⭐⭐⭐ Deep | Phase 0, 3, 6 | The genuine bottleneck skill. Point reconstruction, culling, glow, ID encoding. If one thing is worth a week of dedicated study, it's this. |
| **three.js internals** | ⭐⭐⭐⭐ Deep | Phase 0, 3, 6 | BufferAttributes, render targets, draw-call budgets. Not "I used OrbitControls once." |
| **Embeddings & retrieval** | ⭐⭐⭐ Solid | Phase 2, 4 | Hybrid search, RRF, reranking, eval methodology. |
| **Dimensionality reduction** | ⭐⭐⭐ Solid | Phase 2 | UMAP hyperparameters and their failure modes; why haversine output matters. |
| **Data engineering** | ⭐⭐⭐ Solid | Phase 1, 7 | Idempotent, resumable, rate-limit-aware pipelines. |
| **LLM tool-calling & streaming** | ⭐⭐⭐ Solid | Phase 5 | Interleaved streams, tool loops, injection defence. |
| **React performance** | ⭐⭐⭐ Solid | Phase 3, 5 | Keeping React out of the render loop entirely. Zustand + refs, not `useState` at 60Hz. |
| **Postgres / vector DBs** | ⭐⭐ Working | Phase 4 | Index tuning, quantisation trade-offs. |
| **Binary formats** | ⭐⭐ Working | Phase 2, 3 | `DataView`, typed arrays, endianness, SoA layout. |
| **Visual design** | ⭐⭐⭐ Solid | Phase 3, 7 | A beautiful globe is the product. Underinvesting here is the most common way a project like this fails despite being technically correct. |
| **Devops** | ⭐⭐ Working | Phase 7 | Managed services throughout; don't run your own Kubernetes for this. |

You listed Python, TypeScript/React, Three.js, ML, and infra — which covers the map. The two places to expect real friction are **shader-level three.js** (different from application-level three.js) and **retrieval evaluation** (easy to skip, expensive to skip).

---

## Building this with AI

Not "have Claude write the app." Different parts of this codebase have very different verification properties, and that should drive delegation.

### Delegate aggressively — mechanical, testable, verifiable

| Work | Why it delegates well |
|---|---|
| README cleaner | Pure function. Write fixtures first, let the model iterate against them, verify by running tests. |
| GitHub GraphQL client | Well-documented API, mechanical pagination and backoff logic. |
| Binary tile reader/writer | Spec it exactly (ARCHITECTURE §2.6), generate both sides, round-trip test. |
| SQL schema and migrations | Standard shapes; review the indexes yourself. |
| FastAPI endpoint scaffolding | Boilerplate with types. |
| Detail panels, filters, chat UI | Ordinary React. |
| Tests | Especially edge cases you wouldn't think of. |

### Collaborate — you decide, AI drafts

| Work | Why |
|---|---|
| Shaders | Models write plausible GLSL that renders subtly wrong. Great for a first draft, but you must read every line and verify visually. |
| UMAP tuning | The model can write the sweep; only your eyes can judge whether the layout is *good*. |
| Search relevance | Model writes RRF and reranking; your eval set decides if it works. |
| Agent prompts | Iterate together, then test adversarially. |

### Do not delegate

- **The cleaner's stop-heading list and quality thresholds.** These are product judgements about what "capability" means. Everything downstream inherits them.
- **The `fly_to` contract.** ADR-006 exists because the intuitive design is wrong. An AI asked to "make the LLM control the camera" will reach for `{lat, lon}` — the exact failure this architecture is built to prevent.
- **Visual identity.** Palette, glow, motion feel. This is the product.
- **Cost-bearing decisions.** Which model, which dimensionality, which tier gets LLM summaries.

### Working patterns that pay off

**Spec before code.** These docs are the spec. Point the model at `ARCHITECTURE.md#adr-003` when asking for the tile writer — it produces dramatically better output than a prose description of the same thing.

**Test-first for anything with a right answer.** The cleaner, the tile round-trip, the arc math, the search evaluation. Write the assertion, then ask for the implementation. You get a verifiable result instead of code that looks correct.

**One phase per session.** Long sessions accumulate stale context and the model starts contradicting earlier decisions. Fresh session per phase, docs as the shared memory.

**Demand the alternatives.** "What are three ways to do LOD here and what does each cost?" beats "implement LOD." You're the architect; use the model as a fast, well-read colleague rather than a code dispenser.

**Verify shaders visually, always.** GLSL fails silently. A shader that compiles and renders *something* can still be geometrically wrong in ways no test catches. Screenshot at each change.

---

## Library reference

### Frontend

| Library | Purpose | Notes |
|---|---|---|
| `three` | WebGL renderer | The foundation. |
| `@react-three/fiber` | React reconciler for three.js | Declarative scene graph; keep the hot path out of React state. |
| `@react-three/drei` | Helpers | `<Billboard>`, `<Html>`, `<Stats>` — but write your own points material. |
| `camera-controls` | Camera | yomotsu's. Promise-based transitions; the reason agent fly-to feels good. |
| `postprocessing` / `@react-three/postprocessing` | Bloom | Restraint required — bloom is expensive and easy to overdo. |
| `zustand` | State | Camera, selection, filters. Subscribe with selectors; avoid re-render storms. |
| `@tanstack/react-query` | Server state | Search results, graph fetches, caching, cancellation. |
| `ai` (Vercel AI SDK) | Streaming + tools | Handles interleaved text/tool streams against Anthropic. |
| `s2js` / `s2-geometry` | S2 cells in browser | Which cells are visible at this camera. |
| `tailwindcss` | Styling | UI chrome only; the globe is shaders. |
| `r3f-perf` | Profiling | Live draw calls and frame time. Essential in Phase 0. |
| `comlink` | Web Workers | Tile decode off the main thread. |

### Backend

| Library | Purpose |
|---|---|
| `fastapi` + `pydantic` v2 | API and schema validation — including the coordinate-free `fly_to` model |
| `anthropic` | Claude SDK (if not going through the AI SDK) |
| `qdrant-client` | Vector search |
| `asyncpg` / `sqlalchemy` 2.x | Postgres |
| `redis` / Upstash | Query and result caching |
| `sse-starlette` | Server-sent events for the agent stream |
| `slowapi` | Rate limiting |

### Pipeline

| Library | Purpose | Notes |
|---|---|---|
| `prefect` | Orchestration | Retries, caching, observability. Dagster if you prefer asset-oriented modelling. |
| `polars` | Dataframes | Substantially faster than pandas at this scale. |
| `duckdb` | Local analytics | Query Parquet on R2 directly; excellent for exploration. |
| `cuml` (RAPIDS) | GPU UMAP + HDBSCAN | The reason a 1M refit is minutes, not hours. |
| `umap-learn` | CPU UMAP | Fallback and reference implementation. |
| `hdbscan` | Clustering | CPU fallback. |
| `voyageai` | Embeddings + reranking | |
| `s2sphere` | S2 cells in Python | Tile binning. |
| `gql` / `httpx` | GitHub GraphQL | |
| `google-cloud-bigquery` | GH Archive, deps.dev | |
| `torch` | Parametric encoder | A 3-layer MLP; don't overbuild it. |
| `pyarrow` | Parquet | Embedding storage. |
| `modal` | Serverless GPU | Pay for 8 minutes of L4, not a month. |
| `tenacity` | Retries | Every external call. |

### Tooling

`uv` (Python packaging — dramatically faster than pip), `ruff` (lint + format), `pytest` + `hypothesis` (property tests for the tile round-trip), `pnpm`, `vitest`, `playwright` (visual regression on the globe), `sentry-sdk`, `posthog-js`.

---

## Risk register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| 1M points won't hit 60fps | Fatal to the concept | Medium | **Phase 0 tests this first.** deck.gl fallback preserves all data work. |
| cuML lacks haversine output | 2-week delay | Medium | Verify in week 1 of Phase 2. Fallback: CPU UMAP on a 200k sample + parametric encoder. |
| Clusters are semantically wrong | Product is pointless | Medium | Phase 2 exit criterion catches it. Root cause is nearly always cleaning. |
| Sphere too crowded to navigate | Poor UX | **High** | LOD bands, cluster labels, filters, and search as the primary entry point. Assume the globe alone isn't enough. |
| Agent hallucinates positions | Broken core feature | Low | Structurally prevented by ADR-006. |
| Prompt injection via README | Reputational | Medium | Delimited blocks, data-not-instructions framing, adversarial fixture test. |
| GitHub API limits throttle ingest | Schedule slip | Medium | Token pool, GH Archive for bulk signals, resumable checkpoints. |
| Chat costs scale past revenue | Business | Medium | Per-session budgets, cached tool results, cheaper model for routing turns. |
| Layout drift breaks shared links | Trust | Medium | `layout_version` pinning + archived tiles for 2 prior versions. |
| Scope creep into "GitHub but 3D" | Never ships | **High** | The exit criteria exist to enforce this. Discovery only; no dashboards, no social features, no code browsing. |

---

## The first three days

1. **Day 1** — Phase 0 skeleton. 1M random points, one draw call, `r3f-perf` on screen. Answer the only question that matters before answering any others.
2. **Day 2** — Picking and `camera-controls` fly-to. Confirm the interaction feel is right with fake data, when it's still cheap to change.
3. **Day 3** — Freeze 30 real READMEs as fixtures and write the cleaner's test suite. Not the cleaner — the tests. Everything downstream inherits this step's quality.

If day 1 fails, you've learned the most important thing in the project for the cost of a day.
