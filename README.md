# GitGlobe

**A 3D interactive globe of the open-source universe.** One million repositories placed on a sphere by what they *do*, not what they're called — navigable by dragging, zooming, and talking to an AI that flies the camera for you.

> "Show me lightweight C++ web servers with minimal dependencies" → the globe spins, zooms into the systems-programming continent, and lights up a cluster of eleven repositories you'd never have found through search.

---

## Why this exists

Package discovery is still a text box and a ranked list. That format has three failure modes:

| Failure | What it costs you |
|---|---|
| **Keyword dependence** | A repo solving your exact problem is invisible if it uses different vocabulary. |
| **No sense of neighborhood** | You find one tool, never the eight alternatives sitting next to it. |
| **No sense of density** | You can't see that a niche is crowded (don't build there) or empty (opportunity). |

GitGlobe's premise: **capability is a continuous space, so render it as one.** Repositories that solve similar problems land in the same region of a sphere. Discovery becomes navigation.

---

## How it works

```
GitHub / deps.dev / ecosyste.ms
            │
            ▼
   ┌─────────────────┐
   │  Ingest & clean │   README stripped of badges/boilerplate → capability text
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │    Embed        │   voyage-3-large → 1024-d, Matryoshka-truncated to 512
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Spherical UMAP  │   output_metric="haversine" → (lat, lon) directly on S²
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Tile & quantize│   S2 cells → 4-byte-per-point binary tiles on a CDN
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  WebGL globe    │   one draw call, GPU picking, arc overlays
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Agent camera   │   Claude emits repo IDs; client resolves to coordinates
   └─────────────────┘
```

Four ideas do the heavy lifting:

**1. Semantic proximity over tags.** Topic tags are sparse, inconsistent, and self-reported. Embedding the actual README — after stripping the badge soup — captures what a project *does*.

**2. Native spherical projection.** Most projects run UMAP into 3D and then normalize onto a sphere. That's wrong: it throws away the radial dimension after the algorithm has already spent it. GitGlobe uses UMAP's `output_metric="haversine"`, which optimizes the layout *on the sphere's surface* from the start. No distortion, no wasted dimension. See [ADR-002](docs/ARCHITECTURE.md#adr-002-spherical-projection-method).

**3. The LLM never invents coordinates.** An agent that outputs `fly_to(lat: 42.1, lon: -80.3)` will hallucinate. GitGlobe's agent outputs *repository IDs*; the client looks up their real positions and computes the camera target. The model reasons about software; the renderer owns geometry. See [ADR-006](docs/ARCHITECTURE.md#adr-006-agent-camera-control-protocol).

**4. Edges are demand-loaded.** A million nodes implies tens of millions of dependency edges. Drawing them is neither possible nor useful — it's a hairball. Arcs appear only for the focused node's neighborhood, capped at ~2,000.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| **Renderer** | three.js + React Three Fiber | Full shader control; needed for a single-draw-call 1M-point globe with custom glow. |
| **Camera** | `camera-controls` (yomotsu) | Promise-based `.setLookAt(..., enableTransition)` — purpose-built for scripted fly-to. |
| **Frontend** | React 19 + TypeScript + Vite + Tailwind + Zustand | Fast HMR against a heavy WebGL scene; Zustand keeps camera state out of React's render path. |
| **API** | FastAPI (Python 3.12) + Pydantic | Same language as the ML pipeline; no model-serving bridge. |
| **Vector search** | Qdrant | Scalar quantization + HNSW keeps 1M×512 vectors in ~1GB RAM with sub-20ms p99. |
| **Metadata** | Postgres 16 + `pg_trgm` | Repo rows, adjacency lists, cluster labels; BM25-ish lexical half of hybrid search. |
| **Embeddings** | Voyage `voyage-3-large` | Matryoshka-truncatable; strong on technical prose. `voyage-code-3` for code-heavy repos. |
| **Agent** | Claude Sonnet 4.5 via Vercel AI SDK | Streams text and tool calls in one channel, which is exactly the "talk while flying" UX. |
| **Reduction** | RAPIDS cuML UMAP | 1M×512 in ~8 min on one L4 vs ~4h on CPU. |
| **Pipeline** | Prefect + Polars + DuckDB, GPU steps on Modal | Serverless GPU means you pay for 8 minutes, not a month. |
| **Tiles** | Cloudflare R2 + CDN | Static binary blobs; zero egress fees. |

Full library-by-library breakdown with versions and rationale: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md#library-reference).

---

## Performance budget

These are the numbers the architecture is designed around. If a design choice breaks one, it's the wrong choice.

| Metric | Target | How |
|---|---|---|
| Points rendered | 1,000,000 | Single `THREE.Points`, positions as 2×`int16` lat/lon, xyz reconstructed in the vertex shader |
| Position payload | **4 MB** | 1M × 4 bytes; full per-point record with stars, colour, and ID is ~12 bytes → 12 MB |
| Time to first paint | < 2.5 s | Top-20k-by-stars tile loads first; remainder streams by visible S2 cell |
| Frame time | < 16 ms | One draw call, back-hemisphere culled in the vertex shader |
| Hover pick | < 1 frame | GPU picking into a 1×1 scissored render target |
| Semantic query p99 | < 200 ms | Qdrant HNSW + scalar quantization, `ef=64` |
| Arcs on screen | ≤ 2,000 | Demand-loaded per focused node |

---

## Repository layout

```
gitglobe/
├── pipeline/                  # Python — offline, runs nightly
│   ├── ingest/                # GH Archive, GitHub GraphQL, deps.dev, ecosyste.ms
│   ├── clean/                 # README → capability text (badge/TOC/license stripping)
│   ├── embed/                 # Voyage batching, caching, retry
│   ├── project/               # cuML UMAP (haversine) + parametric encoder for new repos
│   ├── cluster/               # HDBSCAN → LLM-named nebulae
│   └── tile/                  # S2 cell binning → quantized .bin tiles
├── api/                       # FastAPI
│   ├── search/                # hybrid dense + lexical, RRF fusion, reranking
│   ├── graph/                 # k-hop neighborhood for arc rendering
│   └── agent/                 # tool definitions, SSE streaming loop
├── web/                       # React + R3F
│   ├── globe/                 # renderer, shaders, LOD, GPU picking
│   ├── arcs/                  # great-circle bezier edge layer
│   ├── chat/                  # streaming chat, tool-call → camera dispatch
│   └── store/                 # Zustand: camera, selection, filters
├── infra/                     # Terraform / Modal / migrations
└── docs/
    ├── ARCHITECTURE.md        # system design + ADRs
    └── IMPLEMENTATION_PLAN.md # phased build plan, skills, libraries
```

---

## Quickstart

```bash
git clone https://github.com/<you>/gitglobe && cd gitglobe

# --- pipeline ---
cd pipeline
uv venv && uv pip install -e ".[dev]"
cp .env.example .env          # GITHUB_TOKEN, VOYAGE_API_KEY, ANTHROPIC_API_KEY
python -m pipeline.run --sample 5000     # small end-to-end slice, CPU-only

# --- services ---
cd ../infra && docker compose up -d      # postgres + qdrant

# --- api ---
cd ../api && uv run fastapi dev

# --- web ---
cd ../web && pnpm install && pnpm dev
```

`--sample 5000` runs the whole pipeline on CPU in a few minutes so you can see a globe before committing to GPU infrastructure. The full 1M run is `python -m pipeline.run --full --gpu`.

---

## Data sources

| Source | Provides | Access | Notes |
|---|---|---|---|
| **GH Archive** | Event firehose since 2011 | Free, BigQuery | Star velocity and activity signals — better than raw star count |
| **GitHub GraphQL v4** | Metadata, README, topics | Token, 5k pts/hr | Use GraphQL not REST; one query fetches 100 repos with READMEs |
| **deps.dev (Open Source Insights)** | Cross-ecosystem dependency graph | Free, BigQuery + API | The source of truth for explicit edges — Google-maintained, covers npm/PyPI/Go/Maven/Cargo/NuGet |
| **ecosyste.ms** | Repo + package metadata, funding | Free REST | Excellent fallback and cross-registry reconciliation |
| **OSSF Criticality Score** | Project importance | Free dataset | Better node-size signal than stars alone |

Star count is a popularity proxy that heavily favours old repos. Node radius should blend stars, recent commit activity, and criticality score — see [ADR-008](docs/ARCHITECTURE.md#adr-008-node-size-signal).

---

## Roadmap

- [x] Concept and spatial pipeline sketch
- [ ] **Phase 0** — Render 1M synthetic points at 60fps *(de-risks everything else)*
- [ ] **Phase 1** — Ingest 100k real repos, cleaned and enriched
- [ ] **Phase 2** — Embed + spherical UMAP + S2 tiling
- [ ] **Phase 3** — Real data on the globe, hover/click, GPU picking
- [ ] **Phase 4** — Hybrid search API with reranking
- [ ] **Phase 5** — Agent camera control, streaming
- [ ] **Phase 6** — Dependency and semantic arcs
- [ ] **Phase 7** — Scale to 1M, nebula labels, share-a-view URLs, deploy

Detailed tasks and exit criteria per phase: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

---

## Honest limitations

Worth stating up front, because they shape the product.

- **A sphere's surface is two-dimensional.** Projecting a 512-d capability space onto it loses real structure. Some repos will land near neighbours they have little to do with. The globe is a navigational metaphor with strong local fidelity, not a faithful map — treat cluster membership as a hint, not a claim.
- **UMAP is not incremental.** A newly indexed repo can't be placed without either refitting or a parametric encoder. GitGlobe trains a parametric encoder against a frozen reference layout so daily additions land in stable positions; the base layout refits monthly. Coordinates therefore drift between refits, and any shared view URL must pin a layout version.
- **README quality is uneven.** Many repos have a title and a badge. Those embed poorly and cluster in a low-signal blob. Filtering on a minimum cleaned-README length is a quality lever, not a bug.
- **Popularity bias is real.** Any star-derived size or ranking amplifies the already-visible. The criticality-score blend mitigates it; it does not remove it.

---

## Documentation

- [Architecture & ADRs](docs/ARCHITECTURE.md) — components, data model, tile format, projection math, agent protocol, cost model
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — phases, skills matrix, AI-assisted workflow, library reference

## License

MIT — see [LICENSE](LICENSE).
