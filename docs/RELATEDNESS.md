# How GitGlobe connects repositories

The brain of the system. This document exists because the question "how do repos
get connected?" had three partial answers scattered across three phases, and the
most important one was missing entirely.

---

## "Related" means three different things

Collapsing them into one number loses exactly the distinction the product is
built on.

| Kind | Question it answers | Source | Character |
|---|---|---|---|
| **`depends_on`** | *A cannot work without B* | deps.dev | Directed, sparse, high precision |
| **`similar_to`** | *A is an alternative to B* | Embedding kNN | Undirected, dense, from text |
| **`used_with`** | *People who use A use B* | Co-occurrence | Undirected, from behaviour |

### The gap

The product's promise is *"LangChain → ChromaDB"*. Look at what the first two
layers actually say about that pair:

- **Not similar.** One is an orchestration framework, one is a vector database.
  Their READMEs share little vocabulary. Cosine similarity ranks a dozen other
  vector databases above LangChain for ChromaDB.
- **Not necessarily dependent.** You can use either without the other.

They are **used together**. That is a behavioural fact, and neither text nor
package manifests capture it. Without a third layer, the flagship use case is
the one the system is worst at.

### Two more reasons the first two layers are not enough

**The dependency graph is far sparser than it looks.** deps.dev only knows about
packaged software. Awesome-lists, dotfiles, ML model repos, tutorials, most C++
projects and nearly every Jupyter notebook have no package identity, so they have
*zero* dependency edges. In the first live ingest, `(none)` + `C++` +
`Jupyter Notebook` were already ~15% of repositories — all of them invisible to
a dependency-only graph, and all of them stuck at PageRank's teleport floor,
indistinguishable from one another.

**Semantic kNN returns competitors, not collaborators.** The eight nearest
neighbours of `express` are eight other web frameworks. Genuinely useful for
"show me alternatives"; the opposite of what "inspect this stack's ecosystem"
needs.

---

## The `used_with` layer

### Sources

- **Co-starring**, from GH Archive `WatchEvent`. Users who starred A also starred
  B. Covers every repository regardless of packaging — precisely where the
  dependency graph is blind.
- **Co-dependency**, from deps.dev. Packages appearing in the same manifest.
  Narrower, but higher precision for stack questions.

### Why PMI, not counts

Count co-occurrences directly and the top result for *everything* is whatever is
most popular, because popular things co-occur with everything. Pointwise mutual
information divides that out:

```
PMI(a, b) = log( P(a,b) / (P(a) · P(b)) )
```

It asks whether two things appear together *more than their individual
popularity predicts*. Clamped at zero (PPMI) — "co-occur less than chance" is
real information, but not something to draw an arc for.

Measured on a synthetic corpus of 4,000 baskets with three genuine communities
plus one repository appearing in a third of all baskets and related to nothing:

```
  within-community edges : 30/30  (100%)
  edges touching the ubiquitous repo: 0
```

Raw co-occurrence would have made that repository the top result for every
query. PMI gives it zero edges.

### Three corrections that matter

**Basket weighting.** A user who starred 8 repositories is making a statement
about each; one who starred 300 is browsing. Weight by `1/log(n)` — `1/n` punishes
large baskets so hard that only tiny ones count, and tiny baskets are noisiest.
Anyone above 400 is dropped outright: at 5,000 stars they would contribute 12.5
million pairs alone.

**Minimum pair count.** PMI's known weakness is that rarity inflates it. Two
obscure repositories sharing one user produce a spectacular score and mean
nothing. Context-distribution smoothing (marginals to the power 0.75, from Levy
& Goldberg 2015) damps it further.

**Mutual top-k.** This is the one that turns a star graph into a structure.
Every small React component library counts `react` among its strongest
associations; `react` counts none of them. Keeping only pairs that appear in
*each other's* top k removes that entire class of edge — which is popularity
leaking back in through the side door.

---

## How the layers combine

They do **not** get averaged into one score. Each drives a different part of the
system, and that separation is the design.

```
                        ┌───────────────────────┐
    clean_text ────────▶│  embeddings (512-d)   │
                        └───────────┬───────────┘
                                    │
                ┌───────────────────┼────────────────────┐
                ▼                   ▼                    ▼
        spherical UMAP        similar_to (kNN)      search / RAG
                │                   │                    │
                ▼                   ▼                    ▼
         POSITION on globe    "alternatives"       query → nodes
         (geography = meaning)  panel, not arcs

    deps.dev ──▶ depends_on ──┐
                              ├──▶ union graph ──▶ PageRank ──▶ size, LOD, labels
    GH Archive ─▶ used_with ──┘         │
                                        └──────────▶ ARCS on globe
```

**Position comes only from embeddings.** Geography is semantic similarity, full
stop. Mixing edge data into the layout would make the map mean two things at
once and neither clearly.

**Arcs come only from `depends_on` and `used_with`.** Which leads to the
non-obvious consequence below.

**PageRank runs over the union**, not over dependencies alone. That is what
stops the 15% of unpackaged repositories from sitting at an indistinguishable
floor. Nodes still isolated after the union fall back to a criticality/star
blend for sizing.

---

## `similar_to` edges should not be drawn

This follows from the layout and is easy to miss.

If UMAP has done its job, semantically similar repositories are **already
adjacent on the globe**. Drawing an arc between two points that are three pixels
apart communicates nothing — the proximity already said it.

The edges worth drawing are the ones that **span distance**. A dependency or a
co-use link between opposite hemispheres is informative *precisely because*
position did not already tell you. A short arc is visual noise; a long arc is a
finding.

So the arc layer gets a minimum angular separation filter, and `similar_to`
becomes a data structure for search expansion and an "alternatives" list in the
detail panel — never geometry.

---

## What this changes in the plan

| Phase | Change |
|---|---|
| **1** | Add co-star extraction from GH Archive. Store raw `star_event` rows, not pre-aggregated pairs — PPMI parameters get retuned often, the BigQuery scan is the expensive half and should happen once. |
| **2** | PageRank over the union of `depends_on` and `used_with`, not dependencies alone. |
| **6** | Arcs are `depends_on` (solid, directional) and `used_with` (dashed, undirected), filtered by minimum angular separation. `similar_to` moves to the detail panel. |
| **5** | The agent gets `find_related(repo_id, kind)` so it can answer "what works with X" separately from "what is like X". Those are different questions and users ask both. |

### Schema

`edge.kind`: `0 = depends_on`, `1 = similar_to`, `2 = used_with`, plus `ppmi` and
`observations` columns for kind 2. New `star_event` table, new
`repo_relatedness` summary so the API never recomputes.

---

## Honest limitations

- **Co-starring measures attention, not use.** People star things they intend to
  try and never do. It correlates well with real use but is not the same thing.
- **It is popularity-biased at the tail.** A repository with 40 stars has too few
  co-occurrences to place, and no amount of PMI fixes a sample size of three.
  Those nodes rely on `depends_on` and position alone.
- **Trends age.** Co-star data from 2019 relates things nobody pairs today. The
  window should be trailing 12–24 months, not all history.
- **GH Archive stars are public actions.** Aggregate co-occurrence only; no
  per-user data reaches the product, and actor logins should be hashed on the
  way into `star_event`.
