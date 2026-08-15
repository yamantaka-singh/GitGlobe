# Design — Graph-aware cyberpunk globe (Phase 0.5)

**Date:** 2026-08-06
**Status:** Approved
**Supersedes:** nothing. Extends the Phase 0 renderer.

---

## Problem

The Phase 0 globe renders and performs, but fails on three counts:

1. **The dots don't read as repositories.** They read as decoration. Nothing about a dot conveys identity, and the readout sits at the bottom of the screen, visually disconnected from the cursor.
2. **Hover picks the wrong point.** The pick pass inflates every hit area to a 5px square. In dense regions that captures a neighbour 2–3px from the cursor, and the highlight ring visibly snaps away from where the user is pointing. This is a bug, not a preference.
3. **The look is generic.** A single Fresnel shell reads as a soft blue lamp rather than an instrument. There is no sense that the repositories are *connected* to each other, which is the whole premise of the product.

## Goals

- Every point reads as a specific repository with a name and a rank.
- Hover lands on the point under the cursor, and the connection between cursor and datum is visually explicit.
- The globe shows structure — a connection web whose topology means something.
- Visual direction: **hard sci-fi instrument.** Precise, dense, restrained. Not neon costume.

## Non-goals

- Real repository data (Phase 1).
- Real dependency edges (Phase 1, from deps.dev).
- Bloom post-processing. Additive blending already provides glow, and full-screen passes are the mobile fill-rate killer.
- Scanline / glitch / chromatic-aberration overlays. They read as costume and undermine the instrument direction.

---

## Architecture

### 1. Graph artifact — `graph.bin`

The edge graph is a **separate file from the tiles**. Tiles stay a pure spatial-visual record; the graph is its own artifact. This mirrors the production architecture, where edges live in Postgres and never enter a tile.

```
offset  bytes        field
0       4            magic 'GGG1'
4       4            nodeCount   (n)
8       4            edgeCount   (e, CSR entries = 2 x directed edges)
12      4            ambientCount (a)
16      2            layoutVersion
18      6            reserved
24      4n           rank      Float32[n]        PageRank, sums to 1
24+4n   4(n+1)       offsets   Uint32[n+1]       CSR row starts
28+8n   4e           targets   Uint32[e]         CSR neighbours
28+8n+4e 8a          ambient   Uint32[2a]        pre-selected backbone edge pairs
28+8n+4e+8a 2e       weights   Uint16[e]         bit15 = outgoing, bits0-14 = weight
```

All `Uint32` arrays are contiguous and precede the `Uint16` array, which guarantees natural alignment for every view regardless of `n`, `e`, or `a`.

CSR is **undirected** — each directed edge appears in both endpoints' rows, with a direction bit in the weight. That gives O(1) "show me everything this repo touches" on hover, which is the only query the renderer makes.

At 100k nodes, m=3 preferential attachment (~300k directed edges, 600k CSR entries): **~4.4 MB**.

### 2. Graph generation

**Barabási–Albert preferential attachment with intra-domain bias.** Each new node attaches to `m=3` existing nodes chosen with probability proportional to degree, but restricted to its own domain 80% of the time. This produces:

- a genuine power-law degree distribution, so real hubs exist;
- edges that mostly stay inside a domain, which is what dependency graphs actually look like;
- a small number of cross-domain bridges, which are the visually interesting arcs.

**PageRank**, damping 0.85, power iteration to L1 delta < 1e-9 or 200 iterations, with dangling-node mass redistributed uniformly. Implemented on the directed edge list as a pure function over CSR, in `src/graph/pagerank.ts`, tested against known-answer graphs.

### 3. What PageRank drives

| Consumer | Use |
|---|---|
| Node radius | `size = 0.1 + 0.9 * normalise(log(rank))` — replaces the log-star model |
| Node brightness | Same term, applied to alpha |
| LOD band | Nodes sorted by rank; band 0 is the top 2% |
| Ambient arcs | Only edges whose endpoints' combined rank exceeds a threshold, capped at `a` |
| Labels | Top ~40 by rank get a billboard label at close zoom |

Rank is a *node* measure, so it cannot invent edges — the edges come from the attachment process (and from deps.dev in Phase 1). What rank decides is **which of the edges are worth drawing**, which is what turns a hairball into a legible backbone.

### 4. Arc renderer

One GPU-resident mesh, two buffers, one shader.

- **Ambient buffer** — static, ~2000 arcs, built once from `graph.bin`'s ambient list. Faint.
- **Focus buffer** — dynamic, capacity 256 arcs. On hover/select, endpoints are rewritten from the CSR row and `needsUpdate` is set. Bright, with a travelling pulse.

Each arc is a ribbon of `SEGMENTS=20` quads. Per-vertex attributes: `aEndA` (vec3), `aEndB` (vec3), `aT` (float along arc), `aSide` (±1), `aMeta` (weight, phase). The vertex shader:

1. **slerps** A→B along the great circle,
2. lifts by `sin(pi*t) * liftAmount`, where lift scales with the angle between endpoints,
3. expands the ribbon perpendicular to the view direction,
4. computes the pulse: `exp(-((fract(uTime*speed + phase) - t)^2) / width)`.

Because position is computed in the shader, animating the pulse costs one uniform write per frame and zero CPU work.

Triangle cost: 2000 arcs x 20 segments x 2 triangles = **80k triangles**, inside the 500k desktop budget.

### 5. Three-layer atmosphere

Replacing the single Fresnel shell:

- **Core** — near-black sphere carrying a lat/long grid in the fragment shader, brightening at grazing angles so the wireframe appears only near the limb.
- **Rim** — thin hard Fresnel band, `pow ~8`. A crisp edge, not a gradient.
- **Scatter** — wide soft Fresnel, `pow ~2.5`, low intensity, for volume.

### 6. Hover identity

1. `uPickPadding` 5px → 2.5px.
2. **Size-biased picking**: the pick shader offsets depth by `-size`, so when two points overlap the larger one wins. Hovering then feels intentional rather than arbitrary.
3. **Reticle** — screen-space corner brackets projected onto the picked point, drawn in the HUD layer, with a leader line to the card.
4. **Cursor-anchored card** — name, rank, domain, dependents, dependencies.
5. **Procedural repo names.** A deterministic hash of `repoId` indexes per-domain word lists, producing plausible names (`vecstore-rs`, `torchflow`, `hyperscrape`). Zero bytes on the wire, generated client-side. `#48213` reads as a dot; `vecstore-rs` reads as a repository.

### 7. Palette

Background moves to near-pure black. Domain hues pull off the rainbow toward a cooler instrument set. Cyan `#2DE2FF` and mint `#7CF5C0` are reserved for interaction states; pure white is reserved for the picked node. HUD chrome: hairline borders, corner brackets, tabular numerals, `↳` markers, no rounded pills.

---

## Testing

- `pagerank.test.ts` — known-answer graphs: a star (centre must dominate), a directed cycle (uniform), a dangling node (mass conserved), and a hand-computed 4-node example. Rank must sum to 1 within 1e-9.
- `graph/format.test.ts` — CSR round-trip, alignment at odd `n`/`e`/`a`, corrupt-buffer rejection.
- `verify-tiles.ts` extended to cross-check `graph.bin` against the manifest: node count matches, CSR offsets monotonic, every target in range, ambient edges reference real nodes, rank sums to 1.
- The existing 10s benchmark remains the gate. If the arc layer costs more than ~2ms p95, ambient arc count drops until it doesn't.

## Risks

| Risk | Mitigation |
|---|---|
| Ambient web becomes visual noise | Cap at 2000, keep alpha low, drive selection by rank threshold not by count alone |
| Ribbon arcs blow the fill budget on integrated GPUs | Arc count is tier-scaled; `low` tier disables the ambient layer entirely |
| Procedural names read as fake | Word lists are domain-specific and use real naming conventions (suffixes `-rs`, `-js`, `.py`, `-core`) |
| Size-biased picking makes small repos unhoverable | The bias is small (a fraction of the depth range) and only resolves genuine overlaps |
