# GitGlobe — Phase 0: prove the renderer

One question, answered before anything else gets built: **can a browser draw a
hundred thousand — then a million — repositories on a sphere at 60fps, with a
live connection web, accurate hover-picking, and a smooth camera?**

No real data yet. If the answer is no, we find out in a day and switch renderers
([ADR-001](../docs/ARCHITECTURE.md#adr-001-renderer--threejs-vs-deckgl)) with the
entire data layer untouched.

---

## Run it

```bash
cd web
npm install
npm run setup     # generates the world (100k nodes + graph) and verifies it
npm run dev
```

Drag to orbit, scroll to zoom. **Hover a node** to see its identity and light up
its connections; **click** to pin it. `↳ top hub` flies to the single most
depended-upon repository in the graph.

Hit `◈ run benchmark`. That is the exit criterion, measured rather than
eyeballed.

### The stress run — this is the one that matters

```bash
npm run gen:world:stress    # 1,000,000 nodes
```

100k is the comfortable default for iterating on the look. **1M is the number
Phase 0 exists to prove.** A renderer that holds 60fps at 100k and collapses at
1M has not passed.

---

## What the world generator produces

```
public/tiles/
  manifest.json   layout version, bands, domains, cluster centres, graph stats
  band-0.bin      top 2% by PageRank      — always loaded
  band-1.bin      next 18%
  band-2.bin      the remaining 80%
  graph.bin       PageRank + CSR adjacency + the ambient backbone
```

Nothing here is uniform noise, deliberately. Positions come from **von
Mises-Fisher clusters** at varying concentrations; edges come from
**Barabási–Albert preferential attachment** biased toward a node's own domain.
That yields a genuine power law — at 100k nodes: median degree 4, p99 34, max
655 — and mostly intra-domain dependencies with a few cross-domain bridges.
Uniform noise would let a renderer "pass" Phase 0 while being unable to cope
with the clumped, wildly-varying-density data Phase 2 produces.

**PageRank** (damping 0.85, power iteration to L1 < 1e-10) then drives four
things: node radius and brightness, LOD band assignment, which edges form the
ambient backbone, and which nodes are prominent enough to matter.

PageRank ranks *nodes*, so it cannot invent edges — those come from the
attachment process, and from deps.dev in Phase 1. What it decides is **which of
the edges are worth drawing**, which is what turns a hairball into a legible
structure. The verifier measures this: backbone endpoints average **1,638× the
mean node rank**.

---

## Exit criterion

| | Target | Where it's measured |
|---|---|---|
| Dropped frames | < 1% | benchmark panel |
| Worst frame | < 100ms | benchmark panel |
| Headroom | ≥ 2× at the target node count | benchmark panel |
| Hover accuracy | the node under the cursor, every time | reticle |
| Fly-to | frames the target smoothly, no snap | domain tabs |

### Why the gate is not "p95 ≤ 16.7ms"

It was, and that was wrong. `requestAnimationFrame` is vsync-locked: on a 60Hz
display the interval between callbacks is 16.67ms whether the renderer is nearly
idle or one particle from collapse. Gating on that interval measures the
*monitor*, not the scene — a perfectly healthy renderer scores p50 16.7 / p95
17.1 and "fails" a threshold it can never clear.

What "sustained 60fps" actually means is **no dropped frames**, so the benchmark
now:

1. **Calibrates** the display period from the median warmup interval (16.67ms at
   60Hz, 8.33 at 120Hz — it adapts to your monitor).
2. **Counts dropped frames** over a deterministic 6-second orbit. A frame is
   dropped when its interval exceeds 1.5× the period.
3. **Probes headroom** by rendering the scene N extra times per frame and finding
   the largest N that still holds the refresh rate.

Headroom is the number that matters, because vsync hides everything above the
line. Without it you cannot tell "comfortable" from "barely coping" — both report
60fps. **3× headroom at 100k is what predicts that 1M will hold.**

Test on a laptop integrated GPU, not just a discrete one.

---

## Layout

```
scripts/
  gen-world.ts      positions → graph → PageRank → tiles + graph.bin
  verify-world.ts   30 integrity checks over both artifacts
  preview-planet.ts CPU render of the planet shader, with pass/fail gates
src/
  tile/format.ts    12 bytes per node — docs/ARCHITECTURE.md §2.6
  graph/
    pagerank.ts     pure CSR power iteration, tested to known answers
    format.ts       graph.bin encode/decode, CSR neighbour queries
  globe/
    shaders.ts      point display + GPU-pick shaders
    arcShaders.ts   slerped ribbon arcs with a travelling pulse
    ArcLayer.tsx    fixed-capacity GPU-resident arc pool
    PointCloud.tsx  one THREE.Points per LOD band
    Backdrop.tsx    nebula sky, tinted starfield, ice giant, 3 atmosphere shells
    planetShaders.ts  banded ice-giant surface, baked once to a texture
    palette.ts      every colour in one place, shared with the Node preview
    usePicking.ts   1×1 scissored GPU pick pass, layer-isolated, 30Hz
    useAnchor.ts    projects the hovered node to screen space for the reticle
  camera/Rig.tsx    the single owner of the camera
  repo/names.ts     procedural repository names
  ui/Reticle.tsx    corner brackets + leader line + identity card
```

---

## The planet

An **ice giant**, not a terrestrial world. Neptune's character comes from
latitudinal banding, methane cirrus riding above it, a few dark storm ovals —
and critically from *low* contrast between the bands. Crank the contrast and you
get a beach ball.

Geography still means something. There is no land, so a territory is a
persistent **weather system**: each domain owns a great cloud mass tinted within
the blue-violet family, and the strongest clusters carry a storm vortex the way
the Great Dark Spot sits in a belt. Everything is dragged by a shared
differential-rotation shear — faster at the equator than the poles — which is
what produces the long swirls rather than flat stripes.

Baked once into a 2048×1024 equirectangular texture at load. RGB is daylight
albedo; alpha is night-side emissive (polar aurora and storm luminance), which
stops the dark limb becoming a dead black crescent.

```bash
npm run preview:planet
```

Renders the same formula on the CPU to `public/planet-globe.png` and
`public/planet-preview.png`, and **fails** on: territory outside 20–85%, storms
outside 0.5–15%, night glow over 12%, mean luminance over 0.55, or band contrast
under 0.02. That last one is measured across *latitude* specifically — global
variance would pass on a planet that was merely noisy.

It has already caught a planet that was 100% land, a night side that glowed like
a lamp, and padding cluster slots winning the nearest-neighbour test.

The whole surface stays under 0.55 luminance on purpose. A hundred thousand data
points sit on top of it, and the moment the planet competes, the map stops being
readable.

## The sky

Two thresholded noise fields at different scales, plus a broad galactic-plane
band, drawn on a camera-locked box with `gl_Position.xyww` so it sits exactly on
the far plane at zero sorting cost. Stars are tinted across blue-white → amber,
because a monochrome starfield reads as dust on the lens.

## Idle rotation

The globe drifts when you leave it alone and stops the moment you engage. It
pauses on **any** pointer activity over the canvas — not just hover — because
gating on hover alone still let the node you were reaching for slide away before
you got to it. Aiming had become a chase. It resumes after 4 seconds, ramping up
over a second so the restart is not its own jolt.

## Design decisions worth knowing before you edit

**The quantised angles ride in `position`.** three.js derives the draw count from
`geometry.attributes.position`, so the geometry must have one. Rather than pay
12 MB for a dummy Float32 xyz *alongside* the real data, `position` is a Uint16
attribute carrying `(thetaQ, phiQ, sizeQ)` — all three fit in 16 bits. 6 bytes
per node, nothing wasted.

**Arc geometry is computed entirely in the vertex shader.** Each vertex knows
only its two endpoints and its position along the arc; the shader slerps the
great circle, lifts it, and expands the ribbon toward the camera. The travelling
pulse is therefore one uniform write per frame — hovering a hub with 600
neighbours costs the same as hovering a leaf.

**Arcs are two buffers, one shader.** `ambient` is static (the PageRank
backbone); `focus` is a 256-arc pool whose endpoints are rewritten on hover.
Neither ever reallocates geometry.

**Depth does the occlusion.** The core sphere is opaque and writes depth, so arcs
passing behind the globe are hidden correctly without any sorting.

**The atmosphere is three layers, not one.** A single Fresnel shell has no edge,
so the eye never finds the horizon and it reads as a soft lamp. A hard rim
(`pow 9`), a wide scatter (`pow 2.6`), and a grazing-angle grid on the core give
the silhouette a boundary and the surface a sense of curvature.

**Picking is size-biased.** The pick shader offsets depth in proportion to node
size, so overlapping nodes resolve to the more significant one. Combined with
dropping hit padding from 5px to 2.5px, this is the fix for hover landing on a
neighbour instead of the node under the cursor.

**Names are procedural.** `#48213` reads as a dot; `vecstore-rs` reads as a
repository. A deterministic hash of the repo id against per-domain word lists —
zero bytes on the wire. Phase 1 deletes `repo/names.ts` entirely.

**Nothing allocates in `useFrame`.** Vectors are module-scope scratch.

**The camera has exactly one owner.** `globeCamera` in `camera/Rig.tsx`.
`flyToDirections` takes directions derived from real nodes — never a
caller-supplied coordinate. That is what will make
[ADR-006](../docs/ARCHITECTURE.md#adr-006-agent-camera-control-protocol) — the
agent emits repo ids, never lat/lon — enforceable in Phase 5 rather than merely
intended.

---

## Known gaps, to close before Phase 3

- **Tiles load whole, per band.** Phase 3 replaces this with per-S2-cell
  streaming, `AbortController` on cells that scroll away, and LRU eviction.
- **`sceneIndex.resolve` is a linear scan across bands.** Fine at 3; needs a real
  index once cells number in the hundreds.
- **No screen-space label collision.** Nebula labels arrive in Phase 3.
- **No keyboard path.** There must be a way to reach search → results → node
  detail without the canvas. Tracked for Phase 7; do not let it slip past that.
- **The palette has not been through a colour-blindness simulator.** Hues are
  spaced with that in mind; it has not been verified.
- **`graph.bin` is 4.4 MB and loads whole.** At 1M nodes it will need the same
  per-cell treatment as tiles.

---

## Troubleshooting

**"No tile manifest (HTTP 404)"** — run `npm run gen:world`.

**Layout version mismatch** — regenerate; tiles and graph must come from the
same run.

**Everything is one white blob** — node size scale is too high for your DPR.
Drag `node size` down; the default of 32 is calibrated for a 2.6-radius camera
at DPR 2.

**Hover highlights the wrong node** — the display and pick passes have drifted
apart. Both materials must get the same `uSizeScale` and `uRadius`, and
`uPixelRatio` must be 1 on the pick material (its target is 1×1, so point sizes
there are in CSS pixels).

**Arcs flicker or disappear at the limb** — expected near the horizon, where the
ribbon goes edge-on. If they vanish over the *front* of the globe, the core
sphere radius has crept above the arc lift.

**Black screen, no errors** — check the console for a shader compile failure.
Structs and early `return` in GLSL ES 1.00 are supported but some mobile drivers
are strict; the error text names the line.
