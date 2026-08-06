# GitGlobe — Phase 0: prove the renderer

The point of this phase is to answer one question before anything else gets
built: **can a browser draw a hundred thousand — then a million — glowing points
on a sphere at 60fps, with hover-picking and a smooth camera?**

No real data. No embeddings. No API. If the answer is no, we find out in a day
and switch renderers ([ADR-001](../docs/ARCHITECTURE.md#adr-001-renderer--threejs-vs-deckgl))
with the entire data layer untouched.

---

## Run it

```bash
cd web
npm install
npm run setup     # generates 100k synthetic points and verifies the tiles
npm run dev
```

Then open the printed URL. Drag to orbit, scroll to zoom, hover a point, click
to fly to it, or click a domain in the right-hand panel.

**Hit `Run 10s benchmark`.** That is the exit criterion, measured rather than
eyeballed.

### The stress run — this is the one that matters

```bash
npm run gen:tiles:stress    # 1,000,000 points, 12 MB of tiles
```

Reload and benchmark again. 100k is the comfortable working default for
iterating on the look; **1M is the number Phase 0 exists to prove.** A renderer
that holds 60fps at 100k and collapses at 1M has not passed.

---

## Exit criterion

| | Target | Where it's measured |
|---|---|---|
| p95 frame time | ≤ 16.7ms | benchmark panel |
| Worst frame | < 100ms | benchmark panel |
| Draw calls | ≤ 4 | benchmark panel / r3f-perf |
| Hover accuracy | correct point, every time | readout at the bottom |
| Fly-to | frames the target smoothly, no snap | domain buttons |

The benchmark drives a **deterministic** orbit — a full revolution plus polar
and dolly oscillation, identical every run — so two runs are comparable and a
regression is visible. It reports percentiles, not average fps, because average
fps hides the stutter that actually makes a scene feel bad. 60fps average with
two 90ms spikes per second is a failure, and only p95/p99 show it.

Test on a laptop integrated GPU, not just a discrete one.

---

## What's here

```
scripts/
  gen-tiles.ts      synthetic tile generator (vMF clusters, not uniform noise)
  verify-tiles.ts   data-integrity checks — also used on real Phase 2 tiles
src/
  tile/
    format.ts       binary tile encode/decode — docs/ARCHITECTURE.md §2.6
    format.test.ts  round-trip, quantisation error, pole and seam edge cases
    loader.ts       manifest + band fetching, layout-version guard
  globe/
    shaders.ts      the whole renderer, really — GLSL for display and picking
    PointCloud.tsx  one THREE.Points per LOD band, one draw call each
    usePicking.ts   1x1 scissored GPU pick pass, layer-isolated, 30Hz
    Backdrop.tsx    starfield, atmosphere shell, opaque core
    Scene.tsx       lifecycle-organised scene graph + tile loading
  camera/Rig.tsx    the single owner of the camera; flyTo lives here
  bench/            deterministic benchmark harness
  perf/             device tiering and measured demotion
  store/            zustand — kept out of the render hot path
  ui/Hud.tsx        stats, controls, legend, benchmark readout
```

---

## Design decisions worth knowing before you edit

**The quantised angles ride in `position`.** three.js derives the draw count
from `geometry.attributes.position`, so the geometry must have one. Rather than
pay 12 MB for a dummy Float32 xyz *alongside* the real data, `position` is a
Uint16 attribute carrying `(thetaQ, phiQ, sizeQ)` — all three fit in 16 bits.
6 bytes per point, nothing wasted. The vertex shader reconstructs the unit
direction. See `shaders.ts`.

**One shared vertex preamble for display and picking.** If the two passes
computed position separately they would eventually disagree, and a picking bug
caused by drifting position maths is close to undebuggable.

**Culling happens in the vertex shader.** Points on the far hemisphere are
pushed outside the clip volume, which removes roughly half the fragment work
with zero CPU cost and no index rebuild.

**No raycasting anywhere.** `raycast = () => null` on every object. GPU picking
renders ids as colours into a 1×1 target at the cursor and reads one pixel back
— constant cost no matter how many points are on screen. The readback stalls the
pipeline, so it is throttled to 30Hz and skipped entirely during camera flights.

**Nothing allocates in `useFrame`.** Vectors are module-scope scratch. Per-frame
allocation produces the sawtooth GC stutter that shows up as random hitches.

**The camera has exactly one owner.** `globeCamera` in `camera/Rig.tsx`.
`flyToDirections` takes *directions derived from real points* — never a
caller-supplied coordinate. That is what will make
[ADR-006](../docs/ARCHITECTURE.md#adr-006-agent-camera-control-protocol)
(the agent emits repo ids, never lat/lon) enforceable in Phase 5 rather than
merely intended.

**Synthetic data is clustered, not uniform.** Uniform points on a sphere look
nothing like a UMAP layout and would let a renderer "pass" Phase 0 while being
unable to cope with the clumped, wildly-varying-density data Phase 2 produces.
The generator samples von Mises-Fisher clusters at varying concentrations plus a
diffuse background.

**Node size comes from log-scaled Pareto stars.** The first version sampled
`pow(rnd(), 3.2)` and `verify-tiles` immediately caught the consequence: the top
2% of repos all landed within 0.975–1.000, so every important repo would have
rendered the same size. Star counts are power-law and radius scales with *log*
stars ([ADR-008](../docs/ARCHITECTURE.md#adr-008-node-size-signal)).

---

## Known gaps, to close before Phase 3

- **Tiles are loaded whole, per band.** Phase 3 replaces this with per-S2-cell
  streaming, `AbortController` on cells that scroll away, and LRU eviction.
- **`sceneIndex.resolve` is a linear scan across bands.** Fine at 3 bands;
  needs a proper index once cells number in the hundreds.
- **No screen-space label collision.** Nebula labels arrive in Phase 3.
- **Keyboard navigation is not implemented.** There must be a full keyboard path
  to search → results → repo detail that never requires the canvas. Tracked for
  Phase 7, but do not let it slip past that.
- **The palette has not been through a colour-blindness simulator.** Hues are
  spaced with that in mind; it has not been verified.

---

## Troubleshooting

**"No tile manifest (HTTP 404)"** — run `npm run gen:tiles` first.

**Everything is one white blob** — point size scale is too high for your DPR.
Drag the `point size` slider down; the default of 32 is calibrated for a
2.6-radius camera at DPR 2.

**Hover highlights the wrong point** — the display and pick passes have drifted
out of sync. Check that both materials get the same `uSizeScale` and `uRadius`,
and that `uPixelRatio` is 1 on the pick material (its target is 1×1, so point
sizes there are in CSS pixels).

**Black screen, no errors** — check the browser console for a shader compile
failure. Structs and early `return` in GLSL ES 1.00 are supported but some
mobile drivers are strict; the error text will name the line.
