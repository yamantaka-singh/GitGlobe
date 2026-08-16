// Mirrors the sub-pixel fade in shaders.ts (placePoint).
//
// The property that kills the twinkling is energy conservation: a point that
// wants to be smaller than a pixel must emit the same total light as if it had
// been drawn at its true size. Pinning it to 1px at full alpha emits too much,
// and that excess is what flickers as the centre crosses the pixel grid.
const drawnSize = (ideal) => Math.min(Math.max(ideal, 1), 72);
const fade = (ideal) => { const s = Math.min(Math.max(ideal, 0), 1); return s * s; };

// Light emitted ∝ area × alpha.
const emitted = (ideal) => drawnSize(ideal) ** 2 * fade(ideal);
const wanted = (ideal) => ideal ** 2;

for (const ideal of [0.1, 0.25, 0.5, 0.75, 0.99]) {
  const err = Math.abs(emitted(ideal) - wanted(ideal)) / wanted(ideal);
  console.assert(err < 1e-12, `sub-pixel energy wrong at ${ideal}px: ${emitted(ideal)} vs ${wanted(ideal)}`);
}
console.log('sub-pixel points emit exactly the light their true size would');

// Points at or above a pixel must be untouched — the globe's look at normal
// zoom depends on this being a no-op.
for (const ideal of [1, 2, 8, 40, 72, 200]) {
  console.assert(fade(ideal) === 1, `fade must not dim a ${ideal}px point`);
}
console.log('points >= 1px are unaffected (fade = 1)');

// The old behaviour, for contrast: how much extra light a sub-pixel point threw.
const old = (ideal) => drawnSize(ideal) ** 2; // alpha was always 1
console.log('at 0.3px the old path emitted', (old(0.3) / wanted(0.3)).toFixed(1) + 'x too much light');

// Monotonic — no brightness inversion as you zoom.
let prev = -1;
for (let i = 0.05; i <= 3; i += 0.05) {
  const e = emitted(i);
  console.assert(e >= prev - 1e-12, `emitted light dips at ${i}px`);
  prev = e;
}
console.log('emitted light rises monotonically with size');
