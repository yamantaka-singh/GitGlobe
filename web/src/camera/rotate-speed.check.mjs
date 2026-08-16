// Mirrors the rotate-speed curve in Rig.tsx.
//
// Guards the property that matters: a fixed drag should move the globe's
// surface the same distance across the screen at every zoom level. A constant
// term added to this curve breaks that silently and only shows up as "too
// sensitive when zoomed in", so it is asserted rather than eyeballed.
const R = 1, SPAN = 1.6 * R, BASE = 0.55;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

const speed = (d) => BASE * clamp((d - R) / SPAN, 0.02, 1);

/** Screen-space travel of the nearest surface point for one unit of drag. */
const apparent = (d) => (R * speed(d)) / (d - R);

// Above the lower clamp, apparent travel must be flat — that is the whole point.
const sample = [2.6, 2.0, 1.5, 1.3, 1.2, 1.1, 1.05];
const base = apparent(sample[0]);
for (const d of sample) {
  const err = Math.abs(apparent(d) - base) / base;
  console.assert(err < 1e-9, `apparent motion drifts at ${d}R: ${apparent(d)} vs ${base}`);
}
console.log('apparent travel per drag is flat across', sample[0] + 'R..' + sample.at(-1) + 'R');

// The old curve is what this file exists to prevent. Prove it fails the above.
const oldSpeed = (d) => BASE * (0.18 + 0.82 * clamp((d - R) / SPAN, 0, 1));
const oldApparent = (d) => (R * oldSpeed(d)) / (d - R);
const worst = oldApparent(1.05) / oldApparent(2.6);
console.assert(worst > 5, 'expected the old curve to be much worse close in');
console.log('old curve was', worst.toFixed(1) + 'x too sensitive at 1.05R — regression guarded');

// Monotonic: zooming in never speeds rotation up.
for (let d = 1.05; d < 2.6; d += 0.05) {
  console.assert(speed(d) <= speed(d + 0.05) + 1e-12, `not monotonic near ${d}R`);
}
console.log('speed is monotonic in distance');
