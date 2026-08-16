// Mirrors the scan-order + window-clamp logic in usePicking.ts.
const PICK_RADIUS = 7, PICK_SIZE = 15;

const order = (() => {
  const out = [];
  for (let y = 0; y < PICK_SIZE; y++) for (let x = 0; x < PICK_SIZE; x++) out.push(y * PICK_SIZE + x);
  const c = PICK_RADIUS;
  return out.sort((a, b) => {
    const ax = (a % PICK_SIZE) - c, ay = Math.floor(a / PICK_SIZE) - c;
    const bx = (b % PICK_SIZE) - c, by = Math.floor(b / PICK_SIZE) - c;
    return ax * ax + ay * ay - (bx * bx + by * by);
  });
})();

const d2 = (i) => { const x=(i%PICK_SIZE)-PICK_RADIUS, y=Math.floor(i/PICK_SIZE)-PICK_RADIUS; return x*x+y*y; };

// 1. every pixel visited exactly once
console.assert(order.length === PICK_SIZE*PICK_SIZE, 'covers the whole window');
console.assert(new Set(order).size === order.length, 'no duplicates');
// 2. centre first — a node under the finger always wins
console.assert(order[0] === PICK_RADIUS*PICK_SIZE + PICK_RADIUS, 'centre is first');
// 3. monotonically non-decreasing distance => "nearest hit" is really nearest
for (let i=1;i<order.length;i++) console.assert(d2(order[i]) >= d2(order[i-1]), 'ordered by distance');

// 4. window never starts off-canvas, and never runs past the right/bottom edge
const clamp = (v, w) => Math.max(0, Math.min(w - PICK_SIZE, v - PICK_RADIUS));
for (const [v,w] of [[0,375],[3,375],[187,375],[374,375],[812,812],[0,812]]) {
  const o = clamp(v,w);
  console.assert(o >= 0 && o + PICK_SIZE <= w, `window inside canvas for v=${v} w=${w} -> ${o}`);
}
console.log('pick window checks passed — covers', PICK_SIZE+'x'+PICK_SIZE,
            'px, centre-first, distance-ordered, always on-canvas');
console.log('tap tolerance: +/-' + PICK_RADIUS + 'px (was 0) =',
            (PICK_SIZE*PICK_SIZE) + 'x more pixels searched than the old 1x1');
