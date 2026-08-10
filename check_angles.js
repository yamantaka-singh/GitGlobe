const fs = require('fs');

const layoutBuffer = fs.readFileSync('web/public/tiles/layout.bin');
const ldv = new DataView(layoutBuffer.buffer, layoutBuffer.byteOffset, layoutBuffer.byteLength);
const ln = ldv.getUint32(4, true);

const pos = new Float32Array(layoutBuffer.buffer, layoutBuffer.byteOffset + 24 + 4*ln, ln*3);

const graphBuffer = fs.readFileSync('web/public/tiles/graph.bin');
const gdv = new DataView(graphBuffer.buffer, graphBuffer.byteOffset, graphBuffer.byteLength);
const gn = gdv.getUint32(4, true);
const ge = gdv.getUint32(8, true);

const offsets = new Uint32Array(graphBuffer.buffer, graphBuffer.byteOffset + 24 + 4*gn, gn+1);
const targets = new Uint32Array(graphBuffer.buffer, graphBuffer.byteOffset + 24 + 4*gn + 4*(gn+1), ge);

let smallAngles = 0, totalAngles = 0;
for (let i = 0; i < Math.min(gn, 1000); i++) {
  const start = offsets[i], end = offsets[i+1];
  const x1 = pos[i*3], y1 = pos[i*3+1], z1 = pos[i*3+2];
  const mag1 = Math.sqrt(x1*x1 + y1*y1 + z1*z1);
  for (let j = start; j < end; j++) {
    const t = targets[j];
    const x2 = pos[t*3], y2 = pos[t*3+1], z2 = pos[t*3+2];
    const mag2 = Math.sqrt(x2*x2 + y2*y2 + z2*z2);
    const dot = (x1*x2 + y1*y2 + z1*z2) / (mag1 * mag2);
    const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
    totalAngles++;
    if (angle < 0.05) smallAngles++;
  }
}
console.log(`Small Angles: ${smallAngles} / ${totalAngles}`);
