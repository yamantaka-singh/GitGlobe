const fs = require('fs');
const buffer = fs.readFileSync('web/public/tiles/graph.bin');
const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
const n = dv.getUint32(4, true);
const e = dv.getUint32(8, true);
const a = dv.getUint32(12, true);
const rankAt = 24;
const offsetsAt = rankAt + 4 * n;
const targetsAt = offsetsAt + 4 * (n + 1);
const ambientAt = targetsAt + 4 * e;
const weightsAt = ambientAt + 8 * a;
const weights = new Uint16Array(buffer.buffer, buffer.byteOffset + weightsAt, e);

let depends = 0, similar = 0, used = 0;
for (let i = 0; i < e; i++) {
  const w = weights[i];
  const kind = (w & 0x6000) >> 13;
  if (kind === 0) depends++;
  else if (kind === 1) similar++;
  else if (kind === 2) used++;
}
console.log(`Depends: ${depends}, Similar: ${similar}, Used: ${used}`);
