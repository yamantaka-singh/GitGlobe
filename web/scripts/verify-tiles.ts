/**
 * Data-integrity check for generated tiles.
 *
 * Runs against whatever is in public/tiles — synthetic now, real UMAP output
 * from Phase 2 onward. Catches the failure modes that would otherwise show up
 * as "the globe looks wrong" with no clue why: wrong point count, quantisation
 * overflow, id collisions between bands, a band whose sizes are all identical,
 * or a layout-version mismatch.
 *
 * Usage:  npm run verify:tiles
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  decodeTile,
  dequantisePhi,
  dequantiseSize,
  dequantiseTheta,
  directionAt,
  tileByteLength,
  TAU,
} from '../src/tile/format.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const TILES = resolve(HERE, '../public/tiles');

let failures = 0;

function check(name: string, ok: boolean, detail = '') {
  if (ok) {
    console.log(`  ok   ${name}${detail ? ` — ${detail}` : ''}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function readBuffer(file: string): ArrayBuffer {
  const buf = readFileSync(resolve(TILES, file));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function main() {
  const manifest = JSON.parse(readFileSync(resolve(TILES, 'manifest.json'), 'utf8'));
  console.log(
    `Verifying ${manifest.total.toLocaleString()} points, layout v${manifest.layoutVersion}` +
      `${manifest.synthetic ? ' (synthetic)' : ''}\n`,
  );

  let seenTotal = 0;
  let duplicateRepoIds = 0;
  const allRepoIds = new Set<number>();
  const domainHistogram = new Map<number, number>();

  for (const entry of manifest.bands) {
    console.log(`band ${entry.band}  (${entry.file})`);
    const buffer = readBuffer(entry.file);
    const tile = decodeTile(buffer);

    check('count matches manifest', tile.count === entry.count, `${tile.count} vs ${entry.count}`);
    check('byte length exact', buffer.byteLength === tileByteLength(tile.count), `${buffer.byteLength} bytes`);
    check('layout version matches', tile.layoutVersion === manifest.layoutVersion);
    check('lod band matches', tile.lodBand === entry.band);

    let minTheta = Infinity;
    let maxTheta = -Infinity;
    let minPhi = Infinity;
    let maxPhi = -Infinity;
    let minSize = Infinity;
    let maxSize = -Infinity;
    let worstNorm = 0;
    let zeroRepoIds = 0;

    for (let i = 0; i < tile.count; i++) {
      const th = dequantiseTheta(tile.thetaQ[i]);
      const ph = dequantisePhi(tile.phiQ[i]);
      const sz = dequantiseSize(tile.sizeQ[i]);
      if (th < minTheta) minTheta = th;
      if (th > maxTheta) maxTheta = th;
      if (ph < minPhi) minPhi = ph;
      if (ph > maxPhi) maxPhi = ph;
      if (sz < minSize) minSize = sz;
      if (sz > maxSize) maxSize = sz;
      if (tile.repoId[i] === 0) zeroRepoIds++;

      // Sample the direction reconstruction — checking all 800k is slow and
      // every 97th is plenty to catch a systematic error.
      if (i % 97 === 0) {
        const [x, y, z] = directionAt(tile, i);
        worstNorm = Math.max(worstNorm, Math.abs(Math.hypot(x, y, z) - 1));
      }

      domainHistogram.set(tile.domain[i], (domainHistogram.get(tile.domain[i]) ?? 0) + 1);
    }

    check('theta within [0, PI]', minTheta >= 0 && maxTheta <= Math.PI + 1e-6, `[${minTheta.toFixed(3)}, ${maxTheta.toFixed(3)}]`);
    check('phi within [0, TAU)', minPhi >= 0 && maxPhi < TAU + 1e-6, `[${minPhi.toFixed(3)}, ${maxPhi.toFixed(3)}]`);
    check('sizes span a real range', maxSize - minSize > 0.05, `[${minSize.toFixed(3)}, ${maxSize.toFixed(3)}]`);
    check('directions are unit length', worstNorm < 1e-4, `worst deviation ${worstNorm.toExponential(1)}`);
    check('no repoId is 0 (reserved)', zeroRepoIds === 0, `${zeroRepoIds} found`);

    // Picking ids are assigned by offsetting each band by the sum of all lower
    // bands, so a duplicate repoId across bands means a point is unreachable.
    for (let i = 0; i < tile.count; i++) {
      if (allRepoIds.has(tile.repoId[i])) duplicateRepoIds++;
      else allRepoIds.add(tile.repoId[i]);
    }
    check('repoIds unique across all bands so far', duplicateRepoIds === 0, `${duplicateRepoIds} duplicates`);

    seenTotal += tile.count;
    console.log('');
  }

  console.log('across all bands');
  check('band counts sum to manifest total', seenTotal === manifest.total, `${seenTotal} vs ${manifest.total}`);
  check(
    'every domain is populated',
    domainHistogram.size === manifest.domains.length,
    `${domainHistogram.size} of ${manifest.domains.length}`,
  );

  // Band 0 must genuinely be the most important points, or LOD is meaningless.
  const band0 = decodeTile(readBuffer(manifest.bands[0].file));
  const bandLast = decodeTile(readBuffer(manifest.bands[manifest.bands.length - 1].file));
  const mean = (t: ReturnType<typeof decodeTile>) => {
    let s = 0;
    for (let i = 0; i < t.count; i++) s += dequantiseSize(t.sizeQ[i]);
    return s / Math.max(1, t.count);
  };
  const m0 = mean(band0);
  const mLast = mean(bandLast);
  check('band 0 outranks the last band by size', m0 > mLast, `${m0.toFixed(3)} vs ${mLast.toFixed(3)}`);

  console.log(`\n${failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
