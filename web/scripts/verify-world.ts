/**
 * Integrity check for a generated world — tiles AND graph.
 *
 * Runs against whatever is in public/tiles: synthetic now, real UMAP output and
 * real deps.dev edges from Phase 2 onward. Catches the failure modes that would
 * otherwise surface as "the globe looks wrong" with no clue why — quantisation
 * overflow, id collisions, a CSR that points off the end of itself, PageRank
 * that silently stopped summing to 1.
 *
 * Usage:  npm run verify              (checks public/tiles)
 *         npm run verify -- <dir>    (checks any world directory)
 *
 * The directory argument is what lets the Python pipeline be tested against
 * these exact checks: it writes a world to a temp directory and runs this
 * verifier over it. Duplicating the checks in Python would mean maintaining two
 * definitions of "correct" that are free to drift apart.
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
import { decodeGraph, WEIGHT_OUTGOING } from '../src/graph/format.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const TILES = process.argv[2] ? resolve(process.argv[2]) : resolve(HERE, '../public/tiles');

let failures = 0;

function check(name: string, ok: boolean, detail = '') {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
}

function readBuffer(file: string): ArrayBuffer {
  const buf = readFileSync(resolve(TILES, file));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function main() {
  const manifest = JSON.parse(readFileSync(resolve(TILES, 'manifest.json'), 'utf8'));
  console.log(
    `Verifying ${manifest.total.toLocaleString()} nodes, layout v${manifest.layoutVersion}` +
      `${manifest.synthetic ? ' (synthetic)' : ''}\n`,
  );

  // ---- tiles ---------------------------------------------------------------
  let seenTotal = 0;
  const allRepoIds = new Set<number>();
  let duplicateRepoIds = 0;
  const domainHistogram = new Set<number>();

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
      if (allRepoIds.has(tile.repoId[i])) duplicateRepoIds++;
      else allRepoIds.add(tile.repoId[i]);
      domainHistogram.add(tile.domain[i]);
      if (i % 97 === 0) {
        const [x, y, z] = directionAt(tile, i);
        worstNorm = Math.max(worstNorm, Math.abs(Math.hypot(x, y, z) - 1));
      }
    }

    check('theta within [0, PI]', minTheta >= 0 && maxTheta <= Math.PI + 1e-6, `[${minTheta.toFixed(3)}, ${maxTheta.toFixed(3)}]`);
    check('phi within [0, TAU)', minPhi >= 0 && maxPhi < TAU + 1e-6, `[${minPhi.toFixed(3)}, ${maxPhi.toFixed(3)}]`);
    check('sizes span a real range', maxSize - minSize > 0.05, `[${minSize.toFixed(3)}, ${maxSize.toFixed(3)}]`);
    check('directions unit length', worstNorm < 1e-4, `worst ${worstNorm.toExponential(1)}`);
    check('no repoId is 0 (reserved)', zeroRepoIds === 0);
    check('repoIds globally unique', duplicateRepoIds === 0, `${duplicateRepoIds} duplicates`);
    seenTotal += tile.count;
    console.log('');
  }

  console.log('tiles overall');
  check('band counts sum to total', seenTotal === manifest.total, `${seenTotal} vs ${manifest.total}`);
  check('every domain populated', domainHistogram.size === manifest.domains.length,
    `${domainHistogram.size} of ${manifest.domains.length}`);

  const band0 = decodeTile(readBuffer(manifest.bands[0].file));
  const bandLast = decodeTile(readBuffer(manifest.bands[manifest.bands.length - 1].file));
  const meanSize = (t: ReturnType<typeof decodeTile>) => {
    let s = 0;
    for (let i = 0; i < t.count; i++) s += dequantiseSize(t.sizeQ[i]);
    return s / Math.max(1, t.count);
  };
  check('band 0 outranks the last band', meanSize(band0) > meanSize(bandLast),
    `${meanSize(band0).toFixed(3)} vs ${meanSize(bandLast).toFixed(3)}`);
  console.log('');

  // ---- graph ---------------------------------------------------------------
  if (!manifest.graph) {
    console.log('graph — none in manifest, skipping\n');
  } else {
    console.log(`graph  (${manifest.graph.file})`);
    const graph = decodeGraph(readBuffer(manifest.graph.file));

    check('node count matches manifest', graph.nodeCount === manifest.total,
      `${graph.nodeCount} vs ${manifest.total}`);
    check('layout version matches', graph.layoutVersion === manifest.layoutVersion);

    let rankSum = 0;
    let minRank = Infinity;
    let maxRank = -Infinity;
    for (let i = 0; i < graph.nodeCount; i++) {
      rankSum += graph.rank[i];
      if (graph.rank[i] < minRank) minRank = graph.rank[i];
      if (graph.rank[i] > maxRank) maxRank = graph.rank[i];
    }
    // Float32 storage of 100k values that sum to 1 loses precision, so the
    // tolerance is generous — this catches "we dropped the dangling mass", not
    // rounding.
    check('pagerank sums to ~1', Math.abs(rankSum - 1) < 1e-3, rankSum.toFixed(9));
    check('all ranks positive', minRank > 0, `min ${minRank.toExponential(2)}`);
    check('rank spans orders of magnitude', maxRank / minRank > 100,
      `max/min = ${(maxRank / minRank).toFixed(0)}x — a power law, not a bell curve`);

    // Node ids are assigned in rank order, so rank must decrease monotonically.
    // If this breaks, LOD banding and "node 0 is the top hub" both silently lie.
    let monotonic = true;
    for (let i = 1; i < graph.nodeCount; i++) {
      if (graph.rank[i] > graph.rank[i - 1] + 1e-12) {
        monotonic = false;
        break;
      }
    }
    check('rank decreases with node id', monotonic);

    let offsetsMonotonic = graph.offsets[0] === 0;
    for (let i = 1; i <= graph.nodeCount; i++) {
      if (graph.offsets[i] < graph.offsets[i - 1]) offsetsMonotonic = false;
    }
    check('csr offsets monotonic and zero-based', offsetsMonotonic);
    check('csr offsets end at edgeCount', graph.offsets[graph.nodeCount] === graph.edgeCount,
      `${graph.offsets[graph.nodeCount]} vs ${graph.edgeCount}`);

    let badTarget = -1;
    let outgoing = 0;
    for (let k = 0; k < graph.edgeCount; k++) {
      if (graph.targets[k] >= graph.nodeCount) { badTarget = k; break; }
      if (graph.weights[k] & WEIGHT_OUTGOING) outgoing++;
    }
    check('every csr target in range', badTarget < 0, badTarget >= 0 ? `index ${badTarget}` : '');
    check('half the csr entries are outgoing', Math.abs(outgoing * 2 - graph.edgeCount) <= 1,
      `${outgoing} of ${graph.edgeCount}`);

    // The undirected CSR must be symmetric: if A lists B, B must list A.
    // A one-sided edge means hovering A shows a link that hovering B denies.
    let asymmetric = 0;
    const sampleStep = Math.max(1, Math.floor(graph.nodeCount / 2000));
    for (let i = 0; i < graph.nodeCount; i += sampleStep) {
      for (let k = graph.offsets[i]; k < graph.offsets[i + 1]; k++) {
        const j = graph.targets[k];
        let found = false;
        for (let m = graph.offsets[j]; m < graph.offsets[j + 1]; m++) {
          if (graph.targets[m] === i) { found = true; break; }
        }
        if (!found) asymmetric++;
      }
    }
    check('csr is symmetric (sampled)', asymmetric === 0, `${asymmetric} one-sided edges`);

    let badAmbient = 0;
    let selfLoops = 0;
    for (let i = 0; i < graph.ambientCount; i++) {
      const a = graph.ambient[i * 2];
      const b = graph.ambient[i * 2 + 1];
      if (a >= graph.nodeCount || b >= graph.nodeCount) badAmbient++;
      if (a === b) selfLoops++;
    }
    check('ambient endpoints in range', badAmbient === 0, `${badAmbient} bad`);
    check('no ambient self-loops', selfLoops === 0, `${selfLoops} found`);
    check('ambient count matches manifest', graph.ambientCount === manifest.graph.ambientArcs);

    // The backbone is only useful if it actually selects hubs. If ambient edges
    // averaged the same rank as random edges, PageRank is not doing its job.
    let ambientRankSum = 0;
    for (let i = 0; i < graph.ambientCount; i++) {
      ambientRankSum += graph.rank[graph.ambient[i * 2]] + graph.rank[graph.ambient[i * 2 + 1]];
    }
    const ambientMean = ambientRankSum / Math.max(1, graph.ambientCount * 2);
    const globalMean = rankSum / graph.nodeCount;
    check('backbone selects high-rank nodes', ambientMean > globalMean * 5,
      `${(ambientMean / globalMean).toFixed(1)}x the average node rank`);
    console.log('');
  }

  console.log(failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
