/**
 * Phase 0.5 world generator — positions, dependency graph, PageRank, tiles.
 *
 * Replaces gen-tiles.ts. The graph has to exist before the tiles do, because
 * PageRank decides node size, brightness, and LOD band assignment. Splitting
 * this into two scripts would mean two passes over the same 100k nodes and an
 * ordering constraint nobody would remember in three weeks.
 *
 * Nothing here is uniform noise. Positions come from von Mises-Fisher clusters,
 * and edges from Barabasi-Albert preferential attachment biased toward a node's
 * own domain — which produces a genuine power law, real hubs, and mostly
 * intra-domain dependencies with a few cross-domain bridges. That is what a
 * dependency graph actually looks like, and it is what the renderer has to cope
 * with in Phase 1.
 *
 * Usage:  npm run gen:world [-- --total 100000 --seed 7 --m 3]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { encodeTileStreaming, TAU, FLAG_LOW_SIGNAL, FLAG_ARCHIVED } from '../src/tile/format.ts';
import { encodeGraph, WEIGHT_OUTGOING, WEIGHT_MASK } from '../src/graph/format.ts';
import { pagerank, buildCsr } from '../src/graph/pagerank.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../public/tiles');

const LAYOUT_VERSION = 2;
const BAND_FRACTIONS = [0.02, 0.18, 0.8];
const AMBIENT_ARC_CAP = 2000;

const DOMAINS = [
  'AI / ML',
  'Web frameworks',
  'Databases',
  'DevOps / infra',
  'Languages / compilers',
  'Systems / embedded',
  'Data engineering',
  'Security',
  'Graphics / games',
  'Mobile',
  'Scraping / automation',
  'Scientific computing',
];

// ---------------------------------------------------------------- rng

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- sphere math

type Vec3 = [number, number, number];

function normalise(v: Vec3): Vec3 {
  const m = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / m, v[1] / m, v[2] / m];
}

function uniformDirection(rnd: () => number): Vec3 {
  const z = 2 * rnd() - 1;
  const t = rnd() * TAU;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(t), r * Math.sin(t), z];
}

function basisFrom(n: Vec3): [Vec3, Vec3] {
  const a: Vec3 = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t1 = normalise([
    a[1] * n[2] - a[2] * n[1],
    a[2] * n[0] - a[0] * n[2],
    a[0] * n[1] - a[1] * n[0],
  ]);
  const t2: Vec3 = [
    n[1] * t1[2] - n[2] * t1[1],
    n[2] * t1[0] - n[0] * t1[2],
    n[0] * t1[1] - n[1] * t1[0],
  ];
  return [t1, t2];
}

/** von Mises-Fisher on S^2 (Wood 1994). kappa ~8 is a smear, ~400 a tight knot. */
function sampleVMF(mu: Vec3, kappa: number, t1: Vec3, t2: Vec3, rnd: () => number): Vec3 {
  const u = rnd();
  const w = 1 + Math.log(u + (1 - u) * Math.exp(-2 * kappa)) / kappa;
  const s = Math.sqrt(Math.max(0, 1 - w * w));
  const v = rnd() * TAU;
  const cv = Math.cos(v) * s;
  const sv = Math.sin(v) * s;
  return normalise([
    w * mu[0] + cv * t1[0] + sv * t2[0],
    w * mu[1] + cv * t1[1] + sv * t2[1],
    w * mu[2] + cv * t1[2] + sv * t2[2],
  ]);
}

function toSpherical(d: Vec3) {
  const theta = Math.acos(Math.max(-1, Math.min(1, d[1])));
  let phi = Math.atan2(d[2], d[0]);
  if (phi < 0) phi += TAU;
  return { theta, phi };
}

// ---------------------------------------------------------------- clusters

interface Cluster {
  id: number;
  label: string;
  domain: number;
  mu: Vec3;
  t1: Vec3;
  t2: Vec3;
  kappa: number;
  weight: number;
}

/**
 * Domain poles — one anchor per domain, spread as evenly as possible.
 *
 * Farthest-point sampling rather than pure random: 12 random directions on a
 * sphere clump badly, and clumped poles make neighbouring domains overlap into
 * one indistinguishable region.
 */
function buildDomainPoles(rnd: () => number, count: number): Vec3[] {
  const poles: Vec3[] = [uniformDirection(rnd)];
  while (poles.length < count) {
    let best: Vec3 = uniformDirection(rnd);
    let bestScore = -2;
    // Sample candidates, keep whichever is furthest from everything chosen.
    for (let k = 0; k < 64; k++) {
      const cand = uniformDirection(rnd);
      let nearest = 2;
      for (const p of poles) {
        const d = 1 - (cand[0] * p[0] + cand[1] * p[1] + cand[2] * p[2]);
        if (d < nearest) nearest = d;
      }
      if (nearest > bestScore) {
        bestScore = nearest;
        best = cand;
      }
    }
    poles.push(best);
  }
  return poles;
}

function buildClusters(rnd: () => number, n: number): Cluster[] {
  // Domains must be SPATIALLY coherent. Assigning `domain = i % 12` to randomly
  // placed clusters scatters each domain across the whole globe, which breaks
  // three things at once: "fly to domain" frames points on opposite
  // hemispheres, the domain filter lights up the entire sphere, and no
  // continent can represent a sector. Real UMAP output is spatially coherent by
  // construction; the synthetic generator has to imitate that deliberately.
  const poles = buildDomainPoles(rnd, DOMAINS.length);
  const perDomain = new Int32Array(DOMAINS.length);

  const clusters: Cluster[] = [];
  for (let i = 0; i < n; i++) {
    // Bias placement toward a domain pole so clusters gather into territories,
    // then let the vMF spread of the nodes themselves blur the borders.
    const targetDomain = i % DOMAINS.length;
    const pole = poles[targetDomain];
    const [pt1, pt2] = basisFrom(pole);

    let mu = sampleVMF(pole, 9, pt1, pt2, rnd);
    for (let attempt = 0; attempt < 24; attempt++) {
      const tooClose = clusters.some((c) => c.mu[0] * mu[0] + c.mu[1] * mu[1] + c.mu[2] * mu[2] > 0.93);
      if (!tooClose) break;
      mu = sampleVMF(pole, 9, pt1, pt2, rnd);
    }

    // Assign by nearest pole rather than by target, so the occasional cluster
    // that lands over a border belongs to the region it is actually in.
    let domain = targetDomain;
    let nearest = -2;
    for (let d = 0; d < poles.length; d++) {
      const dot = poles[d][0] * mu[0] + poles[d][1] * mu[1] + poles[d][2] * mu[2];
      if (dot > nearest) {
        nearest = dot;
        domain = d;
      }
    }

    const [t1, t2] = basisFrom(mu);
    perDomain[domain]++;
    clusters.push({
      id: i,
      label: `${DOMAINS[domain]} ${perDomain[domain]}`,
      domain,
      mu,
      t1,
      t2,
      kappa: 30 + Math.pow(rnd(), 2) * 900,
      weight: 0.25 + Math.pow(rnd(), 1.6) * 2.5,
    });
  }
  return clusters;
}

function cumulative(weights: number[]): Float64Array {
  const out = new Float64Array(weights.length);
  let acc = 0;
  for (let i = 0; i < weights.length; i++) {
    acc += weights[i];
    out[i] = acc;
  }
  for (let i = 0; i < out.length; i++) out[i] /= acc;
  return out;
}

function pickWeighted(cdf: Float64Array, u: number): number {
  let lo = 0;
  let hi = cdf.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (u <= cdf[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

// ---------------------------------------------------------------- graph

interface DirectedEdges {
  src: Uint32Array;
  dst: Uint32Array;
}

/**
 * Barabasi-Albert preferential attachment, biased toward a node's own domain.
 *
 * Each new node attaches to `m` existing nodes chosen with probability
 * proportional to degree. The classic trick: keep an array in which every node
 * appears once per unit of degree, then sample it uniformly. That is exactly
 * preferential attachment, with no cumulative-weight rebuild per node.
 */
function buildDependencyGraph(
  total: number,
  domainOf: Uint8Array,
  m: number,
  intraDomainBias: number,
  rnd: () => number,
): DirectedEdges {
  const domainCount = DOMAINS.length;
  const globalPool: number[] = [];
  const domainPools: number[][] = Array.from({ length: domainCount }, () => []);

  const seedCount = Math.max(m + 1, 8);
  const src: number[] = [];
  const dst: number[] = [];

  // Seed: a small fully-connected core so preferential attachment has something
  // to prefer. Without it the first few nodes attach to nothing and the whole
  // degree distribution starts flat.
  for (let i = 0; i < seedCount; i++) {
    for (let j = 0; j < i; j++) {
      src.push(i);
      dst.push(j);
      globalPool.push(i, j);
      domainPools[domainOf[i]].push(i);
      domainPools[domainOf[j]].push(j);
    }
  }

  const chosen = new Set<number>();
  for (let i = seedCount; i < total; i++) {
    const d = domainOf[i];
    const own = domainPools[d];
    chosen.clear();

    let guard = 0;
    while (chosen.size < m && guard < m * 40) {
      guard++;
      const useOwn = own.length > 0 && rnd() < intraDomainBias;
      const pool = useOwn ? own : globalPool;
      if (pool.length === 0) break;
      const candidate = pool[(rnd() * pool.length) | 0];
      if (candidate !== i) chosen.add(candidate);
    }

    for (const target of chosen) {
      src.push(i);
      dst.push(target);
      globalPool.push(i, target);
      domainPools[d].push(i);
      domainPools[domainOf[target]].push(target);
    }
  }

  return { src: Uint32Array.from(src), dst: Uint32Array.from(dst) };
}

/** Undirected CSR from a directed edge list — each edge lands in both rows. */
function buildUndirectedCsr(n: number, edges: DirectedEdges, weightOf: (k: number) => number) {
  const e = edges.src.length;
  const degree = new Uint32Array(n);
  for (let k = 0; k < e; k++) {
    degree[edges.src[k]]++;
    degree[edges.dst[k]]++;
  }

  const offsets = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) offsets[i + 1] = offsets[i] + degree[i];

  const entries = offsets[n];
  const targets = new Uint32Array(entries);
  const weights = new Uint16Array(entries);
  const cursor = offsets.slice(0, n);

  for (let k = 0; k < e; k++) {
    const a = edges.src[k];
    const b = edges.dst[k];
    const w = Math.min(weightOf(k), WEIGHT_MASK);
    targets[cursor[a]] = b;
    weights[cursor[a]] = w | WEIGHT_OUTGOING;
    cursor[a]++;
    targets[cursor[b]] = a;
    weights[cursor[b]] = w;
    cursor[b]++;
  }

  return { offsets, targets, weights };
}

// ---------------------------------------------------------------- main

function arg(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
}

function main() {
  const total = arg('total', 100_000);
  const seed = arg('seed', 7);
  const clusterCount = arg('clusters', 60);
  const m = arg('m', 3);
  const backgroundFrac = 0.12;
  const intraDomainBias = 0.8;

  const rnd = mulberry32(seed);
  const clusters = buildClusters(rnd, clusterCount);
  const cdf = cumulative(clusters.map((c) => c.weight));

  console.log(`GitGlobe world generator — ${total.toLocaleString()} nodes, seed ${seed}\n`);
  const t0 = Date.now();

  // ---- 1. positions -------------------------------------------------------
  process.stdout.write('  positions… ');
  const theta = new Float64Array(total);
  const phi = new Float64Array(total);
  const domainOf = new Uint8Array(total);
  const clusterOf = new Int16Array(total);

  for (let i = 0; i < total; i++) {
    let d: Vec3;
    let ci = -1;
    if (rnd() < backgroundFrac) {
      d = uniformDirection(rnd);
    } else {
      ci = pickWeighted(cdf, rnd());
      const c = clusters[ci];
      d = sampleVMF(c.mu, c.kappa, c.t1, c.t2, rnd);
    }
    const sp = toSpherical(d);
    theta[i] = sp.theta;
    phi[i] = sp.phi;
    clusterOf[i] = ci;
    domainOf[i] = ci < 0 ? DOMAINS.length - 1 : clusters[ci].domain;
  }
  console.log('done');

  // ---- 2. dependency graph ------------------------------------------------
  process.stdout.write(`  dependency graph (BA, m=${m}, intra-domain ${intraDomainBias})… `);
  const edges = buildDependencyGraph(total, domainOf, m, intraDomainBias, rnd);
  console.log(`${edges.src.length.toLocaleString()} directed edges`);

  // ---- 3. PageRank --------------------------------------------------------
  process.stdout.write('  pagerank… ');
  const directed = buildCsr(total, edges.src, edges.dst);
  const pr = pagerank(directed, { damping: 0.85, tolerance: 1e-10, maxIterations: 300 });
  const rankSum = pr.rank.reduce((s, v) => s + v, 0);
  console.log(
    `${pr.iterations} iterations, delta ${pr.delta.toExponential(2)}, ` +
      `${pr.converged ? 'converged' : 'HIT ITERATION CAP'}, sum ${rankSum.toFixed(12)}`,
  );
  if (!pr.converged) console.warn('  ! pagerank did not converge — ranks are not trustworthy');

  // ---- 4. rank ordering ---------------------------------------------------
  // Global picking ids are assigned in rank order, so band 0 is genuinely the
  // most important 2% and `newId` doubles as an importance index.
  process.stdout.write('  ranking… ');
  const order = new Uint32Array(total);
  for (let i = 0; i < total; i++) order[i] = i;
  order.sort((a, b) => pr.rank[b] - pr.rank[a]);

  const newId = new Uint32Array(total);
  for (let position = 0; position < total; position++) newId[order[position]] = position;

  // Node radius scales with LOG rank (ADR-008): raw PageRank is power-law, so
  // using it directly makes one node enormous and every other node invisible.
  const logRank = new Float64Array(total);
  let minLog = Infinity;
  let maxLog = -Infinity;
  for (let i = 0; i < total; i++) {
    const v = Math.log(pr.rank[i] + 1e-12);
    logRank[i] = v;
    if (v < minLog) minLog = v;
    if (v > maxLog) maxLog = v;
  }
  const span = Math.max(1e-9, maxLog - minLog);
  const sizeOf = (i: number) => 0.1 + 0.9 * ((logRank[i] - minLog) / span);
  console.log('done');

  // ---- 5. tiles -----------------------------------------------------------
  mkdirSync(OUT_DIR, { recursive: true });

  const bandSizes = BAND_FRACTIONS.map((f) => Math.round(total * f));
  bandSizes[bandSizes.length - 1] = total - bandSizes.slice(0, -1).reduce((a, b) => a + b, 0);

  let cursor = 0;
  const manifestBands: Array<{ band: number; count: number; bytes: number; file: string }> = [];

  for (let band = 0; band < bandSizes.length; band++) {
    const count = bandSizes[band];
    const start = cursor;
    cursor += count;

    const buf = encodeTileStreaming(count, { layoutVersion: LAYOUT_VERSION, lodBand: band }, (i, out) => {
      const srcIndex = order[start + i];
      out.theta = theta[srcIndex];
      out.phi = phi[srcIndex];
      // repoId is the rank position + 1, so it is stable, unique, and ordered.
      out.repoId = start + i + 1;
      out.size = sizeOf(srcIndex);
      out.domain = domainOf[srcIndex];
      const isolated = directed.offsets[srcIndex + 1] === directed.offsets[srcIndex];
      out.flags =
        (clusterOf[srcIndex] < 0 ? FLAG_LOW_SIGNAL : 0) | (isolated && band === 2 ? FLAG_ARCHIVED : 0);
    });

    const file = `band-${band}.bin`;
    writeFileSync(resolve(OUT_DIR, file), Buffer.from(buf));
    manifestBands.push({ band, count, bytes: buf.byteLength, file });
    console.log(
      `  band ${band}: ${count.toLocaleString().padStart(9)} nodes  ${(buf.byteLength / 1e6).toFixed(2).padStart(6)} MB  → ${file}`,
    );
  }

  // ---- 6. graph artifact --------------------------------------------------
  process.stdout.write('  graph artifact… ');

  // Remap every endpoint to its rank-order id, so the client needs no
  // translation layer between picking ids and graph node ids.
  const remappedSrc = new Uint32Array(edges.src.length);
  const remappedDst = new Uint32Array(edges.dst.length);
  for (let k = 0; k < edges.src.length; k++) {
    remappedSrc[k] = newId[edges.src[k]];
    remappedDst[k] = newId[edges.dst[k]];
  }

  // Edge weight: how strong the connection looks. Combined endpoint rank is the
  // honest proxy — an edge between two hubs matters more than one between two
  // leaves — normalised into the 15 bits the format gives us.
  const combined = new Float64Array(edges.src.length);
  let maxCombined = 0;
  for (let k = 0; k < edges.src.length; k++) {
    const v = Math.log(pr.rank[edges.src[k]] + pr.rank[edges.dst[k]] + 1e-12) - 2 * minLog;
    combined[k] = v;
    if (v > maxCombined) maxCombined = v;
  }
  const weightOf = (k: number) => Math.round((combined[k] / Math.max(1e-9, maxCombined)) * WEIGHT_MASK);

  const csr = buildUndirectedCsr(
    total,
    { src: remappedSrc, dst: remappedDst },
    weightOf,
  );

  // Ambient backbone: the strongest edges by combined endpoint rank. This is
  // what turns a hairball into a legible structure — PageRank does not decide
  // which nodes connect, it decides which connections are worth drawing.
  const edgeOrder = new Uint32Array(edges.src.length);
  for (let k = 0; k < edgeOrder.length; k++) edgeOrder[k] = k;
  edgeOrder.sort((a, b) => combined[b] - combined[a]);

  const ambientCount = Math.min(AMBIENT_ARC_CAP, edgeOrder.length);
  const ambient = new Uint32Array(ambientCount * 2);
  for (let i = 0; i < ambientCount; i++) {
    const k = edgeOrder[i];
    ambient[i * 2] = remappedSrc[k];
    ambient[i * 2 + 1] = remappedDst[k];
  }

  const rankByNewId = new Float32Array(total);
  for (let i = 0; i < total; i++) rankByNewId[newId[i]] = pr.rank[i];

  const graphBuf = encodeGraph({
    rank: rankByNewId,
    offsets: csr.offsets,
    targets: csr.targets,
    weights: csr.weights,
    ambient,
    layoutVersion: LAYOUT_VERSION,
  });
  writeFileSync(resolve(OUT_DIR, 'graph.bin'), Buffer.from(graphBuf));
  console.log(
    `${(graphBuf.byteLength / 1e6).toFixed(2)} MB  (${csr.targets.length.toLocaleString()} CSR entries, ` +
      `${ambientCount.toLocaleString()} ambient arcs)`,
  );

  // ---- 7. manifest --------------------------------------------------------
  // Degree stats, so the manifest can prove the graph is actually scale-free
  // rather than us assuming it.
  const degrees = new Uint32Array(total);
  for (let i = 0; i < total; i++) degrees[i] = csr.offsets[i + 1] - csr.offsets[i];
  const sortedDeg = Array.from(degrees).sort((a, b) => b - a);
  const meanDeg = sortedDeg.reduce((s, v) => s + v, 0) / total;

  const manifest = {
    layoutVersion: LAYOUT_VERSION,
    generatedAt: new Date().toISOString(),
    seed,
    total,
    synthetic: true,
    bands: manifestBands,
    domains: DOMAINS,
    graph: {
      file: 'graph.bin',
      bytes: graphBuf.byteLength,
      directedEdges: edges.src.length,
      csrEntries: csr.targets.length,
      ambientArcs: ambientCount,
      pagerank: {
        damping: 0.85,
        iterations: pr.iterations,
        converged: pr.converged,
        delta: pr.delta,
      },
      degree: {
        mean: Number(meanDeg.toFixed(2)),
        max: sortedDeg[0],
        p50: sortedDeg[Math.floor(total * 0.5)],
        p99: sortedDeg[Math.floor(total * 0.01)],
      },
    },
    clusters: clusters.map((c) => {
      const sp = toSpherical(c.mu);
      return { id: c.id, label: c.label, domain: c.domain, theta: sp.theta, phi: sp.phi, kappa: c.kappa };
    }),
  };
  writeFileSync(resolve(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));

  const tileBytes = manifestBands.reduce((a, b) => a + b.bytes, 0);
  console.log(
    `\nDone in ${((Date.now() - t0) / 1000).toFixed(1)}s — ` +
      `${((tileBytes + graphBuf.byteLength) / 1e6).toFixed(1)} MB total\n` +
      `  degree: mean ${meanDeg.toFixed(1)}, median ${manifest.graph.degree.p50}, ` +
      `p99 ${manifest.graph.degree.p99}, max ${sortedDeg[0]}  ← a power law, not a bell curve`,
  );
}

main();
