/**
 * Phase 0 synthetic tile generator.
 *
 * Deliberately NOT uniform noise. Uniform points on a sphere look nothing like
 * a real UMAP layout, and would let us "pass" Phase 0 with a renderer that
 * falls over on the clumped, wildly-varying-density data Phase 2 produces.
 *
 * Instead: von Mises-Fisher clusters (the sphere's analogue of a Gaussian) at
 * varying concentrations, plus a diffuse background — which is what an actual
 * semantic layout looks like.
 *
 * Usage:  npm run gen:tiles [-- --total 1000000 --seed 7]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodeTileStreaming, TAU, FLAG_LOW_SIGNAL, FLAG_ARCHIVED } from '../src/tile/format.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../public/tiles');

const LAYOUT_VERSION = 1;

/** LOD bands, in points. Sums to `total`; band 0 is the always-loaded set. */
const BAND_FRACTIONS = [0.02, 0.18, 0.8];

/** Domain names — Phase 3 replaces these with LLM-generated cluster labels. */
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

/** Uniform on S² — inverse-CDF on cos(theta), not uniform theta (which poles-clumps). */
function uniformDirection(rnd: () => number): Vec3 {
  const z = 2 * rnd() - 1;
  const t = rnd() * TAU;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(t), r * Math.sin(t), z];
}

/** An orthonormal basis with `n` as its third axis. Branchless-ish, no degenerate case. */
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

/**
 * Sample the von Mises-Fisher distribution on S² (Wood 1994).
 * `kappa` is concentration: ~8 is a loose smear, ~400 is a tight knot.
 */
function sampleVMF(mu: Vec3, kappa: number, t1: Vec3, t2: Vec3, rnd: () => number): Vec3 {
  const u = rnd();
  // Numerically stable for large kappa, where exp(-2k) underflows to 0 harmlessly.
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

function toSpherical(d: Vec3): { theta: number; phi: number } {
  // three.js is Y-up: theta measured from +Y, phi in the XZ plane.
  const theta = Math.acos(Math.max(-1, Math.min(1, d[1])));
  let phi = Math.atan2(d[2], d[0]);
  if (phi < 0) phi += TAU;
  return { theta, phi };
}

// ---------------------------------------------------------------- cluster model

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

function buildClusters(rnd: () => number, n: number): Cluster[] {
  const clusters: Cluster[] = [];
  for (let i = 0; i < n; i++) {
    // Rejection-sample so cluster centres don't sit on top of each other —
    // real UMAP output has visible gaps between nebulae.
    let mu = uniformDirection(rnd);
    for (let attempt = 0; attempt < 24; attempt++) {
      const tooClose = clusters.some((c) => c.mu[0] * mu[0] + c.mu[1] * mu[1] + c.mu[2] * mu[2] > 0.86);
      if (!tooClose) break;
      mu = uniformDirection(rnd);
    }
    const [t1, t2] = basisFrom(mu);
    const domain = i % DOMAINS.length;
    clusters.push({
      id: i,
      label: `${DOMAINS[domain]} ${Math.floor(i / DOMAINS.length) + 1}`,
      domain,
      mu,
      t1,
      t2,
      // Wide spread of concentrations — some tight niches, some sprawling fields.
      kappa: 12 + Math.pow(rnd(), 2) * 900,
      weight: 0.25 + Math.pow(rnd(), 1.6) * 2.5,
    });
  }
  return clusters;
}

/** Normalised cumulative weights, for weighted sampling. */
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

/** Binary search — called 1M times, so the linear scan actually shows up. */
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

// ---------------------------------------------------------------- main

function arg(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
}

function main() {
  // 100k is the working default: fast to regenerate, fast to reload, enough
  // density to judge how the layout reads. The Phase 0 exit criterion is still
  // measured at 1M — `npm run gen:tiles:stress` — because a renderer that only
  // holds 60fps at 100k tells us nothing about the target.
  const total = arg('total', 100_000);
  const seed = arg('seed', 7);
  const clusterCount = arg('clusters', 60);
  const backgroundFrac = 0.12;

  const rnd = mulberry32(seed);
  const clusters = buildClusters(rnd, clusterCount);
  const cdf = cumulative(clusters.map((c) => c.weight));

  console.log(`Generating ${total.toLocaleString()} synthetic points across ${clusterCount} clusters (seed ${seed})…`);
  const t0 = Date.now();

  // Pass 1: positions, sizes, cluster assignment. Typed arrays only — building
  // 1M objects here would cost hundreds of MB and a GC pause.
  const theta = new Float64Array(total);
  const phi = new Float64Array(total);
  const size = new Float32Array(total);
  const clusterOf = new Int16Array(total);

  for (let i = 0; i < total; i++) {
    let d: Vec3;
    let ci = -1;
    if (rnd() < backgroundFrac) {
      d = uniformDirection(rnd); // diffuse background — the long tail of odd repos
    } else {
      ci = pickWeighted(cdf, rnd());
      const c = clusters[ci];
      d = sampleVMF(c.mu, c.kappa, c.t1, c.t2, rnd);
    }
    const sp = toSpherical(d);
    theta[i] = sp.theta;
    phi[i] = sp.phi;
    clusterOf[i] = ci;

    // Star counts follow a power law, and node radius scales with LOG stars
    // (ADR-008). Sampling `pow(rnd(), k)` directly saturates the top of the
    // range — the most important 2% all come out the same size, which defeats
    // the entire point of encoding importance in radius. Sample Pareto stars,
    // then take the log, which is what the real pipeline will do.
    const stars = 5 * Math.pow(1 - rnd(), -1 / 1.15);
    size[i] = Math.log1p(stars);
  }

  // Normalise log-sizes to [0, 1], then lift off the floor so the smallest
  // repos are still faintly visible rather than sub-pixel.
  let maxLog = 0;
  for (let i = 0; i < total; i++) if (size[i] > maxLog) maxLog = size[i];
  for (let i = 0; i < total; i++) size[i] = 0.1 + 0.9 * (size[i] / maxLog);

  // Pass 2: rank by size so LOD band 0 is genuinely "the most important 2%".
  // TypedArray.prototype.sort takes a comparator and sorts in place, so this
  // never boxes 1M numbers into a regular Array.
  const sorted = new Uint32Array(total);
  for (let i = 0; i < total; i++) sorted[i] = i;
  sorted.sort((a, b) => size[b] - size[a]);

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
      const src = sorted[start + i];
      out.theta = theta[src];
      out.phi = phi[src];
      out.repoId = src + 1; // 0 is reserved for "nothing picked"
      // One global monotone scale, no per-band remap: a band-local rescale
      // would make a band-2 point look the same size as a band-0 point, and
      // relative importance has to survive crossing an LOD boundary.
      out.size = size[src];
      const c = clusterOf[src];
      out.domain = c < 0 ? DOMAINS.length - 1 : clusters[c].domain;
      out.flags = (c < 0 ? FLAG_LOW_SIGNAL : 0) | (size[src] < 0.13 ? FLAG_ARCHIVED : 0);
    });

    const file = `band-${band}.bin`;
    writeFileSync(resolve(OUT_DIR, file), Buffer.from(buf));
    manifestBands.push({ band, count, bytes: buf.byteLength, file });
    console.log(
      `  band ${band}: ${count.toLocaleString().padStart(9)} points  ${(buf.byteLength / 1e6).toFixed(2).padStart(6)} MB  → ${file}`,
    );
  }

  const manifest = {
    layoutVersion: LAYOUT_VERSION,
    generatedAt: new Date().toISOString(),
    seed,
    total,
    synthetic: true,
    bands: manifestBands,
    domains: DOMAINS,
    clusters: clusters.map((c) => {
      const sp = toSpherical(c.mu);
      return { id: c.id, label: c.label, domain: c.domain, theta: sp.theta, phi: sp.phi, kappa: c.kappa };
    }),
  };
  writeFileSync(resolve(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));

  const totalBytes = manifestBands.reduce((a, b) => a + b.bytes, 0);
  console.log(
    `\nDone in ${((Date.now() - t0) / 1000).toFixed(1)}s — ${(totalBytes / 1e6).toFixed(1)} MB total ` +
      `(${(totalBytes / total).toFixed(1)} bytes/point)`,
  );
}

main();
