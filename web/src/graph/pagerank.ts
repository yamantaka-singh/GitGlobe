/**
 * PageRank over a directed graph in CSR form.
 *
 * Pure function, no I/O, no dependencies — so it can be tested against
 * known-answer graphs rather than eyeballed. See pagerank.test.ts.
 *
 * Uses the "push" formulation, which works directly from out-edge CSR:
 *
 *   r'[j] = (1-d)/N  +  d * (danglingMass/N)  +  d * SUM over i->j of r[i]/outdeg(i)
 *
 * The dangling term matters. Nodes with no outgoing edges are common in real
 * dependency graphs (leaf libraries that depend on nothing), and dropping their
 * mass makes the vector stop summing to 1 — which silently rescales every rank
 * and breaks any threshold built on top of it.
 */

export interface CsrGraph {
  /** Number of nodes. */
  n: number;
  /** Row starts, length n+1. `offsets[i]`..`offsets[i+1]` indexes `targets`. */
  offsets: Uint32Array;
  /** Neighbour node indices. */
  targets: Uint32Array;
}

export interface PageRankOptions {
  /** Probability of following an edge rather than teleporting. */
  damping?: number;
  /** Stop when the L1 change between iterations drops below this. */
  tolerance?: number;
  maxIterations?: number;
}

export interface PageRankResult {
  rank: Float64Array;
  iterations: number;
  /** Final L1 delta — if this is >= tolerance, the run hit maxIterations. */
  delta: number;
  converged: boolean;
}

export function pagerank(graph: CsrGraph, options: PageRankOptions = {}): PageRankResult {
  const { n, offsets, targets } = graph;
  const damping = options.damping ?? 0.85;
  const tolerance = options.tolerance ?? 1e-9;
  const maxIterations = options.maxIterations ?? 200;

  if (n === 0) {
    return { rank: new Float64Array(0), iterations: 0, delta: 0, converged: true };
  }
  if (offsets.length !== n + 1) {
    throw new Error(`CSR offsets must have length n+1 (${n + 1}), got ${offsets.length}`);
  }

  // Float64 throughout: at 1M nodes the individual ranks are ~1e-6, and Float32
  // accumulation error is large enough to disturb the ordering of near-ties.
  let rank = new Float64Array(n).fill(1 / n);
  let next = new Float64Array(n);

  const outDegree = new Uint32Array(n);
  for (let i = 0; i < n; i++) outDegree[i] = offsets[i + 1] - offsets[i];

  const teleport = (1 - damping) / n;
  let iterations = 0;
  let delta = Infinity;

  for (; iterations < maxIterations; iterations++) {
    let danglingMass = 0;
    for (let i = 0; i < n; i++) {
      if (outDegree[i] === 0) danglingMass += rank[i];
    }

    const base = teleport + (damping * danglingMass) / n;
    next.fill(base);

    for (let i = 0; i < n; i++) {
      const deg = outDegree[i];
      if (deg === 0) continue;
      const share = (damping * rank[i]) / deg;
      const start = offsets[i];
      const end = offsets[i + 1];
      for (let k = start; k < end; k++) next[targets[k]] += share;
    }

    delta = 0;
    for (let i = 0; i < n; i++) delta += Math.abs(next[i] - rank[i]);

    const swap = rank;
    rank = next;
    next = swap;

    if (delta < tolerance) {
      iterations++;
      break;
    }
  }

  return { rank, iterations, delta, converged: delta < tolerance };
}

/**
 * Build an out-edge CSR from an edge list. Edges are sorted by source in a
 * counting pass, so this is O(n + e) rather than O(e log e).
 */
export function buildCsr(n: number, sources: Uint32Array, destinations: Uint32Array): CsrGraph {
  const e = sources.length;
  if (destinations.length !== e) {
    throw new Error(`sources and destinations must be the same length (${e} vs ${destinations.length})`);
  }

  const offsets = new Uint32Array(n + 1);
  for (let k = 0; k < e; k++) {
    const s = sources[k];
    if (s >= n) throw new Error(`edge source ${s} out of range for ${n} nodes`);
    offsets[s + 1]++;
  }
  for (let i = 0; i < n; i++) offsets[i + 1] += offsets[i];

  const targets = new Uint32Array(e);
  const cursor = offsets.slice(0, n);
  for (let k = 0; k < e; k++) {
    const d = destinations[k];
    if (d >= n) throw new Error(`edge destination ${d} out of range for ${n} nodes`);
    targets[cursor[sources[k]]++] = d;
  }

  return { n, offsets, targets };
}
