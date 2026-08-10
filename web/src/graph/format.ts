/**
 * GitGlobe graph artifact — see docs/superpowers/specs/2026-08-06-cyberpunk-graph-globe-design.md
 *
 * The edge graph is a SEPARATE file from the tiles. Tiles stay a pure
 * spatial-visual record; the graph is its own artifact. That mirrors the
 * production architecture, where edges live in Postgres and never enter a tile.
 *
 * Layout (little-endian):
 *
 *   offset            bytes     field
 *   0                 4         magic 'GGG1'
 *   4                 4         nodeCount    (n)
 *   8                 4         edgeCount    (e — CSR entries, = 2x directed edges)
 *   12                4         ambientCount (a)
 *   16                2         layoutVersion
 *   18                6         reserved
 *   24                4n        rank      Float32[n]     PageRank, sums to 1
 *   24 + 4n           4(n+1)    offsets   Uint32[n+1]    CSR row starts
 *   28 + 8n           4e        targets   Uint32[e]      CSR neighbours
 *   28 + 8n + 4e      8a        ambient   Uint32[2a]     backbone edge pairs
 *   28 + 8n + 4e + 8a 2e        weights   Uint16[e]      bit15 = outgoing
 *
 * Every 4-byte array precedes the single 2-byte array, which guarantees natural
 * alignment for all views regardless of n, e or a. Get that ordering wrong and
 * `new Uint32Array(buffer, offset, len)` throws on odd inputs only — the kind of
 * bug that passes every test until real data arrives.
 *
 * The CSR is UNDIRECTED: each directed edge appears in both endpoints' rows,
 * with a direction bit in the weight. The renderer's only query is "everything
 * this repo touches", and that wants both directions in one O(1) lookup.
 */

export const GRAPH_MAGIC = 0x47474731; // 'GGG1'
export const GRAPH_HEADER_BYTES = 24;

/** Weight field: high bit marks the edge as outgoing from the row's node. */
export const WEIGHT_OUTGOING = 0x8000;
/**
 * Bits 13-14: edge kind. Must match KIND_SHIFT / KIND_MASK in
 * pipeline/src/gitglobe/tiles/format.py — the cross-language format test is
 * what keeps the two from drifting.
 */
export const KIND_SHIFT = 13;
export const KIND_MASK = 0x6000;
export const WEIGHT_MASK = 0x1fff;

/** Edge kinds, matching `edge.kind` in migration 002. */
export const EDGE_DEPENDS_ON = 0;
export const EDGE_SIMILAR_TO = 1;
export const EDGE_USED_WITH = 2;

export function edgeKind(weight: number): number {
  return (weight & KIND_MASK) >> KIND_SHIFT;
}

export interface GraphHeader {
  nodeCount: number;
  edgeCount: number;
  ambientCount: number;
  layoutVersion: number;
}

export interface RepoGraph extends GraphHeader {
  rank: Float32Array;
  offsets: Uint32Array;
  targets: Uint32Array;
  /** Flat [srcA, dstA, srcB, dstB, ...] pairs. */
  ambient: Uint32Array;
  weights: Uint16Array;
}

export function graphByteLength(n: number, e: number, a: number): number {
  return GRAPH_HEADER_BYTES + 4 * n + 4 * (n + 1) + 4 * e + 8 * a + 2 * e;
}

function views(buffer: ArrayBuffer, n: number, e: number, a: number) {
  const rankAt = GRAPH_HEADER_BYTES;
  const offsetsAt = rankAt + 4 * n;
  const targetsAt = offsetsAt + 4 * (n + 1);
  const ambientAt = targetsAt + 4 * e;
  const weightsAt = ambientAt + 8 * a;
  return {
    rank: new Float32Array(buffer, rankAt, n),
    offsets: new Uint32Array(buffer, offsetsAt, n + 1),
    targets: new Uint32Array(buffer, targetsAt, e),
    ambient: new Uint32Array(buffer, ambientAt, 2 * a),
    weights: new Uint16Array(buffer, weightsAt, e),
  };
}

export function encodeGraph(g: {
  rank: Float32Array | Float64Array;
  offsets: Uint32Array;
  targets: Uint32Array;
  weights: Uint16Array;
  ambient: Uint32Array;
  layoutVersion: number;
}): ArrayBuffer {
  const n = g.rank.length;
  const e = g.targets.length;
  const a = g.ambient.length / 2;

  if (g.offsets.length !== n + 1) throw new Error(`offsets must be length n+1 (${n + 1}), got ${g.offsets.length}`);
  if (g.weights.length !== e) throw new Error(`weights must be length e (${e}), got ${g.weights.length}`);
  if (!Number.isInteger(a)) throw new Error(`ambient must hold whole pairs, got ${g.ambient.length} entries`);

  const buffer = new ArrayBuffer(graphByteLength(n, e, a));
  const dv = new DataView(buffer);
  dv.setUint32(0, GRAPH_MAGIC, true);
  dv.setUint32(4, n, true);
  dv.setUint32(8, e, true);
  dv.setUint32(12, a, true);
  dv.setUint16(16, g.layoutVersion, true);

  const v = views(buffer, n, e, a);
  v.rank.set(g.rank instanceof Float32Array ? g.rank : Float32Array.from(g.rank));
  v.offsets.set(g.offsets);
  v.targets.set(g.targets);
  v.ambient.set(g.ambient);
  v.weights.set(g.weights);
  return buffer;
}

export function decodeGraph(buffer: ArrayBuffer): RepoGraph {
  if (buffer.byteLength < GRAPH_HEADER_BYTES) {
    throw new Error(`Graph too short: ${buffer.byteLength} bytes`);
  }
  const dv = new DataView(buffer);
  const magic = dv.getUint32(0, true);
  if (magic !== GRAPH_MAGIC) {
    throw new Error(`Bad graph magic 0x${magic.toString(16)} (expected 0x${GRAPH_MAGIC.toString(16)})`);
  }
  const nodeCount = dv.getUint32(4, true);
  const edgeCount = dv.getUint32(8, true);
  const ambientCount = dv.getUint32(12, true);
  const expected = graphByteLength(nodeCount, edgeCount, ambientCount);
  if (buffer.byteLength !== expected) {
    throw new Error(`Graph length mismatch: got ${buffer.byteLength}, expected ${expected}`);
  }

  return {
    nodeCount,
    edgeCount,
    ambientCount,
    layoutVersion: dv.getUint16(16, true),
    ...views(buffer, nodeCount, edgeCount, ambientCount),
  };
}

export interface Neighbour {
  node: number;
  weight: number;
  outgoing: boolean;
}

/**
 * Everything a node touches, both directions, strongest first.
 *
 * `limit` exists because hub nodes in a scale-free graph have thousands of
 * neighbours and the arc layer draws at most a couple of hundred. Taking the
 * strongest is more useful than taking an arbitrary prefix.
 */
export function neighboursOf(graph: RepoGraph, node: number, limit = 64): Neighbour[] {
  if (node < 0 || node >= graph.nodeCount) return [];
  const start = graph.offsets[node];
  const end = graph.offsets[node + 1];

  const out: Neighbour[] = [];
  for (let k = start; k < end; k++) {
    const w = graph.weights[k];
    out.push({ node: graph.targets[k], weight: w & WEIGHT_MASK, outgoing: (w & WEIGHT_OUTGOING) !== 0 });
  }
  if (out.length <= limit) return out;
  out.sort((p, q) => q.weight - p.weight);
  out.length = limit;
  return out;
}

export function degreeOf(graph: RepoGraph, node: number): { in: number; out: number } {
  if (node < 0 || node >= graph.nodeCount) return { in: 0, out: 0 };
  let inCount = 0;
  let outCount = 0;
  for (let k = graph.offsets[node]; k < graph.offsets[node + 1]; k++) {
    if (graph.weights[k] & WEIGHT_OUTGOING) outCount++;
    else inCount++;
  }
  return { in: inCount, out: outCount };
}

/**
 * Rank expressed as a percentile in [0,1], for display.
 *
 * `rank` must be a value read out of a Float32Array, not a float64 literal:
 * float32(0.05) is 0.050000000745, which is *greater* than the float64 0.05,
 * so a literal walks straight past its own entry and returns 0.
 */
export function rankPercentile(sortedRanksDesc: Float32Array, rank: number): number {
  let lo = 0;
  let hi = sortedRanksDesc.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedRanksDesc[mid] > rank) lo = mid + 1;
    else hi = mid;
  }
  return 1 - lo / Math.max(1, sortedRanksDesc.length);
}
