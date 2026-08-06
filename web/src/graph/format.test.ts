import { describe, it, expect } from 'vitest';
import {
  encodeGraph,
  decodeGraph,
  graphByteLength,
  neighboursOf,
  degreeOf,
  rankPercentile,
  GRAPH_HEADER_BYTES,
  WEIGHT_OUTGOING,
} from './format';

/**
 * A tiny hand-built graph:
 *   0 -> 1, 0 -> 2, 1 -> 2, 3 isolated
 * stored undirected, so node 2's row holds both incoming edges.
 */
function fixture(nodeCount = 4) {
  const adjacency: Array<Array<{ to: number; out: boolean; w: number }>> = Array.from(
    { length: nodeCount },
    () => [],
  );
  const link = (a: number, b: number, w: number) => {
    adjacency[a].push({ to: b, out: true, w });
    adjacency[b].push({ to: a, out: false, w });
  };
  link(0, 1, 10);
  link(0, 2, 20);
  link(1, 2, 30);

  const offsets = new Uint32Array(nodeCount + 1);
  for (let i = 0; i < nodeCount; i++) offsets[i + 1] = offsets[i] + adjacency[i].length;
  const e = offsets[nodeCount];
  const targets = new Uint32Array(e);
  const weights = new Uint16Array(e);
  let k = 0;
  for (let i = 0; i < nodeCount; i++) {
    for (const edge of adjacency[i]) {
      targets[k] = edge.to;
      weights[k] = edge.w | (edge.out ? WEIGHT_OUTGOING : 0);
      k++;
    }
  }

  return {
    rank: Float32Array.from([0.4, 0.3, 0.25, 0.05]).subarray(0, nodeCount),
    offsets,
    targets,
    weights,
    ambient: Uint32Array.from([0, 2, 1, 2]),
    layoutVersion: 1,
  };
}

describe('graph format', () => {
  it('computes byte length from the layout', () => {
    // n=4, e=6, a=2 → 24 + 16 + 20 + 24 + 16 + 12
    expect(graphByteLength(4, 6, 2)).toBe(GRAPH_HEADER_BYTES + 16 + 20 + 24 + 16 + 12);
  });

  it('round-trips every array exactly', () => {
    const src = fixture();
    const g = decodeGraph(encodeGraph(src));

    expect(g.nodeCount).toBe(4);
    expect(g.edgeCount).toBe(6);
    expect(g.ambientCount).toBe(2);
    expect(g.layoutVersion).toBe(1);
    expect(Array.from(g.offsets)).toEqual(Array.from(src.offsets));
    expect(Array.from(g.targets)).toEqual(Array.from(src.targets));
    expect(Array.from(g.weights)).toEqual(Array.from(src.weights));
    expect(Array.from(g.ambient)).toEqual([0, 2, 1, 2]);
    for (let i = 0; i < 4; i++) expect(g.rank[i]).toBeCloseTo(src.rank[i], 6);
  });

  it('stays aligned for odd node, edge and ambient counts', () => {
    // Misordered arrays would throw "start offset of Uint32Array should be a
    // multiple of 4" here and nowhere else, so this is the load-bearing test.
    for (const [n, e, a] of [
      [1, 1, 1],
      [3, 5, 7],
      [7, 3, 1],
      [101, 203, 51],
    ]) {
      const offsets = new Uint32Array(n + 1);
      for (let i = 0; i <= n; i++) offsets[i] = Math.min(i, e);
      const src = {
        rank: new Float32Array(n).fill(1 / n),
        offsets,
        targets: new Uint32Array(e),
        weights: new Uint16Array(e),
        ambient: new Uint32Array(2 * a),
        layoutVersion: 3,
      };
      const buf = encodeGraph(src);
      expect(buf.byteLength).toBe(graphByteLength(n, e, a));
      expect(() => decodeGraph(buf)).not.toThrow();
    }
  });

  it('returns both directions from neighboursOf', () => {
    const g = decodeGraph(encodeGraph(fixture()));

    const n0 = neighboursOf(g, 0);
    expect(n0.map((x) => x.node).sort()).toEqual([1, 2]);
    expect(n0.every((x) => x.outgoing)).toBe(true);

    const n2 = neighboursOf(g, 2);
    expect(n2.map((x) => x.node).sort()).toEqual([0, 1]);
    expect(n2.every((x) => !x.outgoing)).toBe(true);
    expect(n2.find((x) => x.node === 1)?.weight).toBe(30);

    expect(neighboursOf(g, 3)).toEqual([]);
  });

  it('takes the strongest neighbours when over the limit', () => {
    const g = decodeGraph(encodeGraph(fixture()));
    const limited = neighboursOf(g, 2, 1);
    expect(limited).toHaveLength(1);
    expect(limited[0].weight).toBe(30);
  });

  it('separates in and out degree', () => {
    const g = decodeGraph(encodeGraph(fixture()));
    expect(degreeOf(g, 0)).toEqual({ in: 0, out: 2 });
    expect(degreeOf(g, 1)).toEqual({ in: 1, out: 1 });
    expect(degreeOf(g, 2)).toEqual({ in: 2, out: 0 });
    expect(degreeOf(g, 3)).toEqual({ in: 0, out: 0 });
  });

  it('handles out-of-range nodes without throwing', () => {
    const g = decodeGraph(encodeGraph(fixture()));
    expect(neighboursOf(g, -1)).toEqual([]);
    expect(neighboursOf(g, 999)).toEqual([]);
    expect(degreeOf(g, 999)).toEqual({ in: 0, out: 0 });
  });

  it('rejects corrupt buffers loudly', () => {
    expect(() => decodeGraph(new ArrayBuffer(8))).toThrow(/too short/);

    const bad = encodeGraph(fixture());
    new DataView(bad).setUint32(0, 0xdeadbeef, true);
    expect(() => decodeGraph(bad)).toThrow(/magic/);

    expect(() => decodeGraph(encodeGraph(fixture()).slice(0, 40))).toThrow(/length mismatch/);
  });

  it('rejects inconsistent inputs at encode time', () => {
    const src = fixture();
    expect(() => encodeGraph({ ...src, offsets: new Uint32Array(2) })).toThrow(/offsets must be length/);
    expect(() => encodeGraph({ ...src, weights: new Uint16Array(1) })).toThrow(/weights must be length/);
    expect(() => encodeGraph({ ...src, ambient: new Uint32Array(3) })).toThrow(/whole pairs/);
  });

  it('computes rank percentiles', () => {
    const sorted = Float32Array.from([0.5, 0.3, 0.15, 0.05]);
    // Ranks must come out of the array, not from float64 literals — see the
    // note on rankPercentile. Passing 0.05 directly returns 0, not 0.25.
    expect(rankPercentile(sorted, sorted[0])).toBeCloseTo(1, 6);
    expect(rankPercentile(sorted, sorted[2])).toBeCloseTo(0.5, 6);
    expect(rankPercentile(sorted, sorted[3])).toBeCloseTo(0.25, 6);
    expect(rankPercentile(sorted, 0.0001)).toBeCloseTo(0, 6);
  });
});
