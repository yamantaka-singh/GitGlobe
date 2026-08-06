import { describe, it, expect } from 'vitest';
import { pagerank, buildCsr } from './pagerank';

const sum = (a: Float64Array) => a.reduce((s, v) => s + v, 0);

function graph(n: number, edges: [number, number][]) {
  return buildCsr(
    n,
    new Uint32Array(edges.map((e) => e[0])),
    new Uint32Array(edges.map((e) => e[1])),
  );
}

describe('pagerank', () => {
  it('conserves total mass', () => {
    const g = graph(6, [
      [0, 1],
      [1, 2],
      [2, 0],
      [3, 0],
      [4, 0],
      // node 5 is dangling — its mass must be redistributed, not lost
    ]);
    const { rank } = pagerank(g);
    expect(sum(rank)).toBeCloseTo(1, 9);
  });

  it('gives every node equal rank on a directed cycle', () => {
    const g = graph(4, [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
    ]);
    const { rank, converged } = pagerank(g);
    expect(converged).toBe(true);
    for (let i = 0; i < 4; i++) expect(rank[i]).toBeCloseTo(0.25, 9);
  });

  it('gives every node equal rank when there are no edges at all', () => {
    const g = graph(5, []);
    const { rank } = pagerank(g);
    for (let i = 0; i < 5; i++) expect(rank[i]).toBeCloseTo(0.2, 9);
  });

  it('ranks the centre of an in-star highest', () => {
    // 1,2,3,4 all point at 0. Node 0 points nowhere.
    const g = graph(5, [
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
    ]);
    const { rank } = pagerank(g);
    expect(rank[0]).toBeGreaterThan(rank[1]);
    // The leaves are symmetric, so they must be exactly equal.
    for (let i = 2; i < 5; i++) expect(rank[i]).toBeCloseTo(rank[1], 12);
    expect(sum(rank)).toBeCloseTo(1, 9);
  });

  it('matches a hand-computed two-node chain', () => {
    // 0 -> 1, and 1 is dangling.
    // teleport = 0.15/2 = 0.075; dangling mass r1 is spread over both nodes.
    //   r0 = 0.075 + 0.425*r1
    //   r1 = 0.075 + 0.425*r1 + 0.85*r0
    // With r1 = 1 - r0:  1.425*r0 = 0.5  =>  r0 = 0.5/1.425
    const r0 = 0.5 / 1.425; // 0.3508771929...
    const g = graph(2, [[0, 1]]);
    const { rank } = pagerank(g, { tolerance: 1e-14, maxIterations: 5000 });
    expect(rank[0]).toBeCloseTo(r0, 9);
    expect(rank[1]).toBeCloseTo(1 - r0, 9);
    expect(sum(rank)).toBeCloseTo(1, 12);
  });

  it('ranks a hub above a leaf in a scale-free-ish graph', () => {
    // Node 0 is a hub: 20 nodes depend on it. Node 1 has a single dependent.
    const edges: [number, number][] = [];
    for (let i = 2; i < 22; i++) edges.push([i, 0]);
    edges.push([22, 1]);
    const g = graph(23, edges);
    const { rank } = pagerank(g);
    expect(rank[0]).toBeGreaterThan(rank[1]);
    expect(rank[1]).toBeGreaterThan(rank[5]);
  });

  it('converges on an irregular graph', () => {
    // Out-degrees must vary, or the uniform vector is already the fixed point
    // and the test passes in one iteration while proving nothing.
    const edges: [number, number][] = [];
    for (let i = 0; i < 200; i++) {
      const degree = 1 + (i % 5);
      for (let j = 0; j < degree; j++) edges.push([i, (i * 7 + j * 31 + 3) % 200]);
    }
    const { rank, iterations, converged } = pagerank(graph(200, edges), { tolerance: 1e-9 });
    expect(converged).toBe(true);
    expect(iterations).toBeGreaterThan(1);
    expect(iterations).toBeLessThan(200);
    // A genuinely irregular graph must produce a genuinely non-uniform result.
    const spread = Math.max(...rank) - Math.min(...rank);
    expect(spread).toBeGreaterThan(1e-4);
  });

  it('respects the damping factor', () => {
    const g = graph(5, [
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
    ]);
    const low = pagerank(g, { damping: 0.1 }).rank;
    const high = pagerank(g, { damping: 0.95 }).rank;
    // Less damping means more teleport, which flattens the distribution.
    expect(high[0] - high[1]).toBeGreaterThan(low[0] - low[1]);
  });

  it('handles an empty graph', () => {
    const { rank, converged } = pagerank({ n: 0, offsets: new Uint32Array(1), targets: new Uint32Array(0) });
    expect(rank.length).toBe(0);
    expect(converged).toBe(true);
  });

  it('rejects a malformed CSR', () => {
    expect(() => pagerank({ n: 3, offsets: new Uint32Array(2), targets: new Uint32Array(0) })).toThrow(
      /offsets must have length/,
    );
  });
});

describe('buildCsr', () => {
  it('groups edges by source', () => {
    const g = graph(3, [
      [2, 0],
      [0, 1],
      [0, 2],
      [2, 1],
    ]);
    expect(Array.from(g.offsets)).toEqual([0, 2, 2, 4]);
    expect(Array.from(g.targets.slice(0, 2)).sort()).toEqual([1, 2]);
    expect(Array.from(g.targets.slice(2, 4)).sort()).toEqual([0, 1]);
  });

  it('rejects out-of-range endpoints', () => {
    expect(() => buildCsr(2, new Uint32Array([5]), new Uint32Array([0]))).toThrow(/source 5 out of range/);
    expect(() => buildCsr(2, new Uint32Array([0]), new Uint32Array([9]))).toThrow(/destination 9 out of range/);
  });

  it('rejects mismatched edge arrays', () => {
    expect(() => buildCsr(2, new Uint32Array([0, 1]), new Uint32Array([0]))).toThrow(/same length/);
  });
});
