import { describe, it, expect } from 'vitest';
import {
  encodeTile,
  decodeTile,
  tileByteLength,
  dequantiseTheta,
  dequantisePhi,
  dequantiseSize,
  directionAt,
  TILE_MAGIC,
  HEADER_BYTES,
  BYTES_PER_POINT,
  FLAG_LOW_SIGNAL,
  FLAG_ARCHIVED,
  TAU,
  type PointInput,
} from './format';

/** Deterministic PRNG so failures are reproducible. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomPoints(n: number, seed = 1): PointInput[] {
  const rnd = mulberry32(seed);
  return Array.from({ length: n }, () => ({
    theta: Math.acos(2 * rnd() - 1),
    phi: rnd() * TAU,
    repoId: Math.floor(rnd() * 4_000_000_000),
    size: rnd(),
    domain: Math.floor(rnd() * 12),
    flags: rnd() < 0.1 ? FLAG_LOW_SIGNAL : 0,
  }));
}

describe('tile format', () => {
  it('has the byte budget the architecture promises', () => {
    expect(BYTES_PER_POINT).toBe(12);
    expect(HEADER_BYTES).toBe(16);
    // 1M points must be 12 MB total, 4 MB of which is position.
    expect(tileByteLength(1_000_000)).toBe(16 + 12_000_000);
  });

  it('writes a readable header', () => {
    const buf = encodeTile(randomPoints(10), { layoutVersion: 7, lodBand: 2 });
    const dv = new DataView(buf);
    expect(dv.getUint32(0, true)).toBe(TILE_MAGIC);
    const tile = decodeTile(buf);
    expect(tile.count).toBe(10);
    expect(tile.layoutVersion).toBe(7);
    expect(tile.lodBand).toBe(2);
  });

  it('round-trips repoId, domain and flags exactly', () => {
    const pts = randomPoints(5000, 42);
    const tile = decodeTile(encodeTile(pts, { layoutVersion: 1, lodBand: 0 }));
    for (let i = 0; i < pts.length; i++) {
      expect(tile.repoId[i]).toBe(pts[i].repoId);
      expect(tile.domain[i]).toBe(pts[i].domain);
      expect(tile.flags[i]).toBe(pts[i].flags);
    }
  });

  it('keeps angular quantisation below 0.006 degrees', () => {
    const pts = randomPoints(20000, 7);
    const tile = decodeTile(encodeTile(pts, { layoutVersion: 1, lodBand: 0 }));
    let maxThetaErr = 0;
    let maxPhiErr = 0;
    for (let i = 0; i < pts.length; i++) {
      maxThetaErr = Math.max(maxThetaErr, Math.abs(dequantiseTheta(tile.thetaQ[i]) - pts[i].theta));
      maxPhiErr = Math.max(maxPhiErr, Math.abs(dequantisePhi(tile.phiQ[i]) - pts[i].phi));
    }
    // Half a quantisation step, in degrees.
    expect((maxThetaErr * 180) / Math.PI).toBeLessThan(0.006);
    expect((maxPhiErr * 180) / Math.PI).toBeLessThan(0.006);
  });

  it('keeps size quantisation below 1e-4', () => {
    const pts = randomPoints(5000, 11);
    const tile = decodeTile(encodeTile(pts, { layoutVersion: 1, lodBand: 0 }));
    for (let i = 0; i < pts.length; i++) {
      expect(Math.abs(dequantiseSize(tile.sizeQ[i]) - pts[i].size)).toBeLessThan(1e-4);
    }
  });

  it('reconstructs unit-length directions', () => {
    const tile = decodeTile(encodeTile(randomPoints(2000, 3), { layoutVersion: 1, lodBand: 0 }));
    for (let i = 0; i < tile.count; i++) {
      const [x, y, z] = directionAt(tile, i);
      expect(Math.hypot(x, y, z)).toBeCloseTo(1, 5);
    }
  });

  it('handles the poles and the phi wrap seam', () => {
    const edge: PointInput[] = [
      { theta: 0, phi: 0, repoId: 1, size: 0, domain: 0, flags: 0 },
      { theta: Math.PI, phi: 0, repoId: 2, size: 1, domain: 11, flags: FLAG_ARCHIVED },
      { theta: Math.PI / 2, phi: TAU - 1e-9, repoId: 3, size: 0.5, domain: 3, flags: 0 },
      { theta: Math.PI / 2, phi: TAU, repoId: 4, size: 0.5, domain: 3, flags: 0 },
      { theta: Math.PI / 2, phi: -0.1, repoId: 5, size: 0.5, domain: 3, flags: 0 },
    ];
    const tile = decodeTile(encodeTile(edge, { layoutVersion: 1, lodBand: 0 }));
    // No quantised value may overflow its field.
    for (let i = 0; i < tile.count; i++) {
      expect(tile.thetaQ[i]).toBeGreaterThanOrEqual(0);
      expect(tile.thetaQ[i]).toBeLessThanOrEqual(32767);
      expect(tile.phiQ[i]).toBeGreaterThanOrEqual(0);
      expect(tile.phiQ[i]).toBeLessThanOrEqual(65535);
    }
    // phi = TAU wraps to 0, phi = -0.1 wraps to TAU - 0.1.
    expect(dequantisePhi(tile.phiQ[3])).toBeCloseTo(0, 4);
    expect(dequantisePhi(tile.phiQ[4])).toBeCloseTo(TAU - 0.1, 3);
    // Poles reconstruct to +Y and -Y.
    expect(directionAt(tile, 0)[1]).toBeCloseTo(1, 5);
    expect(directionAt(tile, 1)[1]).toBeCloseTo(-1, 5);
  });

  it('rejects corrupt buffers loudly', () => {
    expect(() => decodeTile(new ArrayBuffer(4))).toThrow(/too short/);

    const bad = encodeTile(randomPoints(4), { layoutVersion: 1, lodBand: 0 });
    new DataView(bad).setUint32(0, 0xdeadbeef, true);
    expect(() => decodeTile(bad)).toThrow(/magic/);

    const truncated = encodeTile(randomPoints(4), { layoutVersion: 1, lodBand: 0 }).slice(0, 20);
    expect(() => decodeTile(truncated)).toThrow(/length mismatch/);
  });

  it('handles an empty tile', () => {
    const tile = decodeTile(encodeTile([], { layoutVersion: 1, lodBand: 0 }));
    expect(tile.count).toBe(0);
    expect(tile.thetaQ.length).toBe(0);
  });
});
