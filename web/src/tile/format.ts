/**
 * GitGlobe binary tile format — see docs/ARCHITECTURE.md §2.6
 *
 * Layout (little-endian, structure-of-arrays so every array maps directly
 * onto a THREE.BufferAttribute with zero per-point JS work):
 *
 *   offset            bytes        field
 *   0                 4            magic  'GGT1'
 *   4                 4            count
 *   8                 2            layoutVersion
 *   10                2            lodBand
 *   12                4            reserved
 *   16                2*count      thetaQ   Int16   theta / PI     * 32767
 *   16 +  2c          2*count      phiQ     Uint16  phi / (2*PI)   * 65535
 *   16 +  4c          4*count      repoId   Uint32
 *   16 +  8c          2*count      sizeQ    Uint16  normalised node radius
 *   16 + 10c          1*count      domain   Uint8   categorical colour index
 *   16 + 11c          1*count      flags    Uint8   bitfield
 *   ------------------------------------------------------------------
 *   total = 16 + 12*count          → 12 MB for 1,000,000 points
 *
 * Alignment: the Uint32 array sits at 16 + 4c, and both terms are divisible
 * by 4 for any count, so every typed-array view is naturally aligned.
 *
 * Endianness: typed-array views inherit platform byte order. Every realistic
 * target (x86, ARM, WASM) is little-endian; `assertLittleEndian` fails loudly
 * on anything else rather than rendering garbage.
 */

export const TILE_MAGIC = 0x47475431; // 'GGT1'
export const HEADER_BYTES = 16;
export const BYTES_PER_POINT = 12;

export const THETA_QUANT = 32767; // theta ∈ [0, PI]
export const PHI_QUANT = 65535; // phi   ∈ [0, 2PI)
export const SIZE_QUANT = 65535; // size  ∈ [0, 1]

/** Node flag bits. */
export const FLAG_LOW_SIGNAL = 1 << 0;
export const FLAG_ARCHIVED = 1 << 1;
export const FLAG_FORK = 1 << 2;

export interface TileHeader {
  count: number;
  layoutVersion: number;
  lodBand: number;
}

export interface Tile extends TileHeader {
  thetaQ: Int16Array;
  phiQ: Uint16Array;
  repoId: Uint32Array;
  sizeQ: Uint16Array;
  domain: Uint8Array;
  flags: Uint8Array;
}

/** Point data in human units, used by the generator and by tests. */
export interface PointInput {
  /** polar angle, radians, [0, PI] */
  theta: number;
  /** azimuth, radians, [0, 2PI) */
  phi: number;
  repoId: number;
  /** normalised node radius, [0, 1] */
  size: number;
  /** categorical colour index, [0, 255] */
  domain: number;
  flags?: number;
}

export function assertLittleEndian(): void {
  const probe = new Uint8Array(new Uint32Array([1]).buffer);
  if (probe[0] !== 1) {
    throw new Error('GitGlobe tiles are little-endian; this platform is big-endian.');
  }
}

export function tileByteLength(count: number): number {
  return HEADER_BYTES + BYTES_PER_POINT * count;
}

export function quantiseTheta(theta: number): number {
  return Math.round(clamp(theta, 0, Math.PI) / Math.PI * THETA_QUANT);
}

export function quantisePhi(phi: number): number {
  const wrapped = ((phi % TAU) + TAU) % TAU;
  // Round-to-nearest can land exactly on PHI_QUANT+1 for phi just under TAU.
  return Math.min(Math.round((wrapped / TAU) * PHI_QUANT), PHI_QUANT);
}

export function quantiseSize(size: number): number {
  return Math.round(clamp(size, 0, 1) * SIZE_QUANT);
}

export function dequantiseTheta(q: number): number {
  return (q / THETA_QUANT) * Math.PI;
}

export function dequantisePhi(q: number): number {
  return (q / PHI_QUANT) * TAU;
}

export function dequantiseSize(q: number): number {
  return q / SIZE_QUANT;
}

export const TAU = Math.PI * 2;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Views onto an existing buffer. Used by both the encoder (to write) and the
 * decoder (to read), so the two can never drift out of sync.
 */
function views(buffer: ArrayBuffer, count: number): Omit<Tile, keyof TileHeader> {
  const c = count;
  return {
    thetaQ: new Int16Array(buffer, HEADER_BYTES, c),
    phiQ: new Uint16Array(buffer, HEADER_BYTES + 2 * c, c),
    repoId: new Uint32Array(buffer, HEADER_BYTES + 4 * c, c),
    sizeQ: new Uint16Array(buffer, HEADER_BYTES + 8 * c, c),
    domain: new Uint8Array(buffer, HEADER_BYTES + 10 * c, c),
    flags: new Uint8Array(buffer, HEADER_BYTES + 11 * c, c),
  };
}

/**
 * Encode without materialising an array of `count` objects — `fill` writes
 * point `i` into a reusable scratch object. At 1M points this is the difference
 * between ~10 MB of scratch and several hundred MB of short-lived garbage.
 */
export function encodeTileStreaming(
  count: number,
  header: { layoutVersion: number; lodBand: number },
  fill: (i: number, out: PointInput) => void,
): ArrayBuffer {
  assertLittleEndian();
  const buffer = new ArrayBuffer(tileByteLength(count));

  const dv = new DataView(buffer);
  dv.setUint32(0, TILE_MAGIC, true);
  dv.setUint32(4, count, true);
  dv.setUint16(8, header.layoutVersion, true);
  dv.setUint16(10, header.lodBand, true);
  dv.setUint32(12, 0, true);

  const v = views(buffer, count);
  const scratch: PointInput = { theta: 0, phi: 0, repoId: 0, size: 0, domain: 0, flags: 0 };
  for (let i = 0; i < count; i++) {
    scratch.flags = 0;
    fill(i, scratch);
    v.thetaQ[i] = quantiseTheta(scratch.theta);
    v.phiQ[i] = quantisePhi(scratch.phi);
    v.repoId[i] = scratch.repoId;
    v.sizeQ[i] = quantiseSize(scratch.size);
    v.domain[i] = scratch.domain;
    v.flags[i] = scratch.flags ?? 0;
  }
  return buffer;
}

export function encodeTile(
  points: readonly PointInput[],
  header: { layoutVersion: number; lodBand: number },
): ArrayBuffer {
  return encodeTileStreaming(points.length, header, (i, out) => {
    const p = points[i];
    out.theta = p.theta;
    out.phi = p.phi;
    out.repoId = p.repoId;
    out.size = p.size;
    out.domain = p.domain;
    out.flags = p.flags ?? 0;
  });
}

export function decodeTile(buffer: ArrayBuffer): Tile {
  assertLittleEndian();
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error(`Tile too short: ${buffer.byteLength} bytes`);
  }
  const dv = new DataView(buffer);
  const magic = dv.getUint32(0, true);
  if (magic !== TILE_MAGIC) {
    throw new Error(`Bad tile magic 0x${magic.toString(16)} (expected 0x${TILE_MAGIC.toString(16)})`);
  }
  const count = dv.getUint32(4, true);
  const expected = tileByteLength(count);
  if (buffer.byteLength !== expected) {
    throw new Error(`Tile length mismatch: got ${buffer.byteLength}, expected ${expected} for count ${count}`);
  }
  return {
    count,
    layoutVersion: dv.getUint16(8, true),
    lodBand: dv.getUint16(10, true),
    ...views(buffer, count),
  };
}

/** Unit-sphere direction for a decoded point, in three.js Y-up world space. */
export function directionAt(tile: Tile, i: number, out: [number, number, number] = [0, 0, 0]) {
  const theta = dequantiseTheta(tile.thetaQ[i]);
  const phi = dequantisePhi(tile.phiQ[i]);
  const st = Math.sin(theta);
  out[0] = st * Math.cos(phi);
  out[1] = Math.cos(theta);
  out[2] = st * Math.sin(phi);
  return out;
}
