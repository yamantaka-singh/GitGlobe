/**
 * CPU preview of the planet surface.
 *
 * A faithful port of BAKE_FRAG, run in Node and written out as a PNG, so the
 * land/sea balance and continent structure can be checked without a browser.
 * Shipping a terrain shader nobody has ever seen the output of is how you get a
 * planet that turns out to be 95% ocean on someone else's machine.
 *
 * This is a diagnostic, not a build step. If it diverges from the shader, the
 * shader is the source of truth — but the formula is short enough that keeping
 * them in step is easy, and the value of being able to *look* is high.
 *
 * Usage:  npm run preview:planet
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { SUN_VEC } from '../src/globe/sun.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const TILES = resolve(HERE, '../public/tiles');
const OUT = resolve(HERE, '../public/planet-preview.png');
const OUT_GLOBE = resolve(HERE, '../public/planet-globe.png');

const WIDTH = 1024;
const HEIGHT = 512;
const SEA_LEVEL = 0.0;

const argNum = (name: string, fallback: number) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
};
const K = argNum('k', 36);
const AMP = argNum('amp', 0.62);
const OFFSET = argNum('offset', 0.68);

// ---------------------------------------------------------------- noise

/** Classic Perlin-style gradient noise. Not simplex, but same character. */
const PERM = new Uint8Array(512);
{
  const p = Array.from({ length: 256 }, (_, i) => i);
  let seed = 1337;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
}

const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
const lerp = (a: number, b: number, t: number) => a + t * (b - a);

function grad(hash: number, x: number, y: number, z: number): number {
  const h = hash & 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}

function noise3(x: number, y: number, z: number): number {
  const X = Math.floor(x) & 255;
  const Y = Math.floor(y) & 255;
  const Z = Math.floor(z) & 255;
  x -= Math.floor(x);
  y -= Math.floor(y);
  z -= Math.floor(z);
  const u = fade(x);
  const v = fade(y);
  const w = fade(z);
  const A = PERM[X] + Y;
  const AA = PERM[A & 255] + Z;
  const AB = PERM[(A + 1) & 255] + Z;
  const B = PERM[(X + 1) & 255] + Y;
  const BA = PERM[B & 255] + Z;
  const BB = PERM[(B + 1) & 255] + Z;
  return lerp(
    lerp(
      lerp(grad(PERM[AA & 255], x, y, z), grad(PERM[BA & 255], x - 1, y, z), u),
      lerp(grad(PERM[AB & 255], x, y - 1, z), grad(PERM[BB & 255], x - 1, y - 1, z), u),
      v,
    ),
    lerp(
      lerp(grad(PERM[(AA + 1) & 255], x, y, z - 1), grad(PERM[(BA + 1) & 255], x - 1, y, z - 1), u),
      lerp(grad(PERM[(AB + 1) & 255], x, y - 1, z - 1), grad(PERM[(BB + 1) & 255], x - 1, y - 1, z - 1), u),
      v,
    ),
    w,
  );
}

function fbm(x: number, y: number, z: number, octaves: number): number {
  let v = 0;
  let a = 0.5;
  for (let i = 0; i < octaves; i++) {
    v += a * noise3(x, y, z);
    x *= 2.02;
    y *= 2.02;
    z *= 2.02;
    a *= 0.5;
  }
  return v;
}

// ---------------------------------------------------------------- png

function crc32(buf: Buffer): number {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

function chunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

/** Minimal RGB PNG writer — no dependencies, and this is a diagnostic. */
function writePng(path: string, width: number, height: number, rgb: Uint8Array) {
  const raw = Buffer.alloc(height * (1 + width * 3));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + width * 3)] = 0; // filter: none
    Buffer.from(rgb.buffer, y * width * 3, width * 3).copy(raw, y * (1 + width * 3) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: truecolour
  writeFileSync(
    path,
    Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk('IHDR', ihdr),
      chunk('IDAT', deflateSync(raw, { level: 9 })),
      chunk('IEND', Buffer.alloc(0)),
    ]),
  );
}

// ---------------------------------------------------------------- main

const SURFACE = {
  deepOcean: [0.024, 0.055, 0.118],
  shelf: [0.055, 0.145, 0.262],
  coast: [0.118, 0.235, 0.29],
  lowland: [0.196, 0.235, 0.18],
  highland: [0.478, 0.416, 0.298],
  ice: [0.784, 0.839, 0.886],
};

const TINT = [
  [0.106, 0.267, 0.353], [0.353, 0.243, 0.145], [0.145, 0.322, 0.259], [0.337, 0.302, 0.176],
  [0.235, 0.216, 0.361], [0.196, 0.318, 0.365], [0.098, 0.263, 0.278], [0.361, 0.196, 0.196],
  [0.294, 0.184, 0.341], [0.169, 0.224, 0.322], [0.267, 0.322, 0.18], [0.243, 0.259, 0.29],
];

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0 || 1e-9));
  return t * t * (3 - 2 * t);
};
const mix3 = (a: number[], b: number[], t: number) => [
  lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t),
];

function main() {
  const manifest = JSON.parse(readFileSync(resolve(TILES, 'manifest.json'), 'utf8'));
  const clusters = manifest.clusters.slice(0, 64).map((c: { theta: number; phi: number; domain: number }) => {
    const st = Math.sin(c.theta);
    return { x: st * Math.cos(c.phi), y: Math.cos(c.theta), z: st * Math.sin(c.phi), d: c.domain };
  });
  const seed = (manifest.seed % 97) * 0.37;

  const rgb = new Uint8Array(WIDTH * HEIGHT * 3);
  let landCount = 0;
  let iceCount = 0;
  let lightSum = 0;
  let areaTotal = 0;
  const albedoMap = new Float32Array(WIDTH * HEIGHT * 3);
  const lightMap = new Float32Array(WIDTH * HEIGHT);
  const domainArea = new Map<number, number>();

  for (let py = 0; py < HEIGHT; py++) {
    const v = py / (HEIGHT - 1);
    const theta = (1 - (1 - v)) * Math.PI; // matches shader: (1 - uv.y) * PI with uv.y = 1 - v
    for (let px = 0; px < WIDTH; px++) {
      const u = px / (WIDTH - 1);
      const phi = u * Math.PI * 2;
      const st = Math.sin(theta);
      const dx = -Math.cos(phi) * st;
      const dy = Math.cos(theta);
      const dz = Math.sin(phi) * st;

      let potential = 0;
      let nearest = -2;
      let domain = 0;
      for (const c of clusters) {
        const d = dx * c.x + dy * c.y + dz * c.z;
        potential += Math.exp(-K * (1 - d));
        if (d > nearest) {
          nearest = d;
          domain = c.d;
        }
      }

      const qx = dx * 1.9 + seed;
      const qy = dy * 1.9 + seed;
      const qz = dz * 1.9 + seed;
      const wx = fbm(qx + 13.7, qy + 13.7, qz + 13.7, 3) * 0.62;
      const wy = fbm(qx + 41.3, qy + 41.3, qz + 41.3, 3) * 0.62;
      const wz = fbm(qx + 77.1, qy + 77.1, qz + 77.1, 3) * 0.62;
      const detail = fbm(dx * 2.7 + wx + seed, dy * 2.7 + wy + seed, dz * 2.7 + wz + seed, 5);
      const ridge = 1 - Math.abs(fbm(dx * 5.1 + seed * 0.7, dy * 5.1 + seed * 0.7, dz * 5.1 + seed * 0.7, 3));

      const h = potential * AMP + detail * 0.78 + ridge * 0.12 - OFFSET;

      const shelf = smoothstep(SEA_LEVEL - 0.3, SEA_LEVEL, h);
      const ocean = mix3(SURFACE.deepOcean, SURFACE.shelf, shelf * shelf);

      const e = smoothstep(SEA_LEVEL, SEA_LEVEL + 0.46, h);
      let terrain = mix3(SURFACE.coast, SURFACE.lowland, smoothstep(0, 0.22, e));
      terrain = mix3(terrain, SURFACE.highland, smoothstep(0.34, 0.92, e));
      const tint = TINT[domain % 12];
      terrain = mix3(terrain, [
        terrain[0] * 0.62 + tint[0] * 0.38,
        terrain[1] * 0.62 + tint[1] * 0.38,
        terrain[2] * 0.62 + tint[2] * 0.38,
      ], 0.55);

      const land = smoothstep(SEA_LEVEL - 0.008, SEA_LEVEL + 0.008, h);
      let col = mix3(ocean, terrain, land);

      const coastline = 1 - smoothstep(0, 0.013, Math.abs(h - SEA_LEVEL));
      const glow = [tint[0] * 0.5 + 0.16, tint[1] * 0.5 + 0.3, tint[2] * 0.5 + 0.36];
      col = [col[0] + glow[0] * coastline * 0.75, col[1] + glow[1] * coastline * 0.75, col[2] + glow[2] * coastline * 0.75];

      const ice = smoothstep(0.9, 0.988, Math.abs(dy) + detail * 0.05);
      col = mix3(col, SURFACE.ice, ice * lerp(0.55, 1, land));

      const coastal = 1 - smoothstep(0, 0.17, h - SEA_LEVEL);
      const conurbation = smoothstep(0.32, 1.15, potential);
      const fine = fbm(dx * 78 + seed, dy * 78 + seed, dz * 78 + seed, 3);
      const lights = clamp01(
        conurbation * land * (1 - ice) * (0.3 + 0.7 * coastal) * smoothstep(0.06, 0.3, fine),
      );

      const w = Math.sin(theta); // equirectangular rows are not equal area
      areaTotal += w;
      if (land > 0.5) {
        landCount += w;
        domainArea.set(domain, (domainArea.get(domain) ?? 0) + 1);
      }
      if (ice > 0.5) iceCount += w;
      lightSum += lights * w;

      const o = (py * WIDTH + px) * 3;
      albedoMap[o] = col[0];
      albedoMap[o + 1] = col[1];
      albedoMap[o + 2] = col[2];
      lightMap[py * WIDTH + px] = lights;
      rgb[o] = Math.round(clamp01(col[0]) * 255);
      rgb[o + 1] = Math.round(clamp01(col[1]) * 255);
      rgb[o + 2] = Math.round(clamp01(col[2]) * 255);
    }
  }

  writePng(OUT, WIDTH, HEIGHT, rgb);

  renderGlobe(albedoMap, lightMap);

  // Weighted by sin(theta): equirectangular rows near the poles cover far less
  // sphere than rows near the equator, and an unweighted count makes the ice
  // caps look about three times bigger than they are.
  const total = areaTotal;
  console.log(`Planet preview → ${OUT}   (k=${K} amp=${AMP} offset=${OFFSET})`);
  console.log(`Globe render   → ${OUT_GLOBE}`);
  console.log(`  land       ${((landCount / total) * 100).toFixed(1)}%   (target 25-45%, by true area)`);
  console.log(`  ice        ${((iceCount / total) * 100).toFixed(1)}%   (target < 12%)`);
  console.log(`  city light ${((lightSum / total) * 100).toFixed(1)}% mean intensity`);
  console.log(`  territories ${domainArea.size} of 12 domains have land`);

  const landFrac = landCount / total;
  const ok = landFrac > 0.18 && landFrac < 0.55 && domainArea.size >= 10 && iceCount / total < 0.14;
  console.log(ok ? '\n  Looks like a planet.' : '\n  OUT OF RANGE — tune uSeaLevel or the potential weight.');
  process.exit(ok ? 0 : 1);
}


/**
 * Orthographic globe render — the same surface as the user will see it, lit by
 * the same sun direction, with city lights on the night side. Checking only the
 * flat map hides everything the lighting does.
 */
function renderGlobe(albedo: Float32Array, lights: Float32Array) {
  const S = 640;
  const out = new Uint8Array(S * S * 3);
  // The same constant the app uses — imported, not copied.
  const sl = Math.hypot(...SUN_VEC);
  const sun = SUN_VEC.map((v) => v / sl);
  const CITY = [1.0, 0.612, 0.278];
  const RIM = [0.431, 0.545, 0.91];
  const SPACE = [0.008, 0.008, 0.016];

  const sample = (dx: number, dy: number, dz: number) => {
    // Inverse of the bake mapping.
    const theta = Math.acos(Math.max(-1, Math.min(1, dy)));
    let phi = Math.atan2(dz, -dx);
    if (phi < 0) phi += Math.PI * 2;
    const u = phi / (Math.PI * 2);
    const vv = 1 - theta / Math.PI;
    const px = Math.min(WIDTH - 1, Math.max(0, Math.round(u * (WIDTH - 1))));
    const py = Math.min(HEIGHT - 1, Math.max(0, Math.round((1 - vv) * (HEIGHT - 1))));
    const o = (py * WIDTH + px) * 3;
    return { c: [albedo[o], albedo[o + 1], albedo[o + 2]], l: lights[py * WIDTH + px] };
  };

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      // Slight tilt so the poles are not edge-on, matching the default camera.
      const nx = (x / S) * 2 - 1;
      const ny = 1 - (y / S) * 2;
      const r2 = nx * nx + ny * ny;
      const o = (y * S + x) * 3;
      let col = [...SPACE];

      if (r2 <= 1.0) {
        const nz = Math.sqrt(1 - r2);
        const tilt = 0.32;
        const dy2 = ny * Math.cos(tilt) + nz * Math.sin(tilt);
        const dz2 = -ny * Math.sin(tilt) + nz * Math.cos(tilt);
        const { c, l } = sample(nx, dy2, dz2);
        const s = nx * sun[0] + dy2 * sun[1] + dz2 * sun[2];
        const day = smoothstep(-0.3, 0.3, s);
        const night = 1 - day;
        col = [
          c[0] * (0.1 + 0.9 * day) + CITY[0] * l * night * night * 1.35,
          c[1] * (0.1 + 0.9 * day) + CITY[1] * l * night * night * 1.35,
          c[2] * (0.1 + 0.9 * day) + CITY[2] * l * night * night * 1.35,
        ];
        const fres = Math.pow(1 - nz, 6.5) * 0.78 * lerp(1, smoothstep(-0.45, 0.9, s), 0.62);
        col = [col[0] + RIM[0] * fres, col[1] + RIM[1] * fres, col[2] + RIM[2] * fres];
      } else if (r2 < 1.44) {
        // Outer scatter halo.
        const d = Math.sqrt(r2);
        const a = Math.pow(Math.max(0, 1 - (d - 1) / 0.2), 3) * 0.5;
        col = [SPACE[0] + 0.478 * a * 0.4, SPACE[1] + 0.4 * a * 0.4, SPACE[2] + 0.788 * a * 0.4];
      }

      out[o] = Math.round(clamp01(col[0]) * 255);
      out[o + 1] = Math.round(clamp01(col[1]) * 255);
      out[o + 2] = Math.round(clamp01(col[2]) * 255);
    }
  }
  writePng(OUT_GLOBE, S, S, out);
}

main();
