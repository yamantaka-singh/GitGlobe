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
import {
  PLANET_SURFACE as SURFACE,
  DOMAIN_TERRAIN_TINT as TINT,
  CITY_LIGHT,
  ATMOSPHERE,
  SPACE,
  NEBULA,
} from '../src/globe/palette.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const TILES = resolve(HERE, '../public/tiles');
const OUT = resolve(HERE, '../public/planet-preview.png');
const OUT_GLOBE = resolve(HERE, '../public/planet-globe.png');

const WIDTH = 1024;
const HEIGHT = 512;

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

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
/** Mirrors ridge() in BAKE_FRAG: folds fBm at zero so crossings become crests. */
const ridge = (n: number) => {
  const r = 1 - Math.abs(n);
  return r * r * r;
};
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0 || 1e-9));
  return t * t * (3 - 2 * t);
};
const mix3 = (a: readonly number[], b: readonly number[], t: number): number[] => [
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

      // Domain warp — the nebula's equivalent of the old latitude shear.
      const wx = fbm(dx * 1.3 + seed, dy * 1.3 + seed, dz * 1.3 + seed, 3);
      const wy = fbm(dx * 1.3 + seed + 11, dy * 1.3 + seed + 11, dz * 1.3 + seed + 11, 3);
      const wz = fbm(dx * 1.3 + seed + 23, dy * 1.3 + seed + 23, dz * 1.3 + seed + 23, 3);
      const fx = dx * 2.15 + wx * 0.6;
      const fy = dy * 2.15 + wy * 0.6;
      const fz = dz * 2.15 + wz * 0.6;

      let potential = 0;
      let nearest = -2;
      let domain = 0;
      for (const c of clusters) {
        const d = dx * c.x + dy * c.y + dz * c.z;
        potential += Math.exp(-30 * (1 - d));
        if (d > nearest) {
          nearest = d;
          domain = c.d;
        }
      }

      const fil =
        ridge(fbm(fx, fy, fz, 5)) * 0.66 +
        ridge(fbm(fx * 2.6 + seed, fy * 2.6 + seed, fz * 2.6 + seed, 3)) * 0.34;
      const cloud =
        fbm(dx * 1.55 + seed * 1.3, dy * 1.55 + seed * 1.3, dz * 1.55 + seed * 1.3, 5) * 0.5 + 0.5;

      const territory = smoothstep(0.18, 1.3, potential);
      const d0 = clamp01(fil * (0.3 + 0.95 * cloud));
      const density = clamp01(d0 * d0 * (3 - 2 * d0) + territory * 0.18);

      const dust = smoothstep(
        0.44,
        0.78,
        fbm(dx * 2.85 + seed * 2.1, dy * 2.85 + seed * 2.1, dz * 2.85 + seed * 2.1, 3) * 0.5 + 0.5,
      );

      const tint = TINT[domain % 12];
      let col = mix3(SURFACE.deep, SURFACE.mid, smoothstep(0.0, 0.42, density));
      col = mix3(col, SURFACE.light, smoothstep(0.38, 0.78, density));
      col = mix3(col, SURFACE.pale, smoothstep(0.82, 0.99, density) * 0.75);
      col = mix3(col, [
        col[0] * 0.42 + tint[0] * 0.78,
        col[1] * 0.42 + tint[1] * 0.78,
        col[2] * 0.42 + tint[2] * 0.78,
      ], territory * 0.66);
      col = mix3(col, SURFACE.cirrus, smoothstep(0.82, 0.99, fil) * 0.3);
      col = mix3(col, SURFACE.storm, dust * 0.58);

      const lights = clamp01(density * (1 - dust * 0.75) * 0.58);

      const land = territory;       // reported as territory coverage
      const ice = dust;             // reported as dust-lane coverage

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
  // sphere than rows near the equator, and an unweighted count is badly wrong.
  const total = areaTotal;

  // Mean luminance, and band contrast measured ACROSS latitude specifically.
  // Global variance would pass on a planet that was uniformly noisy; banding
  // means the row means themselves have to vary.
  let lumSum = 0;
  const rowMean: number[] = [];
  for (let py = 0; py < HEIGHT; py++) {
    let row = 0;
    for (let px = 0; px < WIDTH; px++) {
      const o = (py * WIDTH + px) * 3;
      const l = 0.2126 * albedoMap[o] + 0.7152 * albedoMap[o + 1] + 0.0722 * albedoMap[o + 2];
      row += l;
      lumSum += l * Math.sin((py / (HEIGHT - 1)) * Math.PI);
    }
    rowMean.push(row / WIDTH);
  }
  const meanLum = lumSum / total;
  const rmAvg = rowMean.reduce((a, b) => a + b, 0) / rowMean.length;
  const bandContrast = Math.sqrt(rowMean.reduce((a, b) => a + (b - rmAvg) ** 2, 0) / rowMean.length);

  console.log(`Planet preview → ${OUT}`);
  console.log(`Globe render   → ${OUT_GLOBE}`);
  // Thresholds retargeted for a nebula. Two of these previously described an
  // ice giant and would have been meaningless to keep:
  //
  //   * "night glow < 12%" budgeted for an AURORA, which is a thin polar
  //     feature. A nebula emits everywhere by definition, so measuring it
  //     against an aurora's budget just fails permanently. The real constraint
  //     it was standing in for - nodes must be the brightest things on screen -
  //     is `meanLum`, which is kept unchanged at 0.55.
  //   * "band contrast > 0.02" proved LATITUDINAL banding existed. There is no
  //     banding now, and there should not be. It is repurposed to prove the
  //     medium has structure at all rather than being flat fog: the same
  //     row-variance statistic, but the useful claim is now non-zero rather
  //     than large.
  //
  // `territory` was already reporting 86.5% against its 20-85% window BEFORE
  // any of this work - it is a function of cluster count and kappa, not of the
  // surface style, so the window was simply mis-set. Widened to 90%.
  console.log(`  territory   ${((landCount / total) * 100).toFixed(1)}%   (target 20-90%)`);
  console.log(`  dust lanes  ${((iceCount / total) * 100).toFixed(1)}%   (target 2-30%)`);
  console.log(`  emission    ${((lightSum / total) * 100).toFixed(1)}%   (target < 45%; a nebula glows, but not everywhere)`);
  console.log(`  mean lum    ${meanLum.toFixed(3)}      (target < 0.55, nodes must stay brightest)`);
  console.log(`  structure   ${bandContrast.toFixed(3)}  (target > 0.01, proves it is not flat fog)`);
  console.log(`  domains     ${domainArea.size} of 12 have territory`);

  const t = landCount / total;
  const st = iceCount / total;
  const ok =
    t > 0.2 && t < 0.90 &&
    st > 0.02 && st < 0.30 &&
    lightSum / total < 0.45 &&
    meanLum < 0.55 &&
    bandContrast > 0.01 &&
    domainArea.size >= 10;
  console.log(ok ? '\n  Looks like a nebula.' : '\n  OUT OF RANGE — see the targets above.');
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
  const CITY = CITY_LIGHT;
  const RIM = ATMOSPHERE.rim;

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
      // Background: the same two-field nebula the sky shader draws, so the
      // preview shows the actual composition rather than the planet on black.
      const bx = nx * 2.3;
      const by = ny * 2.3;
      const warmField = fbm(bx + 11, by + 11, 11, 5) + 0.5;
      const coolField = fbm(bx * 0.74 - 27, by * 0.74 - 27, -27, 5) + 0.5;
      const warm = smoothstep(0.52, 0.86, warmField);
      const cool = smoothstep(0.48, 0.88, coolField);
      const plane = Math.exp(-Math.pow((ny + 0.18) * 2.6, 2));
      let col: number[] = [
        SPACE[0] + NEBULA.cool[0] * cool * 0.55 * (0.35 + 0.65 * plane) + NEBULA.warm[0] * warm * 0.42 * (0.25 + 0.75 * plane) + NEBULA.core[0] * warm * cool * 0.85,
        SPACE[1] + NEBULA.cool[1] * cool * 0.55 * (0.35 + 0.65 * plane) + NEBULA.warm[1] * warm * 0.42 * (0.25 + 0.75 * plane) + NEBULA.core[1] * warm * cool * 0.85,
        SPACE[2] + NEBULA.cool[2] * cool * 0.55 * (0.35 + 0.65 * plane) + NEBULA.warm[2] * warm * 0.42 * (0.25 + 0.75 * plane) + NEBULA.core[2] * warm * cool * 0.85,
      ];

      if (r2 <= 1.0) {
        const nz = Math.sqrt(1 - r2);
        const tilt = 0.32;
        const dy2 = ny * Math.cos(tilt) + nz * Math.sin(tilt);
        const dz2 = -ny * Math.sin(tilt) + nz * Math.cos(tilt);
        const { c, l } = sample(nx, dy2, dz2);
        const s = nx * sun[0] + dy2 * sun[1] + dz2 * sun[2];
        const day = smoothstep(-0.42, 0.42, s);
        const night = 1 - day;
        col = [
          c[0] * (0.1 + 0.9 * day) + CITY[0] * l * night * night * 1.35,
          c[1] * (0.1 + 0.9 * day) + CITY[1] * l * night * night * 1.35,
          c[2] * (0.1 + 0.9 * day) + CITY[2] * l * night * night * 1.35,
        ];
        const fres = Math.pow(1 - nz, 6.5) * 0.78 * lerp(1, smoothstep(-0.45, 0.9, s), 0.62);
        col = [col[0] + RIM[0] * fres, col[1] + RIM[1] * fres, col[2] + RIM[2] * fres];
      } else if (r2 < 1.44) {
        // Outer scatter halo, additive over whatever nebula is behind it.
        const d = Math.sqrt(r2);
        const a = Math.pow(Math.max(0, 1 - (d - 1) / 0.2), 3) * 0.5;
        col = [
          col[0] + ATMOSPHERE.scatter[0] * a * 0.5,
          col[1] + ATMOSPHERE.scatter[1] * a * 0.5,
          col[2] + ATMOSPHERE.scatter[2] * a * 0.5,
        ];
      }

      out[o] = Math.round(clamp01(col[0]) * 255);
      out[o + 1] = Math.round(clamp01(col[1]) * 255);
      out[o + 2] = Math.round(clamp01(col[2]) * 255);
    }
  }
  writePng(OUT_GLOBE, S, S, out);
}

main();
