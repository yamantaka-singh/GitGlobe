/**
 * Planet surface generation — baked once into an equirectangular texture.
 *
 * The globe is a *map*, not Earth. Its geography is derived from the semantic
 * layout: continents form where repository clusters are dense, so the AI/ML
 * landmass exists because AI/ML repositories exist there. Sectors are literally
 * territories.
 *
 * ## Why bake
 *
 * Generating terrain means fBm noise plus a loop over every cluster centre, per
 * fragment. Doing that every frame would cost more than the entire point cloud.
 * Instead it runs once into a render target at load, and the sphere afterwards
 * costs a single texture fetch. This is how real planet renderers work, and it
 * is the difference between a technique that scales to mobile and one that
 * doesn't.
 *
 * ## Output packing
 *
 *   RGB = daylight albedo (terrain, ocean, ice, coastlines — fully shaded)
 *   A   = city-light intensity, for the night side
 *
 * One RGBA texture, one fetch. City lights concentrate where cluster potential
 * is high and the surface is land — so the night side glows exactly where the
 * repositories are, which is the whole conceit of the reference image.
 *
 * ## Geographic plausibility
 *
 * Real coastlines are fractal, so a plain distance threshold around each cluster
 * gives circular blobs that read as CGI. Domain-warping the noise before it
 * thresholds is what produces peninsulas, inland seas, and archipelagos — the
 * cues the eye uses to accept something as a landmass.
 */

export const MAX_CLUSTERS = 64;

/** Ashima / Stefan Gustavson simplex noise, MIT. Shared by both stages. */
const NOISE = /* glsl */ `
  vec4 permute(vec4 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0))
             + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 1.0 / 7.0;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  float fbm3(vec3 p) {
    float v = 0.0; float a = 0.5;
    for (int i = 0; i < 3; i++) { v += a * snoise(p); p *= 2.02; a *= 0.5; }
    return v;
  }

  float fbm5(vec3 p) {
    float v = 0.0; float a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * snoise(p); p *= 2.02; a *= 0.5; }
    return v;
  }
`;

export const BAKE_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const BAKE_FRAG = /* glsl */ `
  precision highp float;
${NOISE}

  varying vec2 vUv;

  // xyz = unit direction of a cluster centre, w = its domain index.
  uniform vec4  uClusters[${MAX_CLUSTERS}];
  uniform vec3  uDomainTint[12];
  uniform float uSeaLevel;
  uniform float uSeed;

  uniform vec3 uDeepOcean;
  uniform vec3 uShelf;
  uniform vec3 uCoast;
  uniform vec3 uLowland;
  uniform vec3 uHighland;
  uniform vec3 uIce;

  const float PI  = 3.141592653589793;
  const float TAU = 6.283185307179586;

  void main() {
    // Must match three.js SphereGeometry's UV convention exactly, or the map
    // slides against the nodes sitting on top of it:
    //   x = -cos(phi) * sin(theta),  y = cos(theta),  z = sin(phi) * sin(theta)
    float theta = (1.0 - vUv.y) * PI;
    float phi   = vUv.x * TAU;
    float st = sin(theta);
    vec3 dir = vec3(-cos(phi) * st, cos(theta), sin(phi) * st);

    // ---- continental potential from the semantic layout --------------------
    // Each cluster of repositories pulls land up around itself. Continents are
    // therefore where the data is, not decoration laid on top of it.
    float potential = 0.0;
    float nearest = -2.0;
    float domain = 0.0;
    for (int i = 0; i < ${MAX_CLUSTERS}; i++) {
      vec3 mu = uClusters[i].xyz;
      // Padding entries are the zero vector; their dot product is 0 and their
      // gaussian is negligible, so no branch is needed (branches on uniforms
      // are the classic way to lose 20% on mobile).
      float d = dot(dir, mu);
      potential += exp(-11.0 * (1.0 - d)) * length(mu);
      if (d > nearest) { nearest = d; domain = uClusters[i].w; }
    }

    // ---- elevation ---------------------------------------------------------
    vec3 q = dir * 1.9 + uSeed;
    // Domain warping. Without it the thresholded potential produces circles,
    // and circles read as computer graphics rather than as coastline.
    vec3 warp = vec3(fbm3(q + 13.7), fbm3(q + 41.3), fbm3(q + 77.1)) * 0.62;
    float detail = fbm5(dir * 2.7 + warp + uSeed);
    float ridge = 1.0 - abs(fbm3(dir * 5.1 + uSeed * 0.7));

    float h = potential * 0.52 + detail * 0.78 + ridge * 0.12 - 0.46;

    // ---- ocean -------------------------------------------------------------
    float shelf = smoothstep(uSeaLevel - 0.30, uSeaLevel, h);
    vec3 ocean = mix(uDeepOcean, uShelf, shelf * shelf);

    // ---- land --------------------------------------------------------------
    float e = smoothstep(uSeaLevel, uSeaLevel + 0.46, h);
    vec3 terrain = mix(uCoast, uLowland, smoothstep(0.0, 0.22, e));
    terrain = mix(terrain, uHighland, smoothstep(0.34, 0.92, e));

    // Dynamic indexing of a uniform array is illegal in ES 1.00 fragment
    // shaders, so the domain tint is selected by masked accumulation.
    vec3 tint = vec3(0.0);
    for (int i = 0; i < 12; i++) {
      tint += uDomainTint[i] * step(abs(float(i) - domain), 0.5);
    }
    // Restraint: territories are *tinted*, not painted. The planet stays a
    // muted body so the data points on top of it remain the brightest thing on
    // screen — the moment the surface competes, the map stops being readable.
    terrain = mix(terrain, terrain * 0.62 + tint * 0.38, 0.55);

    float land = smoothstep(uSeaLevel - 0.008, uSeaLevel + 0.008, h);
    vec3 albedo = mix(ocean, terrain, land);

    // A thin lit band exactly at sea level. Coastlines are what the eye uses to
    // parse a sphere as a map, and one bright pixel of shoreline does more than
    // any amount of terrain detail.
    float coastline = 1.0 - smoothstep(0.0, 0.016, abs(h - uSeaLevel));
    albedo += tint * coastline * 0.30;

    // ---- ice ---------------------------------------------------------------
    // Latitude-driven, roughened by the same noise so the margin is ragged.
    float lat = abs(dir.y);
    float ice = smoothstep(0.78, 0.94, lat + detail * 0.10);
    albedo = mix(albedo, uIce, ice * mix(0.65, 1.0, land));

    // ---- city lights -------------------------------------------------------
    // Dense where cluster potential is high, broken up so they read as
    // settlements rather than a wash. Never on ice, never at sea.
    float urban = smoothstep(0.30, 1.25, potential) * land * (1.0 - ice);
    float grain = fbm3(dir * 34.0 + uSeed);
    float lights = urban * smoothstep(0.02, 0.55, grain);
    lights += urban * smoothstep(0.55, 0.95, fbm3(dir * 12.0)) * 0.5;

    gl_FragColor = vec4(albedo, clamp(lights, 0.0, 1.0));
  }
`;
