/**
 * Ice-giant surface generation — baked once into an equirectangular texture.
 *
 * The globe is an ice giant, not a terrestrial world. Neptune's character comes
 * from latitudinal banding, methane cirrus riding above it, and a handful of
 * dark storm ovals — and critically from LOW contrast between those bands.
 * Crank the contrast and you get a beach ball.
 *
 * ## Geography on a gas giant
 *
 * There is no land, so a territory cannot be a continent. It is a persistent
 * weather system: each domain owns a great cloud mass, tinted within the
 * blue-violet family, sitting inside the banding. The strongest clusters get a
 * storm vortex at their centre, the way the Great Dark Spot sits in a belt.
 *
 * Sectors are still legible, but the planet reads as one body.
 *
 * ## Why bake
 *
 * Banding, domain-warped turbulence, cirrus and a 64-cluster loop, per fragment,
 * every frame, would cost more than the entire point cloud. Baked once into a
 * render target at load, the sphere afterwards is a single texture fetch.
 *
 * ## Output packing
 *
 *   RGB = daylight albedo (bands, cirrus, storms, territory tint)
 *   A   = night-side emissive — aurora and storm luminance
 *
 * `scripts/preview-planet.ts` ports this to the CPU and renders it headless, so
 * the result can be looked at without a browser.
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
  uniform float uSeed;

  uniform vec3 uDeep;
  uniform vec3 uMid;
  uniform vec3 uLight;
  uniform vec3 uPale;
  uniform vec3 uCirrus;
  uniform vec3 uStorm;

  const float PI  = 3.141592653589793;
  const float TAU = 6.283185307179586;

  // Longitude shear as a function of latitude. Real gas giants rotate faster at
  // the equator than the poles, and that differential is what drags features
  // into the long swirls Neptune is known for.
  float shearAt(float lat) {
    return (1.0 - lat * lat) * 1.35 - 0.55;
  }

  void main() {
    // Must match three.js SphereGeometry's UV convention exactly, or the map
    // slides against the nodes sitting on top of it.
    float theta = (1.0 - vUv.y) * PI;
    float phi   = vUv.x * TAU;
    float st = sin(theta);
    vec3 dir = vec3(-cos(phi) * st, cos(theta), sin(phi) * st);
    float lat = dir.y;

    // Sheared sampling position. Everything downstream reads from 'flow' rather
    // than 'dir', so bands, cirrus and storms all get dragged by the same
    // differential rotation and stay coherent with each other.
    float shear = shearAt(lat);
    float sphi = phi + shear;
    vec3 flow = vec3(-cos(sphi) * st, lat, sin(sphi) * st);

    // ---- territory: great cloud systems ------------------------------------
    float potential = 0.0;
    float nearest = -2.0;
    float domain = 0.0;
    for (int i = 0; i < ${MAX_CLUSTERS}; i++) {
      vec3 mu = uClusters[i].xyz;
      // Padding entries are the zero vector. Their gaussian is harmlessly zero,
      // but their DOT PRODUCT is zero too — which beats any real cluster on the
      // far side of the planet and would tint every remote region with domain 0.
      float valid = step(0.5, length(mu));
      float d = dot(dir, mu) * valid + (valid - 1.0) * 3.0;
      potential += exp(-30.0 * (1.0 - d)) * valid;
      if (d > nearest) { nearest = d; domain = uClusters[i].w; }
    }

    // ---- banding -----------------------------------------------------------
    // Turbulence perturbs the band coordinate rather than the colour. Perturbing
    // colour gives noise laid over stripes; perturbing the coordinate makes the
    // bands themselves wander, which is what atmospheres actually do.
    float turb = fbm5(flow * 2.1 + uSeed) * 0.30
               + fbm3(flow * 5.4 + uSeed * 1.7) * 0.10;
    float bandCoord = lat * 6.5 + turb * 2.4;
    float band = sin(bandCoord * PI);

    // Asymmetric: zones (bright, rising gas) are narrower than belts (dark).
    float zone = smoothstep(-0.10, 0.85, band);
    float deepBelt = smoothstep(0.35, -0.75, band);

    vec3 albedo = mix(uMid, uLight, zone);
    albedo = mix(albedo, uDeep, deepBelt * 0.85);
    // A thin pale crown at the very top of the brightest zones.
    albedo = mix(albedo, uPale, smoothstep(0.80, 0.99, band) * 0.55);

    // ---- territory tint ----------------------------------------------------
    // Dynamic indexing of a uniform array is illegal in ES 1.00 fragment
    // shaders, so the domain tint is selected by masked accumulation.
    vec3 tint = vec3(0.0);
    for (int i = 0; i < 12; i++) {
      tint += uDomainTint[i] * step(abs(float(i) - domain), 0.5);
    }
    // Restraint. Territories shift the band colour; they never replace it. The
    // moment the surface competes with the nodes, the map stops being readable.
    float territory = smoothstep(0.18, 1.30, potential);
    albedo = mix(albedo, albedo * 0.45 + tint * 0.55, territory * 0.62);

    // ---- storms ------------------------------------------------------------
    // Dark ovals where a territory is strongest, stretched in longitude by the
    // same shear. Bright companion cloud on the leading edge, exactly as on
    // Neptune's Great Dark Spot.
    float stormMask = smoothstep(0.85, 1.9, potential);
    float swirl = fbm3(flow * 7.5 + uSeed * 2.3);
    float storm = stormMask * smoothstep(-0.25, 0.35, swirl);
    albedo = mix(albedo, uStorm, storm * 0.80);
    float companion = stormMask * smoothstep(0.30, 0.62, swirl) * (1.0 - storm);
    albedo = mix(albedo, uCirrus, companion * 0.55);

    // ---- methane cirrus ----------------------------------------------------
    // Sampled with latitude scaled up, which stretches the noise into long
    // longitudinal streaks. Thresholded high so it stays wispy.
    vec3 cirrusP = vec3(flow.x * 3.4, lat * 26.0, flow.z * 3.4) + uSeed;
    float cirrus = fbm3(cirrusP) + fbm3(cirrusP * 2.7) * 0.4;
    cirrus = smoothstep(0.14, 0.46, cirrus);
    // Densest in the mid latitudes, as on Neptune — thin at the equator, gone
    // at the poles.
    cirrus *= smoothstep(0.02, 0.35, abs(lat)) * smoothstep(0.97, 0.72, abs(lat));
    albedo = mix(albedo, uCirrus, cirrus * 0.55);

    // ---- polar hood --------------------------------------------------------
    // A brighter cap, softly edged. Not ice — high-altitude haze.
    float hood = smoothstep(0.72, 0.99, abs(lat) + turb * 0.10);
    albedo = mix(albedo, mix(uPale, uCirrus, 0.35), hood * 0.55);

    // ---- night-side emissive ----------------------------------------------
    // Aurora at the poles plus a glow in the storm cores. Without this the dark
    // limb is a dead black crescent and the planet looks bitten into.
    // Poles and storm cores ONLY. Feeding 'territory' in here was a mistake:
    // territories cover most of the planet, so the entire night side lit up
    // like a lamp instead of showing a thin aurora against the dark.
    float aurora = smoothstep(0.86, 0.995, abs(lat)) * (0.35 + 0.65 * fbm3(flow * 9.0 + uSeed));
    float glow = clamp(aurora * 0.95 + storm * 0.12, 0.0, 1.0);

    gl_FragColor = vec4(albedo, glow);
  }
`;
