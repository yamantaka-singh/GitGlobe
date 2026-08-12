/**
 * Nebula surface generation — baked once into an equirectangular texture.
 *
 * The globe was an ice giant: latitudinal banding, methane cirrus, dark storm
 * ovals, lit from one side. It is now an emission nebula, which is a different
 * object in the two ways that drive every line below.
 *
 * **It has no preferred axis.** A gas giant's entire character comes from
 * latitude — bands, zones, belts, polar hood. Every one of those is a function
 * of `lat`, and none of them survives here. A nebula is shaped by turbulence,
 * so structure comes from domain-warped noise instead: the sampling coordinate
 * is displaced by noise before being read, which is what produces sinuous
 * curdled filaments rather than stripes.
 *
 * **It emits rather than reflects.** There is no albedo and no terminator. The
 * alpha channel is no longer a night-only aurora but omnidirectional emission,
 * and `Backdrop` holds the sun term down to a weak modelling cue so the sphere
 * still reads as a sphere.
 *
 * ## Geography in a nebula
 *
 * A territory cannot be a continent or a weather system. It is a star-forming
 * core: gas concentrates where a domain's clusters are, and denser gas is
 * hotter and brighter. The `potential` field driving this is unchanged from the
 * banded version, so territories sit in exactly the same places — only their
 * expression changed.
 *
 * Ridged noise is what separates this from fog. Folding fBm at zero with
 * 1 - |n| turns every zero crossing into a sharp crest; those crests are the
 * filaments. Plain fBm gives clouds, and clouds do not read as a nebula.
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
  uniform float uClusterWeights[${MAX_CLUSTERS}];
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

  // Ridged noise. Ordinary fBm gives soft clouds; folding it at zero with
  // 1 - |n| turns every zero crossing into a sharp crest, and those crests are
  // the filaments that make a nebula read as a nebula rather than as fog.
  // Cubing sharpens them further and pushes the space between them to black.
  float ridge(float n) {
    float r = 1.0 - abs(n);
    return r * r * r;
  }

  void main() {
    // Must match three.js SphereGeometry's UV convention exactly, or the map
    // slides against the nodes sitting on top of it.
    float theta = (1.0 - vUv.y) * PI;
    float phi   = vUv.x * TAU;
    float st = sin(theta);
    vec3 dir = vec3(-cos(phi) * st, cos(theta), sin(phi) * st);
    float lat = dir.y;

    // Domain-warped sampling position: the coordinate is displaced by noise
    // before being read. A gas giant's features are dragged into bands by
    // differential rotation, so the old code sheared by latitude. A nebula has
    // no rotation and no preferred axis - it is shaped by turbulence, and
    // warping the coordinate is what produces the curdled, sinuous structure
    // that latitude shear cannot express.
    vec3 warp = vec3(
      fbm3(dir * 1.30 + uSeed),
      fbm3(dir * 1.30 + uSeed + 11.0),
      fbm3(dir * 1.30 + uSeed + 23.0)
    );
    vec3 flow = dir * 2.15 + warp * 0.60;

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

      // Dynamic decay based on cluster weight (repo count/density):
      // Larger repo counts -> smaller decay (e.g. 10.0) -> wider territory on the globe!
      // Smaller repo counts -> larger decay (e.g. 45.0) -> smaller tight territory.
      float weight = uClusterWeights[i];
      float decay = mix(45.0, 10.0, clamp(weight, 0.0, 1.0));
      potential += (0.4 + 0.8 * weight) * exp(-decay * (1.0 - d)) * valid;

      if (d > nearest) { nearest = d; domain = uClusters[i].w; }
    }

    // ---- the medium --------------------------------------------------------
    // Two ridged octaves for filaments, plus a smooth low-frequency term for
    // where the gas is simply piled up. Filaments alone look like cracked
    // glaze; the smooth term gives them something to sit in.
    float fil = ridge(fbm5(flow)) * 0.66
              + ridge(fbm3(flow * 2.60 + uSeed)) * 0.34;
    float cloud = fbm5(dir * 1.55 + uSeed * 1.3) * 0.5 + 0.5;

    // No additive floor. An earlier version added cloud * 0.26 unconditionally,
    // which filled the voids and left the whole sphere emitting - 79% of it
    // above the glow threshold, against a 12% budget. A nebula is mostly empty;
    // the dark is what the filaments are legible against.
    float density = clamp(fil * (0.30 + 0.95 * cloud), 0.0, 1.0);
    // S-curve: darkens the mid-tones and keeps the crests, which widens the
    // gap between void and filament rather than raising everything together.
    density = density * density * (3.0 - 2.0 * density);

    // ---- territory ---------------------------------------------------------
    // A domain's cluster is a star-forming core: gas concentrates there. This
    // is the same 'potential' the banding used, so territories stay in exactly
    // the same places - only their expression changes.
    float territory = smoothstep(0.18, 1.30, potential);
    density = clamp(density + territory * 0.18, 0.0, 1.0);

    // Dynamic indexing of a uniform array is illegal in ES 1.00 fragment
    // shaders, so the domain tint is selected by masked accumulation.
    vec3 tint = vec3(0.0);
    for (int i = 0; i < 12; i++) {
      tint += uDomainTint[i] * step(abs(float(i) - domain), 0.5);
    }

    // ---- dust lanes --------------------------------------------------------
    // Dust does not glow, it OCCLUDES - the dark rifts in a nebula are the
    // foreground, not gaps. So it is applied last and it subtracts from
    // emission rather than adding a dark colour. Thresholded hard, because
    // real dust lanes have edges.
    float dust = smoothstep(0.44, 0.78, fbm3(dir * 2.85 + uSeed * 2.1) * 0.5 + 0.5);

    // ---- emission colour ---------------------------------------------------
    // A subtle, non-uniform dimmed navy blue background using the cloud noise.
    vec3 navy = vec3(0.02, 0.03, 0.12);
    // Removed dust occlusion completely so random procedural noise doesn't dim specific domains (like Mobile)
    vec3 gas = navy * (cloud * 0.8 + density * 0.5);

    // ---- Celestialsapien / Alien X Stars -----------------------------------
    // (Removed artificial surface stars so they don't look like repos)

    // ---- emission ----------------------------------------------------------
    // No aurora or city light glow (which is tinted blue). Just output the stars directly into the albedo.
    float glow = 0.0;

    gl_FragColor = vec4(gas, glow);
  }
`;
