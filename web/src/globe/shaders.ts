/**
 * Point-cloud shaders. GLSL ES 1.00 (three.js ShaderMaterial default).
 *
 * Two decisions worth knowing before editing:
 *
 * 1. **The quantised angles ride in `position`.** three.js derives the draw
 *    count from `geometry.attributes.position`, so the geometry must have one.
 *    Rather than pay 12 MB for a dummy Float32 xyz alongside our real data, we
 *    pack (thetaQ, phiQ, sizeQ) into a single Uint16 `position` attribute —
 *    all three fit in 16 bits, and three.js declares `attribute vec3 position`
 *    for us. 6 bytes per point, no waste.
 *
 * 2. **No `gl_VertexID`.** It would save the 4 MB index attribute but requires
 *    glslVersion: GLSL3, which changes every keyword in both shaders. 4 MB of
 *    VRAM is cheaper than a class of silent compile differences.
 *
 * The vertex preamble is shared between the display and pick passes so the two
 * can never disagree about where a point is. A picking bug caused by drifting
 * position maths is close to undebuggable.
 */

/** Reconstructs world position and the facing term from quantised angles. */
const VERTEX_COMMON = /* glsl */ `
  // position = (thetaQ, phiQ, sizeQ), Uint16, declared by three.js
  //   thetaQ  theta / PI   * 32767
  //   phiQ    phi   / TAU  * 65535
  //   sizeQ   size         * 65535
  attribute float aDomain;   // Uint8
  attribute float aFlags;    // Uint8   bit0 low-signal, bit1 archived
  attribute float aIndex;    // Float32 index within this tile

  uniform float uRadius;
  uniform float uSizeScale;
  uniform float uPixelRatio;
  uniform float uCullBias;

  const float PI  = 3.141592653589793;
  const float TAU = 6.283185307179586;

  struct Placed {
    vec4  clip;
    float pointSize;
    float facing;
    float size;
    vec3  dir;      // unit direction on the sphere, for sun-side shading
  };

  Placed placePoint() {
    float theta = position.x * (PI  / 32767.0);
    float phi   = position.y * (TAU / 65535.0);
    float st    = sin(theta);

    // three.js is Y-up: theta from +Y, phi around the XZ plane.
    vec3 dir = vec3(st * cos(phi), cos(theta), st * sin(phi));

    vec3 worldPos = (modelMatrix * vec4(dir * uRadius, 1.0)).xyz;
    vec3 worldNrm = normalize(mat3(modelMatrix) * dir);
    vec3 toCamera = normalize(cameraPosition - worldPos);

    Placed p;
    p.facing = dot(worldNrm, toCamera);
    p.size = position.z / 65535.0;
    p.dir = worldNrm;

    // Cull the far hemisphere here rather than on the CPU: no index rebuild,
    // no per-frame JS, and it removes ~half the fragment work for free.
    if (p.facing < uCullBias) {
      p.clip = vec4(2.0, 2.0, 2.0, 1.0);   // outside the clip volume
      p.pointSize = 0.0;
      return p;
    }

    vec4 mv = viewMatrix * vec4(worldPos, 1.0);

    // Perspective-correct: -mv.z is view-space depth.
    p.pointSize = clamp(p.size * uSizeScale * uPixelRatio / max(-mv.z, 0.001), 1.0, 28.0);
    p.clip = projectionMatrix * mv;
    return p;
  }
`;

export const POINTS_VERT = /* glsl */ `
${VERTEX_COMMON}

  uniform vec3  uPalette[12];
  uniform float uHoverIndex;
  uniform float uDimLowSignal;
  uniform float uDomainFilter;   // -1 = show everything
  uniform vec3  uSunDir;
  uniform float uNightDim;

  varying vec3  vColor;
  varying float vAlpha;
  varying float vHover;
  varying float vSize;

  void main() {
    Placed p = placePoint();
    gl_Position  = p.clip;
    gl_PointSize = p.pointSize;
    vSize = p.size;

    if (p.pointSize == 0.0) {
      vColor = vec3(0.0);
      vAlpha = 0.0;
      vHover = 0.0;
      return;
    }

    // Dynamic indexing into a uniform array is legal in ES 1.00 vertex shaders
    // (it is the *fragment* stage that restricts it).
    int di = int(clamp(aDomain, 0.0, 11.0));
    vColor = uPalette[di];

    // Fade points in as they rotate over the limb instead of popping.
    float limb = smoothstep(uCullBias, uCullBias + 0.18, p.facing);

    float lowSignal = mod(aFlags, 2.0);              // bit 0
    float archived  = mod(floor(aFlags / 2.0), 2.0); // bit 1
    float dim = mix(1.0, uDimLowSignal, lowSignal) * mix(1.0, 0.55, archived);

    // Domain filter. Dim rather than hide, so filtering reads as "the rest is
    // still there, just quiet" — which preserves the sense of a whole map.
    float matches = step(uDomainFilter, -0.5) + step(abs(aDomain - uDomainFilter), 0.5);
    dim *= mix(0.045, 1.0, min(matches, 1.0));

    // Nodes on the unlit side dim slightly. Enough to reinforce that this is a
    // lit body, not so much that data on the night side becomes unreadable.
    float night = smoothstep(-0.4, 0.5, dot(p.dir, normalize(uSunDir)));
    dim *= mix(uNightDim, 1.0, night);

    vAlpha = limb * dim;
    vHover = step(abs(aIndex - uHoverIndex), 0.5);

    if (vHover > 0.5) {
      vColor = vec3(1.0);
      vAlpha = 1.0;
      gl_PointSize = max(p.pointSize * 2.4, 9.0);
    }
  }
`;

export const POINTS_FRAG = /* glsl */ `
  precision mediump float;

  uniform float uHubThreshold;

  varying vec3  vColor;
  varying float vAlpha;
  varying float vHover;
  varying float vSize;

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float r = sqrt(r2);
    float falloff = 1.0 - r;
    // Hard core, tight falloff, minimal bloom. Using pow() is expensive,
    // so we approximate pow(falloff, 3.6) with falloff^4 and pow(falloff, 1.4)
    // with falloff * sqrt(falloff) to save ALUs.
    float core = falloff * falloff;
    core = core * core;
    float halo = falloff * sqrt(falloff) * 0.16;
    float a = (core + halo) * vAlpha;

    vec3 rgb = mix(vColor, vec3(1.0), core * 0.62);

    // High-rank nodes get a containment ring. It is the cheapest possible
    // signal that a point is a significant thing rather than a speck, and it
    // only appears where there are enough pixels to draw it.
    if (vSize > uHubThreshold) {
      float strength = smoothstep(uHubThreshold, 1.0, vSize);
      float ring = smoothstep(0.60, 0.70, r) * (1.0 - smoothstep(0.80, 0.92, r));
      a += ring * 0.55 * strength * vAlpha;
      rgb = mix(rgb, vec3(0.62, 0.94, 1.0), ring * strength);
    }

    // Hover: a crisp bracket ring that stays legible inside a dense cluster.
    if (vHover > 0.5) {
      float ring = smoothstep(0.58, 0.68, r) * (1.0 - smoothstep(0.84, 0.96, r));
      rgb += ring * 1.8;
      a = max(a, ring);
    }

    if (a < 0.004) discard;
    gl_FragColor = vec4(rgb, a);
  }
`;

export const PICK_VERT = /* glsl */ `
${VERTEX_COMMON}

  uniform float uPickPadding;
  uniform float uIdOffset;
  uniform float uSizeBias;

  varying vec3 vPickColor;

  void main() {
    Placed p = placePoint();
    gl_Position = p.clip;

    // Bias depth toward the camera in proportion to node size, so when two
    // points overlap the more significant one wins. Without this, picking in a
    // dense cluster resolves to whichever speck happens to be nearer by a
    // fraction of a millimetre, and hovering feels arbitrary.
    gl_Position.z -= p.size * uSizeBias * gl_Position.w;

    // Inflate slightly (+4px margin, min 6px) so points are easy to hit without
    // giant fixed boxes swallowing neighboring nodes in dense clusters when zoomed in.
    gl_PointSize = p.pointSize > 0.0 ? max(p.pointSize + 4.0, uPickPadding) : 0.0;

    // Global id + 1, so a cleared black buffer decodes to "nothing".
    float id = aIndex + uIdOffset + 1.0;
    vPickColor = vec3(
      mod(id, 256.0),
      mod(floor(id / 256.0), 256.0),
      mod(floor(id / 65536.0), 256.0)
    ) / 255.0;
  }
`;

export const PICK_FRAG = /* glsl */ `
  precision highp float;

  varying vec3 vPickColor;

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    if (dot(uv, uv) > 1.0) discard;
    gl_FragColor = vec4(vPickColor, 1.0);
  }
`;

/**
 * Twelve domain colours — the "hard sci-fi instrument" direction.
 *
 * Deliberately cold-dominant: nine cool hues, three warm accents. A full
 * rainbow reads as a data-viz default; a cold field with a few warm signals
 * reads as an instrument, and the warm nodes become the ones your eye finds.
 *
 * All are bright and saturated because they composite additively onto black —
 * pastels wash out to grey here. Confusable pairs (the red/green axis) are
 * separated by large lightness differences for deuteranopia. Verify with a
 * simulator before Phase 3 ships, per web3d-interaction-ux.
 */
export const DOMAIN_PALETTE: readonly (readonly [number, number, number])[] = [
  [0.26, 0.78, 1.00], // AI / ML — signal cyan
  [1.00, 0.62, 0.24], // Web frameworks — amber (warm accent)
  [0.36, 0.95, 0.72], // Databases — mint
  [0.94, 0.82, 0.42], // DevOps — pale gold (warm accent)
  [0.62, 0.58, 1.00], // Languages — periwinkle
  [0.58, 0.92, 1.00], // Systems — ice
  [0.20, 0.72, 0.80], // Data engineering — deep teal
  [1.00, 0.42, 0.42], // Security — alert red (warm accent)
  [0.82, 0.48, 1.00], // Graphics — orchid
  [0.42, 0.60, 0.92], // Mobile — steel blue
  [0.70, 0.94, 0.46], // Scraping — chartreuse
  [0.64, 0.72, 0.84], // Scientific — cool grey
];
