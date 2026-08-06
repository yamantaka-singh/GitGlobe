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

    // Cull the far hemisphere here rather than on the CPU: no index rebuild,
    // no per-frame JS, and it removes ~half the fragment work for free.
    if (p.facing < uCullBias) {
      p.clip = vec4(2.0, 2.0, 2.0, 1.0);   // outside the clip volume
      p.pointSize = 0.0;
      return p;
    }

    vec4 mv = viewMatrix * vec4(worldPos, 1.0);
    float size = position.z / 65535.0;

    // Perspective-correct: -mv.z is view-space depth.
    p.pointSize = clamp(size * uSizeScale * uPixelRatio / max(-mv.z, 0.001), 1.0, 28.0);
    p.clip = projectionMatrix * mv;
    return p;
  }
`;

export const POINTS_VERT = /* glsl */ `
${VERTEX_COMMON}

  uniform vec3  uPalette[12];
  uniform float uHoverIndex;
  uniform float uDimLowSignal;

  varying vec3  vColor;
  varying float vAlpha;
  varying float vHover;

  void main() {
    Placed p = placePoint();
    gl_Position  = p.clip;
    gl_PointSize = p.pointSize;

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

  varying vec3  vColor;
  varying float vAlpha;
  varying float vHover;

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float r = sqrt(r2);
    // Tight core, soft halo — reads as a star rather than a flat disc.
    float core = pow(1.0 - r, 2.4);
    float halo = pow(1.0 - r, 0.7) * 0.28;
    float a = (core + halo) * vAlpha;
    if (a < 0.004) discard;

    vec3 rgb = mix(vColor, vec3(1.0), core * 0.55);

    // Hover gets a crisp ring so it stays legible inside a dense cluster.
    if (vHover > 0.5) {
      float ring = smoothstep(0.62, 0.72, r) * (1.0 - smoothstep(0.86, 0.96, r));
      rgb += ring * 1.6;
      a = max(a, ring);
    }

    gl_FragColor = vec4(rgb, a);
  }
`;

export const PICK_VERT = /* glsl */ `
${VERTEX_COMMON}

  uniform float uPickPadding;
  uniform float uIdOffset;

  varying vec3 vPickColor;

  void main() {
    Placed p = placePoint();
    gl_Position = p.clip;
    // Inflate slightly so small points stay hittable without demanding
    // sub-pixel mouse accuracy.
    gl_PointSize = p.pointSize > 0.0 ? max(p.pointSize, uPickPadding) : 0.0;

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
 * Twelve domain colours. Bright and saturated because they composite additively
 * onto near-black; pastel palettes wash out to grey here. Hues are spaced to
 * stay distinguishable under deuteranopia — the pairs that confuse (red/green)
 * are separated by large lightness differences. Verify with a simulator before
 * Phase 3 ships, per web3d-interaction-ux.
 */
export const DOMAIN_PALETTE: readonly (readonly [number, number, number])[] = [
  [0.42, 0.72, 1.0], // AI / ML — blue
  [1.0, 0.55, 0.28], // Web frameworks — orange
  [0.55, 0.9, 0.62], // Databases — green
  [0.98, 0.83, 0.35], // DevOps — yellow
  [0.76, 0.6, 1.0], // Languages — violet
  [0.4, 0.92, 0.9], // Systems — cyan
  [1.0, 0.62, 0.78], // Data engineering — pink
  [0.62, 0.78, 0.55], // Security — olive
  [1.0, 0.45, 0.45], // Graphics — coral
  [0.55, 0.65, 0.95], // Mobile — periwinkle
  [0.85, 0.95, 0.55], // Scraping — lime
  [0.72, 0.74, 0.8], // Scientific — cool grey
];
