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
    float fade;     // <1 when the point wanted to be smaller than one pixel
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
    float ideal = p.size * uSizeScale * uPixelRatio / max(-mv.z, 0.001);
    p.pointSize = clamp(ideal, 1.0, 72.0);

    // A point cannot be drawn smaller than one pixel, but pinning it there at
    // full brightness is what makes the field twinkle when the globe is zoomed
    // out or turning: every sub-pixel node keeps its full intensity while its
    // centre crosses the pixel grid, so the rasteriser flickers it on and off.
    //
    // Carrying the shortfall into alpha instead is the standard fix — the point
    // stays one pixel wide but dims by the area it was denied, so shrinking past
    // a pixel fades smoothly rather than scintillating. Squared because the
    // shortfall is a length and coverage goes as area.
    float shortfall = clamp(ideal, 0.0, 1.0);
    p.fade = shortfall * shortfall;
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

    vAlpha = limb * dim * p.fade;
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

  uniform float uHubGain;

  varying vec3  vColor;
  varying float vAlpha;
  varying float vHover;
  varying float vSize;

  /**
   * A star in a nebula, not a target reticle.
   *
   * The previous profile drew a literal annulus at r in [0.60, 0.92] on every
   * node above a size threshold. Rendered offline it is unmistakably a
   * bullseye: bright core, dark gap, hard outer ring. It fired on 1,657 of
   * 87,227 nodes — only 1.9%, which is why it was easy to dismiss on the
   * numbers, but those are the band-0 nodes, the largest things on screen at up
   * to 28px. Visual weight is not headcount.
   *
   * Two properties replace it:
   *
   *  1. **Monotonic alpha.** A ring needs alpha to rise again as r grows.
   *     Everything here is a positive multiple of a decreasing window, so no
   *     combination of parameters can reintroduce an annulus.
   *  2. **Importance is continuous.** Rank drives bloom gain, not a branch.
   *     A threshold means two nodes either side of it render differently for a
   *     difference of 0.001 in rank, and that discontinuity is exactly what the
   *     eye reads as a distinct class of object.
   *
   * The window is (1 - r2), which is exactly 0 at the rim, so the disc fades
   * out instead of terminating on the discard. Powers of it are products, so
   * this is cheaper than the pow()/sqrt() it replaces.
   */
  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float w  = 1.0 - r2;
    float w2 = w * w;
    float w3 = w2 * w;
    float core = w3 * w3;   // w^6 - tight pin-point centre
    float glow = w2;        // controlled body

    // Tighter body so dense clusters retain their individual star nature and rich color
    float gain = 0.2 + uHubGain * vSize * 0.5;
    float a = (core * 0.75 + glow * 0.10 * gain) * vAlpha;

    // Retain pure, vivid domain colors without washing out the center to white
    vec3 rgb = vColor;

    // Hover stays a crisp cursor ring
    if (vHover > 0.5) {
      float r = sqrt(r2);
      float ring = smoothstep(0.58, 0.68, r) * (1.0 - smoothstep(0.84, 0.96, r));
      rgb += vec3(1.0) * ring * 1.5;
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
 * Twelve domain colours, generated in OKLCH rather than chosen by eye.
 *
 * The previous palette described itself as "nine cool hues, three warm
 * accents". It was the opposite: eleven of twelve sat between 0° and 133° —
 * gold, orange, neon yellow, tangerine, lime, coral, crimson, peach, lemon and
 * three pinks. No blues, no greens, no cyans. That is why domains were hard to
 * tell apart, and the docstring had been asserting the reverse for months.
 *
 * Construction: twelve hues 30° apart at constant chroma, with lightness
 * ALTERNATING between 0.70 and 0.92. The alternation is the important part —
 * hues that collapse together under colour blindness are then separated by
 * brightness instead, which no amount of hue spacing can achieve.
 *
 * Measured in OKLab, minimum pairwise distance across all 66 pairs:
 *
 *                      before   after
 *   normal vision      0.0754  0.1080   1.43x
 *   deuteranopia       0.0311  0.0553   1.78x
 *   tritanopia         0.0148  0.0500   3.38x
 *   vs the planet      0.3671  0.3288   still well clear
 *
 * Every colour stays bright because nodes composite additively onto near-black
 * and must remain the most luminous thing on screen; and every one is checked
 * against the planet's mid-blue so no domain disappears into the surface.
 *
 * Regenerate rather than hand-edit: adjusting one entry by eye is how a palette
 * drifts back into a single hue family.
 */
export const DOMAIN_PALETTE: readonly (readonly [number, number, number])[] = [
  [0.891, 0.463, 0.607], // AI / ML — rose
  [1.000, 0.753, 0.677], // Web frontend — pale salmon
  [0.865, 0.529, 0.212], // Data and storage — amber
  [1.000, 0.879, 0.429], // Infrastructure — light gold
  [0.579, 0.669, 0.225], // Languages and compilers — olive
  [0.615, 1.000, 0.700], // Systems and embedded — pale mint
  [0.000, 0.729, 0.634], // Data engineering — teal
  [0.327, 0.994, 1.000], // Security — pale cyan
  [0.216, 0.660, 0.925], // Graphics and games — azure
  [0.754, 0.877, 1.000], // Mobile — pale periwinkle
  [0.675, 0.536, 0.910], // Automation and tooling — violet
  [1.000, 0.760, 1.000], // Science and numerics — pale orchid
];
