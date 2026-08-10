/**
 * Arc ribbon shaders. GLSL ES 1.00.
 *
 * Every arc's geometry is computed on the GPU from its two endpoints, so the
 * CPU never touches a vertex after upload. That is what makes the travelling
 * pulse free: it is one uniform write per frame, not a buffer rewrite.
 *
 * Attributes, per vertex:
 *   position  vec3   endpoint A, unit vector    (named `position` because
 *                                                three.js derives draw count
 *                                                from it — see PointCloud)
 *   aEndB     vec3   endpoint B, unit vector
 *   aParams   vec2   (t along arc 0..1, side -1/+1)
 *   aMeta     vec2   (weight 0..1, kind + phase — integer part is the
 *                        edge kind, fraction is the pulse phase)
 *   aNodes    vec2   (global id of A, global id of B)
 *
 * The ribbon is expanded in world space perpendicular to both the arc tangent
 * and the view direction, with a distance term so screen width stays constant.
 * Clip-space expansion (what Line2 does) would also work; world space is fewer
 * moving parts and the arcs are never near the camera.
 */

export const ARC_VERT = /* glsl */ `
  attribute vec3 aEndB;
  attribute vec2 aParams;
  attribute vec2 aMeta;
  attribute vec2 aNodes;

  uniform float uRadius;
  uniform float uWidthPx;
  uniform float uWorldPerPixel;   // 2*tan(fov/2) / viewportHeight
  uniform float uTime;
  uniform float uPulseSpeed;
  uniform float uPulseWidth;
  uniform float uLiftBase;
  uniform float uLiftScale;
  uniform float uFocusNode;
  uniform float uFocusBoost;

  varying float vWeight;
  varying float vPulse;
  varying float vFocus;
  varying float vEdge;
  varying float vFacing;
  varying float vKind;

  const float PI = 3.141592653589793;

  vec3 slerp(vec3 a, vec3 b, float t, float omega, float sinOmega) {
    if (sinOmega < 1.0e-4) return normalize(mix(a, b, t));
    return (sin((1.0 - t) * omega) / sinOmega) * a + (sin(t * omega) / sinOmega) * b;
  }

  vec3 arcPoint(vec3 a, vec3 b, float t, float omega, float sinOmega, float lift) {
    vec3 dir = slerp(a, b, t, omega, sinOmega);
    // sin(PI*t) peaks at the midpoint and is exactly 0 at both ends, so arcs
    // always land precisely on their nodes rather than hovering above them.
    return dir * (uRadius * (1.0 + lift * sin(PI * t)));
  }

  void main() {
    vec3 a = position;
    vec3 b = aEndB;
    float t = aParams.x;
    float side = aParams.y;

    float cosOmega = clamp(dot(a, b), -1.0, 1.0);
    float omega = acos(cosOmega);
    float sinOmega = sin(omega);

    // Long arcs lift higher, so a link across the globe reads as a deliberate
    // span rather than a line dragged through the surface.
    float lift = uLiftBase + uLiftScale * (omega / PI);

    vec3 p = arcPoint(a, b, t, omega, sinOmega, lift);
    // Finite-difference tangent. Cheaper than an analytic derivative and
    // indistinguishable at 20 segments.
    float dt = 0.012;
    vec3 pNext = arcPoint(a, b, clamp(t + dt, 0.0, 1.0), omega, sinOmega, lift);
    vec3 pPrev = arcPoint(a, b, clamp(t - dt, 0.0, 1.0), omega, sinOmega, lift);
    vec3 tangent = normalize(pNext - pPrev);

    vec3 worldPos = (modelMatrix * vec4(p, 1.0)).xyz;
    vec3 toCam = cameraPosition - worldPos;
    float dist = length(toCam);
    vec3 viewDir = toCam / max(dist, 1.0e-5);

    vec3 offsetDir = cross(tangent, viewDir);
    float offsetLen = length(offsetDir);
    // Degenerate when the arc runs straight at the camera; any perpendicular
    // will do there because the ribbon is edge-on and invisible anyway.
    offsetDir = offsetLen > 1.0e-4 ? offsetDir / offsetLen : vec3(0.0, 1.0, 0.0);

    float halfWidth = 0.5 * uWidthPx * uWorldPerPixel * dist;
    worldPos += offsetDir * side * halfWidth;

    vWeight = aMeta.x;
    vEdge = side;
    vFacing = dot(normalize(p), viewDir);

    float focusA = step(abs(aNodes.x - uFocusNode), 0.5);
    float focusB = step(abs(aNodes.y - uFocusNode), 0.5);
    vFocus = max(focusA, focusB) * uFocusBoost;

    // Travelling pulse: a gaussian bump chasing 't' around the wire.
    // aMeta.y packs kind in the integer part and phase in the fraction.
    vKind = floor(aMeta.y);
    float head = fract(uTime * uPulseSpeed + fract(aMeta.y));
    float d = t - head;
    d -= floor(d + 0.5);                      // wrap into [-0.5, 0.5]
    vPulse = exp(-(d * d) / (uPulseWidth * uPulseWidth));

    gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
  }
`;

export const ARC_FRAG = /* glsl */ `
  precision mediump float;

  uniform vec3  uColor;
  uniform vec3  uKindColor[3];
  uniform vec3  uPulseColor;
  uniform vec3  uFocusColor;
  uniform float uBaseAlpha;
  uniform float uPulseGain;

  varying float vWeight;
  varying float vPulse;
  varying float vFocus;
  varying float vEdge;
  varying float vFacing;
  varying float vKind;

  void main() {
    // Soft edges across the ribbon's width — a hard-edged quad reads as a
    // rectangle, not a wire.
    float across = clamp(1.0 - abs(vEdge), 0.0, 1.0);
    // pow(across, 0.75) is very expensive. across * (2.0 - across) is a fast
    // quadratic approximation that achieves the same soft profile visually.
    float profile = across * (2.0 - across);

    // Arcs on the far limb fade rather than pop. The opaque core already
    // occludes anything genuinely behind the globe.
    float limb = smoothstep(-0.55, -0.1, vFacing);

    float weight = 0.3 + 0.7 * vWeight;
    float base = uBaseAlpha * weight * (1.0 + vFocus * 3.0);
    float pulse = vPulse * uPulseGain * (0.4 + 0.6 * vFocus);

    float alpha = (base + pulse) * profile * limb;
    if (alpha < 0.003) discard;

    // Colour by relationship type. Every arc used to be one colour, so the
    // wires said "these two repos are connected" and nothing more — the single
    // most information-free channel on the globe. Dynamic indexing of a uniform
    // array is illegal in ES 1.00 fragment shaders, so this is the same masked
    // accumulation the domain tint uses in BAKE_FRAG.
    vec3 kindColor = vec3(0.0);
    for (int i = 0; i < 3; i++) {
      kindColor += uKindColor[i] * step(abs(float(i) - vKind), 0.5);
    }

    vec3 rgb = mix(kindColor, uPulseColor, clamp(vPulse, 0.0, 1.0));
    rgb = mix(rgb, uFocusColor, clamp(vFocus, 0.0, 1.0) * 0.75);

    gl_FragColor = vec4(rgb, alpha);
  }
`;
