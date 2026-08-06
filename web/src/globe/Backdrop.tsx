import { useMemo } from 'react';
import * as THREE from 'three';

import { SUN_DIR } from './lighting';

/**
 * Static subtree — nothing here changes after mount, so it can be built once
 * and forgotten (web3d-scene-architect's lifecycle rule).
 *
 * The atmosphere is three separate layers, not one. A single Fresnel shell is
 * why the first version read as a soft blue lamp: it has no edge, so the eye
 * never finds the horizon. Splitting it into a hard rim, a wide scatter, and a
 * gridded core gives the silhouette a defined boundary and the surface a sense
 * of curvature — which is what makes it read as an instrument.
 */

// ---------------------------------------------------------------- starfield

const STAR_VERT = /* glsl */ `
  attribute float aMag;
  varying float vMag;
  void main() {
    vMag = aMag;
    gl_PointSize = aMag * 2.2;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const STAR_FRAG = /* glsl */ `
  precision mediump float;
  varying float vMag;
  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = dot(uv, uv);
    if (r > 1.0) discard;
    gl_FragColor = vec4(vec3(0.66, 0.76, 0.95), (1.0 - r) * vMag * 0.24);
  }
`;

export function Starfield({ count = 2600, radius = 60 }: { count?: number; radius?: number }) {
  const { geometry, material } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const mag = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const z = 2 * Math.random() - 1;
      const t = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      pos[i * 3] = r * Math.cos(t) * radius;
      pos[i * 3 + 1] = z * radius;
      pos[i * 3 + 2] = r * Math.sin(t) * radius;
      mag[i] = 0.3 + Math.pow(Math.random(), 3) * 1.3;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('aMag', new THREE.Float32BufferAttribute(mag, 1));
    return {
      geometry: geo,
      material: new THREE.ShaderMaterial({
        vertexShader: STAR_VERT,
        fragmentShader: STAR_FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    };
  }, [count, radius]);

  return <points geometry={geometry} material={material} frustumCulled={false} raycast={() => null} />;
}

// ---------------------------------------------------------------- core

const CORE_VERT = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  varying vec3 vLocal;
  void main() {
    vLocal = normalize(position);
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const CORE_FRAG = /* glsl */ `
  precision highp float;

  uniform vec3  uBase;
  uniform vec3  uGrid;
  uniform vec3  uSunDir;      // world-space, normalised
  uniform vec3  uSunTint;
  uniform float uParallels;
  uniform float uMeridians;
  uniform float uGridGain;
  uniform float uTerminator;

  varying vec3 vNormalW;
  varying vec3 vPosW;
  varying vec3 vLocal;

  const float PI  = 3.141592653589793;
  const float TAU = 6.283185307179586;

  // Derivative-aware line: constant screen width regardless of how badly the
  // surface is foreshortened. A fixed-width smoothstep shears apart at the limb.
  float gridLine(float coord, float count, float weight) {
    float scaled = coord * count;
    float f = abs(fract(scaled) - 0.5);
    return 1.0 - smoothstep(0.0, fwidth(scaled) * weight, f);
  }

  void main() {
    vec3 n = normalize(vNormalW);
    vec3 toCam = normalize(cameraPosition - vPosW);
    float facing = max(dot(n, toCam), 0.0);

    float theta = acos(clamp(vLocal.y, -1.0, 1.0)) / PI;
    float phi = (atan(vLocal.z, vLocal.x) + PI) / TAU;

    // Two grid densities: a fine mesh plus a heavier line every fifth division.
    // A single uniform grid reads as graph paper; a hierarchy reads as an
    // instrument that someone graduated on purpose.
    float fine = max(gridLine(theta, uParallels, 1.1), gridLine(phi, uMeridians, 1.1)) * 0.34;
    float major = max(gridLine(theta, uParallels / 3.0, 1.5), gridLine(phi, uMeridians / 6.0, 1.5));
    float lines = max(fine, major);

    // Meridians converge at the poles into a solid blob — fade them out there.
    lines *= smoothstep(0.0, 0.14, theta) * smoothstep(0.0, 0.14, 1.0 - theta);

    // THE thing the previous version was missing. Without a light direction the
    // sphere is uniformly dark and reads as a flat disc no matter how good the
    // rim is. A terminator gives it volume: one limb catches light, the
    // opposite side falls away, and the eye finally reads it as a solid body.
    float sun = dot(n, normalize(uSunDir));
    float day = smoothstep(-uTerminator, uTerminator, sun);

    // Grazing angles only, so the grid describes curvature at the limb rather
    // than tiling the whole surface like wallpaper.
    float grazing = pow(1.0 - facing, 2.2);

    vec3 rgb = uBase * (0.55 + 0.45 * day);
    rgb += uGrid * lines * grazing * uGridGain * (0.28 + 0.72 * day);
    // A warm sliver exactly at the terminator — the strongest single cue that
    // this is a lit body and not a coloured circle.
    rgb += uSunTint * pow(1.0 - abs(sun), 22.0) * 0.55 * grazing;

    gl_FragColor = vec4(rgb, 1.0);
  }
`;

/**
 * Opaque inner sphere. Load-bearing in two ways: it stops you seeing straight
 * through to the far hemisphere's points, and it writes depth, which is what
 * correctly occludes arcs passing behind the globe.
 */
export function Core({ radius }: { radius: number }) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uBase: { value: new THREE.Color(0.010, 0.020, 0.038) },
          uGrid: { value: new THREE.Color(0.13, 0.48, 0.62) },
          uSunDir: { value: SUN_DIR.clone() },
          uSunTint: { value: new THREE.Color(0.42, 0.86, 1.0) },
          uParallels: { value: 24 },
          uMeridians: { value: 48 },
          uGridGain: { value: 1.35 },
          // How soft the day/night boundary is. Hard enough to read as a
          // terminator, soft enough not to alias into a jagged line.
          uTerminator: { value: 0.42 },
        },
        vertexShader: CORE_VERT,
        fragmentShader: CORE_FRAG,
        transparent: false,
        depthWrite: true,
      }),
    [],
  );

  return (
    <mesh material={material} raycast={() => null} renderOrder={0}>
      <sphereGeometry args={[radius * 0.988, 128, 80]} />
    </mesh>
  );
}

// ---------------------------------------------------------------- shells

const SHELL_VERT = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const SHELL_FRAG = /* glsl */ `
  precision mediump float;
  uniform vec3  uColor;
  uniform vec3  uSunDir;
  uniform float uIntensity;
  uniform float uPower;
  uniform float uSunBias;
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vec3 n = normalize(vNormalW);
    vec3 toCam = normalize(cameraPosition - vPosW);
    float fres = pow(1.0 - max(dot(n, toCam), 0.0), uPower);

    // Scatter concentrates on the lit limb. A uniform halo is the tell-tale of
    // a fake atmosphere — real ones are brightest where the light grazes.
    float sun = smoothstep(-0.5, 0.85, dot(n, normalize(uSunDir)));
    float a = fres * uIntensity * mix(1.0, sun, uSunBias);
    if (a < 0.002) discard;
    gl_FragColor = vec4(uColor, a);
  }
`;

function Shell({
  radius,
  color,
  intensity,
  power,
  sunBias,
  renderOrder,
}: {
  radius: number;
  color: THREE.Color;
  intensity: number;
  power: number;
  sunBias: number;
  renderOrder: number;
}) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: color },
          uSunDir: { value: SUN_DIR.clone() },
          uIntensity: { value: intensity },
          uPower: { value: power },
          uSunBias: { value: sunBias },
        },
        vertexShader: SHELL_VERT,
        fragmentShader: SHELL_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      }),
    [color, intensity, power, sunBias],
  );

  return (
    <mesh material={material} raycast={() => null} frustumCulled={false} renderOrder={renderOrder}>
      <sphereGeometry args={[radius, 64, 48]} />
    </mesh>
  );
}

/**
 * Three shells, each doing one job:
 *
 * - **edge** — very high exponent, almost on the surface. A hairline. This is
 *   what makes the silhouette look cut rather than airbrushed.
 * - **rim** — high exponent, sun-biased. The lit crescent.
 * - **scatter** — low exponent, wide, faint, strongly sun-biased. Volume.
 *
 * One shell forces one exponent, and a single exponent can only ever produce a
 * gradient — which is why the first version read as a blue smudge.
 */
export function Atmosphere({ radius }: { radius: number }) {
  const edge = useMemo(() => new THREE.Color(0.72, 0.96, 1.0), []);
  const rim = useMemo(() => new THREE.Color(0.24, 0.78, 1.0), []);
  const scatter = useMemo(() => new THREE.Color(0.08, 0.30, 0.68), []);
  return (
    <>
      <Shell radius={radius * 1.004} color={edge} intensity={0.55} power={20} sunBias={0.35} renderOrder={3} />
      <Shell radius={radius * 1.03} color={rim} intensity={0.70} power={7.5} sunBias={0.6} renderOrder={4} />
      <Shell radius={radius * 1.19} color={scatter} intensity={0.34} power={2.4} sunBias={0.8} renderOrder={5} />
    </>
  );
}

// ---------------------------------------------------------------- equator

const RING_FRAG = /* glsl */ `
  precision mediump float;
  uniform vec3 uColor;
  uniform float uIntensity;
  varying vec2 vUv;
  void main() {
    float edge = abs(vUv.y - 0.5) * 2.0;
    float a = (1.0 - smoothstep(0.0, 1.0, edge)) * uIntensity;
    if (a < 0.002) discard;
    gl_FragColor = vec4(uColor, a);
  }
`;

const RING_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/**
 * A thin ring at the equator. Small detail, disproportionate effect: it gives
 * the globe an axis and a sense of scale, which is the difference between a
 * ball of dots and an instrument someone calibrated.
 */
export function EquatorRing({ radius }: { radius: number }) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color(0.24, 0.72, 0.88) },
          uIntensity: { value: 0.30 },
        },
        vertexShader: RING_VERT,
        fragmentShader: RING_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  return (
    <mesh
      material={material}
      rotation={[-Math.PI / 2, 0, 0]}
      raycast={() => null}
      frustumCulled={false}
      renderOrder={1}
    >
      <ringGeometry args={[radius * 1.22, radius * 1.245, 160]} />
    </mesh>
  );
}
