import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

import { SUN_DIR } from './lighting';
import { ATMOSPHERE, CITY_LIGHT, PLANET_SURFACE } from './palette';

/**
 * Static subtree — nothing here changes after mount (web3d-scene-architect's
 * lifecycle rule), so it is built once and forgotten.
 *
 * The globe is a planet, not a wireframe ball: a baked terrain map with a
 * day/night terminator, amber city lights on the dark side, and a three-layer
 * periwinkle-to-violet atmosphere. Structure follows the reference — a
 * naturalistic, muted body so that the data standing on it stays the brightest
 * thing on screen.
 */

// ---------------------------------------------------------------- starfield

const STAR_VERT = /* glsl */ `
  attribute float aMag;
  varying float vMag;
  void main() {
    vMag = aMag;
    gl_PointSize = aMag * 2.1;
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
    gl_FragColor = vec4(vec3(0.70, 0.76, 0.92), (1.0 - r) * vMag * 0.20);
  }
`;

export function Starfield({ count = 2200, radius = 60 }: { count?: number; radius?: number }) {
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
      mag[i] = 0.28 + Math.pow(Math.random(), 3.2) * 1.25;
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

// ---------------------------------------------------------------- planet

const PLANET_VERT = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const PLANET_FRAG = /* glsl */ `
  precision highp float;

  uniform sampler2D uSurface;
  uniform float uHasSurface;
  uniform vec3  uSunDir;
  uniform vec3  uCityLight;
  uniform vec3  uFallback;
  uniform vec3  uGraticule;
  uniform float uGraticuleGain;
  uniform float uNightFloor;
  uniform float uTerminator;

  varying vec3 vNormalW;
  varying vec3 vPosW;
  varying vec2 vUv;

  const float PI  = 3.141592653589793;
  const float TAU = 6.283185307179586;

  float gridLine(float coord, float count, float weight) {
    float scaled = coord * count;
    float f = abs(fract(scaled) - 0.5);
    return 1.0 - smoothstep(0.0, fwidth(scaled) * weight, f);
  }

  void main() {
    vec3 n = normalize(vNormalW);
    vec3 toCam = normalize(cameraPosition - vPosW);
    float facing = max(dot(n, toCam), 0.0);

    vec4 surf = texture2D(uSurface, vUv);
    // Until the bake lands, fall back to flat ocean rather than black — a
    // one-frame black sphere reads as a load failure.
    vec3 albedo = mix(uFallback, surf.rgb, uHasSurface);
    float cityLights = surf.a * uHasSurface;

    // ---- day / night -------------------------------------------------------
    float sun = dot(n, normalize(uSunDir));
    float day = smoothstep(-uTerminator, uTerminator, sun);

    vec3 rgb = albedo * (uNightFloor + (1.0 - uNightFloor) * day);

    // Cities burn only on the unlit side, brightest deep into the dark where
    // nothing competes with them. This is the single strongest cue in the
    // reference image that the planet is inhabited.
    float night = 1.0 - day;
    rgb += uCityLight * cityLights * night * night * 1.35;

    // A warm sliver exactly at the terminator, plus a hint of scatter bleeding
    // onto the dark side of the boundary.
    rgb += uCityLight * pow(1.0 - abs(sun), 26.0) * 0.16;

    // ---- graticule ---------------------------------------------------------
    // Kept very faint now that there is terrain to read. It exists to say
    // "this is an instrument", not to be looked at.
    float theta = vUv.y;
    float phi = vUv.x;
    float lines = max(gridLine(theta, 9.0, 1.4), gridLine(phi, 18.0, 1.4));
    lines *= smoothstep(0.0, 0.10, theta) * smoothstep(0.0, 0.10, 1.0 - theta);
    float grazing = pow(1.0 - facing, 2.4);
    rgb += uGraticule * lines * grazing * uGraticuleGain * (0.25 + 0.75 * day);

    gl_FragColor = vec4(rgb, 1.0);
  }
`;

/**
 * The planet body. Opaque, and load-bearing twice over: it stops you seeing
 * through to the far hemisphere's nodes, and it writes depth, which is what
 * correctly occludes arcs passing behind the globe.
 */
export function Planet({ radius, surface }: { radius: number; surface: THREE.Texture | null }) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uSurface: { value: null as THREE.Texture | null },
          uHasSurface: { value: 0 },
          uSunDir: { value: SUN_DIR.clone() },
          uCityLight: { value: new THREE.Color(...CITY_LIGHT) },
          uFallback: { value: new THREE.Color(...PLANET_SURFACE.deepOcean) },
          uGraticule: { value: new THREE.Color(0.24, 0.44, 0.62) },
          uGraticuleGain: { value: 0.42 },
          // Never fully black: an unlit hemisphere with zero albedo loses its
          // silhouette against space and the globe looks bitten into.
          uNightFloor: { value: 0.10 },
          uTerminator: { value: 0.30 },
        },
        vertexShader: PLANET_VERT,
        fragmentShader: PLANET_FRAG,
        transparent: false,
        depthWrite: true,
      }),
    [],
  );

  useEffect(() => {
    material.uniforms.uSurface.value = surface;
    material.uniforms.uHasSurface.value = surface ? 1 : 0;
  }, [material, surface]);

  useEffect(() => () => material.dispose(), [material]);

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
    // a fake atmosphere — real ones are brightest where light grazes.
    float sun = smoothstep(-0.45, 0.9, dot(n, normalize(uSunDir)));
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
  color: RGBTuple;
  intensity: number;
  power: number;
  sunBias: number;
  renderOrder: number;
}) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color(...color) },
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

  useEffect(() => () => material.dispose(), [material]);

  return (
    <mesh material={material} raycast={() => null} frustumCulled={false} renderOrder={renderOrder}>
      <sphereGeometry args={[radius, 64, 48]} />
    </mesh>
  );
}

type RGBTuple = readonly [number, number, number];

/**
 * Three shells, each doing one job:
 *
 * - **edge** — very high exponent, almost on the surface. A hairline that makes
 *   the silhouette look cut rather than airbrushed.
 * - **rim** — periwinkle, sun-biased. The lit crescent.
 * - **scatter** — violet, wide, faint. The halo that bleeds into space.
 *
 * The periwinkle-to-violet shift across the layers is the detail that makes the
 * reference read as photographic. A single-hue glow always looks synthetic.
 */
export function Atmosphere({ radius }: { radius: number }) {
  return (
    <>
      <Shell radius={radius * 1.004} color={ATMOSPHERE.edge} intensity={0.42} power={22} sunBias={0.45} renderOrder={3} />
      <Shell radius={radius * 1.035} color={ATMOSPHERE.rim} intensity={0.78} power={6.5} sunBias={0.62} renderOrder={4} />
      <Shell radius={radius * 1.20} color={ATMOSPHERE.scatter} intensity={0.40} power={2.2} sunBias={0.78} renderOrder={5} />
    </>
  );
}
