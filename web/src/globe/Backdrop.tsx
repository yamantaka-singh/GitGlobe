import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

import { SUN_DIR } from './lighting';
import { ATMOSPHERE, CITY_LIGHT, NEBULA, PLANET_SURFACE, SPACE } from './palette';

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

// ---------------------------------------------------------------- deep space

const SKY_VERT = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = normalize(position);
    // Depth forced to the far plane so the sky never occludes anything and
    // never needs sorting against the scene.
    vec4 p = projectionMatrix * viewMatrix * vec4(position + cameraPosition, 1.0);
    gl_Position = p.xyww;
  }
`;

const SKY_FRAG = /* glsl */ `
  precision mediump float;

  uniform vec3 uSpace;
  uniform vec3 uWarm;
  uniform vec3 uCool;
  uniform vec3 uCore;

  varying vec3 vDir;

  // Cheap value noise — the sky is huge in screen area, so this is the one
  // place in the project where fragment cost really is the whole budget.
  float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
          mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
      mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
          mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
      f.z);
  }

  float fbmSky(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p *= 2.1; a *= 0.5; }
    return v;
  }

  void main() {
    vec3 d = normalize(vDir);

    // Two independent cloud fields at different scales and offsets. One field
    // reads as a texture; two overlapping ones read as depth.
    float warmField = fbmSky(d * 2.3 + 11.0);
    float coolField = fbmSky(d * 1.7 - 27.0);

    // Thresholded hard — nebulae are mostly empty sky with a few dense regions,
    // and a soft threshold gives you uniform haze instead.
    float warm = smoothstep(0.52, 0.86, warmField);
    float cool = smoothstep(0.48, 0.88, coolField);

    // A broad band of denser dust, like looking along a galactic plane. Gives
    // the sky an orientation, which stops it reading as random.
    float plane = exp(-pow((d.y + 0.18) * 2.6, 2.0));

    vec3 rgb = uSpace;
    rgb += uCool * cool * 0.55 * (0.35 + 0.65 * plane);
    rgb += uWarm * warm * 0.42 * (0.25 + 0.75 * plane);
    // Where both fields are dense, a brighter core.
    rgb += uCore * warm * cool * 0.85;

    gl_FragColor = vec4(rgb, 1.0);
  }
`;

/**
 * Nebula backdrop.
 *
 * A plain starfield on flat black reads as a screensaver. Two thresholded noise
 * fields plus a galactic-plane band give the sky depth and orientation, and the
 * planet suddenly looks like it is somewhere rather than floating on a colour.
 *
 * Rendered as an inside-out box locked to the camera with `gl_Position.xyww`,
 * so it always sits exactly on the far plane at zero depth-sorting cost.
 */
export function Nebula() {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uSpace: { value: new THREE.Color(...SPACE) },
          uWarm: { value: new THREE.Color(...NEBULA.warm) },
          uCool: { value: new THREE.Color(...NEBULA.cool) },
          uCore: { value: new THREE.Color(...NEBULA.core) },
        },
        vertexShader: SKY_VERT,
        fragmentShader: SKY_FRAG,
        side: THREE.BackSide,
        depthWrite: false,
        depthTest: false,
      }),
    [],
  );

  useEffect(() => () => material.dispose(), [material]);

  return (
    <mesh material={material} raycast={() => null} frustumCulled={false} renderOrder={-1}>
      <boxGeometry args={[2, 2, 2]} />
    </mesh>
  );
}

// ---------------------------------------------------------------- starfield

const STAR_VERT = /* glsl */ `
  attribute float aMag;
  attribute vec3 aTint;
  varying float vMag;
  varying vec3 vTint;
  void main() {
    vMag = aMag;
    vTint = aTint;
    gl_PointSize = aMag * 2.4;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const STAR_FRAG = /* glsl */ `
  precision mediump float;
  varying float vMag;
  varying vec3 vTint;
  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = dot(uv, uv);
    if (r > 1.0) discard;
    float core = 1.0 - r;
    gl_FragColor = vec4(vTint, core * core * vMag * 0.55);
  }
`;

export function Starfield({ count = 3600, radius = 60 }: { count?: number; radius?: number }) {
  const { geometry, material } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const mag = new Float32Array(count);
    const tint = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const z = 2 * Math.random() - 1;
      const t = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      pos[i * 3] = r * Math.cos(t) * radius;
      pos[i * 3 + 1] = z * radius;
      pos[i * 3 + 2] = r * Math.sin(t) * radius;
      mag[i] = 0.25 + Math.pow(Math.random(), 3.4) * 1.5;

      // Real starfields are not white. A spread from cool blue-white through
      // to amber is the difference between "stars" and "dust on the lens".
      const k = Math.random();
      const c =
        k < 0.62 ? [0.78, 0.84, 1.0] :
        k < 0.86 ? [1.0, 0.98, 0.92] :
        k < 0.96 ? [1.0, 0.86, 0.68] :
                   [1.0, 0.72, 0.62];
      tint[i * 3] = c[0];
      tint[i * 3 + 1] = c[1];
      tint[i * 3 + 2] = c[2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('aMag', new THREE.Float32BufferAttribute(mag, 1));
    geo.setAttribute('aTint', new THREE.Float32BufferAttribute(tint, 3));
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

    // ---- emission ----------------------------------------------------------
    // A nebula is self-luminous, so there is no terminator to speak of. The sun
    // term is kept only as a weak modelling cue: with no directional variation
    // at all a sphere of glowing gas flattens into a disc and the globe stops
    // reading as a globe. uNightFloor is therefore high rather than near
    // zero - it sets how much of the surface is visible independent of the
    // light, which for gas is nearly all of it.
    float sun = dot(n, normalize(uSunDir));
    float day = smoothstep(-uTerminator, uTerminator, sun);

    vec3 rgb = albedo * (uNightFloor + (1.0 - uNightFloor) * day);

    // The gas glows its OWN colour. Routing emission through a single fixed
    // tint - which is what the ice giant's aurora did - would erase the domain
    // hue exactly where the gas is densest, and density is where territories
    // are. A small cyan lift stands in for the OIII line sitting on top.
    float night = 1.0 - day;
    vec3 emissive = albedo * 1.45 + uCityLight * 0.22;
    rgb += emissive * cityLights * (0.62 + 0.38 * night);

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
          uFallback: { value: new THREE.Color(...PLANET_SURFACE.mid) },
          uGraticule: { value: new THREE.Color(0.24, 0.44, 0.62) },
          uGraticuleGain: { value: 0.16 },
          // Never fully black: an unlit hemisphere with zero albedo loses its
          // silhouette against space and the globe looks bitten into.
          // High, not near-zero: emissive gas is visible all the way round.
          // The residual 0.28 of directional variation is what keeps the sphere
          // from flattening into a disc.
          uNightFloor: { value: 0.72 },
          // A gas giant's atmosphere is deep, so its terminator is soft. A hard
          // edge is the tell that you are looking at a lit sphere, not a world.
          uTerminator: { value: 0.42 },
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
      {/* Alien X Outline - uniform 360 glow, zero sunBias so it's a perfect silhouette cut-out */}
      <Shell radius={radius * 1.008} color={ATMOSPHERE.edge} intensity={0.9} power={20} sunBias={0.0} renderOrder={3} />
      <Shell radius={radius * 1.02} color={ATMOSPHERE.rim} intensity={0.5} power={8} sunBias={0.0} renderOrder={4} />
    </>
  );
}
