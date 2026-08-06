import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Static subtree, per web3d-scene-architect's lifecycle rule: nothing here ever
 * changes, so it can be built once and forgotten. Two draw calls total.
 */

const STAR_VERT = /* glsl */ `
  attribute float aMag;
  varying float vMag;
  void main() {
    vMag = aMag;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aMag * 2.6;
    gl_Position = projectionMatrix * mv;
  }
`;

const STAR_FRAG = /* glsl */ `
  precision mediump float;
  varying float vMag;
  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = dot(uv, uv);
    if (r > 1.0) discard;
    gl_FragColor = vec4(vec3(0.72, 0.78, 0.95), (1.0 - r) * vMag * 0.30);
  }
`;

export function Starfield({ count = 2400, radius = 60 }: { count?: number; radius?: number }) {
  const { geometry, material } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const mag = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      // Uniform on the sphere: inverse-CDF on z, not uniform in theta.
      const z = 2 * Math.random() - 1;
      const t = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      pos[i * 3] = r * Math.cos(t) * radius;
      pos[i * 3 + 1] = z * radius;
      pos[i * 3 + 2] = r * Math.sin(t) * radius;
      mag[i] = 0.35 + Math.pow(Math.random(), 3) * 1.4;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('aMag', new THREE.Float32BufferAttribute(mag, 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: STAR_VERT,
      fragmentShader: STAR_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return { geometry: geo, material: mat };
  }, [count, radius]);

  return <points geometry={geometry} material={material} frustumCulled={false} raycast={() => null} />;
}

const ATMO_VERT = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const ATMO_FRAG = /* glsl */ `
  precision mediump float;
  uniform vec3 uColor;
  uniform float uIntensity;
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vec3 toCam = normalize(cameraPosition - vPosW);
    // Fresnel: bright where the surface turns away, which is the limb.
    float fres = pow(1.0 - max(dot(normalize(vNormalW), toCam), 0.0), 3.2);
    gl_FragColor = vec4(uColor, fres * uIntensity);
  }
`;

/**
 * Backside-rendered shell slightly larger than the point cloud. Gives the globe
 * a horizon so it reads as a sphere rather than a floating cloud of dots.
 */
export function Atmosphere({ radius }: { radius: number }) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color(0.28, 0.46, 0.9) },
          uIntensity: { value: 0.55 },
        },
        vertexShader: ATMO_VERT,
        fragmentShader: ATMO_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  return (
    <mesh material={material} raycast={() => null} frustumCulled={false}>
      <sphereGeometry args={[radius * 1.14, 48, 32]} />
    </mesh>
  );
}

/**
 * Opaque inner sphere. Without it you see straight through to the far
 * hemisphere's points, and the globe reads as a hollow shell of noise.
 */
export function Core({ radius }: { radius: number }) {
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.016, 0.022, 0.045),
        transparent: false,
      }),
    [],
  );
  return (
    <mesh material={material} raycast={() => null}>
      <sphereGeometry args={[radius * 0.985, 64, 48]} />
    </mesh>
  );
}
