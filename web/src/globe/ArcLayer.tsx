import { useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import { ARC_FRAG, ARC_VERT } from './arcShaders';
import { ARC_KIND_COLOR } from './palette';
import { useGlobeStore } from '../store/useGlobeStore';

const SEGMENTS = 20;
const VERTS_PER_ARC = (SEGMENTS + 1) * 2;
const INDICES_PER_ARC = SEGMENTS * 6;

export interface ArcEndpoints {
  /** Unit direction of endpoint A. */
  a: THREE.Vector3;
  b: THREE.Vector3;
  /** 0..1 — drives brightness and width. */
  weight: number;
  nodeA: number;
  nodeB: number;
  /**
   * Relationship type: 0=depends_on, 1=similar_to, 2=used_with. Drives the
   * arc's colour, so a link's appearance says what kind of link it is.
   */
  kind?: number;
}

export interface ArcStyle {
  color: THREE.Color;
  pulseColor: THREE.Color;
  focusColor: THREE.Color;
  baseAlpha: number;
  pulseGain: number;
  pulseSpeed: number;
  pulseWidth: number;
  widthPx: number;
  liftBase: number;
  liftScale: number;
  focusBoost: number;
}

/**
 * A fixed-capacity pool of GPU-resident arcs.
 *
 * Geometry is allocated once at `capacity` and never reallocated. Updating the
 * set of visible arcs writes endpoint attributes and moves the draw range —
 * no geometry rebuild, so hovering a hub with 600 neighbours costs the same as
 * hovering a leaf.
 */
export class ArcPool {
  readonly geometry: THREE.BufferGeometry;
  readonly material: THREE.ShaderMaterial;
  private readonly endA: THREE.BufferAttribute;
  private readonly endB: THREE.BufferAttribute;
  private readonly meta: THREE.BufferAttribute;
  private readonly nodes: THREE.BufferAttribute;
  private activeArcs = 0;

  constructor(
    readonly capacity: number,
    radius: number,
    style: ArcStyle,
  ) {
    const verts = capacity * VERTS_PER_ARC;

    const endAArray = new Float32Array(verts * 3);
    const endBArray = new Float32Array(verts * 3);
    const paramsArray = new Float32Array(verts * 2);
    const metaArray = new Float32Array(verts * 2);
    const nodesArray = new Float32Array(verts * 2);
    const indices = new Uint32Array(capacity * INDICES_PER_ARC);

    // `aParams` and the index buffer are pure topology — identical for every
    // arc, written once, never touched again.
    for (let arc = 0; arc < capacity; arc++) {
      const vBase = arc * VERTS_PER_ARC;
      for (let s = 0; s <= SEGMENTS; s++) {
        const t = s / SEGMENTS;
        for (let side = 0; side < 2; side++) {
          const v = vBase + s * 2 + side;
          paramsArray[v * 2] = t;
          paramsArray[v * 2 + 1] = side === 0 ? -1 : 1;
        }
      }
      const iBase = arc * INDICES_PER_ARC;
      for (let s = 0; s < SEGMENTS; s++) {
        const v = vBase + s * 2;
        const o = iBase + s * 6;
        indices[o] = v;
        indices[o + 1] = v + 1;
        indices[o + 2] = v + 2;
        indices[o + 3] = v + 1;
        indices[o + 4] = v + 3;
        indices[o + 5] = v + 2;
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.endA = new THREE.BufferAttribute(endAArray, 3);
    this.endB = new THREE.BufferAttribute(endBArray, 3);
    this.meta = new THREE.BufferAttribute(metaArray, 2);
    this.nodes = new THREE.BufferAttribute(nodesArray, 2);
    this.endA.setUsage(THREE.DynamicDrawUsage);
    this.endB.setUsage(THREE.DynamicDrawUsage);
    this.meta.setUsage(THREE.DynamicDrawUsage);
    this.nodes.setUsage(THREE.DynamicDrawUsage);

    this.geometry.setAttribute('position', this.endA);
    this.geometry.setAttribute('aEndB', this.endB);
    this.geometry.setAttribute('aParams', new THREE.BufferAttribute(paramsArray, 2));
    this.geometry.setAttribute('aMeta', this.meta);
    this.geometry.setAttribute('aNodes', this.nodes);
    this.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), radius * 1.6);
    this.geometry.setDrawRange(0, 0);

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uRadius: { value: radius },
        uWidthPx: { value: style.widthPx },
        uWorldPerPixel: { value: 0.002 },
        uTime: { value: 0 },
        uPulseSpeed: { value: style.pulseSpeed },
        uPulseWidth: { value: style.pulseWidth },
        uLiftBase: { value: style.liftBase },
        uLiftScale: { value: style.liftScale },
        uFocusNode: { value: -1 },
        uFocusBoost: { value: style.focusBoost },
        uColor: { value: style.color.clone() },
        uKindColor: {
          value: ARC_KIND_COLOR.map(([r, g, b]) => new THREE.Vector3(r, g, b)),
        },
        uPulseColor: { value: style.pulseColor.clone() },
        uFocusColor: { value: style.focusColor.clone() },
        uBaseAlpha: { value: style.baseAlpha },
        uPulseGain: { value: style.pulseGain },
      },
      vertexShader: ARC_VERT,
      fragmentShader: ARC_FRAG,
      transparent: true,
      depthWrite: false,
      // Depth test against the opaque core is what hides arcs on the far side.
      depthTest: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
  }

  get count() {
    return this.activeArcs;
  }

  setArcs(arcs: readonly ArcEndpoints[]) {
    const n = Math.min(arcs.length, this.capacity);
    const ea = this.endA.array as Float32Array;
    const eb = this.endB.array as Float32Array;
    const me = this.meta.array as Float32Array;
    const nd = this.nodes.array as Float32Array;

    for (let arc = 0; arc < n; arc++) {
      const { a, b, weight, nodeA, nodeB, kind = 0 } = arcs[arc];
      // A stable per-arc phase, so pulses don't march in lockstep. Derived from
      // the node ids rather than random, so it survives a re-render unchanged.
      //
      // Kind rides in the INTEGER part of the same float. Phase is [0,1) by
      // construction, so kind + phase is exactly recoverable with floor/fract,
      // and it avoids widening aMeta to a vec3 — which would mean reallocating
      // the buffer, changing itemSize in three places, and touching the upload
      // range arithmetic, all to carry two bits.
      const phase = ((nodeA * 2654435761 + nodeB * 40503) % 1000) / 1000;
      const vBase = arc * VERTS_PER_ARC;
      for (let v = vBase; v < vBase + VERTS_PER_ARC; v++) {
        ea[v * 3] = a.x;
        ea[v * 3 + 1] = a.y;
        ea[v * 3 + 2] = a.z;
        eb[v * 3] = b.x;
        eb[v * 3 + 1] = b.y;
        eb[v * 3 + 2] = b.z;
        me[v * 2] = weight;
        me[v * 2 + 1] = kind + phase;
        nd[v * 2] = nodeA;
        nd[v * 2 + 1] = nodeB;
      }
    }

    // Only the touched range is re-uploaded. Without this, swapping 12 focus
    // arcs would push the whole 256-arc buffer to the GPU every hover.
    const touched = n * VERTS_PER_ARC;
    for (const attr of [this.endA, this.endB, this.meta, this.nodes]) {
      attr.addUpdateRange(0, touched * attr.itemSize);
      attr.needsUpdate = true;
    }

    this.activeArcs = n;
    this.geometry.setDrawRange(0, n * INDICES_PER_ARC);
  }

  clear() {
    this.activeArcs = 0;
    this.geometry.setDrawRange(0, 0);
  }

  setFocus(nodeId: number) {
    this.material.uniforms.uFocusNode.value = nodeId;
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}

export const AMBIENT_STYLE: ArcStyle = {
  color: new THREE.Color(0.85, 0.45, 0.15), // warm copper
  pulseColor: new THREE.Color(1.0, 0.85, 0.3), // gold pulse
  focusColor: new THREE.Color(1.0, 0.4, 0.4), // fiery red
  baseAlpha: 0.015,
  pulseGain: 0.36,
  pulseSpeed: 0.09,
  pulseWidth: 0.055,
  widthPx: 1.5,
  liftBase: 0.012,
  liftScale: 0.30,
  focusBoost: 1,
};

export const FOCUS_STYLE: ArcStyle = {
  color: new THREE.Color(1.0, 0.65, 0.2), // bright orange
  pulseColor: new THREE.Color(1.0, 1.0, 0.9), // almost white hot
  focusColor: new THREE.Color(1.0, 0.3, 0.3), // intense red
  baseAlpha: 0.34,
  pulseGain: 0.95,
  pulseSpeed: 0.34,
  pulseWidth: 0.05,
  widthPx: 8.0,
  liftBase: 0.02,
  liftScale: 0.34,
  focusBoost: 0,
};

export function ArcLayer({ pool, radius: _radius }: { pool: ArcPool; radius: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera, size } = useThree();
  const reducedMotion = useGlobeStore((s) => s.reducedMotion);

  // Screen-constant ribbon width needs the vertical world-units-per-pixel at
  // unit distance. It changes only on resize or fov change, not per frame.
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    const worldPerPixel = (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2)) / size.height;
    pool.material.uniforms.uWorldPerPixel.value = worldPerPixel;
  }, [camera, size, pool]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (mesh) {
      mesh.frustumCulled = false;
      mesh.raycast = () => null;
      mesh.renderOrder = 2;
    }
  }, []);

  useFrame((_, delta) => {
    // Freezing time rather than skipping the uniform keeps the pulses where
    // they were instead of snapping when motion is re-enabled.
    if (!reducedMotion) pool.material.uniforms.uTime.value += delta;
  });

  return <mesh ref={meshRef} geometry={pool.geometry} material={pool.material} />;
}
