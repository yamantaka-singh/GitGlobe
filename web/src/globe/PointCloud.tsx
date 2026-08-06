import { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import type { LoadedBand } from '../tile/loader';
import { DOMAIN_PALETTE, PICK_FRAG, PICK_VERT, POINTS_FRAG, POINTS_VERT } from './shaders';
import { PICK_LAYER } from './layers';
import { SUN_DIR } from './lighting';
import { useGlobeStore } from '../store/useGlobeStore';

export interface PointCloudHandle {
  points: THREE.Points;
  pickMaterial: THREE.ShaderMaterial;
  displayMaterial: THREE.ShaderMaterial;
  idOffset: number;
  count: number;
}

interface Props {
  band: LoadedBand;
  radius: number;
  onReady?: (handle: PointCloudHandle) => void;
  onDispose?: (idOffset: number) => void;
}

/**
 * One THREE.Points per LOD band — one draw call each, three total at 1M points.
 *
 * Everything the GPU needs is uploaded once at mount. Nothing in useFrame
 * allocates, per web3d-performance-budget's allocation rule; the only per-frame
 * writes are two uniform scalars.
 */
export function PointCloud({ band, radius, onReady, onDispose }: Props) {
  const { gl } = useThree();
  const pointsRef = useRef<THREE.Points>(null);

  const { geometry, displayMaterial, pickMaterial } = useMemo(() => {
    const { tile, idOffset } = band;
    const n = tile.count;

    // (thetaQ, phiQ, sizeQ) packed into `position` — three.js needs a position
    // attribute to derive the draw count, so we make it carry real payload
    // rather than 12 MB of dummy xyz. See shaders.ts.
    const packed = new Uint16Array(n * 3);
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      packed[o] = tile.thetaQ[i]; // non-negative by construction
      packed[o + 1] = tile.phiQ[i];
      packed[o + 2] = tile.sizeQ[i];
    }

    const index = new Float32Array(n);
    for (let i = 0; i < n; i++) index[i] = i;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Uint16BufferAttribute(packed, 3));
    geo.setAttribute('aDomain', new THREE.Uint8BufferAttribute(tile.domain, 1));
    geo.setAttribute('aFlags', new THREE.Uint8BufferAttribute(tile.flags, 1));
    geo.setAttribute('aIndex', new THREE.Float32BufferAttribute(index, 1));

    // `position` holds quantised angles, not coordinates, so a computed bounding
    // sphere would be nonsense. Set it by hand: the cloud is a shell of `radius`.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), radius * 1.05);

    const palette = DOMAIN_PALETTE.map(([r, g, b]) => new THREE.Vector3(r, g, b));

    const shared = {
      uRadius: { value: radius },
      uSizeScale: { value: 32 },
      uPixelRatio: { value: Math.min(gl.getPixelRatio(), 2) },
      // Slightly past the true limb so points fade out rather than vanish.
      uCullBias: { value: -0.06 },
    };

    const display = new THREE.ShaderMaterial({
      uniforms: {
        ...shared,
        uPalette: { value: palette },
        uHoverIndex: { value: -1 },
        uDimLowSignal: { value: 0.32 },
        // Nodes above this normalised size get a containment ring.
        uHubThreshold: { value: 0.62 },
        uDomainFilter: { value: -1 },
        uSunDir: { value: SUN_DIR.clone() },
        uNightDim: { value: 0.42 },
      },
      vertexShader: POINTS_VERT,
      fragmentShader: POINTS_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      // Additive is what makes dense regions read as nebulae rather than as a
      // flat sheet of discs. It also means we never need to sort points.
      blending: THREE.AdditiveBlending,
    });

    const pick = new THREE.ShaderMaterial({
      uniforms: {
        // Separate uniform objects: sharing them would couple the two passes'
        // pixel ratio, and the pick target is always 1:1.
        uRadius: { value: radius },
        uSizeScale: { value: 32 },
        uPixelRatio: { value: Math.min(gl.getPixelRatio(), 2) },
        uCullBias: { value: -0.06 },
        // 2.5px, down from 5. Every extra pixel of hit padding is another pixel
        // of "I pointed there and it selected something else" — which is
        // exactly what made the first version feel like the dots and the repos
        // were in different places.
        uPickPadding: { value: 6.0 },
        uIdOffset: { value: idOffset },
        // Depth bias in NDC units. Small enough that it only resolves genuine
        // overlaps, large enough to always beat the sphere's own curvature.
        uSizeBias: { value: 0.004 },
      },
      vertexShader: PICK_VERT,
      fragmentShader: PICK_FRAG,
      transparent: false,
      depthWrite: true,
      depthTest: true,
      blending: THREE.NoBlending,
    });

    return { geometry: geo, displayMaterial: display, pickMaterial: pick };
  }, [band, radius, gl]);

  // R3F auto-disposes declarative objects, but these were created imperatively.
  useEffect(
    () => () => {
      geometry.dispose();
      displayMaterial.dispose();
      pickMaterial.dispose();
    },
    [geometry, displayMaterial, pickMaterial],
  );

  useEffect(() => {
    const p = pointsRef.current;
    if (!p) return;
    p.layers.enable(PICK_LAYER);
    // GPU picking replaces raycasting entirely. Without this, R3F would
    // raycast a million points on every pointer move.
    p.raycast = () => null;
    p.frustumCulled = false; // the globe is always at least partly on screen
    onReady?.({
      points: p,
      pickMaterial,
      displayMaterial,
      idOffset: band.idOffset,
      count: band.tile.count,
    });
    // Without this, demoting the device tier unmounts a band while the picking
    // loop still holds its handle — and then renders a disposed geometry.
    return () => onDispose?.(band.idOffset);
  }, [band, displayMaterial, pickMaterial, onReady, onDispose]);

  useFrame(() => {
    const { hoveredId, sizeScale, activeDomain } = useGlobeStore.getState();
    const local = hoveredId >= 0 ? hoveredId - band.idOffset : -1;
    const inThisBand = local >= 0 && local < band.tile.count;
    displayMaterial.uniforms.uHoverIndex.value = inThisBand ? local : -1;
    displayMaterial.uniforms.uSizeScale.value = sizeScale;
    // The pick pass must use the same size, or the hit area drifts away from
    // what the user can see and hovering starts missing.
    pickMaterial.uniforms.uSizeScale.value = sizeScale;
    displayMaterial.uniforms.uDomainFilter.value = activeDomain;
  });

  return <points ref={pointsRef} geometry={geometry} material={displayMaterial} />;
}
