import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';

import { Atmosphere, Core, Starfield } from './Backdrop';
import { PointCloud, type PointCloudHandle } from './PointCloud';
import { usePicking } from './usePicking';
import { Rig, globeCamera } from '../camera/Rig';
import { Benchmark } from '../bench/Benchmark';
import { useDeviceTier, TIER_BUDGET } from '../perf/useDeviceTier';
import { fetchBand, fetchManifest, type LoadedBand, type TileManifest } from '../tile/loader';
import { dequantisePhi, dequantiseTheta } from '../tile/format';
import { useGlobeStore } from '../store/useGlobeStore';

export const GLOBE_RADIUS = 1;

/** Resolved from a global picking id, for the HUD and for fly-to. */
export interface PointRef {
  id: number;
  repoId: number;
  band: number;
  domain: number;
  direction: THREE.Vector3;
}

/** Populated once tiles load; read by the HUD without going through React state. */
export const sceneIndex = {
  manifest: null as TileManifest | null,
  bands: [] as LoadedBand[],

  resolve(globalId: number): PointRef | null {
    if (globalId < 0) return null;
    for (const b of this.bands) {
      const local = globalId - b.idOffset;
      if (local < 0 || local >= b.tile.count) continue;
      const theta = dequantiseTheta(b.tile.thetaQ[local]);
      const phi = dequantisePhi(b.tile.phiQ[local]);
      const st = Math.sin(theta);
      return {
        id: globalId,
        repoId: b.tile.repoId[local],
        band: b.band,
        domain: b.tile.domain[local],
        direction: new THREE.Vector3(st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi)),
      };
    }
    return null;
  },
};

export function Scene() {
  const gl = useThree((s) => s.gl);
  const tier = useGlobeStore((s) => s.tier);
  const [bands, setBands] = useState<LoadedBand[]>([]);
  const [clouds, setClouds] = useState<PointCloudHandle[]>([]);
  const cloudMap = useRef(new Map<number, PointCloudHandle>());

  useDeviceTier();

  // Load bands up to whatever the detected tier can afford, awaiting each in
  // turn so band 0 paints immediately instead of waiting on the long tail.
  useEffect(() => {
    const ac = new AbortController();
    let cancelled = false;

    (async () => {
      try {
        const manifest = sceneIndex.manifest ?? (await fetchManifest(ac.signal));
        if (cancelled) return;
        sceneIndex.manifest = manifest;

        const maxBand = TIER_BUDGET(tier).maxBand;
        const loaded: LoadedBand[] = [];
        for (let b = 0; b <= maxBand; b++) {
          const existing = sceneIndex.bands.find((x) => x.band === b);
          const band = existing ?? (await fetchBand(manifest, b, ac.signal));
          if (cancelled) return;
          loaded.push(band);
          sceneIndex.bands = [...loaded];
          setBands([...loaded]);
          const total = loaded.reduce((sum, x) => sum + x.tile.count, 0);
          useGlobeStore.getState().setLoaded(loaded.map((x) => x.band), total);
        }
        useGlobeStore.getState().setLoadError(null);
      } catch (err) {
        if ((err as Error).name === 'AbortError' || cancelled) return;
        useGlobeStore.getState().setLoadError((err as Error).message);
      }
    })();

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [tier]);

  const handleReady = useCallback((handle: PointCloudHandle) => {
    cloudMap.current.set(handle.idOffset, handle);
    setClouds([...cloudMap.current.values()]);
  }, []);

  const handleDispose = useCallback((idOffset: number) => {
    cloudMap.current.delete(idOffset);
    setClouds([...cloudMap.current.values()]);
  }, []);

  usePicking(clouds, true);

  // Compile every program during load rather than mid-interaction. A shader
  // that first appears when the user zooms in freezes for 100-300ms.
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    if (clouds.length === 0) return;
    gl.compile(scene, camera);
  }, [clouds, gl, scene, camera]);

  // Click to select and fly. Guarded against drags: releasing an orbit gesture
  // over a point fires a `click`, and flying away every time the user finishes
  // rotating the globe is maddening.
  useEffect(() => {
    const el = gl.domElement;
    const down = { x: 0, y: 0, at: 0 };
    const DRAG_SLOP_PX = 5;
    const CLICK_MAX_MS = 400;

    const onPointerDown = (e: PointerEvent) => {
      down.x = e.clientX;
      down.y = e.clientY;
      down.at = performance.now();
    };
    const onPointerUp = (e: PointerEvent) => {
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      if (moved > DRAG_SLOP_PX || performance.now() - down.at > CLICK_MAX_MS) return;

      const id = useGlobeStore.getState().hoveredId;
      useGlobeStore.getState().setSelected(id);
      const ref = sceneIndex.resolve(id);
      if (ref) void globeCamera.flyToDirections([ref.direction], { padding: 0.05 });
    };

    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointerup', onPointerUp);
    return () => {
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointerup', onPointerUp);
    };
  }, [gl]);

  const dprCap = useMemo(() => TIER_BUDGET(tier).dprCap, [tier]);
  useEffect(() => {
    gl.setPixelRatio(Math.min(window.devicePixelRatio, dprCap));
  }, [gl, dprCap]);

  return (
    <>
      {/* STATIC — never changes */}
      <Starfield />
      <Core radius={GLOBE_RADIUS} />
      <Atmosphere radius={GLOBE_RADIUS} />

      {/* DRIVEN — the single owner of the camera */}
      <Rig radius={GLOBE_RADIUS} />

      {/* INTERACTIVE — one draw call per LOD band */}
      {bands.map((band) => (
        <PointCloud
          key={band.band}
          band={band}
          radius={GLOBE_RADIUS}
          onReady={handleReady}
          onDispose={handleDispose}
        />
      ))}

      <Benchmark />
    </>
  );
}
