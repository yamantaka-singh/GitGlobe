import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';

import { Atmosphere, Planet, Starfield } from './Backdrop';
import { usePlanetTexture } from './usePlanetTexture';
import { PointCloud, type PointCloudHandle } from './PointCloud';
import { ArcLayer, ArcPool, AMBIENT_STYLE, FOCUS_STYLE, type ArcEndpoints } from './ArcLayer';
import { usePicking } from './usePicking';
import { useAnchor } from './useAnchor';
import { Rig, globeCamera } from '../camera/Rig';
import { Benchmark } from '../bench/Benchmark';
import { useDeviceTier, TIER_BUDGET } from '../perf/useDeviceTier';
import { fetchBand, fetchManifest, type LoadedBand, type TileManifest } from '../tile/loader';
import { fetchGraph, sortedRanks } from '../graph/loader';
import { neighboursOf, type RepoGraph } from '../graph/format';
import { dequantisePhi, dequantiseTheta } from '../tile/format';
import { useGlobeStore } from '../store/useGlobeStore';

export const GLOBE_RADIUS = 1;
const FOCUS_ARC_CAPACITY = 256;

export interface PointRef {
  id: number;
  repoId: number;
  band: number;
  domain: number;
  direction: THREE.Vector3;
}

/**
 * Populated once tiles and graph load. Read directly by the HUD and reticle
 * rather than mirrored into React state — this data is large, immutable, and
 * read at 60Hz, none of which suits a store.
 */
export const sceneIndex = {
  manifest: null as TileManifest | null,
  bands: [] as LoadedBand[],
  graph: null as RepoGraph | null,
  sortedRanks: null as Float32Array | null,

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

  /** Unit direction for a node id, written into `out` to avoid allocating. */
  directionInto(globalId: number, out: THREE.Vector3): boolean {
    for (const b of this.bands) {
      const local = globalId - b.idOffset;
      if (local < 0 || local >= b.tile.count) continue;
      const theta = dequantiseTheta(b.tile.thetaQ[local]);
      const phi = dequantisePhi(b.tile.phiQ[local]);
      const st = Math.sin(theta);
      out.set(st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi));
      return true;
    }
    return false;
  },
};

export function Scene() {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const tier = useGlobeStore((s) => s.tier);
  const showAmbientArcs = useGlobeStore((s) => s.showAmbientArcs);

  const [bands, setBands] = useState<LoadedBand[]>([]);
  const [clouds, setClouds] = useState<PointCloudHandle[]>([]);
  const [graphVersion, setGraphVersion] = useState(0);
  const cloudMap = useRef(new Map<number, PointCloudHandle>());

  useDeviceTier();
  useAnchor(GLOBE_RADIUS);

  // Terrain is baked from the cluster centres, so it can only be built once the
  // manifest has arrived. Continents grow where the repositories actually are.
  const [manifestState, setManifestState] = useState<TileManifest | null>(null);
  const surface = usePlanetTexture(manifestState, tier);

  const ambientPool = useMemo(
    () => new ArcPool(TIER_BUDGET(tier).ambientArcs, GLOBE_RADIUS, AMBIENT_STYLE),
    [tier],
  );
  const focusPool = useMemo(() => new ArcPool(FOCUS_ARC_CAPACITY, GLOBE_RADIUS, FOCUS_STYLE), []);

  useEffect(() => () => ambientPool.dispose(), [ambientPool]);
  useEffect(() => () => focusPool.dispose(), [focusPool]);

  // ---- load tiles then graph ------------------------------------------------
  useEffect(() => {
    const ac = new AbortController();
    let cancelled = false;

    (async () => {
      try {
        const manifest = sceneIndex.manifest ?? (await fetchManifest(ac.signal));
        if (cancelled) return;
        sceneIndex.manifest = manifest;
        setManifestState(manifest);

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

        // Graph last: it is the largest single file and the globe is already
        // interactive without it.
        if (!sceneIndex.graph && manifest.graph) {
          const graph = await fetchGraph(manifest.graph.file, manifest.layoutVersion, ac.signal);
          if (cancelled) return;
          sceneIndex.graph = graph;
          sceneIndex.sortedRanks = sortedRanks(graph);
          setGraphVersion((v) => v + 1);
          useGlobeStore.getState().setGraphReady(true, graph.ambientCount);
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

  // ---- ambient backbone -----------------------------------------------------
  useEffect(() => {
    const graph = sceneIndex.graph;
    if (!graph || bands.length === 0) return;

    if (!showAmbientArcs) {
      ambientPool.clear();
      return;
    }

    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const arcs: ArcEndpoints[] = [];
    const limit = Math.min(graph.ambientCount, ambientPool.capacity);

    for (let i = 0; i < limit; i++) {
      const nodeA = graph.ambient[i * 2];
      const nodeB = graph.ambient[i * 2 + 1];
      // Endpoints can fall outside the loaded LOD bands on a low tier. Skipping
      // is correct — an arc to a node that isn't drawn points at nothing.
      if (!sceneIndex.directionInto(nodeA, a) || !sceneIndex.directionInto(nodeB, b)) continue;
      arcs.push({
        a: a.clone(),
        b: b.clone(),
        weight: 1 - i / limit,
        nodeA,
        nodeB,
      });
    }

    ambientPool.setArcs(arcs);
    useGlobeStore.getState().setGraphReady(true, arcs.length);
  }, [graphVersion, bands, showAmbientArcs, ambientPool]);

  // ---- focus arcs on hover --------------------------------------------------
  useEffect(() => {
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();

    const unsubscribe = useGlobeStore.subscribe((state, prev) => {
      if (state.hoveredId === prev.hoveredId && state.selectedId === prev.selectedId) return;

      // Selection pins the web; hover previews it. Pinning matters because the
      // arcs vanish the moment you move the cursor toward them otherwise.
      const focus = state.selectedId >= 0 ? state.selectedId : state.hoveredId;
      ambientPool.setFocus(focus);

      const graph = sceneIndex.graph;
      if (!graph || focus < 0 || !sceneIndex.directionInto(focus, a)) {
        focusPool.clear();
        useGlobeStore.getState().setFocusArcCount(0);
        return;
      }

      const neighbours = neighboursOf(graph, focus, FOCUS_ARC_CAPACITY);
      const arcs: ArcEndpoints[] = [];
      const origin = a.clone();
      for (const n of neighbours) {
        if (!sceneIndex.directionInto(n.node, b)) continue;
        arcs.push({
          a: origin,
          b: b.clone(),
          weight: 0.35 + 0.65 * (n.weight / 32767),
          nodeA: focus,
          nodeB: n.node,
        });
      }
      focusPool.setArcs(arcs);
      useGlobeStore.getState().setFocusArcCount(arcs.length);
    });

    return unsubscribe;
  }, [ambientPool, focusPool]);

  // ---- point clouds ---------------------------------------------------------
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
  // that first appears when the user hovers freezes for 100-300ms.
  useEffect(() => {
    if (clouds.length === 0) return;
    gl.compile(scene, camera);
  }, [clouds, graphVersion, surface, gl, scene, camera]);

  // ---- click to select ------------------------------------------------------
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

      const store = useGlobeStore.getState();
      const id = store.hoveredId;
      // Clicking empty space clears the pin rather than doing nothing.
      if (id < 0) {
        store.setSelected(-1);
        return;
      }
      store.setSelected(id === store.selectedId ? -1 : id);
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
      <Planet radius={GLOBE_RADIUS} surface={surface} />
      <Atmosphere radius={GLOBE_RADIUS} />

      {/* DRIVEN — the single owner of the camera */}
      <Rig radius={GLOBE_RADIUS} />

      {/* INTERACTIVE — one draw call per LOD band, one per arc pool */}
      {bands.map((band) => (
        <PointCloud
          key={band.band}
          band={band}
          radius={GLOBE_RADIUS}
          onReady={handleReady}
          onDispose={handleDispose}
        />
      ))}
      <ArcLayer pool={ambientPool} radius={GLOBE_RADIUS} />
      <ArcLayer pool={focusPool} radius={GLOBE_RADIUS} />

      <Benchmark />
    </>
  );
}
