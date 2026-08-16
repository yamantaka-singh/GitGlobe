import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import type { PointCloudHandle } from './PointCloud';
import { PICK_LAYER } from './layers';
import { useGlobeStore } from '../store/useGlobeStore';

const PICK_INTERVAL_MS = 33; // ~30Hz, per the architecture's picking budget

/**
 * Half-width of the pick window, in CSS pixels.
 *
 * This used to read a single pixel. A mouse cursor is a single pixel, so that
 * was fine; a fingertip is about 44, and a node is a couple of pixels wide, so
 * tapping one was largely luck — the single most common complaint about the
 * globe on a phone. Sampling a square around the touch and taking the nearest
 * hit gives the finger a tolerance without moving anything on screen.
 *
 * Coarse pointers get a much larger window than a mouse, where a wide radius
 * would make hover feel magnetic and the reticle jumpy.
 */
const PICK_RADIUS = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches ? 7 : 2;
const PICK_SIZE = PICK_RADIUS * 2 + 1;

/**
 * GPU picking.
 *
 * Raycasting a million points is not an option, so instead we render the point
 * clouds' ids as colours into a 1x1 render target positioned at the cursor via
 * `camera.setViewOffset`, and read back a single pixel. Constant cost, no
 * matter how many points are on screen.
 *
 * Three details that matter:
 *  - The camera is switched to PICK_LAYER so only points are drawn.
 *  - `setViewOffset` takes CSS pixels, matching pointer event coordinates.
 *  - The readback is synchronous and stalls the pipeline, so it is throttled
 *    and skipped entirely while the camera is moving.
 */
export function usePicking(clouds: readonly PointCloudHandle[], enabled: boolean) {
  const { gl, camera, scene, size } = useThree();
  const setHovered = useGlobeStore((s) => s.setHovered);

  const pointer = useRef({ x: -1, y: -1, dirty: false });
  const lastPick = useRef(0);

  const target = useMemo(
    () =>
      new THREE.WebGLRenderTarget(PICK_SIZE, PICK_SIZE, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
        stencilBuffer: false,
      }),
    [],
  );
  const readback = useMemo(() => new Uint8Array(PICK_SIZE * PICK_SIZE * 4), []);
  // Offsets into the pick window ordered by distance from its centre, so the
  // first hit found is the node nearest the finger. Built once.
  const scanOrder = useMemo(() => {
    const out: number[] = [];
    for (let y = 0; y < PICK_SIZE; y++) {
      for (let x = 0; x < PICK_SIZE; x++) out.push(y * PICK_SIZE + x);
    }
    const c = PICK_RADIUS;
    return out.sort((a, b) => {
      const ax = (a % PICK_SIZE) - c, ay = Math.floor(a / PICK_SIZE) - c;
      const bx = (b % PICK_SIZE) - c, by = Math.floor(b / PICK_SIZE) - c;
      return ax * ax + ay * ay - (bx * bx + by * by);
    });
  }, []);
  const savedClear = useMemo(() => new THREE.Color(), []);

  useEffect(() => () => target.dispose(), [target]);

  const onPointerMove = useCallback((e: PointerEvent) => {
    pointer.current.x = e.clientX;
    pointer.current.y = e.clientY;
    pointer.current.dirty = true;
  }, []);

  /**
   * A finger never hovers.
   *
   * Selection reads `hoveredId`, which only this hook sets, and only from
   * `pointermove`. A touch goes down and up with no move in between, so on a
   * phone `hoveredId` was whatever a stray earlier event left behind — usually
   * -1, which the click handler reads as "empty space" and clears the
   * selection. Tapping a node did nothing.
   *
   * Picking on `pointerdown` gives the pick pass the ~50-150ms of a normal tap
   * to run before `pointerup` reads the result. Resetting the throttle matters:
   * without it the 30Hz gate can swallow the one pick that the tap depends on.
   */
  const onPointerDown = useCallback((e: PointerEvent) => {
    pointer.current.x = e.clientX;
    pointer.current.y = e.clientY;
    pointer.current.dirty = true;
    lastPick.current = 0;
  }, []);

  const onPointerLeave = useCallback(() => {
    pointer.current.dirty = false;
    pointer.current.x = -1;
    useGlobeStore.getState().setHovered(-1);
  }, []);

  useEffect(() => {
    const el = gl.domElement;
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointerleave', onPointerLeave);
    return () => {
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointerleave', onPointerLeave);
    };
  }, [gl, onPointerMove, onPointerDown, onPointerLeave]);

  useFrame(() => {
    if (!enabled || clouds.length === 0) return;
    if (pointer.current.x < 0 || pointer.current.y < 0) return;
    // Picking during a fly-to is wasted work and adds a readback stall to the
    // one moment the frame budget is tightest.
    if (useGlobeStore.getState().cameraBusy) return;

    const now = performance.now();
    if (now - lastPick.current < PICK_INTERVAL_MS) return;
    lastPick.current = now;
    pointer.current.dirty = false;

    const rect = gl.domElement.getBoundingClientRect();
    const x = Math.floor(pointer.current.x - rect.left);
    const y = Math.floor(pointer.current.y - rect.top);
    if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return;

    const cam = camera as THREE.PerspectiveCamera;
    const prevLayerMask = cam.layers.mask;
    const prevTarget = gl.getRenderTarget();

    // Swap in the pick materials. Storing and restoring rather than keeping two
    // scenes, because an Object3D can only have one parent.
    for (const c of clouds) c.points.material = c.pickMaterial;

    // Sample a square centred on the pointer rather than the single pixel under
    // it. Clamped so the window never starts off-canvas near an edge.
    const ox = Math.max(0, Math.min(Math.round(rect.width) - PICK_SIZE, x - PICK_RADIUS));
    const oy = Math.max(0, Math.min(Math.round(rect.height) - PICK_SIZE, y - PICK_RADIUS));
    cam.setViewOffset(Math.round(rect.width), Math.round(rect.height), ox, oy, PICK_SIZE, PICK_SIZE);
    cam.layers.set(PICK_LAYER);

    gl.getClearColor(savedClear);
    const prevClearAlpha = gl.getClearAlpha();

    gl.setRenderTarget(target);
    gl.setClearColor(0x000000, 1);
    gl.clear(true, true, false);
    gl.render(scene, cam);
    gl.readRenderTargetPixels(target, 0, 0, PICK_SIZE, PICK_SIZE, readback);

    gl.setRenderTarget(prevTarget);
    gl.setClearColor(savedClear, prevClearAlpha);
    cam.clearViewOffset();
    cam.layers.mask = prevLayerMask;
    for (const c of clouds) c.points.material = c.displayMaterial;

    // Nearest hit to the centre of the window wins.
    let id = 0;
    for (const px of scanOrder) {
      const o = px * 4;
      const v = readback[o] + readback[o + 1] * 256 + readback[o + 2] * 65536;
      if (v !== 0) { id = v; break; }
    }
    setHovered(id === 0 ? -1 : id - 1);
  });

  // Keep the pick pass's pixel ratio pinned to 1 — the target is 1x1 regardless
  // of display DPR, so point sizes must be computed in the same space.
  useEffect(() => {
    for (const c of clouds) {
      c.pickMaterial.uniforms.uPixelRatio.value = 1;
    }
  }, [clouds, size]);
}
