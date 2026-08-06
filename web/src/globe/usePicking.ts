import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import type { PointCloudHandle } from './PointCloud';
import { PICK_LAYER } from './layers';
import { useGlobeStore } from '../store/useGlobeStore';

const PICK_INTERVAL_MS = 33; // ~30Hz, per the architecture's picking budget

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
      new THREE.WebGLRenderTarget(1, 1, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
        stencilBuffer: false,
      }),
    [],
  );
  const readback = useMemo(() => new Uint8Array(4), []);
  const savedClear = useMemo(() => new THREE.Color(), []);

  useEffect(() => () => target.dispose(), [target]);

  const onPointerMove = useCallback((e: PointerEvent) => {
    pointer.current.x = e.clientX;
    pointer.current.y = e.clientY;
    pointer.current.dirty = true;
  }, []);

  const onPointerLeave = useCallback(() => {
    pointer.current.dirty = false;
    pointer.current.x = -1;
    useGlobeStore.getState().setHovered(-1);
  }, []);

  useEffect(() => {
    const el = gl.domElement;
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerleave', onPointerLeave);
    return () => {
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerleave', onPointerLeave);
    };
  }, [gl, onPointerMove, onPointerLeave]);

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

    cam.setViewOffset(Math.round(rect.width), Math.round(rect.height), x, y, 1, 1);
    cam.layers.set(PICK_LAYER);

    gl.getClearColor(savedClear);
    const prevClearAlpha = gl.getClearAlpha();

    gl.setRenderTarget(target);
    gl.setClearColor(0x000000, 1);
    gl.clear(true, true, false);
    gl.render(scene, cam);
    gl.readRenderTargetPixels(target, 0, 0, 1, 1, readback);

    gl.setRenderTarget(prevTarget);
    gl.setClearColor(savedClear, prevClearAlpha);
    cam.clearViewOffset();
    cam.layers.mask = prevLayerMask;
    for (const c of clouds) c.points.material = c.displayMaterial;

    const id = readback[0] + readback[1] * 256 + readback[2] * 65536;
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
