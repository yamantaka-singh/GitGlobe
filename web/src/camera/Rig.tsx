import { useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { CameraControls } from '@react-three/drei';
import CameraControlsImpl from 'camera-controls';
import * as THREE from 'three';

import { useGlobeStore } from '../store/useGlobeStore';

/**
 * The one place that touches the camera, per web3d-scene-architect's Rule 5.
 * Everything else — HUD buttons, the benchmark, and in Phase 5 the agent —
 * goes through `globeCamera`, never through `camera.position` directly.
 */

// Module-scope scratch. Allocating Vector3s inside useFrame is the classic
// source of the sawtooth GC stutter web3d-performance-budget warns about.
const _centroid = new THREE.Vector3();
const _eye = new THREE.Vector3();
const _target = new THREE.Vector3();
const _p = new THREE.Vector3();

export interface FlyToOptions {
  /** Extra padding on the framing, as a multiple of globe radius. */
  padding?: number;
  /** Skip the transition (used by the benchmark to reset instantly). */
  instant?: boolean;
}

class GlobeCamera {
  controls: CameraControlsImpl | null = null;
  radius = 1;

  get ready() {
    return this.controls !== null;
  }

  /**
   * Scale a framing distance so the globe fits the *narrow* axis.
   *
   * three.js `fov` is the vertical field of view, so every distance below was
   * tuned against a landscape viewport where the horizontal fov is the wider
   * one. Rotate a phone to portrait and the horizontal fov becomes the narrow
   * axis — at 375x812 it collapses from 40° to about 19°, so the radius-1
   * sphere needs a distance of ~6 and was being drawn at 2.6. The result was a
   * globe cropped well past both screen edges with no way to zoom out to it.
   *
   * tan(hFov/2) = tan(vFov/2) * aspect, and distance scales as 1/tan(half),
   * so the correction is exactly 1/aspect. Applying it as a multiplier rather
   * than recomputing from scratch keeps the hand-tuned framing of each caller
   * — including its deliberate slight crop — just measured against the axis
   * that actually constrains the shot.
   */
  private fit(distance: number) {
    const cam = this.controls?.camera as THREE.PerspectiveCamera | undefined;
    if (!cam?.isPerspectiveCamera || !(cam.aspect > 0) || cam.aspect >= 1) return distance;
    return distance / cam.aspect;
  }

  /**
   * Frame a set of unit-sphere directions.
   *
   * The camera target is always derived from real point positions — never from
   * a caller-supplied coordinate. That contract is what makes ADR-006 (the
   * agent emits repo ids, not lat/lon) enforceable rather than aspirational.
   */
  async flyToDirections(dirs: readonly THREE.Vector3[], opts: FlyToOptions = {}) {
    if (!this.controls || dirs.length === 0) return;

    _centroid.set(0, 0, 0);
    for (const d of dirs) _centroid.add(d);
    if (_centroid.lengthSq() < 1e-9) _centroid.copy(dirs[0]); // antipodal set
    _centroid.normalize();

    let spread = 0.04;
    for (const d of dirs) spread = Math.max(spread, _centroid.angleTo(d));

    const altitude = this.fit(this.radius * (1.18 + spread * 1.7 + (opts.padding ?? 0)));
    _eye.copy(_centroid).multiplyScalar(altitude);
    _target.set(0, 0, 0);

    const store = useGlobeStore.getState();
    store.setCameraBusy(true);
    try {
      await this.controls.setLookAt(
        _eye.x,
        _eye.y,
        _eye.z,
        _target.x,
        _target.y,
        _target.z,
        !opts.instant && !store.reducedMotion,
      );
    } finally {
      useGlobeStore.getState().setCameraBusy(false);
    }
  }

  /** Convenience for a single spherical coordinate. */
  async flyToSpherical(theta: number, phi: number, opts?: FlyToOptions) {
    const st = Math.sin(theta);
    _p.set(st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi));
    await this.flyToDirections([_p.clone()], opts);
  }

  async reset(instant = false) {
    if (!this.controls) return;
    const d = this.fit(this.radius * 2.6);
    useGlobeStore.getState().setCameraBusy(true);
    try {
      await this.controls.setLookAt(0, d * 0.28, d, 0, 0, 0, !instant);
    } finally {
      useGlobeStore.getState().setCameraBusy(false);
    }
  }

  /**
   * The pre-entry framing: far out, slightly above, so `reset()` has somewhere
   * to travel from when the user presses Start. Instant — this is where the
   * globe already is, not a move the user should see.
   */
  async establish() {
    if (!this.controls) return;
    const d = this.fit(this.radius * 5.4);
    await this.controls.setLookAt(d * 0.34, d * 0.30, d, 0, 0, 0, false);
  }

  /** Used by the benchmark to drive a deterministic orbit. */
  setOrbitAngle(azimuth: number, polar: number, distance: number) {
    this.controls?.rotateTo(azimuth, polar, false);
    this.controls?.dollyTo(distance, false);
  }

  /**
   * Save and restore the framing around something that hijacks the camera.
   *
   * The benchmark drives a deterministic sweep and used to simply stop wherever
   * the sweep ended, dumping the user at a random angle and distance with no
   * way back to what they were looking at. Saving here rather than in the
   * benchmark keeps the rule that this class is the only thing that touches the
   * camera.
   */
  private saved: { eye: THREE.Vector3; target: THREE.Vector3 } | null = null;

  saveState() {
    if (!this.controls) return;
    const eye = new THREE.Vector3();
    const target = new THREE.Vector3();
    this.controls.getPosition(eye);
    this.controls.getTarget(target);
    this.saved = { eye, target };
  }

  async restoreState(smooth = true) {
    if (!this.controls || !this.saved) return;
    const { eye, target } = this.saved;
    this.saved = null;
    const store = useGlobeStore.getState();
    store.setCameraBusy(true);
    try {
      await this.controls.setLookAt(
        eye.x, eye.y, eye.z,
        target.x, target.y, target.z,
        smooth && !store.reducedMotion,
      );
    } finally {
      useGlobeStore.getState().setCameraBusy(false);
    }
  }
}

export const globeCamera = new GlobeCamera();

export function Rig({ radius }: { radius: number }) {
  const ref = useRef<CameraControlsImpl>(null);
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    globeCamera.radius = radius;
  }, [radius]);

  useEffect(() => {
    if (!ref.current) return;
    const c = ref.current;
    globeCamera.controls = c;

    c.minDistance = radius * 1.005; // allow zooming right down to the node clusters
    c.maxDistance = radius * 14; // allow zooming way out into deep space
    // A wheel notch delivers far more dolly per event than a pinch does, so one
    // speed cannot serve both: 0.9 that feels right on a trackpad makes pinch
    // feel like it barely responds. Coarse pointers only, so the desktop feel
    // is unchanged.
    const coarse = window.matchMedia?.('(pointer: coarse)').matches ?? false;
    c.dollySpeed = coarse ? 2.4 : 0.9;
    c.truckSpeed = 0;
    c.smoothTime = 0.32;
    c.draggingSmoothTime = 0.14;
    c.azimuthRotateSpeed = 0.55;
    c.polarRotateSpeed = 0.55;
    // The globe stays centred: panning it off-axis makes the sphere metaphor
    // fall apart and there is no way back without a reset.
    c.mouseButtons.left = CameraControlsImpl.ACTION.ROTATE;
    c.mouseButtons.right = CameraControlsImpl.ACTION.NONE;
    c.mouseButtons.middle = CameraControlsImpl.ACTION.DOLLY;
    c.mouseButtons.wheel = CameraControlsImpl.ACTION.DOLLY;
    c.touches.one = CameraControlsImpl.ACTION.TOUCH_ROTATE;
    c.touches.two = CameraControlsImpl.ACTION.TOUCH_DOLLY;
    c.touches.three = CameraControlsImpl.ACTION.NONE;

    // Start at the establishing framing, not the working one — the Intro
    // overlay is up, and Start needs somewhere to fly from.
    void globeCamera.establish();
    invalidate();
    return () => {
      globeCamera.controls = null;
    };
  }, [radius, invalidate]);

  useFrame((_, delta) => {
    const { autoRotate, cameraBusy, reducedMotion, hoveredId, selectedId } = useGlobeStore.getState();
    if (!autoRotate || cameraBusy || reducedMotion || hoveredId >= 0 || selectedId >= 0) return;

    // Decreased rotation speed as requested
    ref.current?.rotate(delta * 0.0035, 0, false);
  });

  return <CameraControls ref={ref} makeDefault />;
}
