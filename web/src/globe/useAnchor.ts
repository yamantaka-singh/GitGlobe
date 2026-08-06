import { useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import { useGlobeStore } from '../store/useGlobeStore';
import { sceneIndex } from './Scene';

// Module-scope scratch — allocating a Vector3 per frame is the sawtooth-GC
// pattern web3d-performance-budget warns about.
const _world = new THREE.Vector3();

/**
 * Projects the hovered node into screen space every frame so the HUD can draw
 * a reticle exactly on it.
 *
 * This is the piece that makes a dot read as a repository. The bottom-of-screen
 * readout in the first version was correct and useless — it never told you
 * *which* dot it was describing. A bracket locked onto the point, with a leader
 * line to the card, makes the link unambiguous.
 */
export function useAnchor(radius: number) {
  const { camera, size } = useThree();
  const setAnchor = useGlobeStore((s) => s.setAnchor);
  const hidden = useMemo(() => ({ x: 0, y: 0, visible: false }), []);

  useFrame(() => {
    const id = useGlobeStore.getState().hoveredId;
    const ref = id >= 0 ? sceneIndex.resolve(id) : null;
    if (!ref) {
      setAnchor(hidden);
      return;
    }

    _world.copy(ref.direction).multiplyScalar(radius);

    // Behind the globe: the point is occluded by the opaque core, so a reticle
    // there would float over solid geometry pointing at nothing.
    const toCamera = camera.position.clone().sub(_world).normalize();
    if (ref.direction.dot(toCamera) < -0.02) {
      setAnchor(hidden);
      return;
    }

    _world.project(camera);
    if (_world.z > 1) {
      setAnchor(hidden);
      return;
    }

    setAnchor({
      x: (_world.x * 0.5 + 0.5) * size.width,
      y: (-_world.y * 0.5 + 0.5) * size.height,
      visible: true,
    });
  });
}
