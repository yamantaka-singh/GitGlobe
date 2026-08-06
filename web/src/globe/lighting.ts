import * as THREE from 'three';

import { SUN_VEC } from './sun';

/**
 * The sun direction as a three.js vector. The numbers live in `sun.ts`, which
 * has no dependencies, so `scripts/preview-planet.ts` can light its Node-side
 * render with the identical value.
 */
export const SUN_DIR = new THREE.Vector3(...SUN_VEC).normalize();
