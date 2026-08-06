import * as THREE from 'three';

/**
 * The sun direction, in world space.
 *
 * Fixed rather than camera-relative, deliberately: a light that follows the
 * camera produces a sphere that looks identical from every angle, which is
 * exactly the flat-disc problem the terminator exists to solve. Fixed in world
 * space, orbiting moves the terminator across the globe, and the rotation
 * finally reads as travelling around a body.
 *
 * Lives in its own module because the point cloud, the core, and all three
 * atmosphere shells need it — and a shared constant should not force three
 * files to import a component module.
 */
export const SUN_DIR = new THREE.Vector3(-0.55, 0.42, 0.72).normalize();
