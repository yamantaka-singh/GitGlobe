/**
 * Layer 1 holds only the pickable point clouds. During the pick pass the camera
 * is switched to this layer alone, so the starfield and atmosphere can't write
 * colours that would decode as bogus repo ids.
 */
export const PICK_LAYER = 1;
