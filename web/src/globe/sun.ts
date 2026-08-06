/**
 * The sun direction, as a plain array.
 *
 * Deliberately dependency-free so the Node-side planet preview can import the
 * exact same value. A preview lit differently from the app is worse than no
 * preview, because it looks like verification while verifying nothing.
 *
 * Chosen so the terminator crosses the *visible* face at the default camera
 * rather than hiding round the back. Almost all of the reference image's
 * character comes from seeing lit terrain and a glowing night side at once; a
 * sun behind the viewer produces a flatly-lit ball and throws that away.
 *
 * Fixed in world space, not camera-relative: a light that follows the camera
 * looks identical from every angle, which defeats the point of having one.
 */
export const SUN_VEC: readonly [number, number, number] = [-0.78, 0.30, 0.34];
