/**
 * Colour — an ice giant.
 *
 * Neptune's look comes from three things, and getting any one of them wrong
 * makes it read as "a blue ball":
 *
 *  1. **Depth, not hue.** It is not one blue. It is a deep ultramarine base
 *     with paler zones banded across it, and the contrast between them is low —
 *     around 25% — which is why photographs look soft rather than striped.
 *  2. **Methane cirrus.** Bright white streaks sitting *above* the bands,
 *     catching light at a different angle. They are the detail that makes it
 *     look like weather instead of paint.
 *  3. **Dark storms.** A handful of near-black ovals, elongated in longitude,
 *     with bright companion clouds at their edges.
 *
 * The data layer sits on top of all of this, so everything here stays under
 * roughly 0.55 luminance. A hundred thousand nodes have to remain the brightest
 * things on screen.
 */

type RGB = readonly [number, number, number];

export const PLANET_SURFACE = {
  /** Abyssal ultramarine — the deepest visible layer, between the bands. */
  deep: [0.031, 0.078, 0.243] as RGB,
  /** The dominant body colour. */
  mid: [0.106, 0.224, 0.545] as RGB,
  /** Pale zones — upper-atmosphere haze catching more light. */
  light: [0.259, 0.451, 0.804] as RGB,
  /** Highest zones, almost cyan. Used sparingly or the planet turns turquoise. */
  pale: [0.478, 0.702, 0.933] as RGB,
  /** Methane cirrus. Never pure white — that belongs to the selected node. */
  cirrus: [0.847, 0.925, 1.0] as RGB,
  /** Storm cores, darker than the deepest band. */
  storm: [0.016, 0.035, 0.125] as RGB,
} as const;

/**
 * Per-domain tints for the great cloud systems.
 *
 * A territory on a gas giant is a persistent weather system, not a continent.
 * Each is a subtle shift in the band colour — every entry stays inside the
 * blue-violet family so the planet reads as one body rather than a paint chart.
 */
export const DOMAIN_TERRAIN_TINT: readonly RGB[] = [
  [0.180, 0.404, 0.780] as RGB, // AI / ML — cerulean
  [0.400, 0.396, 0.741] as RGB, // Web frameworks — periwinkle
  [0.153, 0.463, 0.639] as RGB, // Databases — teal blue
  [0.451, 0.427, 0.667] as RGB, // DevOps — dusty violet
  [0.322, 0.318, 0.741] as RGB, // Languages — indigo
  [0.235, 0.514, 0.784] as RGB, // Systems — sky
  [0.114, 0.396, 0.596] as RGB, // Data engineering — deep teal
  [0.494, 0.373, 0.643] as RGB, // Security — mauve
  [0.435, 0.353, 0.784] as RGB, // Graphics — violet
  [0.208, 0.373, 0.729] as RGB, // Mobile — cobalt
  [0.286, 0.510, 0.702] as RGB, // Scraping — steel cyan
  [0.337, 0.408, 0.616] as RGB, // Scientific — slate blue
];

/**
 * Night-side glow. On a gas giant there are no cities — this is aurora and
 * storm luminance, which is both physically reasonable and the thing that keeps
 * the dark limb from becoming a dead black crescent.
 */
export const CITY_LIGHT: RGB = [0.451, 0.855, 1.0];

/**
 * Neptune's atmosphere is thin and high, so the rim is a tight bright line
 * rather than the wide bloom Earth gets. The violet outer scatter is what sells
 * an ice giant specifically.
 */
export const ATMOSPHERE = {
  edge: [0.729, 0.882, 1.0] as RGB,
  rim: [0.302, 0.549, 0.973] as RGB,
  scatter: [0.400, 0.353, 0.847] as RGB,
} as const;

/** Deep space with a trace of blue, so the rim has somewhere to fall off to. */
export const SPACE: RGB = [0.004, 0.006, 0.016];

/** Nebula clouds in the background — dim, large, and slow. */
export const NEBULA = {
  warm: [0.361, 0.153, 0.318] as RGB, // dusty magenta
  cool: [0.098, 0.180, 0.400] as RGB, // cold blue
  core: [0.290, 0.318, 0.545] as RGB, // where they overlap
} as const;
