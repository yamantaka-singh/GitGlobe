/**
 * Colour, sampled from the reference.
 *
 * The Populous globe is not a neon cyberpunk cliché — it is a photographic
 * Earth on pure black, with a periwinkle-to-violet atmosphere and amber city
 * lights on the night side. The restraint is the point: the planet is muted and
 * naturalistic, and the *markers* are the only pure-white, high-contrast things
 * on screen.
 *
 * That is exactly the hierarchy GitGlobe needs. The surface must stay quiet or
 * a hundred thousand data points become unreadable on top of it.
 *
 * All values are linear-ish 0..1 RGB triples for direct use in shaders.
 */

type RGB = readonly [number, number, number];

/** Terrain ramp. Deliberately desaturated — territories tint these, not replace them. */
export const PLANET_SURFACE = {
  /** Abyssal. Nearly the background, so oceans read as absence. */
  deepOcean: [0.024, 0.055, 0.118] as RGB,
  /** Continental shelf — where ocean meets land. */
  shelf: [0.055, 0.145, 0.262] as RGB,
  /** The first band of land. */
  coast: [0.118, 0.235, 0.290] as RGB,
  /** Plains and basins. Cool olive, as on the reference's African daylight. */
  lowland: [0.196, 0.235, 0.180] as RGB,
  /** Ranges and plateaus. The tan that dominates the lit half of the reference. */
  highland: [0.478, 0.416, 0.298] as RGB,
  /** Caps and cloud. Never pure white — that is reserved for the selected node. */
  ice: [0.784, 0.839, 0.886] as RGB,
} as const;

/**
 * Per-domain territory tints — muted cousins of the node palette.
 *
 * A domain's continent should be *recognisably* its colour without ever
 * competing with the nodes standing on it, so every entry here is roughly a
 * third of the saturation of its counterpart in DOMAIN_PALETTE.
 */
export const DOMAIN_TERRAIN_TINT: readonly RGB[] = [
  [0.106, 0.267, 0.353], // AI / ML — cold slate blue
  [0.353, 0.243, 0.145] as RGB, // Web frameworks — burnt sienna
  [0.145, 0.322, 0.259] as RGB, // Databases — deep jade
  [0.337, 0.302, 0.176] as RGB, // DevOps — ochre
  [0.235, 0.216, 0.361] as RGB, // Languages — indigo
  [0.196, 0.318, 0.365] as RGB, // Systems — steel
  [0.098, 0.263, 0.278] as RGB, // Data engineering — teal
  [0.361, 0.196, 0.196] as RGB, // Security — oxide red
  [0.294, 0.184, 0.341] as RGB, // Graphics — aubergine
  [0.169, 0.224, 0.322] as RGB, // Mobile — dusk blue
  [0.267, 0.322, 0.180] as RGB, // Scraping — moss
  [0.243, 0.259, 0.290] as RGB, // Scientific — basalt
];

/** Amber. Straight off the night side of the reference image. */
export const CITY_LIGHT: RGB = [1.0, 0.612, 0.278];

/**
 * The atmosphere in the reference runs periwinkle at the surface to violet at
 * the outer edge, not cyan. That shift is most of why it reads as photographic
 * rather than as a sci-fi glow.
 */
export const ATMOSPHERE = {
  edge: [0.780, 0.839, 1.0] as RGB, // near-white hairline at the silhouette
  rim: [0.431, 0.545, 0.910] as RGB, // periwinkle
  scatter: [0.478, 0.400, 0.788] as RGB, // violet, wide and faint
} as const;

/** Space. Not quite black — pure #000 makes the rim look like a sticker. */
export const SPACE: RGB = [0.008, 0.008, 0.016];
