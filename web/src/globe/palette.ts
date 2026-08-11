/**
 * Colour — an emission nebula.
 *
 * The body was an ice giant: latitudinal banding, methane cirrus, dark storm
 * ovals, lit by a sun. A nebula is the opposite object in every respect that
 * matters to a renderer, and the palette has to change with it:
 *
 *  1. **It emits, it does not reflect.** There is no albedo and no terminator.
 *     Brightness tracks density and ionisation, not the angle to a light.
 *  2. **Hue encodes excitation.** Real emission nebulae are not one colour:
 *     ionised hydrogen glows crimson-magenta, doubly-ionised oxygen glows
 *     teal. That two-hue tension is the thing that reads as "nebula" — a
 *     single-hue cloud reads as smoke.
 *  3. **Dust is foreground, not absence.** The dark rifts are opaque dust
 *     lanes in front of the glow, so they are near-black and hard-edged, not
 *     a fade to the background.
 *
 * The one rule carried over unchanged: everything here stays under roughly
 * 0.55 luminance. A hundred thousand nodes have to remain the brightest things
 * on screen, and a nebula is far easier to accidentally blow out than a planet
 * because it is emissive everywhere rather than only on the lit half.
 */

type RGB = readonly [number, number, number];

export const PLANET_SURFACE = {
  /** The void between filaments. Almost pure black for an OLED techy look. */
  deep: [0.008, 0.008, 0.016] as RGB,
  /** Base gas: Very dark indigo/blue to keep the background clean. */
  mid: [0.024, 0.031, 0.071] as RGB,
  /** Mid-density emission — Dimmed techy cyan so dots pop. */
  light: [0.034, 0.245, 0.310] as RGB,
  /** Hot cores — Faint vivid cyan. */
  pale: [0.090, 0.376, 0.392] as RGB,
  /** Ionisation fronts — Dimmed neon magenta/pink for a subtle sharp contrast. */
  cirrus: [0.364, 0.064, 0.256] as RGB,
  /** Dust lanes. Opaque and near black to cut sharply through the neon. */
  storm: [0.004, 0.004, 0.008] as RGB,
} as const;

/**
 * Per-domain tints for the great gas concentrations.
 *
 * These are the SAME TWELVE HUES as `DOMAIN_PALETTE` in `shaders.ts`, pulled
 * down in luminance and mixed toward the nebula's base indigo. That pairing is
 * the point: a domain's nodes and the medium they sit in now share a hue, so
 * the territory reads as those repositories' own region rather than as an
 * unrelated stripe of colour underneath them. Previously the two palettes were
 * chosen independently — nodes warm, terrain uniformly blue-violet — and a
 * territory's colour therefore told you nothing about the nodes on it.
 *
 * The old set stayed inside one hue family so a *planet* would read as one
 * body. A nebula has no body to hold together; it is a medium, and a medium can
 * carry hue variation without falling apart. So the constraint that forced
 * every domain toward blue is gone, and twelve distinguishable hues fit.
 *
 * Luminance lands between 0.28 and 0.41 — well under the 0.55 ceiling, so the
 * nodes still win.
 */
export const DOMAIN_TERRAIN_TINT: readonly RGB[] = [
  [0.421, 0.224, 0.374] as RGB, // AI / ML — deep rose
  [0.465, 0.340, 0.402] as RGB, // Web frontend — deep salmon
  [0.411, 0.251, 0.216] as RGB, // Data and storage — deep amber
  [0.465, 0.391, 0.303] as RGB, // Infrastructure — deep gold
  [0.296, 0.307, 0.221] as RGB, // Languages and compilers — deep olive
  [0.311, 0.439, 0.411] as RGB, // Systems and embedded — deep mint
  [0.065, 0.331, 0.385] as RGB, // Data engineering — deep teal
  [0.196, 0.437, 0.531] as RGB, // Security — deep cyan
  [0.151, 0.303, 0.501] as RGB, // Graphics and games — deep azure
  [0.367, 0.390, 0.531] as RGB, // Mobile — deep periwinkle
  [0.335, 0.253, 0.495] as RGB, // Automation and tooling — deep violet
  [0.465, 0.343, 0.531] as RGB, // Science and numerics — deep orchid
];

/**
 * Night-side glow. On a gas giant there are no cities — this is aurora and
 * storm luminance, which is both physically reasonable and the thing that keeps
 * the dark limb from becoming a dead black crescent.
 */
export const CITY_LIGHT: RGB = [0.451, 0.855, 1.0];

/**
 * Atmosphere Outline
 * A solid black border/silhouette.
 */
export const ATMOSPHERE = {
  edge: [0.0, 0.0, 0.0] as RGB, // Pure black
  rim: [0.0, 0.0, 0.0] as RGB, // Pure black
  scatter: [0.0, 0.0, 0.0] as RGB,
} as const;

/**
 * Arc colour by relationship type — the semantics of an edge, not decoration.
 *
 * Every arc was previously one colour, which meant the wires carried exactly
 * one bit: connected or not. The database has distinguished three kinds since
 * migration 002 and the renderer simply never saw them.
 *
 * The assignment is not arbitrary:
 *
 *  - **depends_on** is amber. It is the only DIRECTED relationship of the
 *    three — A needs B — and the travelling pulse already reads as flow along
 *    the wire. Warm reads as active, and amber is the one hue with no
 *    counterpart in the nebula behind it, so hard structural facts never
 *    camouflage against the medium.
 *  - **similar_to** is cyan. Semantic kinship is symmetric and passive, and
 *    cool recedes. It is also the most numerous kind by a wide margin, so it
 *    has to be the quietest or the globe turns into a ball of string.
 *  - **used_with** is violet. Co-occurrence sits between the two: evidence of
 *    a real relationship, but observed rather than declared.
 *
 * All three are brighter than the 0.55 surface ceiling because an arc has to
 * be legible crossing a lit region of the medium.
 */
export const ARC_KIND_COLOR: readonly RGB[] = [
  [1.0, 0.671, 0.259] as RGB, // 0 depends_on (outdegree) — amber
  [0.302, 0.788, 0.847] as RGB, // 1 similar_to — cyan
  [0.706, 0.510, 0.973] as RGB, // 2 used_with — violet
  [0.9, 0.2, 0.5] as RGB,       // 3 dependent_on (indegree) — magenta
];

/** Deep space: Pure black void so the starry globe stands out. */
export const SPACE: RGB = [0.0, 0.0, 0.0];

/** Nebula clouds in the background — removed for Alien X vibe. */
export const NEBULA = {
  warm: [0.0, 0.0, 0.0] as RGB, 
  cool: [0.0, 0.0, 0.0] as RGB, 
  core: [0.0, 0.0, 0.0] as RGB, 
} as const;
