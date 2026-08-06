/**
 * Superseded by gen-world.ts.
 *
 * Tiles can no longer be generated on their own: PageRank decides node size,
 * brightness and LOD band, so the dependency graph has to exist first. The two
 * steps are one pipeline now.
 *
 * This file is safe to delete.
 */
console.error(
  'gen-tiles.ts has been replaced by gen-world.ts — run `npm run gen:world` instead.\n' +
    'Tiles now depend on PageRank, so the graph and the tiles are generated together.',
);
process.exit(1);
