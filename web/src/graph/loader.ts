import { decodeGraph, type RepoGraph } from './format';

const TILE_ROOT = `${import.meta.env.BASE_URL}tiles/`;

export async function fetchGraph(file: string, layoutVersion: number, signal?: AbortSignal): Promise<RepoGraph> {
  const res = await fetch(`${TILE_ROOT}${file}`, { signal });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${file}: HTTP ${res.status}. Run \`npm run gen:world\`.`);
  }
  const graph = decodeGraph(await res.arrayBuffer());
  if (graph.layoutVersion !== layoutVersion) {
    throw new Error(
      `${file} is layout v${graph.layoutVersion} but the manifest is v${layoutVersion}. Regenerate the world.`,
    );
  }
  return graph;
}

/**
 * Ranks sorted descending, for percentile lookups in the HUD.
 *
 * Node ids are already assigned in rank order by the generator, so this is a
 * copy rather than a sort — but sorting defensively costs nothing at load time
 * and means the HUD stays correct if that invariant ever changes.
 */
export function sortedRanks(graph: RepoGraph): Float32Array {
  const copy = Float32Array.from(graph.rank);
  copy.sort();
  return copy.reverse();
}
