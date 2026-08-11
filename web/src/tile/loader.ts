import { registerNames } from '../repo/names';
import { registerScores } from '../repo/scores';
import { decodeTile, type Tile } from './format';

export interface GraphManifest {
  file: string;
  bytes: number;
  directedEdges: number;
  csrEntries: number;
  ambientArcs: number;
  pagerank: { damping: number; iterations: number; converged: boolean; delta: number };
  degree: { mean: number; max: number; p50: number; p99: number };
}

export interface TileManifest {
  layoutVersion: number;
  generatedAt: string;
  seed: number;
  total: number;
  synthetic: boolean;
  /** `names` is present on pipeline-built worlds and absent on synthetic ones. */
  bands: Array<{
    band: number; count: number; bytes: number; file: string;
    names?: string;
    /** Score columns; absent on worlds built before `calibrate`/`learn` ran. */
    meta?: string;
  }>;
  domains: string[];
  /** Absent on pre-v2 worlds generated before the graph existed. */
  graph?: GraphManifest;
  clusters: Array<{ id: number; label: string; domain: number; theta: number; phi: number; kappa: number }>;
}

export interface LoadedBand {
  band: number;
  tile: Tile;
  /** Index of this band's first point in the global picking id space. */
  idOffset: number;
  bytes: number;
  loadMs: number;
}

const TILE_ROOT = `${import.meta.env.BASE_URL}tiles/`;

export async function fetchManifest(signal?: AbortSignal): Promise<TileManifest> {
  const res = await fetch(`${TILE_ROOT}manifest.json`, { signal });
  if (!res.ok) {
    throw new Error(
      `No tile manifest (HTTP ${res.status}). Run \`npm run gen:tiles\` before starting the dev server.`,
    );
  }
  return res.json();
}

export async function fetchBand(
  manifest: TileManifest,
  band: number,
  signal?: AbortSignal,
): Promise<LoadedBand> {
  const entry = manifest.bands.find((b) => b.band === band);
  if (!entry) throw new Error(`Manifest has no band ${band}`);

  const t0 = performance.now();
  const res = await fetch(`${TILE_ROOT}${entry.file}`, { signal });
  if (!res.ok) throw new Error(`Failed to fetch ${entry.file}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  const tile = decodeTile(buf);

  if (tile.layoutVersion !== manifest.layoutVersion) {
    throw new Error(
      `Tile ${entry.file} is layout v${tile.layoutVersion} but the manifest is v${manifest.layoutVersion}. ` +
        `Regenerate tiles.`,
    );
  }

  // Picking ids must be globally unique across bands, so each band is offset by
  // the total count of every lower band.
  const idOffset = manifest.bands
    .filter((b) => b.band < band)
    .reduce((sum, b) => sum + b.count, 0);

  // Names are fetched after the tile and never block it. A band whose names
  // fail to load still renders — it falls back to procedural labels, which is
  // strictly better than an empty globe.
  if (entry.names) {
    try {
      const nameRes = await fetch(`${TILE_ROOT}${entry.names}`, { signal });
      if (nameRes.ok) {
        const names: string[] = await nameRes.json();
        if (names.length !== tile.count) {
          console.warn(
            `${entry.names} has ${names.length} names for ${tile.count} points — ` +
              `names and tiles are out of sync. Rebuild.`,
          );
        }
        registerNames(idOffset, names);
      }
    } catch (err) {
      if ((err as Error)?.name !== 'AbortError') {
        console.warn(`Could not load ${entry.names}; falling back to generated names.`, err);
      }
    }
  }

  // Same contract as names: fetched after the tile, never blocking it, and a
  // failure degrades to "not scored" rather than to an empty globe.
  if (entry.meta) {
    try {
      const metaRes = await fetch(`${TILE_ROOT}${entry.meta}`, { signal });
      if (metaRes.ok) {
        const columns = await metaRes.json();
        const length = columns.score?.length ?? columns.starRank?.length ?? 0;
        if (length && length !== tile.count) {
          console.warn(
            `${entry.meta} has ${length} rows for ${tile.count} points — ` +
              `scores and tiles are out of sync. Rebuild.`,
          );
        } else {
          registerScores(idOffset, tile.count, columns);
        }
      }
    } catch (err) {
      if ((err as Error)?.name !== 'AbortError') {
        console.warn(`Could not load ${entry.meta}; scores will show as unranked.`, err);
      }
    }
  }

  return { band, tile, idOffset, bytes: buf.byteLength, loadMs: performance.now() - t0 };
}
