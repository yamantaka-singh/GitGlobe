import { decodeTile, type Tile } from './format';

export interface TileManifest {
  layoutVersion: number;
  generatedAt: string;
  seed: number;
  total: number;
  synthetic: boolean;
  bands: Array<{ band: number; count: number; bytes: number; file: string }>;
  domains: string[];
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

  return { band, tile, idOffset, bytes: buf.byteLength, loadMs: performance.now() - t0 };
}
