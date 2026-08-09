/**
 * Repository names — real ones when the world has them, procedural otherwise.
 *
 * "#48213" reads as a dot. "vecstore-rs" reads as a repository. That difference
 * is most of why the Phase 0 globe felt like decoration rather than data.
 *
 * Real names arrive as `names-N.json`, one array per band, aligned with that
 * band's tile: entry `i` names point `i`. No id lookup, no map of a million
 * entries — `repoId` is `ordinal + 1` and bands are contiguous slices of the
 * rank order, so a band's names slot straight into a flat array at `idOffset`.
 *
 * The procedural generator stays as the fallback. Synthetic worlds from
 * `gen-world.ts` have no names, and the renderer has to be developable without
 * a database and a Vertex AI bill behind it.
 */

const STEMS: Record<number, string[]> = {
  0: ['torch', 'tensor', 'neural', 'infer', 'diffus', 'grad', 'attn', 'embed', 'logit', 'axon', 'synapse', 'latent'],
  1: ['router', 'hydra', 'ember', 'quill', 'signal', 'render', 'hydrate', 'island', 'slot', 'vane', 'lattice'],
  2: ['vecstore', 'shard', 'ledger', 'kvproxy', 'index', 'chronos', 'columnar', 'quorum', 'wal', 'btree'],
  3: ['helm', 'forge', 'pilot', 'drift', 'ansible', 'orchestr', 'provision', 'rollout', 'sentinel', 'nomad'],
  4: ['lexer', 'parsec', 'bytecode', 'hir', 'monomorph', 'typeck', 'codegen', 'macro', 'ast', 'borrow'],
  5: ['ferrite', 'kernel', 'ringbuf', 'mmio', 'baremetal', 'ioctl', 'scheduler', 'atomics', 'nostd'],
  6: ['pipeline', 'stream', 'batch', 'parquet', 'lakehouse', 'ingest', 'dagger', 'arrow', 'delta'],
  7: ['cipher', 'vault', 'attest', 'sandbox', 'fuzz', 'sigstore', 'zerotrust', 'entropy', 'nonce'],
  8: ['raster', 'shader', 'voxel', 'bevy', 'photon', 'mesh', 'bloom', 'sdf', 'lumen', 'radiance'],
  9: ['compose', 'flutterkit', 'nativebridge', 'haptic', 'swipe', 'appshell', 'coldstart'],
  10: ['scrapy', 'crawler', 'harvest', 'spider', 'headless', 'proxypool', 'throttle', 'sitemap'],
  11: ['numerics', 'simplex', 'quadrature', 'lapack', 'symbolic', 'montecarlo', 'ode', 'fftw'],
};

const MODIFIERS = [
  'hyper', 'micro', 'nano', 'poly', 'meta', 'omni', 'ultra', 'proto', 'quantum', 'auto',
  'fast', 'lite', 'deep', 'open', 'edge', 'zero',
];

const SUFFIXES = ['-rs', '-js', '-py', '-go', '-core', '-kit', '-lab', 'x', 'ly', 'io', '.dev', '-ng', ''];

const ORGS = [
  'apertureworks', 'northgate', 'obsidian-io', 'helios', 'blackbox', 'lumenlabs', 'terrafirma',
  'axiom', 'coldfront', 'signalworks', 'anvil', 'greyscale', 'nimbus', 'ironwood', 'palewire',
];

/** xmur3-style hash: cheap, well-mixed, and stable across engines. */
function hash(value: number, salt: number): number {
  let h = (value ^ salt) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^ (h >>> 16)) >>> 0;
}

const pick = <T>(list: readonly T[], value: number, salt: number): T => list[hash(value, salt) % list.length];

export interface RepoIdentity {
  name: string;
  org: string;
  fullName: string;
}

const cache = new Map<number, RepoIdentity>();

/**
 * Real names by node ordinal (`repoId - 1`). Sparse until every band loads,
 * which is the point: band 0 arrives first and its names are usable
 * immediately, without waiting on the 80% of the corpus in band 2.
 */
const realNames: string[] = [];

/** Called by the tile loader once a band's `names-N.json` arrives. */
export function registerNames(idOffset: number, names: readonly string[]): void {
  for (let i = 0; i < names.length; i++) realNames[idOffset + i] = names[i];
  // Entries cached before the names landed are procedural and now wrong.
  cache.clear();
}

export function hasRealNames(): boolean {
  return realNames.length > 0;
}

/** Test seam — resets module state between cases. */
export function clearNames(): void {
  realNames.length = 0;
  cache.clear();
}

export function repoIdentity(repoId: number, domain: number): RepoIdentity {
  const cached = cache.get(repoId);
  if (cached) return cached;

  const real = realNames[repoId - 1];
  if (real) {
    const slash = real.indexOf('/');
    const identity: RepoIdentity =
      slash > 0
        ? { org: real.slice(0, slash), name: real.slice(slash + 1), fullName: real }
        : { org: '', name: real, fullName: real };
    if (cache.size > 4096) cache.clear();
    cache.set(repoId, identity);
    return identity;
  }

  const stems = STEMS[domain] ?? STEMS[0];
  const stem = pick(stems, repoId, 0x9e37);
  const roll = hash(repoId, 0x85eb) % 100;

  let name: string;
  if (roll < 42) {
    name = stem + pick(SUFFIXES, repoId, 0xc2b2);
  } else if (roll < 72) {
    name = pick(MODIFIERS, repoId, 0x27d4) + stem;
  } else if (roll < 90) {
    name = `${stem}-${pick(stems, repoId, 0x1656)}`;
  } else {
    name = `${pick(MODIFIERS, repoId, 0x165c)}-${stem}${pick(SUFFIXES, repoId, 0x4f2d)}`;
  }

  const identity: RepoIdentity = {
    name,
    org: pick(ORGS, repoId, 0x7feb),
    fullName: `${pick(ORGS, repoId, 0x7feb)}/${name}`,
  };

  // Bounded so a long hover session over band 2 can't grow without limit.
  if (cache.size > 4096) cache.clear();
  cache.set(repoId, identity);
  return identity;
}
