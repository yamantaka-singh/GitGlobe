/**
 * Procedural repository names.
 *
 * "#48213" reads as a dot. "vecstore-rs" reads as a repository. That difference
 * is most of why the Phase 0 globe felt like decoration rather than data, and
 * it costs zero bytes on the wire: names are a deterministic function of the
 * repo id, generated on the client.
 *
 * Phase 1 deletes this file and uses real names.
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

export function repoIdentity(repoId: number, domain: number): RepoIdentity {
  const cached = cache.get(repoId);
  if (cached) return cached;

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
