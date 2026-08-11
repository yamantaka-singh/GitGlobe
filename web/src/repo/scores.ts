/**
 * Per-repository scores for the detail panel — global rank and brain composite.
 *
 * These arrive as `meta-N.json` sidecars rather than as extra bytes per point,
 * because nothing here reaches a shader. The panel reads one repository at a
 * time, so widening the tile format would have cost 87k x N bytes on every page
 * load, and forced a version bump on both sides of the binary format, to serve
 * a single selection.
 *
 * Stored as columns rather than objects for the same reason the writer emits
 * them that way: at this row count the repeated key names cost more than the
 * values.
 *
 * `null` means "not scored" and is preserved as `undefined` rather than
 * collapsed to 0. A repository nobody has judged and one judged at the bottom
 * are different claims, and only the second is ours to make.
 */

import { group } from '../ui/num';

export interface RepoScores {
  /** 0-100 composite against the measured global scale. */
  score?: number;
  /** Approximate position by stars among all public repositories. */
  starRank?: number;
  /** Mean of the rubric dimensions the student was allowed to keep. */
  brain?: number;
}

interface Columns {
  score?: (number | null)[];
  starRank?: (number | null)[];
  brain?: (number | null)[];
}

/** idOffset -> columns for that band. Bands load independently and out of order. */
const bands = new Map<number, { count: number; columns: Columns }>();

export function registerScores(idOffset: number, count: number, columns: Columns): void {
  bands.set(idOffset, { count, columns });
}

export function clearScores(): void {
  bands.clear();
}

const at = (column: (number | null)[] | undefined, i: number): number | undefined => {
  const value = column?.[i];
  return value === null || value === undefined ? undefined : value;
};

/**
 * Scores for a global picking id, or an empty object if that band has no
 * sidecar — which is the normal state before `calibrate` and `learn` have run.
 */
export function scoresFor(id: number): RepoScores {
  for (const [offset, { count, columns }] of bands) {
    const i = id - offset;
    if (i < 0 || i >= count) continue;
    return { score: at(columns.score, i), starRank: at(columns.starRank, i), brain: at(columns.brain, i) };
  }
  return {};
}

/**
 * "#4,312 of ~420M · top 1 in 97,000" — the honest global standing.
 *
 * The panel previously showed an in-corpus PageRank percentile labelled only
 * "Top X%". Every repository in this corpus clears ~66 stars, which is already
 * the top 0.2% of GitHub, so that number understated standing by roughly 500x
 * while reading as though it were a global one.
 */
export function describeRank(starRank: number | undefined, totalRepos = 420_000_000): string {
  if (starRank === undefined || !Number.isFinite(starRank)) return 'not ranked';
  return `#${group(starRank)} of ~${Math.round(totalRepos / 1e6)}M · ${rarity(starRank, totalRepos)}`;
}

/**
 * Just the rarity — "top 1 in 3,442,623" or "top 0.00103%".
 *
 * Split out so the hover card, which has room for one short line, uses the same
 * 1e-6 cutover as the detail panel. Duplicating the threshold is how the two
 * would come to disagree about the same repository.
 */
export function rarity(starRank: number | undefined, totalRepos = 420_000_000): string {
  if (starRank === undefined || !Number.isFinite(starRank)) return 'unranked';
  const share = starRank / totalRepos;
  return share < 1e-6
    ? `top 1 in ${group(1 / Math.max(share, 1e-12))}`
    : `top ${(share * 100).toPrecision(3)}%`;
}
