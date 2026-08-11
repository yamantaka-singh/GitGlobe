import { describe, it, expect, beforeEach } from 'vitest';
import { clearScores, describeRank, rarity, registerScores, scoresFor } from './scores';

/**
 * The failure this guards against is not a crash — it is a confident wrong
 * number in the panel. Two ways that happens:
 *
 * 1. `null` (not scored) collapsing to 0, which renders as "we judged this and
 *    it came last" rather than "nothing has judged this".
 * 2. Band offsets applied wrongly, so the panel shows one repository's score
 *    next to another repository's name. Nothing errors; it just lies.
 */
beforeEach(clearScores);

describe('scoresFor', () => {
  it('reads a value at the right index within a band', () => {
    registerScores(0, 3, { score: [10, 20, 30], starRank: [1, 2, 3], brain: [40, 50, 60] });
    expect(scoresFor(1)).toEqual({ score: 20, starRank: 2, brain: 50 });
  });

  it('applies the band offset', () => {
    // Band 1 starts at id 100. Reading id 101 must give that band's index 1,
    // not index 101 of some other array.
    registerScores(0, 2, { score: [1, 2] });
    registerScores(100, 2, { score: [8, 9] });
    expect(scoresFor(101).score).toBe(9);
  });

  it('keeps null as undefined rather than zero', () => {
    registerScores(0, 2, { score: [null, 55] });
    expect(scoresFor(0).score).toBeUndefined();
    expect(scoresFor(1).score).toBe(55);
  });

  it('returns empty for an id outside every band', () => {
    registerScores(0, 2, { score: [1, 2] });
    expect(scoresFor(99)).toEqual({});
  });

  it('returns empty when no sidecar loaded at all', () => {
    // Normal state before `calibrate` and `learn` have ever run.
    expect(scoresFor(0)).toEqual({});
  });

  it('tolerates a column being absent', () => {
    registerScores(0, 1, { score: [70] });
    expect(scoresFor(0)).toEqual({ score: 70, starRank: undefined, brain: undefined });
  });
});

describe('describeRank', () => {
  it('switches to one-in-N only for the genuinely rare', () => {
    // The cutover is share < 1e-6, i.e. rank < 420. Values were checked
    // against the running function, not assumed — my first draft of this test
    // asserted "top 1 in 97,403" for rank 4,312, which is the wrong branch.
    expect(describeRank(122)).toBe('#122 of ~420M · top 1 in 3,442,623');
    expect(describeRank(4312)).toBe('#4,312 of ~420M · top 0.00103%');
  });

  it('formats a common repository as a percentage', () => {
    expect(describeRank(5_000_000)).toBe('#5,000,000 of ~420M · top 1.19%');
  });

  it('says so instead of inventing a rank', () => {
    // The whole point of preserving null: an unscored repo must not render as
    // "#0 of ~420M", which would read as the best repository on GitHub.
    expect(describeRank(undefined)).toBe('not ranked');
    expect(describeRank(NaN)).toBe('not ranked');
  });

  it('is far more precise than the in-corpus percentile it replaced', () => {
    // A repo at the median of an 87k corpus used to render as "Top 50%".
    // Its real standing is around one in ten thousand.
    expect(describeRank(43_000)).toBe('#43,000 of ~420M · top 0.0102%');
  });

  it('matches the Python StarScale.describe it mirrors', () => {
    // Same 1e-6 threshold and same wording as global_scale.py. If these drift,
    // the CLI and the panel report different rarities for the same repository.
    expect(describeRank(400)).toContain('top 1 in');
    expect(describeRank(500)).toContain('%');
  });

  it('agrees with the compact form the hover card uses', () => {
    // The hover card and the detail panel describe the same repository one
    // click apart. They previously disagreed outright — the card showed an
    // in-corpus PageRank percentile, the panel a measured global rank. Sharing
    // `rarity` is what keeps them from drifting again.
    for (const rank of [122, 400, 4312, 43_000, 5_000_000]) {
      expect(describeRank(rank)).toContain(rarity(rank));
    }
  });

  it('rarity has its own unranked wording rather than a fake number', () => {
    expect(rarity(undefined)).toBe('unranked');
    expect(rarity(NaN)).toBe('unranked');
  });

  it('groups thousands the same way regardless of system locale', () => {
    // Caught on an en-IN machine: bare toLocaleString() produced "#50,00,000"
    // there and "#5,000,000" elsewhere, so the same repository read differently
    // depending on who opened it — and neither matched Python's f"{n:,}".
    expect(describeRank(5_000_000)).toContain('#5,000,000');
    expect(describeRank(122)).toContain('3,442,623');
  });
});
