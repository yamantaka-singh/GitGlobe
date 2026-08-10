import { beforeEach, describe, expect, it } from 'vitest';

import { clearNames, hasRealNames, registerNames, repoIdentity } from './names';

describe('repository names', () => {
  beforeEach(clearNames);

  describe('without a names file', () => {
    it('generates a plausible name so synthetic worlds stay developable', () => {
      expect(hasRealNames()).toBe(false);
      expect(repoIdentity(7, 0).fullName).toContain('/');
    });

    it('is deterministic — the same node keeps its name across hovers', () => {
      expect(repoIdentity(7, 0).fullName).toBe(repoIdentity(7, 0).fullName);
    });
  });

  describe('with real names', () => {
    it('maps repoId to the band-aligned entry', () => {
      // repoId is ordinal + 1, and band 0 starts at idOffset 0.
      registerNames(0, ['pytorch/pytorch', 'facebook/react', 'torvalds/linux']);
      expect(hasRealNames()).toBe(true);
      expect(repoIdentity(1, 0).fullName).toBe('pytorch/pytorch');
      expect(repoIdentity(3, 0).fullName).toBe('torvalds/linux');
    });

    it('splits org and name', () => {
      registerNames(0, ['rust-lang/rust']);
      expect(repoIdentity(1, 0)).toMatchObject({ org: 'rust-lang', name: 'rust' });
    });

    it('offsets later bands correctly', () => {
      registerNames(0, ['a/one', 'a/two', 'a/three']);
      registerNames(3, ['b/four', 'b/five']);
      expect(repoIdentity(4, 0).fullName).toBe('b/four');
      expect(repoIdentity(5, 0).fullName).toBe('b/five');
    });

    it('invalidates names cached before its band arrived', () => {
      // Band 2 loads last, so a node hovered early gets a procedural name and
      // caches it. Without invalidation it keeps that fake name for the rest of
      // the session while its neighbours show real ones.
      const generated = repoIdentity(4, 0).fullName;
      registerNames(3, ['nodejs/node']);
      expect(repoIdentity(4, 0).fullName).toBe('nodejs/node');
      expect(repoIdentity(4, 0).fullName).not.toBe(generated);
    });

    it('still falls back for ids no band has covered yet', () => {
      registerNames(0, ['a/one']);
      expect(repoIdentity(9999, 0).fullName).toContain('/');
    });

    it('handles a name with no slash without inventing an org', () => {
      registerNames(0, ['justaname']);
      expect(repoIdentity(1, 0)).toMatchObject({ org: '', fullName: 'justaname' });
    });
  });
});
