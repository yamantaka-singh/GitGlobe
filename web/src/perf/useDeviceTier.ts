import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import type * as THREE from 'three';
import { useGlobeStore, type Tier } from '../store/useGlobeStore';

/**
 * Device tiering, per web3d-performance-budget.
 *
 * Static detection picks the starting budget once, and nothing lowers it at
 * runtime — see the note at the bottom of this file for why the frame-time
 * sampler that used to live here was removed rather than tuned. Runtime quality
 * is governed by pixel ratio in `Scene`, which never costs the user any data.
 */
export function TIER_BUDGET(tier: Tier) {
  return {
    // The ambient arc layer is fill-rate bound — ribbons are big translucent
    // quads — so it is the first thing to go on a weak GPU, before point count.
    low: { maxBand: 0, dprCap: 1.0, targetMs: 33, ambientArcs: 0 },
    mid: { maxBand: 1, dprCap: 1.5, targetMs: 20, ambientArcs: 900 },
    high: { maxBand: 2, dprCap: 2.0, targetMs: 16.7, ambientArcs: 2000 },
  }[tier];
}

export function detectTier(gl: THREE.WebGLRenderer): Tier {
  let renderer = '';
  try {
    const ctx = gl.getContext();
    const dbg = ctx.getExtension('WEBGL_debug_renderer_info') as { UNMASKED_RENDERER_WEBGL: number } | null;
    if (dbg) renderer = String(ctx.getParameter(dbg.UNMASKED_RENDERER_WEBGL) ?? '');
  } catch {
    // Some browsers block the extension entirely. Fall through to heuristics.
  }

  const cores = navigator.hardwareConcurrency ?? 4;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
  const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

  if (/Apple (M\d|GPU)/i.test(renderer) && !mobile) return 'high';
  if (/(SwiftShader|llvmpipe|Software)/i.test(renderer)) return 'low';
  if (mobile && mem <= 3) return 'low';
  // Phones get the full corpus. They used to be pinned to `mid`, which caps
  // loading at band 1 — 39,747 of 198,731 nodes. All bands are one draw call
  // each, so the extra points cost geometry bandwidth once at load, not frame
  // time; what actually costs frame time is fragments, and the DPR governor
  // owns that. The `cores <= 4` rule that used to force `low` is gone as well:
  // iOS under-reports hardwareConcurrency, so it demoted exactly the devices
  // most able to cope.
  // ponytail: no runtime downgrade path at all. If a genuinely weak phone is
  // observed struggling at the lowest DPR, gate this on `mem >= 6`.
  if (mobile) return 'high';
  if (cores <= 4 || mem <= 4) return 'mid';
  return 'high';
}

/**
 * One decision per page load, not one per effect run.
 *
 * `setTier(initial)` used to run on every invocation of the effect below, which
 * meant any remount reset the tier to its optimistic starting value and kicked
 * off a fresh 158k-point band load. Module scope rather than a ref: the point is
 * to survive a remount, which is exactly what a ref does not do.
 */
let tierDecided = false;

export function useDeviceTier() {
  const gl = useThree((s) => s.gl);
  const setTier = useGlobeStore((s) => s.setTier);
  useEffect(() => {
    if (!tierDecided) {
      tierDecided = true;
      setTier(detectTier(gl));
    }

    // Honour the OS accessibility setting before we start spinning anything.
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyMotion = () => {
      useGlobeStore.getState().setReducedMotion(mq.matches);
      if (mq.matches) useGlobeStore.getState().setAutoRotate(false);
    };
    applyMotion();
    mq.addEventListener('change', applyMotion);

    return () => mq.removeEventListener('change', applyMotion);
  }, [gl, setTier]);
}

/**
 * Why there is no longer a measured demotion here.
 *
 * There used to be a frame-time sampler that dropped the tier when it saw a bad
 * p95. It was removed rather than tuned, because both halves of it were wrong:
 *
 *  - **The measurement could not be trusted.** It sampled `requestAnimationFrame`
 *    deltas, and a browser throttles rAF to roughly 1fps in a background tab —
 *    so backgrounding the page, or switching apps on a phone, produced 1000ms
 *    "frames" and demoted instantly. That is the reported "nodes keep dropping":
 *    it fired on a 10-core desktop with 16GB purely because the tab was not in
 *    front. Its earlier window also covered the 158,984-point upload, measuring
 *    the load instead of the steady state.
 *
 *  - **The remedy was worse than the problem.** Demotion lowers `maxBand`, which
 *    unloads LOD bands — 198,731 nodes collapsing to 39,747 or 3,975. Throwing
 *    away 80–98% of the corpus is the most destructive possible response to one
 *    slow frame, and it is the thing users actually notice.
 *
 * Runtime quality is now handled entirely by the DPR governor in `Scene`
 * (drei's `PerformanceMonitor` driving `dprScale`). That is the correct lever:
 * this scene is fragment-bound, not geometry-bound — all bands are one draw
 * call each, while zooming in multiplies shaded pixels per point. Pixel ratio
 * is quadratic in fragment cost, recovers automatically when load drops, and
 * costs a little sharpness instead of most of the data.
 *
 * `detectTier` still runs once for the *starting* DPR cap and ambient-arc
 * budget. It is static, so a throttled tab cannot corrupt it.
 */
