import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import type * as THREE from 'three';
import { useGlobeStore, type Tier } from '../store/useGlobeStore';

/**
 * Device tiering, per web3d-performance-budget.
 *
 * Static detection gets us a starting guess; measured demotion is what actually
 * works, because a laptop that renders the first second fine can still collapse
 * once the GPU warms up. We watch the first ~2 seconds of real frame times and
 * demote if the device can't hold the budget.
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
  if (mobile && (mem <= 3 || cores <= 4)) return 'low';
  if (mobile) return 'mid';
  if (cores <= 4 || mem <= 4) return 'mid';
  return 'high';
}

export function useDeviceTier() {
  const gl = useThree((s) => s.gl);
  const setTier = useGlobeStore((s) => s.setTier);

  useEffect(() => {
    const initial = detectTier(gl);
    setTier(initial);

    // Honour the OS accessibility setting before we start spinning anything.
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyMotion = () => {
      useGlobeStore.getState().setReducedMotion(mq.matches);
      if (mq.matches) useGlobeStore.getState().setAutoRotate(false);
    };
    applyMotion();
    mq.addEventListener('change', applyMotion);

    // Measured demotion: sample frame times for 2s, then decide.
    const samples: number[] = [];
    let last = performance.now();
    let raf = 0;
    const tick = () => {
      const now = performance.now();
      samples.push(now - last);
      last = now;
      if (samples.length < 140) {
        raf = requestAnimationFrame(tick);
        return;
      }
      // Ignore the first 40 frames — shader compilation and tile upload happen
      // there and would demote every device.
      const warm = samples.slice(40).sort((a, b) => a - b);
      const p95 = warm[Math.floor(warm.length * 0.95)] ?? 0;
      const current = useGlobeStore.getState().tier;
      if (p95 > 34 && current !== 'low') useGlobeStore.getState().setTier('low');
      else if (p95 > 21 && current === 'high') useGlobeStore.getState().setTier('mid');
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      mq.removeEventListener('change', applyMotion);
    };
  }, [gl, setTier]);
}
