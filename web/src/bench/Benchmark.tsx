import { useCallback, useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';

import { useGlobeStore, type BenchResult } from '../store/useGlobeStore';
import { globeCamera } from '../camera/Rig';

const DURATION_MS = 10_000;
const MAX_SAMPLES = 4096;
/** Frames discarded at the start — shader compilation and buffer upload land here. */
const WARMUP_FRAMES = 30;

/** Phase 0 exit criterion: 60fps sustained means p95 frame time inside the budget. */
const BUDGET_MS = 16.7;

function percentile(sorted: Float64Array, n: number, p: number): number {
  if (n === 0) return 0;
  const idx = Math.min(n - 1, Math.max(0, Math.round((n - 1) * p)));
  return sorted[idx];
}

/**
 * Measures the exit criterion instead of eyeballing it.
 *
 * Drives a deterministic orbit — same path every run, so two runs are
 * comparable — and records real frame deltas. Reports p50/p95/p99 rather than
 * average fps, because average fps hides exactly the stutter that makes a
 * scene feel bad. A run that averages 60fps but spikes to 90ms twice a second
 * is a failure, and only the percentiles show it.
 */
export function Benchmark() {
  const { gl } = useThree();
  const samples = useRef(new Float64Array(MAX_SAMPLES));
  const state = useRef({
    active: false,
    frames: 0,
    n: 0,
    startedAt: 0,
    last: 0,
    azimuth: 0,
    prevAutoRotate: true,
    maxDrawCalls: 0,
  });

  const finish = useCallback(() => {
    const s = state.current;
    if (!s.active) return;
    s.active = false;

    const n = s.n;
    const sorted = samples.current.slice(0, n).sort();
    const durationMs = performance.now() - s.startedAt;
    let over16 = 0;
    let worst = 0;
    for (let i = 0; i < n; i++) {
      if (sorted[i] > BUDGET_MS) over16++;
      if (sorted[i] > worst) worst = sorted[i];
    }

    const p95 = percentile(sorted, n, 0.95);
    const result: BenchResult = {
      label: new Date().toLocaleTimeString(),
      frames: s.frames,
      durationMs,
      fps: (s.frames / durationMs) * 1000,
      p50: percentile(sorted, n, 0.5),
      p95,
      p99: percentile(sorted, n, 0.99),
      worst,
      over16,
      drawCalls: s.maxDrawCalls,
      points: useGlobeStore.getState().totalPoints,
      passed: p95 <= BUDGET_MS && worst < 100,
    };

    const store = useGlobeStore.getState();
    store.pushBenchResult(result);
    store.setBenchRunning(false);
    store.setAutoRotate(s.prevAutoRotate);

    // eslint-disable-next-line no-console
    console.table({
      points: result.points.toLocaleString(),
      fps: result.fps.toFixed(1),
      'p50 ms': result.p50.toFixed(2),
      'p95 ms': result.p95.toFixed(2),
      'p99 ms': result.p99.toFixed(2),
      'worst ms': result.worst.toFixed(1),
      'frames over 16.7ms': `${result.over16} / ${n}`,
      'draw calls': result.drawCalls,
      verdict: result.passed ? 'PASS' : 'FAIL',
    });
  }, []);

  // Start when the store flag flips.
  useEffect(() => {
    const unsub = useGlobeStore.subscribe((store, prev) => {
      if (store.benchRunning === prev.benchRunning) return;
      const s = state.current;
      if (store.benchRunning) {
        s.prevAutoRotate = prev.autoRotate;
        useGlobeStore.getState().setAutoRotate(false);
        s.active = true;
        s.frames = 0;
        s.n = 0;
        s.maxDrawCalls = 0;
        s.azimuth = 0;
        s.startedAt = performance.now();
        s.last = s.startedAt;
        void globeCamera.reset(true);
      } else if (s.active) {
        finish();
      }
    });
    return unsub;
  }, [finish]);

  useFrame(() => {
    const s = state.current;
    if (!s.active) return;

    const now = performance.now();
    const dt = now - s.last;
    s.last = now;
    s.frames++;

    if (s.frames > WARMUP_FRAMES && s.n < MAX_SAMPLES) {
      samples.current[s.n++] = dt;
      if (gl.info.render.calls > s.maxDrawCalls) s.maxDrawCalls = gl.info.render.calls;
    }

    // Deterministic sweep: a full azimuth revolution plus a polar and dolly
    // oscillation, so the run covers dense clusters, empty regions, and both
    // near and far framings rather than whichever view happened to be up.
    const t = (now - s.startedAt) / DURATION_MS;
    s.azimuth = t * Math.PI * 2;
    const polar = Math.PI * (0.5 + 0.3 * Math.sin(t * Math.PI * 4));
    const distance = globeCamera.radius * (1.5 + 1.3 * (0.5 + 0.5 * Math.cos(t * Math.PI * 6)));
    globeCamera.setOrbitAngle(s.azimuth, polar, distance);

    if (now - s.startedAt >= DURATION_MS) finish();
  });

  return null;
}
