import { useCallback, useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import { useGlobeStore, type BenchResult } from '../store/useGlobeStore';
import { globeCamera } from '../camera/Rig';

/**
 * Frame-budget benchmark.
 *
 * The first version of this gated on "p95 frame interval <= 16.7ms" and was
 * simply wrong. requestAnimationFrame is vsync-locked: on a 60Hz display the
 * interval between callbacks is 16.67ms whether the renderer is doing almost
 * nothing or is right at the edge of dropping frames. Measuring that interval
 * measures the monitor, not the scene. A perfectly healthy renderer scores
 * p50 = 16.7 / p95 = 17.1 and "fails" a gate it can never pass.
 *
 * What "sustained 60fps" actually means is "no dropped frames", so that is what
 * this measures now:
 *
 *  1. **Calibrate.** Take the median interval over the warmup — that is the
 *     display's refresh period (16.67ms at 60Hz, 8.33 at 120Hz).
 *  2. **Phase A — dropped frames.** Orbit deterministically. A frame is dropped
 *     when its interval exceeds 1.5x the period. This is the honest verdict.
 *  3. **Phase B — headroom.** Render the scene N extra times per frame and find
 *     the largest N that still holds the refresh rate. If the scene fits three
 *     times into one frame, there is 3x headroom — which is the number that
 *     actually tells us whether 1M nodes will fit.
 *
 * Headroom is the useful metric because vsync hides everything above the line.
 * Without it you cannot distinguish "comfortable" from "one particle away from
 * collapse", and both report 60fps.
 */

const WARMUP_FRAMES = 40;
const MAX_SAMPLES = 4096;

/**
 * Phone profile.
 *
 * Phase B measures headroom by rendering the whole scene N extra times in a
 * single frame. At the desktop steps that peaks at eight full renders of
 * 198,731 points per frame — on a phone that is a thermal event, not a
 * measurement: the GPU throttles partway through, so the number it produces is
 * wrong *and* it can cost a dropped WebGL context. Small screens get a shorter
 * sweep and stop probing at 3x, which is still enough to answer the only
 * question that matters on a phone ("is there any margin at all?").
 *
 * ponytail: coarse pointer + width, not a device database. Good enough to
 * separate "phone" from "workstation", which is all this gates.
 */
function isSmallScreen() {
  if (typeof window === 'undefined') return false;
  return window.matchMedia?.('(pointer: coarse)').matches || window.innerWidth < 700;
}

const DESKTOP = { phaseA: 6000, steps: [1, 2, 3, 5, 7], probeMs: 1100 };
// ~6s: long enough to read as a measurement rather than a flash, still roughly
// half the desktop run and nowhere near the thermal load of the 5x/7x probes.
const PHONE = { phaseA: 4000, steps: [1, 2], probeMs: 1000 };

/** A frame counts as dropped when it misses its vsync slot by half a period. */
const DROP_FACTOR = 1.5;
/** Gate: fewer than 1% dropped frames, and nothing catastrophic. */
const MAX_DROP_RATIO = 0.01;
const MAX_WORST_MS = 100;
/** A probe step passes if it drops fewer than 5% of frames. */
const PROBE_DROP_TOLERANCE = 0.05;

type Phase = 'idle' | 'warmup' | 'measure' | 'probe';

function percentile(sorted: Float64Array, n: number, p: number): number {
  if (n === 0) return 0;
  return sorted[Math.min(n - 1, Math.max(0, Math.round((n - 1) * p)))];
}

export function Benchmark() {
  const { gl, scene, camera } = useThree();
  const samples = useRef(new Float64Array(MAX_SAMPLES));
  const probeSamples = useRef(new Float64Array(512));

  const s = useRef({
    phase: 'idle' as Phase,
    frames: 0,
    n: 0,
    startedAt: 0,
    phaseStartedAt: 0,
    last: 0,
    period: 16.667,
    prevAutoRotate: true,
    prevCalls: 0,
    prevTris: 0,
    maxCalls: 0,
    maxTris: 0,
    probeIndex: 0,
    probeN: 0,
    probeDrops: 0,
    headroom: 1,
    profile: DESKTOP as { phaseA: number; steps: number[]; probeMs: number },
  });

  const stop = useCallback(
    (aborted: boolean) => {
      const st = s.current;
      if (st.phase === 'idle') return;
      st.phase = 'idle';
      gl.info.autoReset = true;

      const store = useGlobeStore.getState();
      store.setAutoRotate(st.prevAutoRotate);
      store.setBenchRunning(false);
      // Glide back to the pre-run framing rather than stranding the user at
      // whatever angle the sweep happened to finish on. Runs on abort too.
      void globeCamera.restoreState(true);
      if (aborted) return;

      const n = st.n;
      const sorted = samples.current.slice(0, n).sort();
      const dropThreshold = st.period * DROP_FACTOR;
      let dropped = 0;
      let worst = 0;
      for (let i = 0; i < n; i++) {
        if (sorted[i] > dropThreshold) dropped++;
        if (sorted[i] > worst) worst = sorted[i];
      }
      const dropRatio = n > 0 ? dropped / n : 1;

      const result: BenchResult = {
        label: new Date().toLocaleTimeString(),
        frames: st.frames,
        durationMs: performance.now() - st.startedAt,
        refreshHz: 1000 / st.period,
        fps: 1000 / percentile(sorted, n, 0.5),
        p50: percentile(sorted, n, 0.5),
        p95: percentile(sorted, n, 0.95),
        p99: percentile(sorted, n, 0.99),
        worst,
        dropped,
        sampled: n,
        dropRatio,
        headroom: st.headroom,
        drawCalls: st.maxCalls,
        triangles: st.maxTris,
        points: useGlobeStore.getState().totalPoints,
        passed: dropRatio < MAX_DROP_RATIO && worst < MAX_WORST_MS,
      };

      store.pushBenchResult(result);

      // eslint-disable-next-line no-console
      console.table({
        nodes: result.points.toLocaleString(),
        display: `${result.refreshHz.toFixed(0)} Hz (${st.period.toFixed(2)}ms)`,
        'dropped frames': `${dropped} / ${n}  (${(dropRatio * 100).toFixed(2)}%)`,
        'p50 / p95 / p99 ms': `${result.p50.toFixed(2)} / ${result.p95.toFixed(2)} / ${result.p99.toFixed(2)}`,
        'worst ms': result.worst.toFixed(1),
        headroom: `${result.headroom}x  (scene fits ${result.headroom} times per frame)`,
        'draw calls': result.drawCalls,
        triangles: result.triangles.toLocaleString(),
        verdict: result.passed ? 'PASS' : 'FAIL',
      });
    },
    [gl],
  );

  useEffect(() => {
    const unsub = useGlobeStore.subscribe((store, prev) => {
      if (store.benchRunning === prev.benchRunning) return;
      const st = s.current;

      if (store.benchRunning) {
        st.profile = isSmallScreen() ? PHONE : DESKTOP;
        // Remember where the user was looking; the sweep is about to take the
        // camera somewhere arbitrary.
        globeCamera.saveState();
        st.prevAutoRotate = prev.autoRotate;
        useGlobeStore.getState().setAutoRotate(false);
        st.phase = 'warmup';
        st.frames = 0;
        st.n = 0;
        st.maxCalls = 0;
        st.maxTris = 0;
        st.probeIndex = 0;
        st.probeN = 0;
        st.probeDrops = 0;
        st.headroom = 1;
        st.startedAt = performance.now();
        st.phaseStartedAt = st.startedAt;
        st.last = st.startedAt;

        // Take manual control of the counters. Whether three.js or R3F resets
        // them, and when relative to our useFrame, is not something to rely on
        // — reading a delta of a monotonic counter is correct either way.
        // (Getting this wrong is why the first run reported "0 calls, 0 tris".)
        gl.info.autoReset = false;
        gl.info.reset();
        st.prevCalls = 0;
        st.prevTris = 0;
      } else {
        stop(true);
      }
    });
    return unsub;
  }, [gl, stop]);

  useEffect(() => () => {
    gl.info.autoReset = true;
  }, [gl]);

  useFrame(() => {
    const st = s.current;
    if (st.phase === 'idle') return;

    const now = performance.now();
    const dt = now - st.last;
    st.last = now;
    st.frames++;

    // Deterministic sweep, driven by absolute elapsed seconds at fixed rates
    // rather than by normalised progress.
    //
    // Normalised progress tied the travel to the run length, so shortening the
    // phone profile did not calm the sweep down — it crammed the same two full
    // rotations, three polar swings and four dolly cycles into half the time
    // and made it whip. Fixed rates mean the motion looks identical on every
    // device and every profile; a shorter run simply sees less of it, and the
    // per-frame cost being measured is unchanged.
    //
    // The ranges are gentler too: the old dolly reached 1.5x radius, close
    // enough to sit inside the point cloud.
    const secs = (now - st.startedAt) / 1000;
    const polar = Math.PI * (0.5 + 0.17 * Math.sin(secs * 0.85));
    const distance = globeCamera.radius * (2.5 + 0.75 * Math.sin(secs * 0.5));
    globeCamera.setOrbitAngle(secs * 0.55, polar, distance);

    if (st.phase === 'warmup') {
      // Shader compilation and buffer upload land here, so these frames are
      // recorded only to calibrate the display period.
      if (st.n < MAX_SAMPLES) samples.current[st.n++] = dt;
      if (st.frames >= WARMUP_FRAMES) {
        const warm = samples.current.slice(0, st.n).sort();
        st.period = percentile(warm, st.n, 0.5) || 16.667;
        st.n = 0;
        st.phase = 'measure';
        st.phaseStartedAt = now;
        st.prevCalls = gl.info.render.calls;
        st.prevTris = gl.info.render.triangles;
      }
      return;
    }

    if (st.phase === 'measure') {
      if (st.n < MAX_SAMPLES) samples.current[st.n++] = dt;

      const calls = gl.info.render.calls - st.prevCalls;
      const tris = gl.info.render.triangles - st.prevTris;
      st.prevCalls = gl.info.render.calls;
      st.prevTris = gl.info.render.triangles;
      if (calls > st.maxCalls) st.maxCalls = calls;
      if (tris > st.maxTris) st.maxTris = tris;

      if (now - st.phaseStartedAt >= st.profile.phaseA) {
        st.phase = 'probe';
        st.phaseStartedAt = now;
        st.probeIndex = 0;
        st.probeN = 0;
        st.probeDrops = 0;
      }
      return;
    }

    // ---- phase B: headroom -------------------------------------------------
    const extra = st.profile.steps[st.probeIndex];
    for (let i = 0; i < extra; i++) gl.render(scene, camera as THREE.Camera);

    if (st.probeN < probeSamples.current.length) probeSamples.current[st.probeN++] = dt;
    if (dt > st.period * DROP_FACTOR) st.probeDrops++;

    if (now - st.phaseStartedAt >= st.profile.probeMs) {
      // Ignore the first few frames of each step — the first frame at a new
      // render multiplier always overruns while the pipeline fills.
      const ratio = st.probeN > 6 ? st.probeDrops / st.probeN : 1;
      if (ratio < PROBE_DROP_TOLERANCE) st.headroom = extra + 1;

      st.probeIndex++;
      st.probeN = 0;
      st.probeDrops = 0;
      st.phaseStartedAt = now;

      // Stop early once a step fails — headroom is monotonic, so every
      // heavier step would fail too and measuring them wastes four seconds.
      if (st.probeIndex >= st.profile.steps.length || ratio >= PROBE_DROP_TOLERANCE) {
        stop(false);
      }
    }
  });

  return null;
}
