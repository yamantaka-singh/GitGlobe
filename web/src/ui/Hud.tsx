import { useEffect, useState } from 'react';
import * as THREE from 'three';

import { useGlobeStore } from '../store/useGlobeStore';
import { globeCamera } from '../camera/Rig';
import { sceneIndex } from '../globe/Scene';
import { DOMAIN_PALETTE } from '../globe/shaders';
import { Reticle } from './Reticle';
import { Intro } from './Intro';
import { RepoDetailPanel } from './RepoDetailPanel';
import { SearchBox } from './SearchBox';
import { group } from './num';
import { AnimatePresence } from 'framer-motion';

function rgb(i: number) {
  const c = DOMAIN_PALETTE[i % DOMAIN_PALETTE.length];
  return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
}

function useFps() {
  const [fps, setFps] = useState(0);
  useEffect(() => {
    let frames = 0;
    let last = performance.now();
    let raf = 0;
    const tick = () => {
      frames++;
      const now = performance.now();
      if (now - last >= 500) {
        setFps((frames / (now - last)) * 1000);
        frames = 0;
        last = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return fps;
}

/**
 * Chrome is deliberately thin.
 *
 * The previous HUD put two dense panels over the scene, which is why it read as
 * a debug tool. Populous keeps almost nothing on screen: a wordmark, a row of
 * category tabs, and the work itself. Everything diagnostic now lives behind a
 * toggle, because it is for us, not for whoever we show this to.
 */
export function Hud() {
  const entered = useGlobeStore((s) => s.entered);
  const activeDomain = useGlobeStore((s) => s.activeDomain);
  const showTelemetry = useGlobeStore((s) => s.showTelemetry);

  const domains = sceneIndex.manifest?.domains ?? [];

  const flyToDomain = (index: number) => {
    const store = useGlobeStore.getState();
    if (index === store.activeDomain) {
      store.setActiveDomain(-1);
      void globeCamera.reset();
      return;
    }
    store.setActiveDomain(index);
    const clusters = (sceneIndex.manifest?.clusters ?? []).filter((c) => c.domain === index);
    if (clusters.length === 0) return;
    const dirs = clusters.map((c) => {
      const st = Math.sin(c.theta);
      return new THREE.Vector3(st * Math.cos(c.phi), Math.cos(c.theta), st * Math.sin(c.phi));
    });
    void globeCamera.flyToDirections(dirs, { padding: 0.12 });
  };

  return (
    <div className="hud">
      <Intro />
      {entered && <Reticle />}

      <header className={`topbar${entered ? '' : ' topbar--hidden'}`}>
        <button
          className="wordmark"
          onClick={() => {
            useGlobeStore.getState().setActiveDomain(-1);
            useGlobeStore.getState().setSelected(-1);
            void globeCamera.reset();
          }}
        >
          GitGlobe
        </button>

        <nav className="tabs">
          <button
            className={activeDomain === -1 ? 'is-active' : undefined}
            onClick={() => {
              useGlobeStore.getState().setActiveDomain(-1);
              void globeCamera.reset();
            }}
          >
            All
          </button>
          {domains.map((d, i) => (
            <button
              key={d}
              className={activeDomain === i ? 'is-active' : undefined}
              onClick={() => flyToDomain(i)}
              style={activeDomain === i ? { color: rgb(i), borderBottomColor: rgb(i) } : undefined}
            >
              {d}
            </button>
          ))}
        </nav>

        <SearchBox />

        <button
          className="topbar__toggle"
          onClick={() => useGlobeStore.getState().setShowTelemetry(!showTelemetry)}
          aria-pressed={showTelemetry}
        >
          {showTelemetry ? '↳ hide stats' : '↳ stats'}
        </button>
      </header>

      <AnimatePresence mode="wait">
        {entered && showTelemetry && <Telemetry key="telemetry" />}
      </AnimatePresence>

      <AnimatePresence>
        {entered && <RepoDetailPanel />}
      </AnimatePresence>

      {entered && <CursorReadout />}
    </div>
  );
}

function CursorReadout() {
  const selectedId = useGlobeStore((s) => s.selectedId);
  const hoveredId = useGlobeStore((s) => s.hoveredId);

  if (selectedId !== -1) return null;

  // Two hints, one per input model, switched in CSS on `(hover: none)`. A phone
  // has no scroll wheel and no hover, so the desktop copy was instructing touch
  // users to do two things their device cannot do.
  return (
    <div className="readout">
      {hoveredId >= 0 ? (
        <span className="muted">↳ click to pin this node and hold its connections</span>
      ) : (
        <>
          <span className="muted readout__pointer">↳ drag to orbit · scroll to zoom · hover a node</span>
          <span className="muted readout__touch">↳ drag to orbit · pinch to zoom · tap a node</span>
        </>
      )}
    </div>
  );
}

const SHOW_BENCHMARK = true;

import { useShallow } from 'zustand/react/shallow';
import { motion } from 'framer-motion';

function Telemetry() {
  const fps = useFps();
  const {
    totalPoints,
    loadedBands,
    graphReady,
    ambientArcCount,
    focusArcCount,
    tier,
    sizeScale,
    autoRotate,
    showAmbientArcs,
    benchRunning,
    benchResults,
  } = useGlobeStore(
    useShallow((s) => ({
      totalPoints: s.totalPoints,
      loadedBands: s.loadedBands,
      graphReady: s.graphReady,
      ambientArcCount: s.ambientArcCount,
      focusArcCount: s.focusArcCount,
      tier: s.tier,
      sizeScale: s.sizeScale,
      autoRotate: s.autoRotate,
      showAmbientArcs: s.showAmbientArcs,
      benchRunning: s.benchRunning,
      benchResults: s.benchResults,
    }))
  );

  const graphMeta = sceneIndex.manifest?.graph;
  const latest = benchResults[0];

  return (
    <motion.section 
      className="telemetry"
      initial={{ opacity: 0, x: -20, filter: 'blur(4px)' }}
      animate={{ opacity: 1, x: 0, filter: 'blur(0px)', transition: { type: 'spring' as const, damping: 20, stiffness: 100 } }}
      exit={{ opacity: 0, x: -10, filter: 'blur(2px)', transition: { duration: 0.2 } }}
    >
      <span className="telemetry__bracket telemetry__bracket--tl" />
      <span className="telemetry__bracket telemetry__bracket--br" />

      <dl className="stats">
        <dt>nodes</dt>
        <dd>{group(totalPoints)}</dd>
        <dt>bands</dt>
        <dd>{loadedBands.join(' ') || '—'}</dd>
        <dt>edges</dt>
        <dd>{graphMeta ? group(graphMeta.directedEdges) : '—'}</dd>
        <dt>backbone</dt>
        <dd>{graphReady ? group(ambientArcCount) : '—'}</dd>
        <dt>focus arcs</dt>
        <dd className={focusArcCount > 0 ? 'live' : undefined}>{focusArcCount || '—'}</dd>
        <dt>tier</dt>
        <dd>{tier}</dd>
        <dt>fps</dt>
        <dd className={fps > 0 && fps < 55 ? 'warn' : 'live'}>{fps.toFixed(0)}</dd>
      </dl>

      {graphMeta && (
        <p className="note">
          pagerank d=0.85 · {graphMeta.pagerank.iterations} iter ·{' '}
          {graphMeta.pagerank.converged ? 'converged' : 'CAPPED'}
          <br />
          degree med {graphMeta.degree.p50} · p99 {graphMeta.degree.p99} · max{' '}
          {group(graphMeta.degree.max)}
        </p>
      )}

      <label className="control">
        <span>
          node size <em>{sizeScale}</em>
        </span>
        <input
          type="range"
          min={8}
          max={120}
          step={2}
          value={sizeScale}
          onChange={(e) => useGlobeStore.getState().setSizeScale(Number(e.target.value))}
        />
      </label>

      <label className="control control--check">
        <input
          type="checkbox"
          checked={showAmbientArcs}
          onChange={(e) => useGlobeStore.getState().setShowAmbientArcs(e.target.checked)}
        />
        <span>backbone web</span>
      </label>

      <label className="control control--check">
        <input
          type="checkbox"
          checked={autoRotate}
          onChange={(e) => useGlobeStore.getState().setAutoRotate(e.target.checked)}
        />
        <span>auto-rotate</span>
      </label>

      {SHOW_BENCHMARK && (
        <>
          <button
            className="bench"
            disabled={benchRunning || totalPoints === 0}
            onClick={() => useGlobeStore.getState().setBenchRunning(true)}
          >
            {benchRunning ? '◈ measuring — 12s' : '◈ run benchmark'}
          </button>

          {latest && (
            <div className={`bench-result ${latest.passed ? 'pass' : 'fail'}`}>
              <strong>{latest.passed ? 'PASS' : 'FAIL'}</strong>
              <span>
                {latest.dropped}/{latest.sampled} frames dropped ({(latest.dropRatio * 100).toFixed(2)}%)
              </span>
              <span>
                {latest.refreshHz.toFixed(0)}Hz display · worst {latest.worst.toFixed(0)}ms
              </span>
              <span className="headroom">
                {latest.headroom}× headroom · {latest.drawCalls} calls ·{' '}
                {(latest.triangles / 1000).toFixed(0)}k tris
              </span>
              <span className="muted">gate: &lt;1% dropped, no frame ≥ 100ms</span>
            </div>
          )}
        </>
      )}
    </motion.section>
  );
}
