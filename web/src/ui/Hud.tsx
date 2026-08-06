import { useEffect, useState } from 'react';
import * as THREE from 'three';

import { useGlobeStore } from '../store/useGlobeStore';
import { globeCamera } from '../camera/Rig';
import { sceneIndex } from '../globe/Scene';
import { DOMAIN_PALETTE } from '../globe/shaders';

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

export function Hud() {
  const fps = useFps();
  const {
    totalPoints,
    loadedBands,
    loadError,
    hoveredId,
    selectedId,
    tier,
    sizeScale,
    autoRotate,
    benchRunning,
    benchResults,
  } = useGlobeStore();

  const hovered = sceneIndex.resolve(hoveredId);
  const selected = sceneIndex.resolve(selectedId);
  const domains = sceneIndex.manifest?.domains ?? [];

  const flyRandomCluster = () => {
    const clusters = sceneIndex.manifest?.clusters ?? [];
    if (clusters.length === 0) return;
    const c = clusters[Math.floor(Math.random() * clusters.length)];
    void globeCamera.flyToSpherical(c.theta, c.phi);
  };

  const flyToDomain = (domainIndex: number) => {
    const clusters = (sceneIndex.manifest?.clusters ?? []).filter((c) => c.domain === domainIndex);
    if (clusters.length === 0) return;
    // Frame every cluster in the domain at once — the same centroid-and-spread
    // path the agent will use in Phase 5.
    const dirs = clusters.map((c) => {
      const st = Math.sin(c.theta);
      return new THREE.Vector3(st * Math.cos(c.phi), Math.cos(c.theta), st * Math.sin(c.phi));
    });
    void globeCamera.flyToDirections(dirs, { padding: 0.1 });
  };

  const latest = benchResults[0];

  return (
    <div className="hud">
      <div className="panel panel--left">
        <h1>
          GitGlobe <span className="tag">phase 0</span>
        </h1>

        {loadError ? (
          <p className="error">{loadError}</p>
        ) : (
          <dl className="stats">
            <dt>points</dt>
            <dd>{totalPoints.toLocaleString()}</dd>
            <dt>bands</dt>
            <dd>{loadedBands.join(', ') || '—'}</dd>
            <dt>tier</dt>
            <dd>{tier}</dd>
            <dt>fps</dt>
            <dd className={fps > 0 && fps < 55 ? 'warn' : undefined}>{fps.toFixed(0)}</dd>
          </dl>
        )}

        <div className="row">
          <button onClick={flyRandomCluster}>Fly to a cluster</button>
          <button onClick={() => void globeCamera.reset()}>Reset view</button>
        </div>

        <label className="control">
          <span>point size</span>
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
            checked={autoRotate}
            onChange={(e) => useGlobeStore.getState().setAutoRotate(e.target.checked)}
          />
          <span>auto-rotate</span>
        </label>

        <button
          className="bench"
          disabled={benchRunning || totalPoints === 0}
          onClick={() => useGlobeStore.getState().setBenchRunning(true)}
        >
          {benchRunning ? 'Measuring… 10s' : 'Run 10s benchmark'}
        </button>

        {latest && (
          <div className={`bench-result ${latest.passed ? 'pass' : 'fail'}`}>
            <strong>{latest.passed ? 'PASS' : 'FAIL'}</strong>
            <span>
              p50 {latest.p50.toFixed(1)}ms · p95 {latest.p95.toFixed(1)}ms · p99 {latest.p99.toFixed(1)}ms
            </span>
            <span>
              {latest.fps.toFixed(0)} fps avg · worst {latest.worst.toFixed(0)}ms · {latest.drawCalls} draw calls
            </span>
            <span className="muted">exit criterion: p95 ≤ 16.7ms and no frame ≥ 100ms</span>
          </div>
        )}
      </div>

      <div className="panel panel--right">
        <h2>Domains</h2>
        <ul className="legend">
          {domains.map((d, i) => (
            <li key={d}>
              <button onClick={() => flyToDomain(i)}>
                <i style={{ background: rgb(i) }} />
                {d}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="readout">
        {hovered ? (
          <>
            <i style={{ background: rgb(hovered.domain) }} />
            repo #{hovered.repoId} · {domains[hovered.domain] ?? 'unknown'} · band {hovered.band}
          </>
        ) : (
          <span className="muted">hover a point — drag to orbit, scroll to zoom, click to fly</span>
        )}
        {selected && <span className="muted"> · selected #{selected.repoId}</span>}
      </div>
    </div>
  );
}
