import { Suspense, useEffect, useRef, useState, type ComponentType } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

import { SPACE } from './globe/palette';

import { Scene } from './globe/Scene';
import { Hud } from './ui/Hud';

const DEV = import.meta.env.DEV;

// Set to true when performance debugging is needed
const SHOW_PERF = false;

/** Automatic rebuilds allowed after a lost WebGL context, before giving up. */
const MAX_GL_RECOVERIES = 2;

/**
 * Renderer config is a decision, not a default (web3d-scene-architect Rule 3):
 *
 *  - `dpr={[1, 2]}` — uncapped devicePixelRatio on a 3x display is 9x the
 *    fragments. The device tier tightens this further at runtime.
 *  - `antialias: false` — MSAA does nothing useful for GL_POINTS (the shader
 *    already produces a soft radial edge) and costs real fill rate.
 *  - `alpha: false` — the canvas covers the viewport and paints its own
 *    background, so there is no blend pass to pay for.
 *  - `frameloop="always"` — there is continuous idle rotation and a benchmark
 *    that must record real frame deltas. "demand" would produce the classic
 *    "it only animates when I move the mouse" bug.
 *  - `depth: true`, near/far kept tight — a 0.01..100 range on a unit sphere
 *    keeps depth precision high enough that the core sphere occludes cleanly.
 */
/**
 * Is WebGL actually available?
 *
 * Roughly 8% of sessions cannot render a canvas — old hardware, blocklisted
 * drivers, hardware acceleration switched off, some locked-down enterprise
 * browsers. Without this check they get a black rectangle and no explanation,
 * which is indistinguishable from the site being broken.
 */
function hasWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

export function App() {
  const wrapper = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);
  const [webgl] = useState(hasWebGL);
  const [contextLost, setContextLost] = useState(false);
  // Bumping this remounts the canvas, which is what actually rebuilds the GPU
  // resources a context loss destroyed.
  const [glKey, setGlKey] = useState(0);
  // If the context keeps dying the device is genuinely out of memory, and
  // rebuilding 198,731 points on every restore would just lose it again in a
  // loop. After a couple of attempts, stop and leave the notice up.
  const recoveries = useRef(0);

  // Stop rendering when the canvas is off-screen or the tab is backgrounded.
  // Free performance, and the difference between a page that drains a laptop
  // battery and one that doesn't.
  useEffect(() => {
    const el = wrapper.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0 });
    io.observe(el);
    const onVisibility = () => setVisible(!document.hidden && document.visibilityState === 'visible');
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      io.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  if (!webgl) {
    return (
      <div className="app">
        <div className="app__fallback" role="alert">
          <h1>GitGlobe needs WebGL</h1>
          <p>
            This browser can't render the globe. It usually means hardware acceleration is
            switched off, or the GPU driver is blocked.
          </p>
          <p className="muted">
            Try enabling hardware acceleration in your browser settings, or open the site in a
            recent version of Chrome, Firefox, Edge, or Safari.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app" ref={wrapper}>
      <Canvas
        key={glKey}
        frameloop={visible ? 'always' : 'never'}
        dpr={[1, 2]}
        gl={{
          antialias: false,
          alpha: false,
          depth: true,
          stencil: false,
          powerPreference: 'high-performance',
          toneMapping: THREE.NoToneMapping,
        }}
        camera={{ fov: 40, near: 0.0001, far: 100, position: [0, 0.7, 2.6] }}
        onCreated={({ gl }) => {
          // Not quite black. Pure #000 makes the atmosphere rim look like a
          // sticker cut out and pasted on; a few points of blue give it
          // somewhere to fall off to.
          gl.setClearColor(new THREE.Color(...SPACE), 1);

          // The browser drops the WebGL context on memory pressure and — most
          // commonly — when iOS Safari backgrounds the tab for a while. The
          // default action of `webglcontextlost` is to make the loss permanent:
          // without preventDefault the context is never restored, and the user
          // returns from another app to a black rectangle that only a manual
          // reload fixes. Calling preventDefault is what makes the browser
          // willing to hand the context back.
          const canvas = gl.domElement;
          // 3D content is opaque to assistive technology — a screen reader
          // finds an unlabelled graphics element it cannot describe. Everything
          // meaningful (search, domain tabs, the detail panel) already exists in
          // the DOM, so hide the canvas rather than announcing a black box.
          canvas.setAttribute('aria-hidden', 'true');

          // Disposing the previous renderer during a remount fires
          // `webglcontextlost` on the canvas being thrown away. That event is
          // about a dead element, not the live one, and acting on it put the
          // notice straight back up over a scene that had already recovered.
          const isStale = () => !canvas.isConnected;

          const onLost = (e: Event) => {
            e.preventDefault();
            if (isStale()) return;
            setContextLost(true);
          };

          // Restoring the context is not the same as restoring the scene.
          //
          // Losing the context destroys every GPU resource with it — the point
          // buffers, the baked planet texture, the compiled shaders. three.js
          // does not re-upload them for an existing scene graph, so clearing
          // the overlay on `webglcontextrestored` only revealed an empty globe
          // and the message's "reload the page" was the sole real cure.
          // Remounting the canvas rebuilds all of it, which is the reload,
          // minus the user having to know to do it.
          const onRestored = () => {
            if (isStale()) return;
            if (recoveries.current >= MAX_GL_RECOVERIES) return; // leave the notice up
            recoveries.current += 1;
            setContextLost(false);
            setGlKey((k) => k + 1);
          };
          canvas.addEventListener('webglcontextlost', onLost);
          canvas.addEventListener('webglcontextrestored', onRestored);
        }}
      >
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
        {DEV && SHOW_PERF && <PerfOverlay />}
      </Canvas>
      {contextLost && (
        <div className="app__fallback" role="status">
          <p>Rendering paused — restoring the globe…</p>
          <p className="muted">
            If this doesn't clear in a few seconds, reload the page.
          </p>
        </div>
      )}
      <Hud />
    </div>
  );
}

/**
 * r3f-perf is a dev dependency and must never reach the production bundle.
 * Lazy-importing it inside a DEV guard means the tree-shaken build has no
 * reference to it at all.
 */
function PerfOverlay() {
  const [Comp, setComp] = useState<ComponentType<{ position?: string }> | null>(null);
  useEffect(() => {
    let alive = true;
    import('r3f-perf')
      .then((m) => alive && setComp(() => m.Perf as unknown as ComponentType<{ position?: string }>))
      .catch(() => {
        /* optional dependency */
      });
    return () => {
      alive = false;
    };
  }, []);
  return Comp ? <Comp position="top-right" /> : null;
}
