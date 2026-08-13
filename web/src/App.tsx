import { Suspense, useEffect, useRef, useState, type ComponentType } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

import { SPACE } from './globe/palette';

import { Scene } from './globe/Scene';
import { Hud } from './ui/Hud';

const DEV = import.meta.env.DEV;

// Set to true when performance debugging is needed
const SHOW_PERF = false;

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
export function App() {
  const wrapper = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);

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

  return (
    <div className="app" ref={wrapper}>
      <Canvas
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
        }}
      >
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
        {DEV && SHOW_PERF && <PerfOverlay />}
      </Canvas>
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
