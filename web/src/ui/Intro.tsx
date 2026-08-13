import { useEffect, useState } from 'react';
import { useGlobeStore } from '../store/useGlobeStore';
import { globeCamera } from '../camera/Rig';
import { group } from './num';

/**
 * The entry moment.
 *
 * Populous does not drop you into its globe — it shows a headline and a single
 * `↳ Start experience`, and the transition is the first thing you see it do.
 * That one beat is most of the difference between "a tool someone left running"
 * and "a product". It also buys real time: tiles and the 4.4 MB graph finish
 * loading behind the overlay instead of popping in under the cursor.
 */
export function Intro() {
  const entered = useGlobeStore((s) => s.entered);
  const totalPoints = useGlobeStore((s) => s.totalPoints);
  const graphReady = useGlobeStore((s) => s.graphReady);
  const loadError = useGlobeStore((s) => s.loadError);
  const [leaving, setLeaving] = useState(false);

  // Hold the globe still and far out until the user commits, so the reveal has
  // somewhere to travel from.
  useEffect(() => {
    if (entered) return;
    useGlobeStore.getState().setAutoRotate(true);
    void globeCamera.establish();
  }, [entered]);

  if (entered) return null;

  const ready = totalPoints > 0 && graphReady;

  const start = () => {
    setLeaving(true);
    void globeCamera.reset();
    // Matches the CSS fade, so the overlay is gone by the time the camera
    // arrives rather than lingering over a settled scene.
    window.setTimeout(() => useGlobeStore.getState().setEntered(true), 620);
  };

  return (
    <div className={`intro${leaving ? ' intro--leaving' : ''}`}>
      <div className="intro__inner">
        <p className="intro__eyebrow">↳ open source, spatially</p>
        <h1 className="intro__headline">
          Every repository
          <br />
          has neighbours.
        </h1>
        <p className="intro__body">
          Nearly two hundred thousand projects placed by what they do, not what they are called. Connected by
          dependency, weighted by PageRank, navigable by hand or by conversation.
        </p>

        {loadError ? (
          <p className="intro__error">{loadError}</p>
        ) : (
          <button className="intro__start" onClick={start} disabled={!ready}>
            {ready ? '↳ Start' : '↳ Loading the world…'}
          </button>
        )}

        <dl className="intro__meta">
          <div>
            <dt>nodes</dt>
            <dd>{totalPoints ? group(totalPoints) : '—'}</dd>
          </div>
          <div>
            <dt>graph</dt>
            <dd>{graphReady ? 'ready' : 'loading'}</dd>
          </div>
          <div>
            <dt>layout</dt>
            <dd>synthetic v2</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
