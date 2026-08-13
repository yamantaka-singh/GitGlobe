import { useEffect, useState } from 'react';
import { useGlobeStore } from '../store/useGlobeStore';
import { globeCamera } from '../camera/Rig';
import { group } from './num';

function TypewriterName({ firstName, lastName }: { firstName: string; lastName: string }) {
  const [first, setFirst] = useState('');
  const [last, setLast] = useState('');

  useEffect(() => {
    let i = 0;
    const totalFirst = firstName.length;
    const totalLast = lastName.length;

    setFirst('');
    setLast('');

    const timer = setInterval(() => {
      if (i <= totalFirst) {
        setFirst(firstName.slice(0, i));
      } else if (i <= totalFirst + totalLast) {
        setLast(lastName.slice(0, i - totalFirst));
      } else {
        clearInterval(timer);
      }
      i++;
    }, 70);

    return () => clearInterval(timer);
  }, [firstName, lastName]);

  const isDoneFirst = first.length === firstName.length;

  return (
    <span className="typewriter-name">
      <span className="typewriter-line">
        {first}
        {!isDoneFirst && <span className="typewriter-cursor">|</span>}
      </span>
      <br />
      <span className="typewriter-line">
        {last}
        {isDoneFirst && <span className="typewriter-cursor">|</span>}
      </span>
    </span>
  );
}

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

        <div className="intro__team">
          <div className="intro__team-member">
            <a href="https://github.com/ydabas-hue" target="_blank" rel="noreferrer">
              <img src="/maki.jpg" alt="Yashasvi" className="intro__team-avatar" />
            </a>
            <div className="intro__team-info">
              <div className="intro__team-name">
                <TypewriterName firstName="Yashasvi" lastName="Dabas" />
              </div>
              <div className="intro__team-badges">
                <a href="https://github.com/ydabas-hue" target="_blank" rel="noreferrer" className="btn-icon-glossy" title="GitHub">
                  <img src="https://img.icons8.com/?size=100&id=4MhUS4CzoLbx&format=png" alt="GitHub" />
                </a>
                <a href="https://www.linkedin.com/in/yashasvi-the-boss" target="_blank" rel="noreferrer" className="btn-icon-glossy" title="LinkedIn">
                  <img src="https://img.icons8.com/?size=100&id=X8g2OZMx4ET5&format=png" alt="LinkedIn" />
                </a>
              </div>
            </div>
          </div>
          <div className="intro__team-member">
            <a href="https://github.com/yamantaka-singh" target="_blank" rel="noreferrer">
              <img src="/toji.jpg" alt="Mrityunjay" className="intro__team-avatar" />
            </a>
            <div className="intro__team-info">
              <div className="intro__team-name">
                <TypewriterName firstName="Mrityunjay" lastName="Singh" />
              </div>
              <div className="intro__team-badges">
                <a href="https://github.com/yamantaka-singh" target="_blank" rel="noreferrer" className="btn-icon-glossy" title="GitHub">
                  <img src="https://img.icons8.com/?size=100&id=4MhUS4CzoLbx&format=png" alt="GitHub" />
                </a>
                <a href="https://www.linkedin.com/in/yamantakasingh" target="_blank" rel="noreferrer" className="btn-icon-glossy" title="LinkedIn">
                  <img src="https://img.icons8.com/?size=100&id=X8g2OZMx4ET5&format=png" alt="LinkedIn" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
