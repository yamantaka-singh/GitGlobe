import { useEffect, useState } from 'react';
import { useGlobeStore } from '../store/useGlobeStore';

/**
 * First-run walkthrough.
 *
 * Nothing about a globe of 198,731 dots explains itself: the domain strip, the
 * fact that position means capability, and that a tap opens a real repository
 * are all invisible until someone says so.
 *
 * Two rules from the brief, both about not being annoying:
 *  - it must be skippable at any point;
 *  - it must never appear unbidden on a return visit. Seen-ness is persisted,
 *    and after that the only way back in is the "?" button in the topbar.
 */
const SEEN_KEY = 'gitglobe.tutorialSeen';

/** localStorage throws in Safari private mode — never let that break the app. */
function markSeen() {
  try {
    localStorage.setItem(SEEN_KEY, '1');
  } catch {
    /* private mode: the tutorial simply shows again next time */
  }
}

export function hasSeenTutorial() {
  try {
    return localStorage.getItem(SEEN_KEY) === '1';
  } catch {
    // Unreadable storage is treated as "already seen": showing the walkthrough
    // on every single load is far more irritating than never showing it.
    return true;
  }
}

const STEPS = [
  {
    title: 'Position means capability',
    body: 'Every dot is a repository, placed by what it does rather than what it is called. Neighbours solve similar problems, so a crowded region is a crowded niche.',
  },
  {
    title: 'Move around',
    body: 'Drag to orbit. Pinch or scroll to zoom. The globe keeps spinning on its own until you touch it.',
  },
  {
    title: 'Open a repository',
    body: 'Tap a dot for its stars, language, description, and the projects that depend on it. Tap a dependency to travel there.',
  },
  {
    title: 'Narrow it down',
    body: 'The strip along the top filters to one domain. The search box finds a project by name and flies you to it.',
  },
];

export function Tutorial() {
  const entered = useGlobeStore((s) => s.entered);
  const open = useGlobeStore((s) => s.tutorialOpen);
  const setOpen = useGlobeStore((s) => s.setTutorialOpen);
  const [step, setStep] = useState(0);

  // Auto-open once, and only for someone who has never seen it. Waiting for
  // `entered` keeps it from stacking on top of the intro.
  useEffect(() => {
    if (entered && !hasSeenTutorial()) setOpen(true);
  }, [entered, setOpen]);

  const close = () => {
    markSeen();
    setOpen(false);
    setStep(0);
  };

  // Escape closes, matching every other dismissible surface here.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (!open) return null;

  const last = step === STEPS.length - 1;
  const s = STEPS[step];

  return (
    <div className="tutorial" role="dialog" aria-modal="false" aria-label="How GitGlobe works">
      <div className="tutorial__head">
        <span className="tutorial__count">
          {step + 1} / {STEPS.length}
        </span>
        <button className="tutorial__skip" onClick={close}>
          Skip
        </button>
      </div>

      <h2 className="tutorial__title">{s.title}</h2>
      <p className="tutorial__body">{s.body}</p>

      <div className="tutorial__dots" aria-hidden="true">
        {STEPS.map((_, i) => (
          <span key={i} className={i === step ? 'is-active' : undefined} />
        ))}
      </div>

      <div className="tutorial__actions">
        {step > 0 && (
          <button className="tutorial__back" onClick={() => setStep((i) => i - 1)}>
            Back
          </button>
        )}
        <button
          className="tutorial__next"
          onClick={() => (last ? close() : setStep((i) => i + 1))}
        >
          {last ? 'Explore' : 'Next'}
        </button>
      </div>
    </div>
  );
}
