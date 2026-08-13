import { useShallow } from 'zustand/react/shallow';
import { useGlobeStore } from '../store/useGlobeStore';
import { sceneIndex } from '../globe/Scene';
import { repoIdentity } from '../repo/names';
import { rarity, scoresFor } from '../repo/scores';
import { DOMAIN_PALETTE } from '../globe/shaders';
import { degreeOf } from '../graph/format';
import { group } from './num';

function rgb(i: number) {
  const c = DOMAIN_PALETTE[i % DOMAIN_PALETTE.length];
  return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
}

const CARD_W = 232;
const CARD_H = 118;
const OFFSET = 26;

/**
 * Corner brackets locked onto the hovered node, plus a leader line to a card.
 *
 * Drawn in the HUD layer rather than in the scene: screen-space UI belongs in
 * the DOM, where it gets crisp text rendering, real fonts, and selectable
 * content for free — none of which a canvas-drawn label has.
 */
export function Reticle() {
  const anchor = useGlobeStore(useShallow((s) => s.anchor));
  const hoveredId = useGlobeStore((s) => s.hoveredId);
  const selectedId = useGlobeStore((s) => s.selectedId);

  const ref = sceneIndex.resolve(hoveredId);
  if (!ref || !anchor.visible) return null;

  const identity = repoIdentity(ref.repoId, ref.domain);
  const domains = sceneIndex.manifest?.domains ?? [];
  const graph = sceneIndex.graph;

  const degree = graph ? degreeOf(graph, hoveredId) : { in: 0, out: 0 };
  // Same measured global scale as the detail panel. This card used to show an
  // in-corpus PageRank percentile labelled "top X%", which was both a different
  // number and a different meaning from the panel one click later.
  const scores = scoresFor(hoveredId);
  const colour = rgb(ref.domain);

  // Flip the card to whichever side has room, so it never runs off the edge.
  const flipX = anchor.x + OFFSET + CARD_W > window.innerWidth;
  const flipY = anchor.y + OFFSET + CARD_H > window.innerHeight;
  const cardX = flipX ? anchor.x - OFFSET - CARD_W : anchor.x + OFFSET;
  const cardY = flipY ? anchor.y - OFFSET - CARD_H : anchor.y + OFFSET;

  return (
    <div className="reticle-root" aria-hidden="true">
      <svg className="reticle-svg">
        <line
          x1={anchor.x}
          y1={anchor.y}
          x2={flipX ? cardX + CARD_W : cardX}
          y2={cardY + (flipY ? CARD_H : 0)}
          stroke={colour}
          strokeWidth={1}
          strokeOpacity={0.5}
        />
      </svg>

      {/* A rotated square, as in the reference. Reads as a surveyed position
          rather than a UI cursor, and stays legible over bright terrain. */}
      <div className="reticle-pin" style={{ left: anchor.x, top: anchor.y }}>
        <span className="reticle-pin__diamond" style={{ borderColor: colour }} />
        <span className="reticle-pin__core" />
      </div>

      <div className="reticle-card" style={{ left: cardX, top: cardY, borderColor: colour }}>
        <div className="reticle-card__name">
          <span className="dot" style={{ background: colour }} />
          {identity.name}
        </div>
        <div className="reticle-card__org">{identity.org}</div>
        <dl className="reticle-card__stats">
          <dt>domain</dt>
          <dd style={{ color: colour }}>{domains[ref.domain] ?? '—'}</dd>
          <dt>rank</dt>
          <dd>{rarity(scores.starRank)}</dd>
          {scores.score !== undefined && (
            <>
              <dt>score</dt>
              <dd>{scores.score.toFixed(1)}</dd>
            </>
          )}
          <dt>dependents</dt>
          <dd>{group(degree.in)}</dd>
          <dt>depends on</dt>
          <dd>{group(degree.out)}</dd>
        </dl>
        {selectedId === hoveredId && <div className="reticle-card__pinned">↳ selected</div>}
      </div>
    </div>
  );
}
