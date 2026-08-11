import { useGlobeStore } from '../store/useGlobeStore';
import { sceneIndex } from '../globe/Scene';
import { repoIdentity, findGlobalIdByName } from '../repo/names';
import { describeRank, scoresFor } from '../repo/scores';
import { DOMAIN_PALETTE } from '../globe/shaders';
import { degreeOf, neighboursOf, EDGE_SIMILAR_TO } from '../graph/format';
import { globeCamera } from '../camera/Rig';
import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { group } from './num';

function rgb(i: number) {
  const c = DOMAIN_PALETTE[i % DOMAIN_PALETTE.length];
  return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
}

/**
 * All hooks are called unconditionally at the top of this component,
 * before any early returns — this is required by React's Rules of Hooks.
 */
export function RepoDetailPanel() {
  // ---- hooks (unconditional, always in the same order) ----------------------
  const selectedId = useGlobeStore((s) => s.selectedId);
  const setSelected = useGlobeStore((s) => s.setSelected);
  const focusArcCount = useGlobeStore((s) => s.focusArcCount);

  const ref = selectedId >= 0 ? sceneIndex.resolve(selectedId) : null;
  const repoId = ref?.repoId ?? 0;
  const domain = ref?.domain ?? 0;
  // Get the name from the tile names file — needed for GitHub API fallback
  const tileName = ref ? repoIdentity(repoId, domain).fullName : '';

  // Fetch real metadata from the API. Pass the repo name so the backend can
  // fall back to GitHub's public API for repos not in our database.
  const { data, isLoading } = useQuery({
    queryKey: ['repo', repoId, tileName],
    queryFn: async () => {
      const params = tileName ? `?name=${encodeURIComponent(tileName)}` : '';
      const res = await fetch(`http://localhost:8000/repo/${repoId}${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    enabled: !!ref && repoId > 0,
    retry: false,
    staleTime: 5 * 60_000,
  });

  const [showList, setShowList] = useState<'dependents' | 'dependencies' | 'alternatives' | null>(null);

  const { data: graphData, isLoading: isGraphLoading } = useQuery({
    queryKey: ['graph', repoId, tileName],
    queryFn: async () => {
      const params = tileName ? `?name=${encodeURIComponent(tileName)}` : '';
      const res = await fetch(`http://localhost:8000/graph/${repoId}${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    enabled: !!ref && repoId > 0 && showList !== null,
    retry: false,
    staleTime: 5 * 60_000,
  });

  useEffect(() => {
    setShowList(null);
  }, [selectedId]);

  // ---- early exits (after all hooks) ----------------------------------------
  if (!ref || selectedId < 0) return null;

  // ---- derive display values ------------------------------------------------
  // Prefer the API's full_name (real GitHub name) over the procedural generator.
  const apiName = data?.full_name;
  const identity = apiName
    ? (() => {
        const slash = apiName.indexOf('/');
        return slash > 0
          ? { org: apiName.slice(0, slash), name: apiName.slice(slash + 1), fullName: apiName }
          : { org: '', name: apiName, fullName: apiName };
      })()
    : repoIdentity(repoId, domain);

  const domains = sceneIndex.manifest?.domains ?? [];
  const graph = sceneIndex.graph;
  const degree = graph ? degreeOf(graph, selectedId) : { in: 0, out: 0 };
  // Global standing, measured against GitHub's real star distribution rather
  // than against the 87k repositories we happen to hold. The old line here was
  // an in-corpus PageRank percentile labelled only "Top X%" — but every repo in
  // this corpus already clears ~66 stars, which is the top 0.2% of GitHub, so
  // that number understated standing by roughly 500x while reading as global.
  const scores = scoresFor(selectedId);

  const domainIndex = data?.domain != null ? data.domain : domain;
  const colour = rgb(domainIndex);
  const domainName = domains[domainIndex] ?? '—';
  const url = `https://github.com/${identity.fullName}`;

  // Display values — real when loaded, placeholder text while fetching
  const description = isLoading ? 'Loading…' : (data?.description || '—');
  const stars = isLoading ? '…' : (data?.stars !== undefined ? group(data.stars) : '0');
  const language = isLoading ? '…' : (data?.language || 'Unknown');

  const dependentsList = graphData?.edges
    .filter((e: any) => e.target === identity.fullName)
    .map((e: any) => e.source) || [];

  const dependenciesList = graphData?.edges
    .filter((e: any) => e.source === identity.fullName)
    .map((e: any) => e.target) || [];

  const alternativesList = graph
    ? neighboursOf(graph, selectedId, 200)
        .filter((n) => n.kind === EDGE_SIMILAR_TO)
        .map((n) => {
          const ref = sceneIndex.resolve(n.node);
          return ref ? repoIdentity(n.node, ref.domain).fullName : '';
        })
        .filter(Boolean)
    : [];

  const activeList = 
    showList === 'dependents' ? dependentsList : 
    showList === 'dependencies' ? dependenciesList : 
    alternativesList;

  const onSelectNode = (fullName: string) => {
    const globalId = findGlobalIdByName(fullName);
    if (globalId >= 0) {
      setSelected(globalId);
      setShowList(null); // Reset when navigating
      const newRef = sceneIndex.resolve(globalId);
      if (newRef) {
        void globeCamera.flyToDirections([newRef.direction], { padding: 0.05 });
      }
    }
  };

  return (
    <div className="detail-panel" role="dialog" aria-label="Repository Details">
      <button
        className="detail-panel__close"
        onClick={() => setSelected(-1)}
        aria-label="Close details"
      >
        ×
      </button>

      <div className="detail-panel__header">
        <span className="dot" style={{ background: colour }} />
        <h2 className="detail-panel__title">{identity.name}</h2>
      </div>
      <div className="detail-panel__org">{identity.org}</div>

      <p className="detail-panel__desc">{description}</p>

      <div className="detail-panel__meta" style={{ color: '#FFFFFF' }}>
        <span className="meta-item">
          <span className="meta-icon" style={{ color: '#FACC15' }}>★</span>{' '}
          {stars}
        </span>
        <span className="meta-item">
          <span className="meta-icon" style={{ color: colour }}>●</span>{' '}
          {language}
        </span>
      </div>

      <dl className="detail-panel__stats">
        <dt>Domain</dt>
        <dd style={{ color: '#FFFFFF' }}>{domainName}</dd>
        <dt>Global rank</dt>
        <dd style={{ color: '#FFFFFF' }}>{describeRank(scores.starRank)}</dd>
        {scores.score !== undefined && (
          <>
            <dt>Score</dt>
            <dd style={{ color: '#FFFFFF' }}>
              {scores.score.toFixed(1)}
              <span style={{ opacity: 0.55 }}> / 100</span>
            </dd>
          </>
        )}
        {scores.brain !== undefined && (
          <>
            <dt>Quality</dt>
            <dd style={{ color: '#FFFFFF' }}>
              {scores.brain.toFixed(1)}
              <span style={{ opacity: 0.55 }}> / 100</span>
            </dd>
          </>
        )}
        <dt>Dependents</dt>
        <dd style={{ color: '#FFFFFF' }}>
          <button 
            className="hud-link" 
            onClick={() => setShowList(showList === 'dependents' ? null : 'dependents')}
            style={{ textDecoration: 'underline', background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: 0 }}
          >
            {group(degree.in)}
          </button>
        </dd>
        <dt>Dependencies</dt>
        <dd style={{ color: '#FFFFFF' }}>
          <button 
            className="hud-link" 
            onClick={() => setShowList(showList === 'dependencies' ? null : 'dependencies')}
            style={{ textDecoration: 'underline', background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: 0 }}
          >
            {group(degree.out)}
          </button>
        </dd>
        <dt>Alternatives</dt>
        <dd style={{ color: '#FFFFFF' }}>
          <button 
            className="hud-link" 
            onClick={() => setShowList(showList === 'alternatives' ? null : 'alternatives')}
            style={{ textDecoration: 'underline', background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: 0 }}
          >
            {group(alternativesList.length)}
          </button>
        </dd>
        <dt>Focus Arcs</dt>
        <dd style={{ color: '#FFFFFF' }}>{focusArcCount}</dd>
      </dl>

      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="detail-panel__link"
      >
        View on GitHub
      </a>

      {showList && (
        <div className="detail-panel__list-container" style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: '0.5rem' }}>
          <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', color: '#888' }}>
            {showList === 'dependents' ? 'Dependents' : showList === 'dependencies' ? 'Dependencies' : 'Alternatives'}
          </h3>
          <div style={{ maxHeight: '180px', overflowY: 'auto' }}>
            {isGraphLoading && showList !== 'alternatives' && <div style={{ color: '#888', fontSize: '0.8rem' }}>Loading...</div>}
            {!isGraphLoading && activeList.length === 0 && <div style={{ color: '#888', fontSize: '0.8rem' }}>No edges found.</div>}
            {!isGraphLoading && activeList.map((repoName: string) => (
              <button 
                key={repoName} 
                onClick={() => onSelectNode(repoName)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  background: 'none',
                  border: 'none',
                  color: '#fff',
                  cursor: 'pointer',
                  padding: '4px 0',
                  fontSize: '0.875rem'
                }}
              >
                {repoName}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

