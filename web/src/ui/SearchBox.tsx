import { useState, useRef, useEffect } from 'react';
import { API } from '../api';
import { useQuery } from '@tanstack/react-query';
import { useGlobeStore } from '../store/useGlobeStore';
import { sceneIndex } from '../globe/Scene';
import { globeCamera } from '../camera/Rig';
import { findGlobalIdByName } from '../repo/names';

interface SearchResult {
  id: number; // repoId
  name: string;
  org: string;
  domain: number | null;
  description: string | null;
  score: number;
}

export function SearchBox() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounce the query for the API
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  // Highlighted row for keyboard navigation. -1 = the input itself.
  const [active, setActive] = useState(-1);

  // Dismiss on click outside, or on focus moving out of the widget entirely.
  // `mousedown` alone strands keyboard users: tabbing away left the dropdown
  // open over the globe with no way to close it.
  useEffect(() => {
    const onPointer = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    const onFocusIn = (e: FocusEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('focusin', onFocusIn);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('focusin', onFocusIn);
    };
  }, []);

  const { data: results, isLoading } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: async () => {
      if (!debouncedQuery.trim()) return [];
      const res = await fetch(`${API}/search?q=${encodeURIComponent(debouncedQuery)}&limit=5`);
      if (!res.ok) throw new Error('Search failed');
      const rawData = await res.json();
      const mapped: SearchResult[] = rawData.map((item: any) => {
        const repo = item.repo;
        const parts = (repo.full_name || '').split('/');
        const org = parts.length > 1 ? parts[0] : '';
        const name = parts.length > 1 ? parts[1] : (repo.full_name || '');
        const globalId = findGlobalIdByName(repo.full_name);
        
        return {
          id: globalId >= 0 ? globalId : repo.id,
          name,
          org,
          domain: repo.domain,
          description: repo.description,
          score: item.score
        };
      });
      return mapped.slice(0, 6);
    },
    enabled: debouncedQuery.trim().length > 0,
  });

  const onSelect = (repo: SearchResult) => {
    setIsOpen(false);
    
    const store = useGlobeStore.getState();
    store.setSelected(repo.id);

    // Try to resolve in the scene to fly to it
    const ref = sceneIndex.resolve(repo.id);
    if (ref) {
      void globeCamera.flyToDirections([ref.direction], { padding: 0.05 });
    }
  };

  const open = isOpen && debouncedQuery.trim().length > 0;
  const rows = results ?? [];

  // Arrow keys move the highlight, Enter commits it, Escape closes. Without
  // this the dropdown was reachable only by tabbing through every row, and
  // there was no way to dismiss it from the keyboard at all.
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
      setActive(-1);
      return;
    }
    if (!open || rows.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => (i + 1) % rows.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => (i <= 0 ? rows.length - 1 : i - 1));
    } else if (e.key === 'Enter' && active >= 0 && rows[active]) {
      // `rows[active]` guard is load-bearing: results arrive async and can
      // shrink under a stale highlight index.
      e.preventDefault();
      onSelect(rows[active]);
    }
  };

  return (
    <div className="search-box" ref={containerRef}>
      <input
        type="search"
        className="search-box__input"
        placeholder="Search repos..."
        value={query}
        role="combobox"
        aria-expanded={open}
        aria-controls="search-results"
        aria-autocomplete="list"
        aria-activedescendant={active >= 0 && rows[active] ? `search-opt-${rows[active].id}` : undefined}
        onKeyDown={onKeyDown}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(-1);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
      />

      {open && (
        <div 
          className="search-box__results"
          onPointerDown={(e) => e.stopPropagation()}
        >
          {isLoading && <div className="search-box__msg" role="status">Searching...</div>}
          {!isLoading && rows.length === 0 && (
            <div className="search-box__msg" role="status">No results found</div>
          )}
          {!isLoading && rows.length > 0 && (
            <ul className="search-box__list" id="search-results" role="listbox">
              {rows.map((r, i) => (
                <li key={r.id} role="presentation">
                  <button
                    id={`search-opt-${r.id}`}
                    role="option"
                    aria-selected={i === active}
                    className={i === active ? 'is-active' : undefined}
                    onMouseEnter={() => setActive(i)}
                    onPointerDown={(e) => {
                      e.preventDefault();
                      onSelect(r);
                    }}
                    onClick={() => onSelect(r)}
                  >
                    <span className="repo-name">{r.org ? `${r.org}/${r.name}` : r.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
