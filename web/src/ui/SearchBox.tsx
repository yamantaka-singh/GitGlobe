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

  // Click outside to close
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
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
        const [org, name] = repo.full_name.split('/');
        const globalId = findGlobalIdByName(repo.full_name);
        
        return {
          id: globalId,
          name,
          org,
          domain: repo.domain,
          description: repo.description,
          score: item.score
        };
      });
      return mapped.filter(r => r.id >= 0);
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

  return (
    <div className="search-box" ref={containerRef}>
      <input
        type="search"
        className="search-box__input"
        placeholder="Search repos..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
      />
      
      {isOpen && debouncedQuery.trim().length > 0 && (
        <div className="search-box__results">
          {isLoading && <div className="search-box__msg">Searching...</div>}
          {!isLoading && results?.length === 0 && (
            <div className="search-box__msg">No results found</div>
          )}
          {!isLoading && results && results.length > 0 && (
            <ul className="search-box__list">
              {results.map((r) => (
                <li key={r.id}>
                  <button onClick={() => onSelect(r)}>
                    <span className="repo-name">{r.org}/{r.name}</span>
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
