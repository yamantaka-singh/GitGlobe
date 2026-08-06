import { create } from 'zustand';

export type Tier = 'low' | 'mid' | 'high';

export interface BenchResult {
  label: string;
  frames: number;
  durationMs: number;
  fps: number;
  p50: number;
  p95: number;
  p99: number;
  worst: number;
  over16: number;
  drawCalls: number;
  points: number;
  passed: boolean;
}

interface GlobeState {
  // --- data ---
  loadedBands: number[];
  totalPoints: number;
  loadError: string | null;

  // --- interaction ---
  /** Global picking id (index across all bands), or -1. */
  hoveredId: number;
  selectedId: number;
  /** True while a scripted camera transition is in flight. */
  cameraBusy: boolean;

  // --- rendering knobs ---
  tier: Tier;
  sizeScale: number;
  autoRotate: boolean;
  reducedMotion: boolean;

  // --- benchmark ---
  benchRunning: boolean;
  benchResults: BenchResult[];

  setLoaded: (bands: number[], totalPoints: number) => void;
  setLoadError: (message: string | null) => void;
  setHovered: (id: number) => void;
  setSelected: (id: number) => void;
  setCameraBusy: (busy: boolean) => void;
  setTier: (tier: Tier) => void;
  setSizeScale: (scale: number) => void;
  setAutoRotate: (on: boolean) => void;
  setReducedMotion: (on: boolean) => void;
  setBenchRunning: (running: boolean) => void;
  pushBenchResult: (result: BenchResult) => void;
}

export const useGlobeStore = create<GlobeState>((set) => ({
  loadedBands: [],
  totalPoints: 0,
  loadError: null,

  hoveredId: -1,
  selectedId: -1,
  cameraBusy: false,

  tier: 'high',
  sizeScale: 32,
  autoRotate: true,
  reducedMotion: false,

  benchRunning: false,
  benchResults: [],

  setLoaded: (loadedBands, totalPoints) => set({ loadedBands, totalPoints }),
  setLoadError: (loadError) => set({ loadError }),
  // Guarded so a 30Hz pick loop doesn't publish an identical value and wake
  // every subscriber for nothing.
  setHovered: (id) => set((s) => (s.hoveredId === id ? s : { hoveredId: id })),
  setSelected: (selectedId) => set({ selectedId }),
  setCameraBusy: (cameraBusy) => set({ cameraBusy }),
  setTier: (tier) => set({ tier }),
  setSizeScale: (sizeScale) => set({ sizeScale }),
  setAutoRotate: (autoRotate) => set({ autoRotate }),
  setReducedMotion: (reducedMotion) => set({ reducedMotion }),
  setBenchRunning: (benchRunning) => set({ benchRunning }),
  pushBenchResult: (result) => set((s) => ({ benchResults: [result, ...s.benchResults].slice(0, 8) })),
}));
