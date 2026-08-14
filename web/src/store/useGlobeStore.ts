import { create } from 'zustand';

export type Tier = 'low' | 'mid' | 'high';

export interface BenchResult {
  label: string;
  frames: number;
  durationMs: number;
  /** Detected display refresh, from the median warmup interval. */
  refreshHz: number;
  fps: number;
  p50: number;
  p95: number;
  p99: number;
  worst: number;
  /** Frames that missed their vsync slot — the real definition of "not 60fps". */
  dropped: number;
  sampled: number;
  dropRatio: number;
  /** How many times the scene fits into one frame. 1x means no margin at all. */
  headroom: number;
  drawCalls: number;
  triangles: number;
  points: number;
  passed: boolean;
}

/** Screen-space position of the hovered node, for the reticle. */
export interface ScreenAnchor {
  x: number;
  y: number;
  visible: boolean;
}

interface GlobeState {
  // --- data ---
  loadedBands: number[];
  totalPoints: number;
  graphReady: boolean;
  ambientArcCount: number;
  focusArcCount: number;
  loadError: string | null;

  // --- interaction ---
  hoveredId: number;
  selectedId: number;
  cameraBusy: boolean;
  anchor: ScreenAnchor;

  // --- rendering knobs ---
  tier: Tier;
  /**
   * Continuous fill-rate release valve, 0.55–1, multiplied into the tier's DPR
   * cap. Zooming in makes every point cover far more pixels, and the scene is
   * fragment-bound long before it is geometry-bound — so this is the lever that
   * should move under load, not the node count.
   */
  dprScale: number;
  sizeScale: number;
  autoRotate: boolean;
  reducedMotion: boolean;
  showAmbientArcs: boolean;
  showGrid: boolean;
  /** -1 = every domain. Anything else dims the rest to near-invisible. */
  activeDomain: number;
  /** The globe stays still behind the intro until the user commits to entering. */
  entered: boolean;
  showTelemetry: boolean;

  // --- benchmark ---
  benchRunning: boolean;
  benchResults: BenchResult[];

  setLoaded: (bands: number[], totalPoints: number) => void;
  setGraphReady: (ready: boolean, ambientArcs: number) => void;
  setFocusArcCount: (count: number) => void;
  setLoadError: (message: string | null) => void;
  setHovered: (id: number) => void;
  setSelected: (id: number) => void;
  setCameraBusy: (busy: boolean) => void;
  setAnchor: (anchor: ScreenAnchor) => void;
  setTier: (tier: Tier) => void;
  setDprScale: (dprScale: number) => void;
  setSizeScale: (scale: number) => void;
  setAutoRotate: (on: boolean) => void;
  setReducedMotion: (on: boolean) => void;
  setShowAmbientArcs: (on: boolean) => void;
  setShowGrid: (on: boolean) => void;
  setActiveDomain: (domain: number) => void;
  setEntered: (entered: boolean) => void;
  setShowTelemetry: (on: boolean) => void;
  setBenchRunning: (running: boolean) => void;
  pushBenchResult: (result: BenchResult) => void;
}

export const useGlobeStore = create<GlobeState>((set) => ({
  loadedBands: [],
  totalPoints: 0,
  graphReady: false,
  ambientArcCount: 0,
  focusArcCount: 0,
  loadError: null,

  hoveredId: -1,
  selectedId: -1,
  cameraBusy: false,
  anchor: { x: 0, y: 0, visible: false },

  tier: 'high',
  dprScale: 1,
  sizeScale: 18,
  autoRotate: true,
  reducedMotion: false,
  showAmbientArcs: true,
  showGrid: true,
  activeDomain: -1,
  entered: false,
  showTelemetry: false,

  benchRunning: false,
  benchResults: [],

  setLoaded: (loadedBands, totalPoints) => set({ loadedBands, totalPoints }),
  setGraphReady: (graphReady, ambientArcCount) => set({ graphReady, ambientArcCount }),
  setFocusArcCount: (focusArcCount) =>
    set((s) => (s.focusArcCount === focusArcCount ? s : { focusArcCount })),
  setLoadError: (loadError) => set({ loadError }),
  // Guarded so a 30Hz pick loop doesn't publish an identical value and wake
  // every subscriber for nothing.
  setHovered: (id) => set((s) => (s.hoveredId === id ? s : { hoveredId: id })),
  setSelected: (selectedId) => set({ selectedId }),
  setCameraBusy: (cameraBusy) => set({ cameraBusy }),
  setAnchor: (anchor) =>
    set((s) =>
      s.anchor.visible === anchor.visible &&
      Math.abs(s.anchor.x - anchor.x) < 0.5 &&
      Math.abs(s.anchor.y - anchor.y) < 0.5
        ? s
        : { anchor },
    ),
  setTier: (tier) => set({ tier }),
  // Guarded: PerformanceMonitor fires often and an identical value would wake
  // every subscriber and re-run the pixel-ratio effect for nothing.
  setDprScale: (dprScale) =>
    set((s) => (Math.abs(s.dprScale - dprScale) < 0.01 ? s : { dprScale })),
  setSizeScale: (sizeScale) => set({ sizeScale }),
  setAutoRotate: (autoRotate) => set({ autoRotate }),
  setReducedMotion: (reducedMotion) => set({ reducedMotion }),
  setShowAmbientArcs: (showAmbientArcs) => set({ showAmbientArcs }),
  setShowGrid: (showGrid) => set({ showGrid }),
  setActiveDomain: (activeDomain) => set({ activeDomain }),
  setEntered: (entered) => set({ entered }),
  setShowTelemetry: (showTelemetry) => set({ showTelemetry }),
  setBenchRunning: (benchRunning) => set({ benchRunning }),
  pushBenchResult: (result) => set((s) => ({ benchResults: [result, ...s.benchResults].slice(0, 8) })),
}));
