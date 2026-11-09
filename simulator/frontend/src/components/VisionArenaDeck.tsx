import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
  Columns2,
  GitCompareArrows,
  Maximize2,
  SlidersHorizontal,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Crosshair,
  Loader2,
  ScanLine,
} from 'lucide-react';
import type { Detection, ModelResult } from '../types';

interface VisionArenaDeckProps {
  imageUrl: string;
  baselineResult: ModelResult | null;
  improvedResult: ModelResult | null;
  isLoading?: boolean;
  shortcutsEnabled?: boolean;
}

type ViewMode = 'dual' | 'diff' | 'focus';
type LabelStyle = 'full' | 'compact' | 'none';
type AnnoTheme = 'baseline' | 'improved' | 'species';

interface AnnoPalette {
  baseline: string;
  enhanced: string;
  consensus: string;
  focus: string;
  ink: string;
}

const SPECIES_COLORS = [
  '#8fb8e8',
  '#93cfa4',
  '#e2b667',
  '#d79ac0',
  '#b7a5e0',
  '#7fc6cf',
  '#b9d59a',
  '#e6a173',
];

const DEFAULT_PALETTE: AnnoPalette = {
  baseline: '#8fb8e8',
  enhanced: '#93cfa4',
  consensus: '#aeb8ac',
  focus: '#e2b667',
  ink: '#111a12',
};

function readPalette(): AnnoPalette {
  if (typeof window === 'undefined') return DEFAULT_PALETTE;
  const style = getComputedStyle(document.documentElement);
  const get = (key: string, fallback: string) => style.getPropertyValue(key).trim() || fallback;
  return {
    baseline: get('--anno-baseline', DEFAULT_PALETTE.baseline),
    enhanced: get('--anno-enhanced', DEFAULT_PALETTE.enhanced),
    consensus: get('--anno-consensus', DEFAULT_PALETTE.consensus),
    focus: get('--anno-focus', DEFAULT_PALETTE.focus),
    ink: get('--anno-ink', DEFAULT_PALETTE.ink),
  };
}

function computeIoU(
  b1: [number, number, number, number],
  b2: [number, number, number, number]
): number {
  const xLeft = Math.max(b1[0], b2[0]);
  const yTop = Math.max(b1[1], b2[1]);
  const xRight = Math.min(b1[2], b2[2]);
  const yBottom = Math.min(b1[3], b2[3]);
  if (xRight < xLeft || yBottom < yTop) return 0;
  const intersection = (xRight - xLeft) * (yBottom - yTop);
  const a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
  const a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
  const union = a1 + a2 - intersection;
  return union > 0 ? intersection / union : 0;
}

function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, ink: string) {
  ctx.font = '500 11px "IBM Plex Sans", system-ui, sans-serif';
  const width = ctx.measureText(text).width + 12;
  const height = 19;
  // Keep label pills inside the specimen frame, including boxes on the right/bottom edge.
  const labelX = Math.max(1, Math.min(x - 1, Math.max(1, ctx.canvas.width - width - 1)));
  const labelY = Math.max(
    1,
    Math.min(y - height - 2 < 0 ? y + 2 : y - height - 2, ctx.canvas.height - height - 1)
  );
  ctx.fillStyle = color;
  if (typeof ctx.roundRect === 'function') {
    ctx.beginPath();
    ctx.roundRect(labelX, labelY, width, height, 2);
    ctx.fill();
  } else {
    ctx.fillRect(labelX, labelY, width, height);
  }
  ctx.fillStyle = ink;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, labelX + 5, labelY + height / 2 + 0.5);
}

export const VisionArenaDeck: React.FC<VisionArenaDeckProps> = ({
  imageUrl,
  baselineResult,
  improvedResult,
  isLoading = false,
  shortcutsEnabled = true,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('dual');
  const [focusModel, setFocusModel] = useState<'improved' | 'baseline' | 'raw'>('improved');
  const [minConf, setMinConf] = useState<number>(0.25);
  const [labelStyle, setLabelStyle] = useState<LabelStyle>('full');
  const [zoom, setZoom] = useState<number>(1);
  const zoomRef = useRef(zoom);
  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredInsect, setHoveredInsect] = useState<Detection | null>(null);
  const [cursorCoords, setCursorCoords] = useState<{ x: number; y: number } | null>(null);

  const dualLeftCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dualRightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const diffCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const focusCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const imageElementRef = useRef<HTMLImageElement | null>(null);
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const isImageLoaded = loadedUrl === imageUrl;

  // Load source image
  useEffect(() => {
    let active = true;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (active) {
        imageElementRef.current = img;
        setLoadedUrl(imageUrl);
      }
    };
    img.src = imageUrl;
    return () => {
      active = false;
    };
  }, [imageUrl]);

  const baselineDetections = useMemo(
    () => baselineResult?.detections.filter((d) => d.confidence >= minConf) ?? [],
    [baselineResult, minConf]
  );
  const improvedDetections = useMemo(
    () => improvedResult?.detections.filter((d) => d.confidence >= minConf) ?? [],
    [improvedResult, minConf]
  );

  const diffAnalysis = useMemo(() => {
    const consensus: { baseline: Detection; improved: Detection }[] = [];
    const improvedOnly: Detection[] = [];
    const baselineOnly: Detection[] = [];
    const matchedBaseline = new Set<number>();

    improvedDetections.forEach((impr) => {
      let matched = false;
      for (let i = 0; i < baselineDetections.length; i++) {
        if (matchedBaseline.has(i)) continue;
        if (computeIoU(impr.bbox, baselineDetections[i].bbox) >= 0.35) {
          consensus.push({ baseline: baselineDetections[i], improved: impr });
          matchedBaseline.add(i);
          matched = true;
          break;
        }
      }
      if (!matched) improvedOnly.push(impr);
    });
    baselineDetections.forEach((base, idx) => {
      if (!matchedBaseline.has(idx)) baselineOnly.push(base);
    });
    return { consensus, improvedOnly, baselineOnly };
  }, [baselineDetections, improvedDetections]);

  const resetViewport = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const zoomBy = useCallback((step: number) => {
    const next = Math.min(4.5, Math.max(1, Math.round((zoomRef.current + step) * 100) / 100));
    zoomRef.current = next;
    setZoom(next);
    if (next === 1) setPan({ x: 0, y: 0 });
  }, []);

  // Non-passive wheel listener so the page does not scroll while zooming the stage.
  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      zoomBy(e.deltaY < 0 ? 0.15 : -0.15);
    };
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => node.removeEventListener('wheel', onWheel);
  }, [zoomBy]);

  // Inspection-stage shortcuts (V / L / + / - / 0).
  useEffect(() => {
    if (!shortcutsEnabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (document.querySelector('[data-modal-open="true"]')) return;
      const el = document.activeElement;
      if (el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return;
      const key = e.key.toLowerCase();
      if (key === 'v') {
        setViewMode((m) => (m === 'dual' ? 'diff' : m === 'diff' ? 'focus' : 'dual'));
      } else if (key === 'l') {
        setLabelStyle((s) => (s === 'full' ? 'compact' : s === 'compact' ? 'none' : 'full'));
      } else if (key === '+' || key === '=') {
        e.preventDefault();
        zoomBy(0.35);
      } else if (key === '-' || key === '_') {
        e.preventDefault();
        zoomBy(-0.35);
      } else if (key === '0') {
        e.preventDefault();
        resetViewport();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [shortcutsEnabled, zoomBy, resetViewport]);

  const primaryCanvas = () =>
    viewMode === 'dual'
      ? dualLeftCanvasRef.current
      : viewMode === 'diff'
        ? diffCanvasRef.current
        : focusCanvasRef.current;

  /** The bitmap is object-fit:contain inside the canvas box — account for letterboxing. */
  const imageFrame = (canvas: HTMLCanvasElement | null, img: HTMLImageElement | null) => {
    if (!canvas || !img?.naturalWidth) return null;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const drawnW = img.naturalWidth * scale;
    const drawnH = img.naturalHeight * scale;
    return {
      rect,
      drawnW,
      drawnH,
      offsetX: (rect.width - drawnW) / 2,
      offsetY: (rect.height - drawnH) / 2,
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning && zoom > 1) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
    const frame = imageFrame(primaryCanvas(), imageElementRef.current);
    if (frame) {
      const relX = (e.clientX - frame.rect.left - frame.offsetX) / frame.drawnW;
      const relY = (e.clientY - frame.rect.top - frame.offsetY) / frame.drawnH;
      const naturalX = Math.round(relX * (imageElementRef.current?.naturalWidth ?? 0));
      const naturalY = Math.round(relY * (imageElementRef.current?.naturalHeight ?? 0));
      setCursorCoords(
        relX >= 0 && relX <= 1 && relY >= 0 && relY <= 1 ? { x: naturalX, y: naturalY } : null
      );
    }
  };

  const handleMouseUp = () => setIsPanning(false);

  // Draw bounding boxes with corner registration ticks — instrument-style, no glow.
  const drawDetections = useCallback(
    (ctx: CanvasRenderingContext2D, detections: Detection[], theme: AnnoTheme) => {
      const palette = readPalette();
      ctx.save();
      ctx.lineJoin = 'miter';

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const color =
          theme === 'baseline'
            ? palette.baseline
            : theme === 'improved'
              ? palette.enhanced
              : SPECIES_COLORS[det.class_id % SPECIES_COLORS.length];

        const isHovered =
          hoveredInsect !== null &&
          hoveredInsect.species_name === det.species_name &&
          Math.abs(hoveredInsect.bbox[0] - det.bbox[0]) < 2;

        ctx.fillStyle = `${color}14`;
        ctx.fillRect(x1, y1, width, height);
        ctx.lineWidth = isHovered ? 2.2 : 1.5;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, width, height);

        // Corner ticks on every box; extended on hover in the focus accent.
        const tick = Math.min(9, Math.min(width, height) / 4);
        ctx.lineWidth = isHovered ? 2.4 : 1.8;
        ctx.strokeStyle = isHovered ? palette.focus : color;
        ctx.beginPath();
        ctx.moveTo(x1, y1 + tick);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + tick, y1);
        ctx.moveTo(x2 - tick, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - tick);
        ctx.stroke();

        if (labelStyle !== 'none') {
          const text =
            labelStyle === 'compact'
              ? `${(det.confidence * 100).toFixed(0)}%`
              : `${det.species_name} ${(det.confidence * 100).toFixed(0)}%`;
          drawLabel(ctx, text, x1, y1, color, palette.ink);
        }
      });
      ctx.restore();
    },
    [hoveredInsect, labelStyle]
  );

  const paintBase = (canvas: HTMLCanvasElement, img: HTMLImageElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    canvas.width = img.naturalWidth || 640;
    canvas.height = img.naturalHeight || 640;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return ctx;
  };

  // Dual view
  useEffect(() => {
    if (viewMode !== 'dual' || !isImageLoaded) return;
    const img = imageElementRef.current;
    if (!img) return;
    const left = dualLeftCanvasRef.current;
    const right = dualRightCanvasRef.current;
    if (left) {
      const ctx = paintBase(left, img);
      if (ctx) drawDetections(ctx, baselineDetections, 'baseline');
    }
    if (right) {
      const ctx = paintBase(right, img);
      if (ctx) drawDetections(ctx, improvedDetections, 'improved');
    }
  }, [viewMode, isImageLoaded, baselineDetections, improvedDetections, drawDetections]);

  // Agreement diff view
  useEffect(() => {
    if (viewMode !== 'diff' || !isImageLoaded) return;
    const img = imageElementRef.current;
    const canvas = diffCanvasRef.current;
    if (!img || !canvas) return;
    const ctx = paintBase(canvas, img);
    if (!ctx) return;
    const p = readPalette();

    const drawGroup = (
      list: { det: Detection; color: string; prefix: string }[]
    ) => {
      list.forEach(({ det, color, prefix }) => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.fillStyle = `${color}18`;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        if (labelStyle !== 'none') {
          const text =
            labelStyle === 'compact'
              ? `${(det.confidence * 100).toFixed(0)}%`
              : `${prefix} ${det.species_name} ${(det.confidence * 100).toFixed(0)}%`;
          drawLabel(ctx, text, x1, y1, color, p.ink);
        }
      });
    };

    drawGroup(diffAnalysis.consensus.map((c) => ({ det: c.improved, color: p.consensus, prefix: 'Both ·' })));
    drawGroup(diffAnalysis.baselineOnly.map((d) => ({ det: d, color: p.baseline, prefix: 'Baseline ·' })));
    drawGroup(diffAnalysis.improvedOnly.map((d) => ({ det: d, color: p.enhanced, prefix: 'Enhanced ·' })));
  }, [viewMode, isImageLoaded, diffAnalysis, labelStyle]);

  // Single-model / raw view
  useEffect(() => {
    if (viewMode !== 'focus' || !isImageLoaded) return;
    const img = imageElementRef.current;
    const canvas = focusCanvasRef.current;
    if (!img || !canvas) return;
    const ctx = paintBase(canvas, img);
    if (!ctx) return;
    if (focusModel === 'improved') drawDetections(ctx, improvedDetections, 'improved');
    else if (focusModel === 'baseline') drawDetections(ctx, baselineDetections, 'baseline');
  }, [viewMode, focusModel, isImageLoaded, baselineDetections, improvedDetections, drawDetections]);

  const focusOnInsect = (det: Detection) => {
    setHoveredInsect(det);
    const img = imageElementRef.current;
    const frame = imageFrame(primaryCanvas(), img);
    if (img?.naturalWidth && frame) {
      const normX = (det.bbox[0] + det.bbox[2]) / 2 / img.naturalWidth - 0.5;
      const normY = (det.bbox[1] + det.bbox[3]) / 2 / img.naturalHeight - 0.5;
      const nextZoom = 2;
      zoomRef.current = nextZoom;
      setZoom(nextZoom);
      setPan({
        x: (-normX * frame.drawnW * nextZoom) / zoom,
        y: (-normY * frame.drawnH * nextZoom) / zoom,
      });
    }
  };

  const transformStyle: React.CSSProperties = {
    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
    transformOrigin: 'center center',
  };

  const hasResults = baselineResult !== null || improvedResult !== null;

  const modes: { id: ViewMode; label: string; icon: React.ReactNode }[] = [
    { id: 'dual', label: 'Side-by-side', icon: <Columns2 className="h-3.5 w-3.5" /> },
    { id: 'diff', label: 'Agreement diff', icon: <GitCompareArrows className="h-3.5 w-3.5" /> },
    { id: 'focus', label: 'Single model', icon: <Maximize2 className="h-3.5 w-3.5" /> },
  ];

  const renderLedgerGroup = (title: string, list: Detection[], color: string, keyPrefix: string) => (
    <div>
      <div className="flex items-center gap-2 border-b border-line pb-1.5">
        <span className="h-2 w-[3px] rounded-full" style={{ backgroundColor: color }} />
        <span className="eyebrow eyebrow-ink">{title}</span>
        <span className="numeric ml-auto text-[11px] text-ink">{String(list.length).padStart(2, '0')}</span>
      </div>
      {list.length === 0 ? (
        <p className="py-3 text-[11px] leading-relaxed text-ink-3">
          No detections above {(minConf * 100).toFixed(0)}% confidence. Lower the threshold to inspect weaker candidates.
        </p>
      ) : (
        <ul className="divide-y divide-line/70">
          {list.map((det, i) => {
            const active = hoveredInsect === det;
            return (
              <li key={`${keyPrefix}-${i}`}>
                <button
                  type="button"
                  onMouseEnter={() => setHoveredInsect(det)}
                  onMouseLeave={() => setHoveredInsect(null)}
                  onFocus={() => setHoveredInsect(det)}
                  onBlur={() => setHoveredInsect(null)}
                  onClick={() => focusOnInsect(det)}
                  className={`group flex w-full items-start gap-2.5 px-1 py-2 text-left transition-colors ${
                    active ? 'bg-leaf-soft' : 'hover:bg-surface-2'
                  }`}
                  aria-label={`Focus ${det.species_name}, ${(det.confidence * 100).toFixed(1)} percent confidence`}
                  title={`Bounding box [${det.bbox.map(Math.round).join(', ')}]`}
                >
                  <span className="numeric mt-0.5 w-5 shrink-0 text-[10px] text-ink-3">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="truncate text-[12px] font-medium text-ink capitalize">
                        {det.species_name}
                      </span>
                      <span className="numeric shrink-0 text-[11px] font-medium text-ink">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                    </span>
                    <span className="mt-1.5 flex items-center gap-2">
                      <span className="block h-[2px] min-w-0 flex-1 bg-line">
                        <span
                          className="block h-full transition-all duration-300"
                          style={{ width: `${det.confidence * 100}%`, backgroundColor: color }}
                        />
                      </span>
                      <span className="numeric shrink-0 text-[9px] text-ink-3">
                        {Math.round(det.bbox[2] - det.bbox[0])}×{Math.round(det.bbox[3] - det.bbox[1])} px
                      </span>
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );

  return (
    <section className="animate-rise-3" aria-labelledby="stage-heading">
      {/* Section heading */}
      <div className="mb-3.5 flex flex-wrap items-baseline justify-between gap-3">
        <div className="flex flex-wrap items-baseline gap-3">
          <span className="eyebrow eyebrow-ink">02 / Inspection stage</span>
          <h2
            id="stage-heading"
            className="font-display text-[23px] leading-none font-medium tracking-[-0.02em] text-ink"
          >
            Detection viewport
          </h2>
          <span className="hidden text-[12px] text-ink-3 md:inline">
            Synchronized pan and zoom · hover a detection to locate it
          </span>
        </div>
        <span className="eyebrow numeric">
          {baselineDetections.length + improvedDetections.length} visible detections
        </span>
      </div>

      {/* Instrument control rail */}
      <div className="mb-3 flex flex-col gap-3 border-y border-line py-2.5 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex flex-wrap items-center gap-5">
          <div className="flex items-center gap-4">
            {modes.map((mode) => (
              <button
                key={mode.id}
                type="button"
                data-active={viewMode === mode.id}
                onClick={() => setViewMode(mode.id)}
                className="tab-link flex items-center gap-1.5 py-1.5 text-[12px]"
                title={`${mode.label} view (V to cycle)`}
              >
                {mode.icon}
                <span>{mode.label}</span>
              </button>
            ))}
          </div>

          {viewMode === 'focus' && (
            <div className="flex items-center gap-1 border-l border-line pl-4">
              {(
                [
                  ['improved', 'Enhanced'],
                  ['baseline', 'Baseline'],
                  ['raw', 'Raw'],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setFocusModel(id)}
                  data-on={focusModel === id}
                  className="ctl h-7 px-2.5 text-[11px]"
                >
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex items-center gap-2.5">
            <SlidersHorizontal className="h-3.5 w-3.5 text-ink-3" />
            <label htmlFor="confidence-range" className="eyebrow whitespace-nowrap">
              Confidence
            </label>
            <input
              id="confidence-range"
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={minConf}
              onChange={(e) => setMinConf(parseFloat(e.target.value))}
              className="precision-range w-24"
            />
            <span className="numeric w-9 text-right text-[12px] font-medium text-ink">
              {(minConf * 100).toFixed(0)}%
            </span>
          </div>

          <span className="hidden h-5 w-px bg-line xl:block" />

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() =>
                setLabelStyle((s) => (s === 'full' ? 'compact' : s === 'compact' ? 'none' : 'full'))
              }
              className="ctl h-8 gap-1.5 px-2.5 text-[11px] font-medium"
              title="Cycle annotation labels (L)"
            >
              <Layers className="h-3.5 w-3.5" />
              <span>
                {labelStyle === 'full' ? 'Full labels' : labelStyle === 'compact' ? 'Scores' : 'Labels off'}
              </span>
            </button>

            <div className="flex items-center">
              <button
                type="button"
                onClick={() => zoomBy(-0.35)}
                disabled={zoom <= 1}
                className="ctl h-8 w-8 rounded-r-none"
                title="Zoom out (−)"
                aria-label="Zoom out"
              >
                <ZoomOut className="h-3.5 w-3.5" />
              </button>
              <span className="numeric flex h-8 min-w-[52px] items-center justify-center border-y border-line bg-surface-2 px-1 text-[11px] font-medium text-ink">
                {(zoom * 100).toFixed(0)}%
              </span>
              <button
                type="button"
                onClick={() => zoomBy(0.35)}
                disabled={zoom >= 4.5}
                className="ctl h-8 w-8 rounded-l-none"
                style={{ marginLeft: -1 }}
                title="Zoom in (+)"
                aria-label="Zoom in"
              >
                <ZoomIn className="h-3.5 w-3.5" />
              </button>
            </div>

            <button
              type="button"
              onClick={resetViewport}
              className="ctl h-8 w-8"
              title="Reset viewport (0)"
              aria-label="Reset viewport"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Stage + ledger */}
      <div className="grid grid-cols-1 items-stretch gap-5 lg:grid-cols-[minmax(0,1fr)_292px]">
        <div
          ref={stageRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            handleMouseUp();
            setCursorCoords(null);
          }}
          className={`stage-panel stage-shell relative min-h-[380px] overflow-hidden select-none ${
            zoom > 1 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-crosshair'
          }`}
        >
          {/* Registration corners */}
          <span className="pointer-events-none absolute top-2 left-2 z-20 h-3 w-3 border-t border-l border-stage-line" />
          <span className="pointer-events-none absolute top-2 right-2 z-20 h-3 w-3 border-t border-r border-stage-line" />
          <span className="pointer-events-none absolute bottom-2 left-2 z-20 h-3 w-3 border-b border-l border-stage-line" />
          <span className="pointer-events-none absolute right-2 bottom-2 z-20 h-3 w-3 border-r border-b border-stage-line" />

          {viewMode === 'dual' && (
            <div className="grid h-full grid-cols-1 grid-rows-2 sm:grid-cols-2 sm:grid-rows-1">
              {[
                {
                  ref: dualLeftCanvasRef,
                  title: 'Baseline / Paddy YOLO',
                  count: baselineDetections.length,
                  color: 'var(--anno-baseline)',
                },
                {
                  ref: dualRightCanvasRef,
                  title: 'Enhanced / EMA + DySample',
                  count: improvedDetections.length,
                  color: 'var(--anno-enhanced)',
                },
              ].map((v, idx) => (
                <div
                  key={v.title}
                  className={`flex min-w-0 flex-col ${
                    idx === 0 ? 'border-b border-stage-line sm:border-b-0 sm:border-r' : ''
                  }`}
                >
                  <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-stage-line px-3.5">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="h-2 w-[3px] shrink-0 rounded-full" style={{ backgroundColor: v.color }} />
                      <span className="eyebrow eyebrow-stage truncate">{v.title}</span>
                    </div>
                    <span className="numeric shrink-0 text-[10px] text-stage-ink-2">
                      {String(v.count).padStart(2, '0')} det
                    </span>
                  </div>
                  <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden p-4">
                    <div
                      className="flex h-full w-full items-center justify-center transition-transform duration-100 ease-out"
                      style={transformStyle}
                    >
                      <canvas
                        ref={v.ref}
                        role="img"
                        aria-label={`${v.title} detection result`}
                        className="h-full w-full object-contain"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {(viewMode === 'diff' || viewMode === 'focus') && (
            <div className="flex h-full flex-col">
              <div className="flex h-9 shrink-0 items-center justify-between border-b border-stage-line px-3.5">
                <span className="eyebrow eyebrow-stage">
                  {viewMode === 'diff'
                    ? 'Agreement analysis / consensus and exclusive detections'
                    : focusModel === 'raw'
                      ? 'Source image / annotations hidden'
                      : focusModel === 'baseline'
                        ? 'Baseline / Paddy YOLO'
                        : 'Enhanced / Paddy YOLO EMA + DySample'}
                </span>
                <span className="numeric text-[10px] text-stage-ink-2">
                  {viewMode === 'diff'
                    ? `${diffAnalysis.consensus.length + diffAnalysis.improvedOnly.length + diffAnalysis.baselineOnly.length} regions`
                    : focusModel === 'raw'
                      ? '—'
                      : `${focusModel === 'baseline' ? baselineDetections.length : improvedDetections.length} det`}
                </span>
              </div>
              <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden p-4">
                <div
                  className="flex h-full w-full items-center justify-center transition-transform duration-100 ease-out"
                  style={transformStyle}
                >
                  {viewMode === 'diff' ? (
                    <canvas
                      ref={diffCanvasRef}
                      role="img"
                      aria-label="Model agreement diff overlay"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <canvas
                      ref={focusCanvasRef}
                      role="img"
                      aria-label="Single model detection result"
                      className="h-full w-full object-contain"
                    />
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Coordinate readout */}
          <div className="pointer-events-none absolute right-4 bottom-4 z-20 flex items-center gap-2 rounded-[3px] border border-stage-line bg-stage-2/90 px-2.5 py-1 text-[10px] text-stage-ink-2">
            <Crosshair className="h-3 w-3" />
            <span className="numeric">
              {cursorCoords ? `[${cursorCoords.x}, ${cursorCoords.y}]` : '— , —'}
            </span>
            <span className="opacity-40">·</span>
            <span className="numeric text-stage-ink">{(zoom * 100).toFixed(0)}%</span>
          </div>

          {/* Awaiting / processing states */}
          {!hasResults && !isLoading && (
            <div className="pointer-events-none absolute top-12 left-4 z-20 flex items-center gap-2 rounded-[3px] border border-stage-line bg-stage-2/85 px-2.5 py-1.5 text-[11px] text-stage-ink-2">
              <ScanLine className="h-3.5 w-3.5" />
              Raw specimen — awaiting inference result
            </div>
          )}
          {isLoading && (
            <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-stage/78 backdrop-blur-[2px]">
              <Loader2 className="h-5 w-5 animate-spin text-stage-ink" />
              <p className="text-[12px] font-medium text-stage-ink">Running dual inference…</p>
              <div className="relative h-[2px] w-40 overflow-hidden bg-stage-line">
                <span className="animate-sweep absolute inset-y-0 left-0 w-1/3 bg-[var(--anno-enhanced)]" />
              </div>
              <p className="eyebrow eyebrow-stage">Baseline + enhanced architecture</p>
            </div>
          )}
        </div>

        {/* Detection ledger */}
        <aside
          className="panel flex flex-col overflow-hidden"
          style={{ height: 'clamp(380px, 42vw, 560px)', boxShadow: 'var(--shadow-lift)' }}
        >
          <div className="border-b border-line px-4 py-3">
            <div className="flex items-baseline justify-between gap-2">
              <span className="eyebrow eyebrow-ink">03 / Detection ledger</span>
              <span className="numeric text-[10px] whitespace-nowrap text-ink-2">
                <span className="text-baseline">{baselineDetections.length} base</span>
                <span className="mx-1 text-ink-3">·</span>
                <span className="text-leaf">{improvedDetections.length} enh</span>
              </span>
            </div>
            <h3 className="mt-0.5 font-display text-[20px] leading-tight font-medium text-ink">
              Detected pests
            </h3>
          </div>

          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-3.5 py-2.5">
            {renderLedgerGroup('Enhanced', improvedDetections, 'var(--c-leaf)', 'enh')}
            {renderLedgerGroup('Baseline', baselineDetections, 'var(--c-baseline)', 'base')}
          </div>

          <div className="border-t border-line px-4 py-2">
            <p className="text-[10px] leading-relaxed text-ink-3">
              Select a detection to centre and magnify it. Hover to highlight the box on every viewport.
            </p>
          </div>
        </aside>
      </div>

      {/* Diff legend / footnote */}
      {viewMode === 'diff' && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-b border-line pb-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            {[
              { label: 'Enhanced only', value: diffAnalysis.improvedOnly.length, color: 'var(--anno-enhanced)' },
              { label: 'Consensus', value: diffAnalysis.consensus.length, color: 'var(--anno-consensus)' },
              { label: 'Baseline only', value: diffAnalysis.baselineOnly.length, color: 'var(--anno-baseline)' },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-[11px] text-ink-2">{item.label}</span>
                <span className="numeric text-[11px] font-medium text-ink">{item.value}</span>
              </div>
            ))}
          </div>
          <span className="eyebrow">Pairing by IoU ≥ 0.35</span>
        </div>
      )}
    </section>
  );
};
