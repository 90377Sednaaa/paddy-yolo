import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
  Sliders,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Columns,
  Sparkles,
  Layers,
  Crosshair,
  Maximize2,
} from 'lucide-react';
import type { Detection, ModelResult } from '../types';

interface VisionArenaDeckProps {
  imageUrl: string;
  baselineResult: ModelResult | null;
  improvedResult: ModelResult | null;
  isLoading?: boolean;
}

type ViewMode = 'dual' | 'diff' | 'focus';
type LabelStyle = 'full' | 'compact' | 'none';

// Restrained, calm insect species palette (no neon)
const SPECIES_COLORS = [
  '#58a6ff', // soft slate blue
  '#3fb950', // natural leaf green
  '#d29922', // warm amber gold
  '#db61a2', // soft rose
  '#bc8cff', // soft lavender
  '#39c5cf', // soft cyan
  '#7ee787', // soft sage
  '#f0883e', // warm orange
];

function computeIoU(b1: [number, number, number, number], b2: [number, number, number, number]): number {
  const xLeft = Math.max(b1[0], b2[0]);
  const yTop = Math.max(b1[1], b2[1]);
  const xRight = Math.min(b1[2], b2[2]);
  const yBottom = Math.min(b1[3], b2[3]);
  if (xRight < xLeft || yBottom < yTop) return 0;
  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  const b1Area = (b1[2] - b1[0]) * (b1[3] - b1[1]);
  const b2Area = (b2[2] - b2[0]) * (b2[3] - b2[1]);
  const unionArea = b1Area + b2Area - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

export const VisionArenaDeck: React.FC<VisionArenaDeckProps> = ({
  imageUrl,
  baselineResult,
  improvedResult,
  isLoading: _isLoading = false,
}) => {
  // Workbench state - default to side-by-side Dual Sync (no curtains)
  const [viewMode, setViewMode] = useState<ViewMode>('dual');
  const [focusModel, setFocusModel] = useState<'improved' | 'baseline' | 'raw'>('improved');
  const [minConf, setMinConf] = useState<number>(0.25);
  const [labelStyle, setLabelStyle] = useState<LabelStyle>('full');
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredInsect, setHoveredInsect] = useState<Detection | null>(null);
  const [cursorCoords, setCursorCoords] = useState<{ x: number; y: number } | null>(null);

  // Canvases
  const dualLeftCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dualRightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const diffCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const focusCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Loaded image ref
  const imageElementRef = useRef<HTMLImageElement | null>(null);
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const isImageLoaded = loadedUrl === imageUrl;

  // Load source image
  useEffect(() => {
    let active = true;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;
    img.onload = () => {
      if (active) {
        imageElementRef.current = img;
        setLoadedUrl(imageUrl);
      }
    };
    return () => {
      active = false;
    };
  }, [imageUrl]);

  // Filter detections by threshold
  const baselineDetections = useMemo(() => {
    if (!baselineResult) return [];
    return baselineResult.detections.filter((d) => d.confidence >= minConf);
  }, [baselineResult, minConf]);

  const improvedDetections = useMemo(() => {
    if (!improvedResult) return [];
    return improvedResult.detections.filter((d) => d.confidence >= minConf);
  }, [improvedResult, minConf]);

  // Compute Diff categorization
  const diffAnalysis = useMemo(() => {
    const consensus: { baseline: Detection; improved: Detection }[] = [];
    const improvedOnly: Detection[] = [];
    const baselineOnly: Detection[] = [];

    const matchedBaseline = new Set<number>();

    improvedDetections.forEach((impr) => {
      let matched = false;
      for (let i = 0; i < baselineDetections.length; i++) {
        if (matchedBaseline.has(i)) continue;
        const base = baselineDetections[i];
        if (computeIoU(impr.bbox, base.bbox) >= 0.35) {
          consensus.push({ baseline: base, improved: impr });
          matchedBaseline.add(i);
          matched = true;
          break;
        }
      }
      if (!matched) {
        improvedOnly.push(impr);
      }
    });

    baselineDetections.forEach((base, idx) => {
      if (!matchedBaseline.has(idx)) {
        baselineOnly.push(base);
      }
    });

    return { consensus, improvedOnly, baselineOnly };
  }, [baselineDetections, improvedDetections]);

  // Reset viewport
  const resetViewport = useCallback(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.35, 4.5));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.35, 1.0));

  // Mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom((z) => Math.min(z + 0.15, 4.5));
    } else {
      setZoom((z) => {
        const next = Math.max(z - 0.15, 1.0);
        if (next === 1.0) setPan({ x: 0, y: 0 });
        return next;
      });
    }
  };

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1.0) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning && zoom > 1.0) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }

    // Compute natural coordinates
    const container = containerRef.current;
    const img = imageElementRef.current;
    if (container && img && img.naturalWidth) {
      const rect = container.getBoundingClientRect();
      const relX = (e.clientX - rect.left - pan.x) / (rect.width * zoom);
      const relY = (e.clientY - rect.top - pan.y) / (rect.height * zoom);
      const naturalX = Math.round(relX * img.naturalWidth);
      const naturalY = Math.round(relY * img.naturalHeight);
      if (naturalX >= 0 && naturalX <= img.naturalWidth && naturalY >= 0 && naturalY <= img.naturalHeight) {
        setCursorCoords({ x: naturalX, y: naturalY });
      } else {
        setCursorCoords(null);
      }
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  // Draw detections helper (Calm, mature styling, no neon)
  const drawDetections = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      detections: Detection[],
      theme: 'baseline' | 'improved' | 'diff'
    ) => {
      ctx.save();

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        let strokeColor = SPECIES_COLORS[det.class_id % SPECIES_COLORS.length];
        if (theme === 'baseline') strokeColor = '#58a6ff'; // soft slate blue
        if (theme === 'improved') strokeColor = '#3fb950'; // natural botanical green

        const isHovered =
          hoveredInsect &&
          hoveredInsect.species_name === det.species_name &&
          Math.abs(hoveredInsect.bbox[0] - det.bbox[0]) < 2;

        // Subtle box fill
        ctx.fillStyle = `${strokeColor}14`;
        ctx.fillRect(x1, y1, width, height);

        // 1.5px clean boundary line
        ctx.lineWidth = isHovered ? 2.5 : 1.5;
        ctx.strokeStyle = strokeColor;
        ctx.strokeRect(x1, y1, width, height);

        // Subdued crosshair corners if hovered
        if (isHovered) {
          ctx.strokeStyle = '#d29922';
          ctx.lineWidth = 2;
          const cornerLen = 10;
          ctx.beginPath();
          ctx.moveTo(x1 - 3, y1 + cornerLen);
          ctx.lineTo(x1 - 3, y1 - 3);
          ctx.lineTo(x1 + cornerLen, y1 - 3);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(x2 + 3, y2 - cornerLen);
          ctx.lineTo(x2 + 3, y2 + 3);
          ctx.lineTo(x2 - cornerLen, y2 + 3);
          ctx.stroke();
        }

        // Label pill if enabled
        if (labelStyle !== 'none') {
          const text =
            labelStyle === 'compact'
              ? `${(det.confidence * 100).toFixed(0)}%`
              : `${det.species_name} ${(det.confidence * 100).toFixed(0)}%`;

          ctx.font = '500 11px -apple-system, BlinkMacSystemFont, sans-serif';
          const textMetrics = ctx.measureText(text);
          const padding = 5;
          const pillWidth = textMetrics.width + padding * 2;
          const pillHeight = 18;

          let pillY = y1 - pillHeight;
          if (pillY < 0) pillY = y1 + 2;

          ctx.fillStyle = strokeColor;
          ctx.fillRect(x1, pillY, pillWidth, pillHeight);

          ctx.fillStyle = '#0b0f14';
          ctx.textBaseline = 'middle';
          ctx.fillText(text, x1 + padding, pillY + pillHeight / 2);
        }
      });

      ctx.restore();
    },
    [hoveredInsect, labelStyle]
  );

  // Render Synchronized Dual Canvases (Side-by-side)
  useEffect(() => {
    if (viewMode !== 'dual') return;
    const img = imageElementRef.current;
    if (!img || !isImageLoaded) return;

    // Left Canvas: Baseline
    const leftCanvas = dualLeftCanvasRef.current;
    if (leftCanvas) {
      const ctx = leftCanvas.getContext('2d');
      if (ctx) {
        leftCanvas.width = img.naturalWidth || 640;
        leftCanvas.height = img.naturalHeight || 640;
        ctx.drawImage(img, 0, 0, leftCanvas.width, leftCanvas.height);
        drawDetections(ctx, baselineDetections, 'baseline');
      }
    }

    // Right Canvas: Improved
    const rightCanvas = dualRightCanvasRef.current;
    if (rightCanvas) {
      const ctx = rightCanvas.getContext('2d');
      if (ctx) {
        rightCanvas.width = img.naturalWidth || 640;
        rightCanvas.height = img.naturalHeight || 640;
        ctx.drawImage(img, 0, 0, rightCanvas.width, rightCanvas.height);
        drawDetections(ctx, improvedDetections, 'improved');
      }
    }
  }, [viewMode, isImageLoaded, baselineDetections, improvedDetections, drawDetections]);

  // Render Diff Canvas
  useEffect(() => {
    if (viewMode !== 'diff') return;
    const canvas = diffCanvasRef.current;
    const img = imageElementRef.current;
    if (!canvas || !img || !isImageLoaded) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = img.naturalWidth || 640;
    canvas.height = img.naturalHeight || 640;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // 1. Consensus (Soft slate)
    diffAnalysis.consensus.forEach(({ improved }) => {
      const [x1, y1, x2, y2] = improved.bbox;
      ctx.fillStyle = '#8b949e15';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#8b949e';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (labelStyle !== 'none') {
        const text = `Consensus: ${improved.species_name} ${(improved.confidence * 100).toFixed(0)}%`;
        ctx.font = '500 11px sans-serif';
        const w = ctx.measureText(text).width + 8;
        ctx.fillStyle = '#8b949e';
        ctx.fillRect(x1, Math.max(0, y1 - 18), w, 18);
        ctx.fillStyle = '#0b0f14';
        ctx.fillText(text, x1 + 4, Math.max(9, y1 - 9));
      }
    });

    // 2. Enhanced Discoveries (Soft green)
    diffAnalysis.improvedOnly.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      ctx.fillStyle = '#3fb95018';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#3fb950';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (labelStyle !== 'none') {
        const text = `Enhanced Only: ${det.species_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = '500 11px sans-serif';
        const w = ctx.measureText(text).width + 8;
        ctx.fillStyle = '#3fb950';
        ctx.fillRect(x1, Math.max(0, y1 - 18), w, 18);
        ctx.fillStyle = '#0b0f14';
        ctx.fillText(text, x1 + 4, Math.max(9, y1 - 9));
      }
    });

    // 3. Baseline Only (Soft slate blue)
    diffAnalysis.baselineOnly.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      ctx.fillStyle = '#58a6ff15';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#58a6ff';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (labelStyle !== 'none') {
        const text = `Baseline Only: ${det.species_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = '500 11px sans-serif';
        const w = ctx.measureText(text).width + 8;
        ctx.fillStyle = '#58a6ff';
        ctx.fillRect(x1, Math.max(0, y1 - 18), w, 18);
        ctx.fillStyle = '#0b0f14';
        ctx.fillText(text, x1 + 4, Math.max(9, y1 - 9));
      }
    });
  }, [viewMode, isImageLoaded, diffAnalysis, labelStyle]);

  // Render Focus Canvas
  useEffect(() => {
    if (viewMode !== 'focus') return;
    const canvas = focusCanvasRef.current;
    const img = imageElementRef.current;
    if (!canvas || !img || !isImageLoaded) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = img.naturalWidth || 640;
    canvas.height = img.naturalHeight || 640;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (focusModel === 'improved') {
      drawDetections(ctx, improvedDetections, 'improved');
    } else if (focusModel === 'baseline') {
      drawDetections(ctx, baselineDetections, 'baseline');
    }
  }, [viewMode, focusModel, isImageLoaded, baselineDetections, improvedDetections, drawDetections]);

  // Center on insect click
  const focusOnInsect = (det: Detection) => {
    setHoveredInsect(det);
    setZoom(2.0);
    const img = imageElementRef.current;
    if (img && img.naturalWidth) {
      const centerX = (det.bbox[0] + det.bbox[2]) / 2;
      const centerY = (det.bbox[1] + det.bbox[3]) / 2;
      const normX = centerX / img.naturalWidth - 0.5;
      const normY = centerY / img.naturalHeight - 0.5;
      setPan({ x: -normX * 400 * 2.0, y: -normY * 400 * 2.0 });
    }
  };

  return (
    <div className="space-y-4">
      {/* 1. Deck Master Toolbar */}
      <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-3 flex flex-wrap items-center justify-between gap-3 shadow-xs">
        {/* Left: View Mode Switcher (Dual Sync vs Diff vs Focus) */}
        <div className="flex items-center space-x-1 p-1 bg-[#171e28] rounded-lg border border-[#1c2432]">
          <button
            onClick={() => setViewMode('dual')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'dual'
                ? 'bg-[#1f2735] text-white shadow-xs'
                : 'text-[#8b949e] hover:text-white'
            }`}
            title="Side-by-Side Dual Viewports"
          >
            <Columns className="h-3.5 w-3.5" />
            <span>Side-by-Side</span>
          </button>
          <button
            onClick={() => setViewMode('diff')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'diff'
                ? 'bg-[#1f2735] text-white shadow-xs'
                : 'text-[#8b949e] hover:text-white'
            }`}
            title="Model Agreement Analysis (Consensus vs New Detections)"
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>Diff Mode</span>
          </button>
          <button
            onClick={() => setViewMode('focus')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'focus'
                ? 'bg-[#1f2735] text-white shadow-xs'
                : 'text-[#8b949e] hover:text-white'
            }`}
            title="Single Model View"
          >
            <Maximize2 className="h-3.5 w-3.5" />
            <span>Single View</span>
          </button>
        </div>

        {/* Focus Mode Sub-selector */}
        {viewMode === 'focus' && (
          <div className="flex items-center space-x-1 p-0.5 bg-[#171e28] rounded-md border border-[#1c2432] text-xs">
            <button
              onClick={() => setFocusModel('improved')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                focusModel === 'improved'
                  ? 'bg-[#1f2735] text-white'
                  : 'text-[#8b949e] hover:text-white'
              }`}
            >
              Enhanced (EMA + DySample)
            </button>
            <button
              onClick={() => setFocusModel('baseline')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                focusModel === 'baseline'
                  ? 'bg-[#1f2735] text-white'
                  : 'text-[#8b949e] hover:text-white'
              }`}
            >
              Baseline (Paddy YOLO)
            </button>
            <button
              onClick={() => setFocusModel('raw')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                focusModel === 'raw'
                  ? 'bg-[#1f2735] text-white'
                  : 'text-[#8b949e] hover:text-white'
              }`}
            >
              Raw Image
            </button>
          </div>
        )}

        {/* Center: Confidence Threshold Slider */}
        <div className="flex items-center space-x-2.5 px-3 py-1 rounded-lg bg-[#171e28] border border-[#1c2432]">
          <div className="flex items-center space-x-1.5 text-xs text-[#8b949e]">
            <Sliders className="h-3.5 w-3.5" />
            <span className="text-white font-medium text-[11px]">Threshold:</span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={minConf}
            onChange={(e) => setMinConf(parseFloat(e.target.value))}
            className="w-24 accent-[#8b949e] bg-[#11161d] cursor-pointer"
          />
          <span className="text-xs font-mono font-medium text-white tabular-nums w-8 text-right">
            {(minConf * 100).toFixed(0)}%
          </span>
        </div>

        {/* Right: Zoom & Canvas Actions */}
        <div className="flex items-center space-x-1.5">
          {/* Label Style Switcher */}
          <button
            onClick={() =>
              setLabelStyle((s) => (s === 'full' ? 'compact' : s === 'compact' ? 'none' : 'full'))
            }
            className="flex items-center space-x-1 px-2.5 py-1.5 rounded-md bg-[#171e28] hover:bg-[#1f2735] border border-[#1c2432] text-[11px] font-mono text-[#8b949e] hover:text-white transition-colors"
            title="Toggle Bounding Box Labels"
          >
            <Layers className="h-3 w-3" />
            <span>
              {labelStyle === 'full' ? 'Labels' : labelStyle === 'compact' ? 'Scores' : 'Off'}
            </span>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center p-0.5 bg-[#171e28] rounded-md border border-[#1c2432]">
            <button
              onClick={handleZoomOut}
              disabled={zoom <= 1.0}
              className="p-1 rounded text-[#8b949e] hover:text-white disabled:opacity-30 transition-colors"
              title="Zoom Out (-)"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span className="px-1.5 text-[11px] font-mono font-medium text-white tabular-nums">
              {(zoom * 100).toFixed(0)}%
            </span>
            <button
              onClick={handleZoomIn}
              disabled={zoom >= 4.5}
              className="p-1 rounded text-[#8b949e] hover:text-white disabled:opacity-30 transition-colors"
              title="Zoom In (+)"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
          </div>

          <button
            onClick={resetViewport}
            className="p-1.5 rounded-md bg-[#171e28] hover:bg-[#1f2735] border border-[#1c2432] text-[#8b949e] hover:text-white transition-colors"
            title="Reset Viewport (0)"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* 2. Main Inspection Canvas Deck & Specimen Inspector Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 items-start">
        {/* Left: Viewport Deck */}
        <div className="lg:col-span-3 space-y-3">
          <div
            ref={containerRef}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className={`relative w-full aspect-square sm:aspect-[4/3] bg-[#0b0f14] rounded-xl border border-[#1c2432] overflow-hidden flex items-center justify-center select-none ${
              zoom > 1.0 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-crosshair'
            }`}
          >
            {/* View Mode: Synchronized Side-by-Side Dual Viewports */}
            {viewMode === 'dual' && (
              <div
                className="grid grid-cols-2 gap-3 w-full h-full p-3 transition-transform duration-75"
                style={{
                  transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                }}
              >
                {/* Left: Baseline */}
                <div className="relative h-full flex items-center justify-center bg-[#11161d] rounded-lg border border-[#1c2432] overflow-hidden">
                  <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-[#0b0f14]/90 border border-[#1c2432] text-[10px] font-mono text-[#8b949e]">
                    Baseline: Paddy YOLO
                  </div>
                  <canvas ref={dualLeftCanvasRef} className="max-h-full max-w-full object-contain" />
                </div>

                {/* Right: Enhanced */}
                <div className="relative h-full flex items-center justify-center bg-[#11161d] rounded-lg border border-[#1c2432] overflow-hidden">
                  <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-[#0b0f14]/90 border border-[#1c2432] text-[10px] font-mono text-[#8b949e]">
                    Enhanced: EMA + DySample
                  </div>
                  <canvas ref={dualRightCanvasRef} className="max-h-full max-w-full object-contain" />
                </div>
              </div>
            )}

            {/* View Mode: Diff Agreement Analysis */}
            {viewMode === 'diff' && (
              <div
                className="relative max-h-full max-w-full flex items-center justify-center transition-transform duration-75"
                style={{
                  transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                }}
              >
                <canvas
                  ref={diffCanvasRef}
                  className="max-h-full max-w-full object-contain pointer-events-none"
                />
              </div>
            )}

            {/* View Mode: Single Focus */}
            {viewMode === 'focus' && (
              <div
                className="relative max-h-full max-w-full flex items-center justify-center transition-transform duration-75"
                style={{
                  transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                }}
              >
                <canvas
                  ref={focusCanvasRef}
                  className="max-h-full max-w-full object-contain pointer-events-none"
                />
              </div>
            )}

            {/* Bottom Coordinates & Zoom Readout */}
            <div className="absolute bottom-3 right-3 px-2 py-1 rounded bg-[#11161d]/90 border border-[#1c2432] text-[10px] font-mono text-[#8b949e] flex items-center space-x-2 pointer-events-none z-10 shadow-xs">
              {cursorCoords ? (
                <span>
                  [{cursorCoords.x}, {cursorCoords.y}]
                </span>
              ) : (
                <span>Ready</span>
              )}
              <span>·</span>
              <span className="text-white">{(zoom * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Diff Mode Agreement Summary Legend */}
          {viewMode === 'diff' && (
            <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-[#11161d] rounded-xl border border-[#1c2432] text-xs">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1.5">
                  <span className="h-2 w-2 rounded-full bg-[#3fb950]" />
                  <span className="text-[#8b949e]">Enhanced Only:</span>
                  <span className="font-mono text-white font-medium">
                    +{diffAnalysis.improvedOnly.length}
                  </span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="h-2 w-2 rounded-full bg-[#8b949e]" />
                  <span className="text-[#8b949e]">Consensus:</span>
                  <span className="font-mono text-white font-medium">
                    {diffAnalysis.consensus.length}
                  </span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="h-2 w-2 rounded-full bg-[#58a6ff]" />
                  <span className="text-[#8b949e]">Baseline Only:</span>
                  <span className="font-mono text-white font-medium">
                    {diffAnalysis.baselineOnly.length}
                  </span>
                </div>
              </div>
              <span className="text-[11px] text-[#656d76] font-mono">
                IoU overlap threshold ≥ 0.35
              </span>
            </div>
          )}
        </div>

        {/* Right: Pest Specimen Inventory Drawer */}
        <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-4 flex flex-col h-[480px] shadow-xs">
          <div className="flex items-center justify-between pb-3 border-b border-[#1c2432] mb-3">
            <div className="flex items-center space-x-2">
              <Crosshair className="h-4 w-4 text-[#8b949e]" />
              <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                Detected Pests
              </h3>
            </div>
            <div className="flex items-center space-x-1.5 text-[11px] font-mono text-[#8b949e]">
              <span>{baselineDetections.length} Baseline</span>
              <span>/</span>
              <span className="text-white font-medium">{improvedDetections.length} Enhanced</span>
            </div>
          </div>

          {/* List of Detections */}
          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {/* Enhanced Model Detections */}
            <div className="space-y-1.5">
              <span className="text-[10px] font-semibold text-[#8b949e] uppercase tracking-wider block">
                Enhanced ({improvedDetections.length})
              </span>
              {improvedDetections.map((det, i) => (
                <div
                  key={`impr-${i}`}
                  onMouseEnter={() => setHoveredInsect(det)}
                  onMouseLeave={() => setHoveredInsect(null)}
                  onClick={() => focusOnInsect(det)}
                  className={`p-2 rounded-lg border text-xs cursor-pointer transition-colors ${
                    hoveredInsect === det
                      ? 'bg-[#1f2735] border-[#388bfd]'
                      : 'bg-[#171e28] hover:bg-[#1f2735] border-[#1c2432]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-white capitalize truncate max-w-[130px]">
                      {det.species_name}
                    </span>
                    <span className="font-mono font-medium text-white tabular-nums text-[11px]">
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-[#656d76] font-mono tabular-nums">
                    <span>
                      [{Math.round(det.bbox[0])}, {Math.round(det.bbox[1])}]
                    </span>
                    <span>
                      {Math.round(det.bbox[2] - det.bbox[0])}×{Math.round(det.bbox[3] - det.bbox[1])}px
                    </span>
                  </div>
                </div>
              ))}
              {improvedDetections.length === 0 && (
                <p className="text-[11px] text-[#656d76] text-center py-2">
                  No pests detected above {(minConf * 100).toFixed(0)}%
                </p>
              )}
            </div>

            {/* Baseline Model Detections */}
            <div className="pt-2 border-t border-[#1c2432] space-y-1.5">
              <span className="text-[10px] font-semibold text-[#8b949e] uppercase tracking-wider block">
                Baseline ({baselineDetections.length})
              </span>
              {baselineDetections.map((det, i) => (
                <div
                  key={`base-${i}`}
                  onMouseEnter={() => setHoveredInsect(det)}
                  onMouseLeave={() => setHoveredInsect(null)}
                  onClick={() => focusOnInsect(det)}
                  className={`p-2 rounded-lg border text-xs cursor-pointer transition-colors ${
                    hoveredInsect === det
                      ? 'bg-[#1f2735] border-[#388bfd]'
                      : 'bg-[#171e28] hover:bg-[#1f2735] border-[#1c2432]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-[#8b949e] capitalize truncate max-w-[130px]">
                      {det.species_name}
                    </span>
                    <span className="font-mono text-[#8b949e] tabular-nums text-[11px]">
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-[#656d76] font-mono tabular-nums">
                    <span>
                      [{Math.round(det.bbox[0])}, {Math.round(det.bbox[1])}]
                    </span>
                    <span>
                      {Math.round(det.bbox[2] - det.bbox[0])}×{Math.round(det.bbox[3] - det.bbox[1])}px
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-2 border-t border-[#1c2432] text-[10px] text-[#656d76] font-mono text-center">
            Click insect to zoom · Hover to highlight
          </div>
        </div>
      </div>
    </div>
  );
};
