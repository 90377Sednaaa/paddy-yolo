import React, { useEffect, useRef } from 'react';
import type { Detection } from '../types';

interface DetectionCanvasProps {
  imageUrl: string;
  detections: Detection[];
  modelTheme: 'baseline' | 'improved';
  minConfidence?: number;
}

// Visual color palette per insect order
const COLOR_PALETTE = [
  '#38bdf8', // sky blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ec4899', // pink
  '#a855f7', // purple
  '#06b6d4', // cyan
  '#84cc16', // lime
  '#f97316', // orange
];

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  imageUrl,
  detections,
  modelTheme,
  minConfidence = 0.25,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;

    img.onload = () => {
      // Size canvas to image dimensions
      canvas.width = img.naturalWidth || 640;
      canvas.height = img.naturalHeight || 640;

      // Draw background image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw detections
      const filteredDetections = detections.filter(
        (d) => d.confidence >= minConfidence
      );

      filteredDetections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        const baseColor = COLOR_PALETTE[det.class_id % COLOR_PALETTE.length];

        // Draw bounding box fill
        ctx.fillStyle = `${baseColor}1a`;
        ctx.fillRect(x1, y1, width, height);

        // Draw bounding box border
        ctx.lineWidth = 2;
        ctx.strokeStyle = baseColor;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label pill
        const labelText = `${det.species_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 12px ui-monospace, monospace';
        const textMetrics = ctx.measureText(labelText);
        const padding = 6;
        const pillWidth = textMetrics.width + padding * 2;
        const pillHeight = 20;

        let pillY = y1 - pillHeight;
        if (pillY < 0) pillY = y1 + 2;

        ctx.fillStyle = baseColor;
        ctx.fillRect(x1, pillY, pillWidth, pillHeight);

        // Label text in dark high-contrast
        ctx.fillStyle = '#0a0d14';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, x1 + padding, pillY + pillHeight / 2);
      });
    };
  }, [imageUrl, detections, modelTheme, minConfidence]);

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-square bg-[#0a0d14] rounded-lg overflow-hidden border border-[#232d42] flex items-center justify-center"
    >
      <canvas
        ref={canvasRef}
        className="max-h-full max-w-full object-contain"
      />
      {detections.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0a0d14]/40 backdrop-blur-[1px]">
          <span className="text-xs text-[#64748b] font-mono">
            No pest detections recorded
          </span>
        </div>
      )}
    </div>
  );
};
