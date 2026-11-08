import React, { useState } from 'react';
import { Sliders } from 'lucide-react';
import type { ModelResult } from '../types';
import { DetectionCanvas } from './DetectionCanvas';

interface ModelArenaProps {
  imageUrl: string;
  baselineResult: ModelResult | null;
  improvedResult: ModelResult | null;
  isLoading?: boolean;
}

export const ModelArena: React.FC<ModelArenaProps> = ({
  imageUrl,
  baselineResult,
  improvedResult,
}) => {
  const [minConf, setMinConf] = useState<number>(0.20);

  return (
    <div className="space-y-4">
      {/* Confidence threshold filter bar */}
      <div className="flex items-center justify-between bg-[#111622] rounded-lg border border-[#232d42] px-4 py-2.5">
        <div className="flex items-center space-x-2 text-xs text-[#94a3b8]">
          <Sliders className="h-4 w-4 text-[#10b981]" />
          <span className="font-medium text-white">Detection Filter:</span>
          <span>Confidence Threshold</span>
        </div>
        <div className="flex items-center space-x-3">
          <input
            type="range"
            min="0.05"
            max="0.90"
            step="0.05"
            value={minConf}
            onChange={(e) => setMinConf(parseFloat(e.target.value))}
            className="w-32 accent-[#10b981] bg-[#182030] cursor-pointer"
          />
          <span className="text-xs font-mono font-medium text-white tabular-nums w-12 text-right">
            {(minConf * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Side-by-Side Dual Arena */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model 1: Baseline Paddy YOLO */}
        <div className="bg-[#111622] rounded-xl border border-[#232d42] p-5 flex flex-col justify-between">
          <div>
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-[#232d42] mb-4">
              <div>
                <div className="flex items-center space-x-2">
                  <span className="h-2 w-2 rounded-full bg-[#38bdf8]" />
                  <h3 className="font-bold text-white text-base tracking-tight">
                    Paddy YOLO
                  </h3>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#182030] border border-[#232d42] text-[#94a3b8]">
                    Baseline
                  </span>
                </div>
                <p className="text-xs text-[#64748b] mt-0.5">
                  RepGG Backbone + Nearest-Neighbor Neck
                </p>
              </div>

              {baselineResult && (
                <div className="text-right">
                  <span className="text-xs font-mono tabular-nums text-[#38bdf8] font-bold">
                    {baselineResult.timing.total_ms} ms
                  </span>
                  <p className="text-[10px] text-[#64748b] font-mono">
                    {(1000 / baselineResult.timing.total_ms).toFixed(1)} FPS
                  </p>
                </div>
              )}
            </div>

            {/* Canvas View */}
            <DetectionCanvas
              imageUrl={imageUrl}
              detections={baselineResult ? baselineResult.detections : []}
              modelTheme="baseline"
              minConfidence={minConf}
            />
          </div>

          {/* Detections Inventory */}
          <div className="mt-4 pt-3 border-t border-[#232d42]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">
                Detected Pests (
                {baselineResult
                  ? baselineResult.detections.filter((d) => d.confidence >= minConf).length
                  : 0}
                )
              </span>
              {baselineResult?.is_mock && (
                <span className="text-[10px] text-[#94a3b8] font-mono bg-[#182030] px-1.5 py-0.5 rounded border border-[#232d42]">
                  Calibration Mode
                </span>
              )}
            </div>

            <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
              {baselineResult?.detections
                .filter((d) => d.confidence >= minConf)
                .map((d, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs px-2.5 py-1.5 rounded bg-[#0a0d14]/70 border border-[#232d42]"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-[#38bdf8]" />
                      <span className="font-medium text-white capitalize">
                        {d.species_name}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 font-mono text-[11px] tabular-nums text-[#94a3b8]">
                      <span>box: [{d.bbox.map((v) => Math.round(v)).join(', ')}]</span>
                      <span className="font-bold text-[#38bdf8]">
                        {(d.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              {(!baselineResult ||
                baselineResult.detections.filter((d) => d.confidence >= minConf).length === 0) && (
                <p className="text-xs text-[#64748b] text-center py-3">
                  No insect specimens above {Math.round(minConf * 100)}% threshold
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Model 2: Improved Paddy YOLO EMA + Dysample */}
        <div className="bg-[#111622] rounded-xl border border-[#232d42] p-5 flex flex-col justify-between">
          <div>
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-[#232d42] mb-4">
              <div>
                <div className="flex items-center space-x-2">
                  <span className="h-2 w-2 rounded-full bg-[#10b981]" />
                  <h3 className="font-bold text-white text-base tracking-tight">
                    Paddy YOLO EMA + Dysample
                  </h3>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/30">
                    Improved Architecture
                  </span>
                </div>
                <p className="text-xs text-[#64748b] mt-0.5">
                  EMA Multi-Scale Attention + DySample Dynamic Sampling
                </p>
              </div>

              {improvedResult && (
                <div className="text-right">
                  <span className="text-xs font-mono tabular-nums text-[#10b981] font-bold">
                    {improvedResult.timing.total_ms} ms
                  </span>
                  <p className="text-[10px] text-[#64748b] font-mono">
                    {(1000 / improvedResult.timing.total_ms).toFixed(1)} FPS
                  </p>
                </div>
              )}
            </div>

            {/* Canvas View */}
            <DetectionCanvas
              imageUrl={imageUrl}
              detections={improvedResult ? improvedResult.detections : []}
              modelTheme="improved"
              minConfidence={minConf}
            />
          </div>

          {/* Detections Inventory */}
          <div className="mt-4 pt-3 border-t border-[#232d42]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">
                Detected Pests (
                {improvedResult
                  ? improvedResult.detections.filter((d) => d.confidence >= minConf).length
                  : 0}
                )
              </span>
              {improvedResult?.is_mock && (
                <span className="text-[10px] text-[#94a3b8] font-mono bg-[#182030] px-1.5 py-0.5 rounded border border-[#232d42]">
                  Calibration Mode
                </span>
              )}
            </div>

            <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
              {improvedResult?.detections
                .filter((d) => d.confidence >= minConf)
                .map((d, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs px-2.5 py-1.5 rounded bg-[#0a0d14]/70 border border-[#232d42]"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-[#10b981]" />
                      <span className="font-medium text-white capitalize">
                        {d.species_name}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 font-mono text-[11px] tabular-nums text-[#94a3b8]">
                      <span>box: [{d.bbox.map((v) => Math.round(v)).join(', ')}]</span>
                      <span className="font-bold text-[#10b981]">
                        {(d.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              {(!improvedResult ||
                improvedResult.detections.filter((d) => d.confidence >= minConf).length === 0) && (
                <p className="text-xs text-[#64748b] text-center py-3">
                  No insect specimens above {Math.round(minConf * 100)}% threshold
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
