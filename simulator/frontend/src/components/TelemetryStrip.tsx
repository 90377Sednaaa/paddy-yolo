import React from 'react';
import { Activity } from 'lucide-react';
import type { Showdown, Timing } from '../types';

interface TelemetryStripProps {
  showdown: Showdown;
  baselineTiming: Timing;
  improvedTiming: Timing;
}

export const TelemetryStrip: React.FC<TelemetryStripProps> = ({
  showdown,
  baselineTiming,
  improvedTiming,
}) => {
  const isImprovedFaster = showdown.faster_model === 'Paddy YOLO EMA + Dysample';
  const isBaselineFaster = showdown.faster_model === 'Paddy YOLO';

  return (
    <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-3.5 shadow-xs">
      <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3.5">
        {/* Left: Telemetry Headline */}
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-[#171e28] border border-[#1c2432] flex items-center justify-center shrink-0">
            <Activity className="h-4 w-4 text-[#8b949e]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold text-white tracking-tight">
                Execution Latency Telemetry
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#171e28] border border-[#283344] text-white font-medium">
                {showdown.speedup_pct > 0 ? `+${showdown.speedup_pct}% Throughput` : 'Parity'}
              </span>
              <span className="text-[11px] font-mono text-[#8b949e]">
                Δ {showdown.delta_ms.toFixed(1)} ms
              </span>
            </div>
            <p className="text-[11px] text-[#656d76] mt-0.5 font-mono">
              FP16 TensorRT · Batch size 1 · Input resolution 640×640
            </p>
          </div>
        </div>

        {/* Right: Dual Channel Latency Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 flex-1 max-w-xl">
          {/* Baseline Paddy YOLO Channel */}
          <div
            className={`px-3 py-2 rounded-lg border transition-colors ${
              isBaselineFaster
                ? 'bg-[#171e28] border-[#283344]'
                : 'bg-[#0b0f14] border-[#1c2432]'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-medium text-[#8b949e] text-[11px]">
                Paddy YOLO (Baseline)
              </span>
              <div className="flex items-baseline space-x-1 font-mono text-xs">
                <span className="font-semibold text-white tabular-nums">
                  {baselineTiming.total_ms.toFixed(1)}
                </span>
                <span className="text-[10px] text-[#656d76]">ms</span>
                <span className="text-[10px] text-[#8b949e] ml-1 tabular-nums">
                  ({showdown.baseline_fps.toFixed(0)} FPS)
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between text-[10px] text-[#656d76] font-mono tabular-nums pt-1 border-t border-[#1c2432]">
              <span>pre: {baselineTiming.preprocess_ms.toFixed(1)}ms</span>
              <span className="text-[#8b949e]">infer: {baselineTiming.inference_ms.toFixed(1)}ms</span>
              <span>nms: {baselineTiming.postprocess_ms.toFixed(1)}ms</span>
            </div>
          </div>

          {/* Enhanced Paddy YOLO EMA + DySample Channel */}
          <div
            className={`px-3 py-2 rounded-lg border transition-colors ${
              isImprovedFaster
                ? 'bg-[#171e28] border-[#283344]'
                : 'bg-[#0b0f14] border-[#1c2432]'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-medium text-white text-[11px]">
                Paddy YOLO EMA + DySample
              </span>
              <div className="flex items-baseline space-x-1 font-mono text-xs">
                <span className="font-semibold text-white tabular-nums">
                  {improvedTiming.total_ms.toFixed(1)}
                </span>
                <span className="text-[10px] text-[#656d76]">ms</span>
                <span className="text-[10px] text-[#8b949e] ml-1 tabular-nums">
                  ({showdown.improved_fps.toFixed(0)} FPS)
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between text-[10px] text-[#656d76] font-mono tabular-nums pt-1 border-t border-[#1c2432]">
              <span>pre: {improvedTiming.preprocess_ms.toFixed(1)}ms</span>
              <span className="text-[#8b949e]">infer: {improvedTiming.inference_ms.toFixed(1)}ms</span>
              <span>nms: {improvedTiming.postprocess_ms.toFixed(1)}ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
