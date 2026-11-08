import React from 'react';
import { Award } from 'lucide-react';
import type { Showdown, Timing } from '../types';

interface SpeedShowdownBannerProps {
  showdown: Showdown;
  baselineTiming: Timing;
  improvedTiming: Timing;
}

export const SpeedShowdownBanner: React.FC<SpeedShowdownBannerProps> = ({
  showdown,
  baselineTiming,
  improvedTiming,
}) => {
  const isImprovedWinner = showdown.winner === 'Paddy YOLO EMA + Dysample';
  const isBaselineWinner = showdown.winner === 'Paddy YOLO';

  return (
    <div className="bg-[#111622] rounded-xl border border-[#232d42] p-5">
      <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-5">
        {/* Left: Winner Designation */}
        <div className="flex items-center space-x-4">
          <div
            className={`h-12 w-12 rounded-xl flex items-center justify-center border ${
              isImprovedWinner
                ? 'bg-[#10b981]/10 border-[#10b981]/30 text-[#10b981]'
                : isBaselineWinner
                ? 'bg-[#38bdf8]/10 border-[#38bdf8]/30 text-[#38bdf8]'
                : 'bg-[#182030] border-[#334155] text-[#94a3b8]'
            }`}
          >
            <Award className="h-6 w-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#64748b]">
                Speed Benchmark Winner
              </span>
              <span
                className={`text-xs px-2 py-0.5 rounded font-mono font-medium ${
                  isImprovedWinner
                    ? 'bg-[#10b981]/20 text-[#10b981] border border-[#10b981]/30'
                    : 'bg-[#38bdf8]/20 text-[#38bdf8] border border-[#38bdf8]/30'
                }`}
              >
                +{showdown.speedup_pct}% Throughput
              </span>
            </div>
            <h3 className="text-lg font-bold text-white tracking-tight">
              {showdown.winner}
            </h3>
            <p className="text-xs text-[#94a3b8] mt-0.5 font-mono tabular-nums">
              Completed {showdown.delta_ms.toFixed(2)} ms faster than competitor
            </p>
          </div>
        </div>

        {/* Center / Right: Direct Latency Comparison */}
        <div className="grid grid-cols-2 gap-4 flex-1 max-w-xl">
          {/* Baseline Panel */}
          <div
            className={`p-3.5 rounded-lg border transition-all ${
              isBaselineWinner
                ? 'bg-[#182030] border-[#38bdf8]/50 ring-1 ring-[#38bdf8]/20'
                : 'bg-[#0a0d14]/60 border-[#232d42]'
            }`}
          >
            <div className="flex items-center justify-between text-xs text-[#94a3b8] mb-1">
              <span className="font-medium text-[#38bdf8]">Paddy YOLO</span>
              <span className="font-mono tabular-nums">
                {showdown.baseline_fps.toFixed(1)} FPS
              </span>
            </div>
            <div className="flex items-baseline space-x-1.5">
              <span className="text-2xl font-bold font-mono tabular-nums text-white">
                {baselineTiming.total_ms.toFixed(2)}
              </span>
              <span className="text-xs text-[#64748b] font-mono">ms</span>
            </div>
            <div className="mt-2 pt-2 border-t border-[#232d42] flex justify-between text-[11px] text-[#64748b] font-mono tabular-nums">
              <span>pre: {baselineTiming.preprocess_ms}ms</span>
              <span>infer: {baselineTiming.inference_ms}ms</span>
              <span>post: {baselineTiming.postprocess_ms}ms</span>
            </div>
          </div>

          {/* Improved Panel */}
          <div
            className={`p-3.5 rounded-lg border transition-all ${
              isImprovedWinner
                ? 'bg-[#182030] border-[#10b981]/50 ring-1 ring-[#10b981]/20'
                : 'bg-[#0a0d14]/60 border-[#232d42]'
            }`}
          >
            <div className="flex items-center justify-between text-xs text-[#94a3b8] mb-1">
              <span className="font-medium text-[#10b981]">
                Paddy YOLO EMA + Dysample
              </span>
              <span className="font-mono tabular-nums">
                {showdown.improved_fps.toFixed(1)} FPS
              </span>
            </div>
            <div className="flex items-baseline space-x-1.5">
              <span className="text-2xl font-bold font-mono tabular-nums text-white">
                {improvedTiming.total_ms.toFixed(2)}
              </span>
              <span className="text-xs text-[#64748b] font-mono">ms</span>
            </div>
            <div className="mt-2 pt-2 border-t border-[#232d42] flex justify-between text-[11px] text-[#64748b] font-mono tabular-nums">
              <span>pre: {improvedTiming.preprocess_ms}ms</span>
              <span>infer: {improvedTiming.inference_ms}ms</span>
              <span>post: {improvedTiming.postprocess_ms}ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
