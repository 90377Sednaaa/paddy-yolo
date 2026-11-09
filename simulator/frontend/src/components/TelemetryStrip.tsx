import React from 'react';
import { ArrowDownRight, Gauge } from 'lucide-react';
import type { Showdown, Timing } from '../types';

interface TelemetryStripProps {
  showdown: Showdown;
  baselineTiming: Timing;
  improvedTiming: Timing;
}

interface LatencyRowProps {
  label: string;
  qualifier: string;
  timing: Timing;
  fps: number;
  widthPct: number;
  color: string;
  isWinner: boolean;
}

const LatencyRow: React.FC<LatencyRowProps> = ({
  label,
  qualifier,
  timing,
  fps,
  widthPct,
  color,
  isWinner,
}) => (
  <div className="grid grid-cols-1 items-center gap-x-4 gap-y-1.5 sm:grid-cols-[218px_minmax(0,1fr)_92px_62px]">
    <div className="flex min-w-0 items-center gap-2.5">
      <span className="h-7 w-[3px] shrink-0 rounded-full" style={{ backgroundColor: color }} />
      <div className="min-w-0 leading-tight">
        <div className="truncate text-[12px] font-medium text-ink">{label}</div>
        <div className="eyebrow mt-0.5">{qualifier}</div>
      </div>
    </div>
    <div className="h-[7px] w-full overflow-hidden rounded-full bg-surface-2">
      <div
        className="animate-bar h-full origin-left rounded-full"
        style={{ width: `${widthPct}%`, backgroundColor: color }}
      />
    </div>
    <span className="numeric text-right text-[13px] font-medium text-ink">
      {timing.total_ms.toFixed(1)}
      <span className="ml-1 text-[10px] font-normal text-ink-3">ms</span>
    </span>
    <span className="numeric text-right text-[11px] text-ink-3">{fps.toFixed(0)} fps</span>
    {isWinner && (
      <span className="hidden sm:col-start-2 sm:row-start-2 sm:mt-0.5 sm:inline text-[10px] font-medium tracking-wide" style={{ color }}>
        ← fastest
      </span>
    )}
  </div>
);

const Breakdown: React.FC<{ label: string; timing: Timing }> = ({ label, timing }) => (
  <div className="min-w-0">
    <p className="eyebrow mb-1.5">{label}</p>
    <div className="numeric flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-ink-2">
      <span>pre <span className="text-ink">{timing.preprocess_ms.toFixed(1)}</span></span>
      <span>infer <span className="text-ink">{timing.inference_ms.toFixed(1)}</span></span>
      <span>post <span className="text-ink">{timing.postprocess_ms.toFixed(1)}</span></span>
    </div>
  </div>
);

export const TelemetryStrip: React.FC<TelemetryStripProps> = ({
  showdown,
  baselineTiming,
  improvedTiming,
}) => {
  const maxMs = Math.max(baselineTiming.total_ms, improvedTiming.total_ms, 0.01);
  const delta = baselineTiming.total_ms - improvedTiming.total_ms;
  const improvedFaster = delta > 0.05;
  const baselineFaster = delta < -0.05;
  const winnerLabel = improvedFaster
    ? 'Paddy YOLO EMA + DySample'
    : baselineFaster
      ? 'Paddy YOLO'
      : 'Statistical tie';

  return (
    <section className="animate-rise-2" aria-labelledby="telemetry-heading">
      <div className="mb-3.5 flex flex-wrap items-baseline justify-between gap-3">
        <div className="flex flex-wrap items-baseline gap-3">
          <span className="eyebrow eyebrow-ink">01 / Execution telemetry</span>
          <h2 id="telemetry-heading" className="font-display text-[23px] leading-none font-medium tracking-[-0.02em] text-ink">
            Inference latency
          </h2>
          <span className="hidden text-[12px] text-ink-3 md:inline">
            Dual-model wall clock, measured per image
          </span>
        </div>
        <span className="eyebrow">Lower is faster</span>
      </div>

      <div className="panel px-5 py-5" style={{ boxShadow: 'var(--shadow-lift)' }}>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_212px]">
          <div className="space-y-5">
            <LatencyRow
              label="Paddy YOLO"
              qualifier="Baseline"
              timing={baselineTiming}
              fps={showdown.baseline_fps}
              widthPct={(baselineTiming.total_ms / maxMs) * 100}
              color="var(--c-baseline)"
              isWinner={baselineFaster}
            />
            <LatencyRow
              label="Paddy YOLO EMA + DySample"
              qualifier="Enhanced"
              timing={improvedTiming}
              fps={showdown.improved_fps}
              widthPct={(improvedTiming.total_ms / maxMs) * 100}
              color="var(--c-leaf)"
              isWinner={improvedFaster}
            />
          </div>

          {/* Head-to-head verdict */}
          <div className="flex flex-col justify-center border-t border-line pt-4 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6">
            <div className="flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5 text-ink-3" />
              <span className="eyebrow">Head-to-head</span>
            </div>
            <div className="numeric mt-2 flex items-baseline gap-1.5 text-ink">
              <span className="font-display text-[40px] leading-none font-medium tracking-[-0.04em]">
                {delta > 0 ? '−' : delta < 0 ? '+' : ''}
                {Math.abs(delta).toFixed(1)}
              </span>
              <span className="text-[14px] text-ink-2">ms</span>
            </div>
            <p className="mt-1.5 text-[12px] leading-snug text-ink-2">
              {Math.abs(showdown.speedup_pct).toFixed(1)}% {improvedFaster ? 'lower latency' : 'difference'} on this run
            </p>
            <div className="mt-3 inline-flex max-w-full items-center gap-1.5 rounded-[3px] bg-leaf-soft px-2 py-1.5 text-[11px] font-medium text-leaf">
              <ArrowDownRight className="h-3 w-3 shrink-0" />
              <span className="truncate">{winnerLabel}</span>
            </div>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-4 border-t border-line pt-4 sm:grid-cols-2">
          <Breakdown label="Baseline phases / ms" timing={baselineTiming} />
          <Breakdown label="Enhanced phases / ms" timing={improvedTiming} />
        </div>
      </div>
    </section>
  );
};
