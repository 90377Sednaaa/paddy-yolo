import React, { useRef } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, Upload, Loader2 } from 'lucide-react';
import type { SampleImage } from '../types';

interface SpecimenSelectorProps {
  samples: SampleImage[];
  selectedSampleId: string | null;
  onSelectSample: (sample: SampleImage) => void;
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  onStep: (direction: -1 | 1) => void;
}

export const SpecimenSelector: React.FC<SpecimenSelectorProps> = ({
  samples,
  selectedSampleId,
  onSelectSample,
  onFileUpload,
  isLoading,
  onStep,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const currentIndex = samples.findIndex((s) => s.id === selectedSampleId);
  const currentSample = currentIndex >= 0 ? samples[currentIndex] : null;
  const binomial = currentSample?.description.split('(')[0].trim() ?? 'Locally supplied photograph';
  const context = currentSample
    ? currentSample.description.match(/\(([^)]+)\)/)?.[1] ?? 'Benchmark specimen'
    : 'Custom specimen · not part of benchmark set';

  return (
    <section className="animate-rise" aria-labelledby="specimen-heading">
      <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
        {/* Editorial specimen identity */}
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <span className="eyebrow eyebrow-ink">Run header / specimen</span>
            <span className="h-px w-8 bg-line-strong" />
            <span className="eyebrow numeric">
              {currentSample ? String(currentIndex + 1).padStart(2, '0') : '—'} /{' '}
              {String(samples.length).padStart(2, '0')}
            </span>
          </div>
          <h2
            id="specimen-heading"
            className="mt-2 font-display text-[clamp(32px,4.2vw,50px)] leading-[1.02] font-medium tracking-[-0.03em] text-ink"
          >
            {currentSample?.name ?? 'Custom Specimen'}
          </h2>
          <p className="mt-1.5 font-display text-[16px] italic text-ink-2">
            {binomial}
            <span className="ml-2 font-sans text-[12px] not-italic text-ink-3">· {context}</span>
          </p>
        </div>

        {/* Controls */}
        <div className="flex shrink-0 flex-col gap-2.5 lg:items-end">
          <span className="eyebrow">Specimen control</span>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => onStep(-1)}
                disabled={isLoading || samples.length === 0}
                className="ctl h-10 w-10"
                title="Previous specimen (←)"
                aria-label="Previous specimen"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>

              <div className="relative">
                <label htmlFor="specimen-select" className="sr-only">
                  Select benchmark specimen
                </label>
                <select
                  id="specimen-select"
                  value={currentSample ? currentSample.id : ''}
                  onChange={(e) => {
                    const found = samples.find((s) => s.id === e.target.value);
                    if (found) onSelectSample(found);
                  }}
                  disabled={isLoading}
                  className="h-10 w-[228px] appearance-none rounded-[4px] border border-line bg-surface pl-3 pr-9 text-[12px] font-medium text-ink transition-colors hover:border-line-strong focus:border-leaf focus:outline-none disabled:opacity-50 sm:w-[264px]"
                >
                  {samples.map((s, idx) => (
                    <option key={s.id} value={s.id}>
                      {String(idx + 1).padStart(2, '0')} · {s.name}
                    </option>
                  ))}
                  {!currentSample && <option value="">Custom upload</option>}
                </select>
                <ChevronDown className="pointer-events-none absolute top-3 right-3 h-4 w-4 text-ink-3" />
              </div>

              <button
                type="button"
                onClick={() => onStep(1)}
                disabled={isLoading || samples.length === 0}
                className="ctl h-10 w-10"
                title="Next specimen (→)"
                aria-label="Next specimen"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFileUpload(file);
                e.target.value = '';
              }}
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="ctl ctl-solid h-10 gap-2 px-4 text-[12px] font-semibold"
              title="Upload your own specimen photograph"
            >
              {isLoading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Upload className="h-3.5 w-3.5" />
              )}
              {isLoading ? 'Evaluating…' : 'Upload specimen'}
            </button>
          </div>

          {/* Clickable position ticks */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1" role="group" aria-label="Jump to specimen">
              {samples.map((s, idx) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => onSelectSample(s)}
                  disabled={isLoading}
                  title={s.name}
                  aria-label={`Specimen ${idx + 1}: ${s.name}`}
                  aria-current={s.id === selectedSampleId ? 'true' : undefined}
                  className={`h-[3px] w-6 rounded-full transition-all duration-300 ${
                    s.id === selectedSampleId
                      ? 'bg-leaf'
                      : 'bg-line-strong hover:bg-ink-3'
                  }`}
                />
              ))}
            </div>
            <span className="eyebrow ml-1 hidden xl:inline">← → to step</span>
          </div>
        </div>
      </div>

      <div className="mt-6 h-px w-full bg-line" />
    </section>
  );
};
