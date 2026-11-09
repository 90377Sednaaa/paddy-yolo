import React, { useRef } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, UploadCloud, RefreshCw } from 'lucide-react';
import type { SampleImage } from '../types';

interface SpecimenSelectorProps {
  samples: SampleImage[];
  selectedSampleId: string | null;
  onSelectSample: (sample: SampleImage) => void;
  onFileUpload: (file: File) => void;
  isLoading: boolean;
}

export const SpecimenSelector: React.FC<SpecimenSelectorProps> = ({
  samples,
  selectedSampleId,
  onSelectSample,
  onFileUpload,
  isLoading,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const currentIndex = samples.findIndex((s) => s.id === selectedSampleId);
  const currentSample = currentIndex >= 0 ? samples[currentIndex] : null;

  const handlePrev = () => {
    if (samples.length === 0) return;
    const prevIdx = (currentIndex - 1 + samples.length) % samples.length;
    onSelectSample(samples[prevIdx]);
  };

  const handleNext = () => {
    if (samples.length === 0) return;
    const nextIdx = (currentIndex + 1) % samples.length;
    onSelectSample(samples[nextIdx]);
  };

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const found = samples.find((s) => s.id === e.target.value);
    if (found) {
      onSelectSample(found);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-3 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 shadow-xs">
      {/* Left: Clean Specimen Dropdown & Stepper */}
      <div className="flex items-center space-x-2 flex-1">
        <span className="text-xs font-medium text-[#8b949e] whitespace-nowrap">
          Benchmark Specimen:
        </span>

        {/* Stepper < */}
        <button
          type="button"
          onClick={handlePrev}
          disabled={isLoading || samples.length === 0}
          className="p-1.5 rounded-md bg-[#171e28] hover:bg-[#1f2735] border border-[#1c2432] text-[#8b949e] hover:text-white disabled:opacity-40 transition-colors"
          title="Previous specimen (Left arrow)"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>

        {/* Dropdown Menu */}
        <div className="relative flex-1 max-w-md">
          <select
            value={currentSample ? currentSample.id : ''}
            onChange={handleSelectChange}
            disabled={isLoading}
            className="w-full appearance-none bg-[#171e28] hover:bg-[#1f2735] text-xs font-medium text-white pl-3 pr-8 py-2 rounded-lg border border-[#1c2432] focus:outline-none focus:border-[#388bfd] cursor-pointer transition-colors"
          >
            {samples.map((s, idx) => (
              <option key={s.id} value={s.id} className="bg-[#11161d] text-white">
                {idx + 1}. {s.name} ({s.description.split('(')[0].trim()})
              </option>
            ))}
            {!currentSample && <option value="">Custom Uploaded Photo</option>}
          </select>
          <ChevronDown className="absolute right-2.5 top-2.5 h-3.5 w-3.5 text-[#8b949e] pointer-events-none" />
        </div>

        {/* Stepper > */}
        <button
          type="button"
          onClick={handleNext}
          disabled={isLoading || samples.length === 0}
          className="p-1.5 rounded-md bg-[#171e28] hover:bg-[#1f2735] border border-[#1c2432] text-[#8b949e] hover:text-white disabled:opacity-40 transition-colors"
          title="Next specimen (Right arrow)"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      {/* Right: Upload Custom Photo */}
      <div className="shrink-0 flex items-center space-x-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={handleFileInput}
          disabled={isLoading}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg border text-xs font-medium transition-colors ${
            isLoading
              ? 'bg-[#171e28] border-[#1c2432] text-[#8b949e] cursor-wait'
              : 'bg-[#171e28] hover:bg-[#1f2735] border-[#1c2432] hover:border-[#283344] text-white shadow-xs'
          }`}
        >
          {isLoading ? (
            <RefreshCw className="h-3.5 w-3.5 animate-spin text-[#8b949e]" />
          ) : (
            <UploadCloud className="h-3.5 w-3.5 text-[#8b949e]" />
          )}
          <span>{isLoading ? 'Processing...' : 'Upload Photo'}</span>
        </button>
      </div>
    </div>
  );
};
