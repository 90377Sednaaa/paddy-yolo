import React, { useCallback, useState } from 'react';
import { Image as ImageIcon, UploadCloud, RefreshCw } from 'lucide-react';
import type { SampleImage } from '../types';

interface DropzoneProps {
  onImageSelected: (file: File | null, previewUrl: string) => void;
  isLoading: boolean;
  samples: SampleImage[];
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onImageSelected,
  isLoading,
  samples,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedName, setSelectedName] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file (JPG, PNG, WEBP).');
        return;
      }
      setSelectedName(file.name);
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    },
    [handleFile]
  );

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const onDragLeave = () => {
    setIsDragOver(false);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const loadSample = async (sample: SampleImage) => {
    try {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      const file = new File([blob], sample.id, { type: 'image/jpeg' });
      setSelectedName(sample.name);
      const url = URL.createObjectURL(blob);
      onImageSelected(file, url);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  return (
    <div className="bg-[#111622] rounded-xl border border-[#232d42] p-5">
      <div className="flex flex-col md:flex-row gap-5 items-stretch">
        {/* Main Drag-and-Drop Area */}
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`flex-1 flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
            isDragOver
              ? 'border-[#10b981] bg-[#10b981]/5'
              : 'border-[#334155] hover:border-[#64748b] bg-[#0a0d14]/40'
          }`}
          onClick={() => document.getElementById('pest-image-input')?.click()}
        >
          <input
            id="pest-image-input"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onInputChange}
            disabled={isLoading}
          />
          <div className="h-10 w-10 rounded-full bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#94a3b8] mb-3">
            {isLoading ? (
              <RefreshCw className="h-5 w-5 animate-spin text-[#10b981]" />
            ) : (
              <UploadCloud className="h-5 w-5" />
            )}
          </div>
          <p className="text-sm font-medium text-white mb-1">
            {selectedName ? (
              <span className="text-[#10b981]">Selected: {selectedName}</span>
            ) : (
              'Drag and drop an insect photograph here, or browse'
            )}
          </p>
          <p className="text-xs text-[#64748b]">
            Supports RGB field images in JPG, PNG, and WebP format (640x640 recommended)
          </p>
        </div>

        {/* Quick Sample Selector */}
        <div className="md:w-80 flex flex-col justify-between bg-[#0a0d14]/60 rounded-lg p-4 border border-[#232d42]">
          <div>
            <div className="flex items-center space-x-2 mb-2 text-[#94a3b8]">
              <ImageIcon className="h-4 w-4 text-[#10b981]" />
              <span className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">
                Standard Benchmark Samples
              </span>
            </div>
            <p className="text-xs text-[#64748b] mb-3">
              Evaluate instantly with representative field specimens:
            </p>
            <div className="grid grid-cols-2 gap-2">
              {samples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => loadSample(s)}
                  disabled={isLoading}
                  className="px-2.5 py-1.5 rounded bg-[#182030] hover:bg-[#232d42] border border-[#232d42] text-left text-xs text-[#f1f5f9] transition-colors truncate"
                  title={s.description}
                >
                  <span className="block font-medium truncate">{s.name}</span>
                  <span className="block text-[10px] text-[#64748b] truncate">
                    {s.id.replace('sample_', '').replace('.jpg', '')}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
