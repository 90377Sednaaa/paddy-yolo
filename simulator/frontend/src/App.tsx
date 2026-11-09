import React, { useEffect, useState, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { SpecimenSelector } from './components/SpecimenSelector';
import { TelemetryStrip } from './components/TelemetryStrip';
import { VisionArenaDeck } from './components/VisionArenaDeck';
import { StatisticsDashboard } from './components/StatisticsDashboard';
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal';
import { BENCHMARK_SAMPLES, STATIC_STATISTICS } from './data/benchmarkData';
import type { DualPredictionResponse, SampleImage, StatisticsData } from './types';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'simulator' | 'statistics'>('simulator');
  const [samples, setSamples] = useState<SampleImage[]>(BENCHMARK_SAMPLES);
  const [stats, setStats] = useState<StatisticsData>(STATIC_STATISTICS);

  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(BENCHMARK_SAMPLES[0].id);
  const [previewUrl, setPreviewUrl] = useState<string | null>(BENCHMARK_SAMPLES[0].url);
  const [dualResult, setDualResult] = useState<DualPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState<boolean>(false);

  // Execute inference on a File
  const executeInference = useCallback(async (file: File) => {
    setIsLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Inference engine error: ${response.statusText}`);
      }

      const data: DualPredictionResponse = await response.json();
      setDualResult(data);
    } catch (err: any) {
      console.warn('Inference request notice:', err);
      setErrorMsg(err.message || 'Failed to complete inference.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle sample selection
  const handleSelectSample = useCallback(
    async (sample: SampleImage) => {
      setSelectedSampleId(sample.id);
      setPreviewUrl(sample.url);
      setErrorMsg(null);

      try {
        const response = await fetch(sample.url);
        const blob = await response.blob();
        const file = new File([blob], sample.id, { type: 'image/jpeg' });
        await executeInference(file);
      } catch (err) {
        console.error('Failed to load sample file:', err);
      }
    },
    [executeInference]
  );

  // Handle custom file upload
  const handleFileUpload = useCallback(
    async (file: File) => {
      setSelectedSampleId(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      await executeInference(file);
    },
    [executeInference]
  );

  // Initialize data and auto-evaluate initial sample on mount
  useEffect(() => {
    let isMounted = true;

    const fetchInitialData = async () => {
      try {
        const statsRes = await fetch('/api/statistics');
        if (statsRes.ok && isMounted) {
          const statsData = await statsRes.json();
          setStats(statsData);
        }

        const samplesRes = await fetch('/api/samples');
        if (samplesRes.ok && isMounted) {
          const samplesData = await samplesRes.json();
          if (samplesData.samples && samplesData.samples.length > 0) {
            setSamples(samplesData.samples);
          }
        }
      } catch (err) {
        console.warn('Backend connection notice (using benchmark cache):', err);
      }
    };

    const initSample = async () => {
      try {
        const response = await fetch(BENCHMARK_SAMPLES[0].url);
        const blob = await response.blob();
        const file = new File([blob], BENCHMARK_SAMPLES[0].id, { type: 'image/jpeg' });
        if (isMounted) {
          await executeInference(file);
        }
      } catch (err) {
        console.warn('Initial specimen evaluation notice:', err);
      }
    };

    fetchInitialData();
    initSample();

    return () => {
      isMounted = false;
    };
  }, [executeInference]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA' ||
        document.activeElement?.tagName === 'SELECT'
      ) {
        return;
      }

      if (e.key === '?') {
        setIsShortcutsOpen((o) => !o);
      } else if (e.key === 'Escape') {
        setIsShortcutsOpen(false);
      } else if (e.key === 'm' || e.key === 'M') {
        setActiveTab((t) => (t === 'simulator' ? 'statistics' : 'simulator'));
      } else if (['1', '2', '3', '4', '5'].includes(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        if (samples[idx]) {
          handleSelectSample(samples[idx]);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [samples, handleSelectSample]);

  return (
    <div className="min-h-screen bg-[#0b0f14] text-[#e6edf3] flex flex-col font-sans">
      {/* Clean, Quiet Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-5 space-y-4">
        {/* Error notification if any */}
        {errorMsg && (
          <div className="bg-[#2d1b1f] border border-[#432328] rounded-lg p-3 flex items-center space-x-2.5 text-xs text-[#f85149]">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {activeTab === 'simulator' ? (
          <div className="space-y-4">
            {/* Clean Specimen Selector Bar */}
            <SpecimenSelector
              samples={samples}
              selectedSampleId={selectedSampleId}
              onSelectSample={handleSelectSample}
              onFileUpload={handleFileUpload}
              isLoading={isLoading}
            />

            {/* Execution Telemetry */}
            {dualResult && previewUrl && (
              <TelemetryStrip
                showdown={dualResult.showdown}
                baselineTiming={dualResult.baseline.timing}
                improvedTiming={dualResult.improved.timing}
              />
            )}

            {/* Side-by-Side Dual Viewport Arena */}
            {previewUrl && (
              <VisionArenaDeck
                imageUrl={previewUrl}
                baselineResult={dualResult ? dualResult.baseline : null}
                improvedResult={dualResult ? dualResult.improved : null}
                isLoading={isLoading}
              />
            )}
          </div>
        ) : (
          /* Statistics & Research Benchmark Evidence */
          <StatisticsDashboard stats={stats} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1c2432] bg-[#0b0f14] py-3.5 text-center text-[11px] text-[#656d76] font-mono">
        Paddy-YOLO Agricultural Computer Vision Workbench · Dual-Architecture Comparative Evaluation System
      </footer>

      {/* Keyboard Shortcuts Dialog */}
      <KeyboardShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />
    </div>
  );
};

export default App;
