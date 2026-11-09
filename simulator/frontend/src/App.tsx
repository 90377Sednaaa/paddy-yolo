import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Navbar } from './components/Navbar';
import type { Theme, WorkspaceTab } from './components/Navbar';
import { SpecimenSelector } from './components/SpecimenSelector';
import { TelemetryStrip } from './components/TelemetryStrip';
import { VisionArenaDeck } from './components/VisionArenaDeck';
import { StatisticsDashboard } from './components/StatisticsDashboard';
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal';
import { BENCHMARK_SAMPLES, STATIC_STATISTICS } from './data/benchmarkData';
import type { DualPredictionResponse, SampleImage, StatisticsData } from './types';
import { AlertCircle, X, RotateCw } from 'lucide-react';

const UNAVAILABLE_MESSAGE =
  'The inference engine is not responding. Start the backend on 127.0.0.1:8000, then retry — the console keeps the current specimen visible in the meantime.';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('simulator');
  const [samples, setSamples] = useState<SampleImage[]>(BENCHMARK_SAMPLES);
  const [stats, setStats] = useState<StatisticsData>(STATIC_STATISTICS);

  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(BENCHMARK_SAMPLES[0].id);
  const [previewUrl, setPreviewUrl] = useState<string | null>(BENCHMARK_SAMPLES[0].url);
  const [dualResult, setDualResult] = useState<DualPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState<boolean>(false);

  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.dataset.theme === 'darkroom' ? 'darkroom' : 'bench'
  );
  const lastFileRef = useRef<File | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  // Theme is persisted and applied at the document root so canvas + portal layers follow it.
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('paddy-theme', theme);
    } catch {
      /* storage may be unavailable in private mode */
    }
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', theme === 'bench' ? '#f2f0e8' : '#0d120e');
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'bench' ? 'darkroom' : 'bench'));
  }, []);

  const executeInference = useCallback(async (file: File) => {
    lastFileRef.current = file;
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
        throw new Error(`Inference engine returned ${response.status} ${response.statusText}. Check the backend log and retry.`);
      }

      const data: DualPredictionResponse = await response.json();
      setDualResult(data);
    } catch (err) {
      const message = err instanceof TypeError ? UNAVAILABLE_MESSAGE : (err as Error).message;
      setErrorMsg(message || 'Inference could not be completed.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRetry = useCallback(() => {
    if (lastFileRef.current) void executeInference(lastFileRef.current);
  }, [executeInference]);

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

  const handleFileUpload = useCallback(
    async (file: File) => {
      setSelectedSampleId(null);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setPreviewUrl(url);
      await executeInference(file);
    },
    [executeInference]
  );

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  // Initialize data and auto-evaluate the first specimen on mount.
  useEffect(() => {
    let isMounted = true;

    const fetchInitialData = async () => {
      try {
        const statsRes = await fetch('/api/statistics');
        if (statsRes.ok && isMounted) {
          setStats(await statsRes.json());
        }

        const samplesRes = await fetch('/api/samples');
        if (samplesRes.ok && isMounted) {
          const samplesData = await samplesRes.json();
          if (samplesData.samples && samplesData.samples.length > 0) {
            setSamples(samplesData.samples);
          }
        }
      } catch {
        // Benchmark cache is a valid offline state; no user-facing noise here.
      }
    };

    const initSample = async () => {
      try {
        const response = await fetch(BENCHMARK_SAMPLES[0].url);
        const blob = await response.blob();
        const file = new File([blob], BENCHMARK_SAMPLES[0].id, { type: 'image/jpeg' });
        if (isMounted) await executeInference(file);
      } catch (err) {
        console.warn('Initial specimen evaluation notice:', err);
      }
    };

    void fetchInitialData();
    void initSample();

    return () => {
      isMounted = false;
    };
  }, [executeInference]);

  const stepSample = useCallback(
    (direction: -1 | 1) => {
      if (samples.length === 0 || isLoading) return;
      const idx = samples.findIndex((s) => s.id === selectedSampleId);
      const next = idx < 0 ? 0 : (idx + direction + samples.length) % samples.length;
      void handleSelectSample(samples[next]);
    },
    [samples, selectedSampleId, isLoading, handleSelectSample]
  );

  // Global keyboard shortcuts. Inspection-stage keys live in VisionArenaDeck.
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === 'Escape') {
        if (isShortcutsOpen) setIsShortcutsOpen(false);
        return;
      }
      if (e.key === '?') {
        e.preventDefault();
        setIsShortcutsOpen((o) => !o);
        return;
      }
      if (isShortcutsOpen) return;

      const el = document.activeElement;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT')) return;

      const key = e.key.toLowerCase();
      if (key === 'm') {
        setActiveTab((t) => (t === 'simulator' ? 'statistics' : 'simulator'));
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stepSample(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stepSample(1);
      } else if (['1', '2', '3', '4', '5'].includes(e.key)) {
        const sample = samples[parseInt(e.key, 10) - 1];
        if (sample) void handleSelectSample(sample);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isShortcutsOpen, samples, handleSelectSample, stepSample]);

  return (
    <div className="relative flex min-h-screen flex-col">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="relative z-10 mx-auto w-full max-w-[1440px] flex-1 px-5 pb-14 sm:px-8 lg:px-10">
        {errorMsg && (
          <div
            role="alert"
            className="animate-rise mt-6 flex items-start gap-3 rounded-[5px] border border-negative/30 bg-negative-soft px-4 py-3"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-negative" />
            <div className="flex-1">
              <p className="text-[12px] font-semibold text-negative">Inference unavailable</p>
              <p className="mt-0.5 text-[12px] leading-relaxed text-ink-2">{errorMsg}</p>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              className="ctl h-8 shrink-0 gap-1.5 px-2.5 text-[11px] font-medium"
            >
              <RotateCw className="h-3 w-3" />
              Retry
            </button>
            <button
              type="button"
              onClick={() => setErrorMsg(null)}
              className="ctl h-8 w-8 shrink-0"
              aria-label="Dismiss notice"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {activeTab === 'simulator' ? (
          <div className="space-y-7 pt-7">
            <SpecimenSelector
              samples={samples}
              selectedSampleId={selectedSampleId}
              onSelectSample={handleSelectSample}
              onFileUpload={handleFileUpload}
              isLoading={isLoading}
              onStep={stepSample}
            />

            {dualResult && previewUrl && (
              <TelemetryStrip
                showdown={dualResult.showdown}
                baselineTiming={dualResult.baseline.timing}
                improvedTiming={dualResult.improved.timing}
              />
            )}

            {previewUrl && (
              <VisionArenaDeck
                imageUrl={previewUrl}
                baselineResult={dualResult ? dualResult.baseline : null}
                improvedResult={dualResult ? dualResult.improved : null}
                isLoading={isLoading}
                shortcutsEnabled={!isShortcutsOpen}
              />
            )}
          </div>
        ) : (
          <StatisticsDashboard stats={stats} />
        )}
      </main>

      <footer className="relative z-10 border-t border-line">
        <div className="mx-auto flex max-w-[1440px] flex-col gap-2 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-8 lg:px-10">
          <span className="eyebrow">Paddy-YOLO · Field Evaluation Console</span>
          <span className="text-[11px] text-ink-3">
            Comparative evaluation of Paddy YOLO and Paddy YOLO EMA + DySample
          </span>
          <span className="eyebrow">Press <kbd className="numeric text-ink-2">?</kbd> for shortcuts</span>
        </div>
      </footer>

      <KeyboardShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />
    </div>
  );
};

export default App;
