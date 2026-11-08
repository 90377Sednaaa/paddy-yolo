import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Dropzone } from './components/Dropzone';
import { SpeedShowdownBanner } from './components/SpeedShowdownBanner';
import { ModelArena } from './components/ModelArena';
import { StatisticsDashboard } from './components/StatisticsDashboard';
import type { DualPredictionResponse, SampleImage, StatisticsData } from './types';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'simulator' | 'statistics'>('simulator');
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(false);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [stats, setStats] = useState<StatisticsData | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dualResult, setDualResult] = useState<DualPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initialize data on mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const healthRes = await fetch('/api/health');
        if (healthRes.ok) {
          setIsBackendHealthy(true);
        }

        const statsRes = await fetch('/api/statistics');
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          setStats(statsData);
        }

        const samplesRes = await fetch('/api/samples');
        if (samplesRes.ok) {
          const samplesData = await samplesRes.json();
          setSamples(samplesData.samples || []);
        }
      } catch (err) {
        console.warn('Backend connection warning:', err);
        setIsBackendHealthy(false);
      }
    };

    fetchInitialData();
  }, []);

  const handleImageSelected = async (file: File | null, url: string) => {
    setPreviewUrl(url);
    setErrorMsg(null);

    if (file) {
      await executeInference(file);
    }
  };

  const executeInference = async (file: File) => {
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
        throw new Error(`Inference service error: ${response.statusText}`);
      }

      const data: DualPredictionResponse = await response.json();
      setDualResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to execute inference.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0d14] text-[#f1f5f9] flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendHealthy={isBackendHealthy}
      />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {errorMsg && (
          <div className="bg-[#ef4444]/10 border border-[#ef4444]/30 rounded-lg p-3.5 flex items-center space-x-3 text-xs text-[#ef4444]">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {activeTab === 'simulator' ? (
          <div className="space-y-6">
            {/* Image Dropzone & Sample Picker */}
            <Dropzone
              onImageSelected={handleImageSelected}
              isLoading={isLoading}
              samples={samples}
            />

            {/* Asynchronous Speed Showdown Banner */}
            {dualResult && previewUrl && (
              <SpeedShowdownBanner
                showdown={dualResult.showdown}
                baselineTiming={dualResult.baseline.timing}
                improvedTiming={dualResult.improved.timing}
              />
            )}

            {/* Side-by-Side Model Arena */}
            {previewUrl ? (
              <ModelArena
                imageUrl={previewUrl}
                baselineResult={dualResult ? dualResult.baseline : null}
                improvedResult={dualResult ? dualResult.improved : null}
                isLoading={isLoading}
              />
            ) : (
              <div className="bg-[#111622] rounded-xl border border-[#232d42] p-16 text-center">
                <p className="text-sm font-medium text-white mb-1">
                  Simulator Ready for Evaluation
                </p>
                <p className="text-xs text-[#64748b] max-w-md mx-auto">
                  Drag and drop a field insect photo above or click one of the standard benchmark sample specimens to compare detection speed and precision.
                </p>
              </div>
            )}
          </div>
        ) : (
          /* Statistics Dashboard */
          <StatisticsDashboard stats={stats} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#232d42] bg-[#0a0d14] py-4 text-center text-xs text-[#64748b] font-mono">
        Paddy-YOLO Agricultural Vision Evaluation System · Comparative Benchmarking Arena
      </footer>
    </div>
  );
};

export default App;
