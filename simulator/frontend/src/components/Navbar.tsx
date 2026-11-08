import React from 'react';
import { BarChart3, Cpu, Layers } from 'lucide-react';

interface NavbarProps {
  activeTab: 'simulator' | 'statistics';
  setActiveTab: (tab: 'simulator' | 'statistics') => void;
  isBackendHealthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isBackendHealthy,
}) => {
  return (
    <header className="border-b border-[#232d42] bg-[#0a0d14]/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-[#182030] border border-[#334155] flex items-center justify-center text-[#10b981]">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold tracking-tight text-white text-base">
                Paddy-YOLO
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-[#182030] border border-[#232d42] text-[#94a3b8] font-mono">
                Vision Arena
              </span>
            </div>
            <p className="text-xs text-[#64748b]">
              Rice Pest Detection & Architecture Simulator
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 p-1 bg-[#111622] rounded-lg border border-[#232d42]">
          <button
            onClick={() => setActiveTab('simulator')}
            className={`flex items-center space-x-2 px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'simulator'
                ? 'bg-[#182030] text-white shadow-sm border border-[#334155]'
                : 'text-[#94a3b8] hover:text-white'
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>Dual Model Simulator</span>
          </button>
          <button
            onClick={() => setActiveTab('statistics')}
            className={`flex items-center space-x-2 px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'statistics'
                ? 'bg-[#182030] text-white shadow-sm border border-[#334155]'
                : 'text-[#94a3b8] hover:text-white'
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>Statistics Comparison</span>
          </button>
        </nav>

        {/* Backend Status indicator */}
        <div className="flex items-center space-x-2">
          <div
            className={`h-2 w-2 rounded-full ${
              isBackendHealthy ? 'bg-[#10b981]' : 'bg-[#ef4444]'
            }`}
          />
          <span className="text-xs text-[#94a3b8] font-mono">
            {isBackendHealthy ? 'Service Active' : 'Connecting...'}
          </span>
        </div>
      </div>
    </header>
  );
};
