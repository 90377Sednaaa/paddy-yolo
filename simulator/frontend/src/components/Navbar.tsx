import React from 'react';
import { Eye, BarChart3, HelpCircle } from 'lucide-react';
import logoImg from '../assets/logo.png';

interface NavbarProps {
  activeTab: 'simulator' | 'statistics';
  setActiveTab: (tab: 'simulator' | 'statistics') => void;
  onOpenShortcuts: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onOpenShortcuts,
}) => {
  return (
    <header className="border-b border-[#1c2432] bg-[#0b0f14]/95 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        {/* Left: Branding */}
        <div className="flex items-center space-x-3">
          <div className="h-8 w-8 rounded-md bg-[#11161d] border border-[#283344] p-1 flex items-center justify-center shrink-0">
            <img
              src={logoImg}
              alt="Paddy-YOLO Pest Detection"
              className="h-full w-full object-contain"
            />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-white text-sm tracking-tight">
                Paddy-YOLO
              </span>
              <span className="text-[11px] font-mono text-[#8b949e]">
                Model Evaluation Simulator
              </span>
            </div>
          </div>
        </div>

        {/* Center: Clean Segmented Navigation */}
        <nav className="flex items-center p-1 bg-[#11161d] rounded-lg border border-[#1c2432]">
          <button
            onClick={() => setActiveTab('simulator')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'simulator'
                ? 'bg-[#1f2735] text-white shadow-xs'
                : 'text-[#8b949e] hover:text-white'
            }`}
          >
            <Eye className="h-3.5 w-3.5" />
            <span>Dual Model Simulator</span>
          </button>
          <button
            onClick={() => setActiveTab('statistics')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'statistics'
                ? 'bg-[#1f2735] text-white shadow-xs'
                : 'text-[#8b949e] hover:text-white'
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>Statistics Comparison</span>
          </button>
        </nav>

        {/* Right: Keyboard Shortcuts Trigger */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onOpenShortcuts}
            className="flex items-center space-x-1 px-2.5 py-1.5 rounded-md bg-[#11161d] hover:bg-[#171e28] border border-[#1c2432] text-xs text-[#8b949e] hover:text-white transition-colors"
            title="Keyboard Shortcuts (?)"
          >
            <HelpCircle className="h-3.5 w-3.5" />
            <span className="font-mono text-[11px]">?</span>
          </button>
        </div>
      </div>
    </header>
  );
};
