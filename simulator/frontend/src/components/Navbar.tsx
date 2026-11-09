import React from 'react';
import { Sun, Moon, Keyboard } from 'lucide-react';
import { BrandMark } from './BrandMark';

export type WorkspaceTab = 'simulator' | 'statistics';
export type Theme = 'bench' | 'darkroom';

interface NavbarProps {
  activeTab: WorkspaceTab;
  setActiveTab: (tab: WorkspaceTab) => void;
  onOpenShortcuts: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

const TABS: { id: WorkspaceTab; index: string; label: string }[] = [
  { id: 'simulator', index: '01', label: 'Inspection Workbench' },
  { id: 'statistics', index: '02', label: 'Benchmark Evidence' },
];

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onOpenShortcuts,
  theme,
  onToggleTheme,
}) => {
  const nextTheme = theme === 'bench' ? 'Darkroom' : 'Bench';

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-paper/92 backdrop-blur-xl">
      {/* Edition line */}
      <div className="border-b border-line/70">
        <div className="mx-auto flex h-8 max-w-[1440px] items-center justify-between px-5 sm:px-8 lg:px-10">
          <span className="eyebrow">Field station / comparative model evaluation</span>
          <span className="eyebrow hidden sm:inline">Rice pest detection · 16 species · dual architecture</span>
        </div>
      </div>

      {/* Masthead */}
      <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4 px-5 py-3.5 sm:px-8 lg:px-10">
        <div className="flex min-w-0 items-center gap-3.5">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[5px] border border-line-strong text-ink">
            <BrandMark size={30} />
          </span>
          <div className="min-w-0">
            <div className="flex items-baseline gap-2.5">
              <h1 className="truncate font-display text-[26px] leading-none font-medium tracking-[-0.02em] text-ink">
                Paddy-YOLO
              </h1>
              <span className="hidden text-[11px] text-ink-3 sm:inline">/</span>
              <span className="hidden truncate text-[13px] text-ink-2 sm:inline">
                Field Evaluation Console
              </span>
            </div>
            <p className="eyebrow mt-1.5 hidden sm:block">Model evaluation · research instrument</p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={onToggleTheme}
            className="ctl h-9 px-2.5 text-[11px] font-medium tracking-wide"
            title={`Switch to ${nextTheme} theme`}
            aria-label={`Switch to ${nextTheme} theme`}
          >
            {theme === 'bench' ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
            <span className="hidden md:inline">{nextTheme}</span>
          </button>
          <button
            type="button"
            onClick={onOpenShortcuts}
            className="ctl h-9 gap-2 px-2.5 text-[11px] font-medium"
            title="Keyboard reference (?)"
            aria-label="Open keyboard reference"
          >
            <Keyboard className="h-3.5 w-3.5" />
            <kbd className="numeric hidden rounded-[3px] border border-line px-1 text-[10px] leading-4 text-ink-2 md:inline">
              ?
            </kbd>
          </button>
        </div>
      </div>

      {/* Workspace navigation — editorial underline tabs */}
      <div className="mx-auto max-w-[1440px] px-5 sm:px-8 lg:px-10">
        <nav aria-label="Workspaces" className="flex items-end justify-between gap-6 overflow-x-auto">
          <div className="flex items-end gap-7">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                data-active={activeTab === tab.id}
                aria-current={activeTab === tab.id ? 'page' : undefined}
                className="tab-link flex items-center gap-2"
              >
                <span className="numeric text-[10px] tracking-wider opacity-60">{tab.index}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
          <span className="eyebrow hidden shrink-0 pb-3.5 lg:inline">
            {activeTab === 'simulator'
              ? 'Live inference / side-by-side'
              : 'Published results / reproducible audit'}
          </span>
        </nav>
      </div>
    </header>
  );
};
