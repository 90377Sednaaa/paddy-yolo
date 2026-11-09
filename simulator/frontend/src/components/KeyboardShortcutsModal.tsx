import React from 'react';
import { X, Command } from 'lucide-react';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const shortcuts = [
    { key: '1 - 5', desc: 'Select benchmark insect specimen' },
    { key: 'V', desc: 'Cycle inspection mode (Side-by-Side / Diff / Single)' },
    { key: '+ / -', desc: 'Zoom in / Zoom out on viewport' },
    { key: '0', desc: 'Reset zoom to fit view' },
    { key: 'M', desc: 'Toggle Simulator / Statistics tab' },
    { key: 'L', desc: 'Toggle detection labels (Labels / Scores / Off)' },
    { key: '?', desc: 'Toggle keyboard shortcuts cheatsheet' },
    { key: 'Esc', desc: 'Close dialog or deselect insect' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-md bg-[#11161d] border border-[#1c2432] rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#1c2432] bg-[#171e28]">
          <div className="flex items-center space-x-2">
            <Command className="h-4 w-4 text-[#8b949e]" />
            <h3 className="text-sm font-semibold text-white tracking-tight">Workbench Keyboard Shortcuts</h3>
          </div>
          <button
            onClick={onClose}
            className="text-[#656d76] hover:text-white p-1 rounded transition-colors"
            title="Close (Esc)"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5 space-y-2.5">
          {shortcuts.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-[#171e28] transition-colors"
            >
              <span className="text-xs text-[#8b949e]">{item.desc}</span>
              <kbd className="px-2 py-0.5 text-xs font-mono font-medium text-white bg-[#171e28] border border-[#1c2432] rounded shadow-xs">
                {item.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="px-5 py-3 border-t border-[#1c2432] bg-[#0b0f14] text-right">
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 text-xs font-medium text-white bg-[#171e28] hover:bg-[#1f2735] border border-[#1c2432] rounded-md transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
