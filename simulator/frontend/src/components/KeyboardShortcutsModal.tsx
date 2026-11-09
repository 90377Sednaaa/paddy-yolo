import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const GROUPS: { title: string; hint: string; items: { keys: string[]; desc: string }[] }[] = [
  {
    title: 'Workspace',
    hint: 'Navigation and selection',
    items: [
      { keys: ['1', '–', '5'], desc: 'Select a benchmark specimen' },
      { keys: ['←', '→'], desc: 'Previous / next specimen' },
      { keys: ['M'], desc: 'Switch between workspaces' },
      { keys: ['?'], desc: 'Open or close this reference' },
      { keys: ['Esc'], desc: 'Close dialog' },
    ],
  },
  {
    title: 'Inspection stage',
    hint: 'Viewport and annotation controls',
    items: [
      { keys: ['V'], desc: 'Cycle view mode: dual / diff / single' },
      { keys: ['+', '−'], desc: 'Zoom in / out' },
      { keys: ['0'], desc: 'Reset zoom and pan' },
      { keys: ['L'], desc: 'Cycle labels: full / scores / off' },
    ],
  },
];

export const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({ isOpen, onClose }) => {
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (isOpen) closeRef.current?.focus();
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#0a0f0b]/55 p-4 backdrop-blur-[3px]"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      data-modal-open="true"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcuts-title"
        className="animate-rise w-full max-w-[540px] overflow-hidden rounded-[6px] border border-line bg-surface shadow-[var(--shadow-overlay)]"
      >
        <div className="flex items-start justify-between border-b border-line px-6 py-5">
          <div>
            <p className="eyebrow">Console reference</p>
            <h2
              id="shortcuts-title"
              className="mt-1 font-display text-[26px] leading-tight font-medium tracking-[-0.02em] text-ink"
            >
              Keyboard shortcuts
            </h2>
            <p className="mt-1 text-[12px] text-ink-2">
              Shortcuts are disabled while a text field or selector has focus.
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="ctl h-8 w-8 shrink-0"
            aria-label="Close keyboard reference"
            title="Close (Esc)"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="grid grid-cols-1 gap-0 sm:grid-cols-2">
          {GROUPS.map((group, i) => (
            <div key={group.title} className={`px-6 py-5 ${i === 0 ? 'sm:border-r sm:border-line' : ''}`}>
              <p className="eyebrow eyebrow-ink">{group.title}</p>
              <p className="mt-0.5 mb-3 font-display text-[15px] italic text-ink-3">{group.hint}</p>
              <dl className="space-y-2.5">
                {group.items.map((item) => (
                  <div key={item.desc} className="flex items-center justify-between gap-3">
                    <dt className="text-[12px] leading-snug text-ink-2">{item.desc}</dt>
                    <dd className="flex shrink-0 items-center gap-1">
                      {item.keys.map((k, ki) => (
                        <kbd
                          key={`${k}-${ki}`}
                          className={`numeric min-w-[22px] rounded-[3px] border border-line bg-surface-2 px-1.5 py-0.5 text-center text-[10px] leading-[15px] text-ink shadow-[0_1px_0_var(--c-line)] ${
                            k === '–' || k === '←' || k === '→' ? 'border-0 bg-transparent shadow-none' : ''
                          }`}
                        >
                          {k}
                        </kbd>
                      ))}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between border-t border-line bg-surface-2/60 px-6 py-3">
          <span className="eyebrow">Press ? to dismiss</span>
          <button type="button" onClick={onClose} className="ctl h-8 px-3 text-[11px] font-medium">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
