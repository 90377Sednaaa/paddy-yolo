# Design System: Paddy-YOLO Simulator

## Visual Direction
- **Mode**: Operate (high scanability, technical precision, zero AI clichés).
- **Zero Emojis**: Replaced entirely by SVG icons from Lucide React (stroke width 1.5px).
- **Typography**:
  - Headings: High-legibility sans-serif with negative tracking (`tracking-tight`).
  - Numerals & Stats: Strict `tabular-nums font-mono` to prevent visual jitter during live millisecond counter updates.
- **Color Palette**:
  - Canvas & Shell: `#0a0d14` (Deep obsidian neutral)
  - Secondary Cards: `#111622` with `#232d42` border
  - Model 1 (Paddy YOLO): `#38bdf8` (Precision sky blue)
  - Model 2 (Paddy YOLO EMA + Dysample): `#10b981` (Agricultural emerald)
- **Canvas Overlays**:
  - Semi-transparent fill (`#...1a`) with 2px solid stroke per insect species.
  - Micro tags with high-contrast dark text on colored background for species name and confidence percentage.
