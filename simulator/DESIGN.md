# Design System: Paddy-YOLO Field Evaluation Console

## Visual Philosophy
**A field station instrument, not a dashboard.** Warm paper chrome, ink typography,
hairline rules and a permanently dark viewing stage for specimen imagery. The visual
language is editorial (journal masthead, numbered captions, ruled tables) rather than
card-based. Every element earns its place; there are no decorative gradients, glows or
rounded card clusters.

- **Mode**: Operate — fast scanning, high-legibility tabular data, zero decorative emoji.
- **Differentiation**: light "Bench" chrome surrounding a dark "Darkroom" inspection stage.
- **Two themes**: `bench` (default) and `darkroom`, both driven by one semantic token set.
  Components never hardcode color; canvas annotations read the same tokens at runtime.

## Typography
- **Display**: Newsreader (variable optical size) — wordmark, specimen names, page titles,
  large endpoint numbers. Fluid `clamp()` sizing.
- **UI**: IBM Plex Sans — labels, controls, prose.
- **Data**: IBM Plex Mono with `tabular-nums` — every latency, percentage, coordinate.
- Fallbacks are resolved system fonts, so the console remains usable offline.

## Token System (`src/index.css`)
Semantic variables live on `:root` / `[data-theme="darkroom"]` and are exposed to
Tailwind through `@theme inline`, so `bg-surface`, `text-ink-2`, `border-line` etc.
resolve at runtime and both themes work with one set of classes.

| Token | Bench | Darkroom |
| --- | --- | --- |
| paper | `#f2f0e8` | `#0d120e` |
| surface | `#faf9f4` | `#151b16` |
| ink / ink-2 / ink-3 | `#18201a` / `#586357` / `#8c948a` | `#e8ece3` / `#aab4a5` / `#8d9b89` |
| line | `#d6d3c7` | `#2c372b` |
| leaf (enhanced) | `#2e6b4c` | `#83c79c` |
| baseline | `#3d6288` | `#8db4e0` |
| ember (focus / active) | `#9c6b24` | `#dcaa5f` |
| negative | `#a23a31` | `#e08378` |
| stage | `#131a15` | `#080c09` |

Canvas annotation strokes (always on a dark stage): baseline `#8fb8e8`, enhanced
`#93cfa4`, consensus `#aeb8ac`, focus reticle `#e2b667`, label ink `#111a12`.

## Layout & Components
- **Masthead**: edition line → wordmark + theme/keyboard controls → underline tabs with
  workspace numbers `01` / `02`. No status badge clusters.
- **Run header**: specimen name in display type, italic binomial, clickable position ticks,
  stepper + native select + one solid primary action ("Upload specimen").
- **Telemetry**: proportional dual-bar latency race + head-to-head verdict; no four-card
  KPI template. Bar widths animate from zero with an exponential ease.
- **Inspection stage**: ruler-like control rail, registration corners, per-viewport headers,
  object-fit-contained canvases with synchronized zoom/pan, coordinate readout, processing
  overlay. Dual viewports stack vertically below 640px.
- **Detection ledger**: numbered rows with hairline separators and confidence micro-bars,
  not nested cards; keyboard-operable buttons; bottom fade signals scroll.
- **Evidence workspace**: display-number hero with a three-way comparison chart, ruled
  supporting endpoints, numbered section captions, editorial tables with `scope`,
  `aria-sort`, sticky-ish headers, and a definition list for architecture specs.

## Interaction & Accessibility
- Working shortcuts: `1–5`, `←/→`, `M`, `V`, `L`, `+/-`, `0`, `?`, `Esc` — all documented
  in the dialog and all implemented.
- Wheel zoom uses a non-passive native listener; pan is drag-based; zoom resets pan at 100%.
- Focus-visible rings, labelled controls, `role="img"` canvases, `role="dialog"` modal,
  `role="alert"` for engine failure, `prefers-reduced-motion` support.
- Educational empty/error states (offline engine with Retry, no detections with a threshold
  hint, no species match with Clear filters).

## Brand
Custom `BrandMark` (reticle + rice leaf) and matching SVG favicon replace the default Vite
lightning mark. Page title, meta description and theme-color are product-specific.
