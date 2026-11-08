# Product Context: Paddy-YOLO Dual-Model Simulator

## Purpose & Problem
Field-based agricultural insect detection requires balancing detection precision with computational latency on edge hardware. This simulator provides real-time asynchronous evaluation comparing:
- Baseline architecture: **Paddy YOLO**
- Enhanced architecture: **Paddy YOLO EMA + Dysample**

## Target Users
- Precision agriculture researchers
- Entomologists and field agronomists
- Computer vision model evaluators

## Surface Mode
**Operate**: Prioritizes scanability, rapid task execution, high-legibility tabular statistics, zero decorative emojis, and millisecond-level precision timing.

## Constraints & Requirements
- Strict user-facing naming: "Paddy YOLO" vs "Paddy YOLO EMA + Dysample".
- Complete exclusion of internal training configuration jargon ("Config D", "lr0 0.001", "warmup_bias 0").
- Asynchronous dual inference displaying real-time speed competition and winner designation.
- High-contrast interactive canvas bounding boxes with species classification labels.
- Tabular numeral alignment (`tabular-nums`) across all timestamps and percentage metrics.
