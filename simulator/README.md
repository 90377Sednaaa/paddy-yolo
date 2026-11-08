# Paddy-YOLO Dual-Model Comparison Simulator

An interactive computer vision benchmark comparing **Paddy YOLO** against **Paddy YOLO EMA + Dysample** with asynchronous inference benchmarking, live speed competition metrics, and comprehensive 16-species statistical evaluation.

---

## Architecture

- **Backend**: FastAPI + PyTorch / Ultralytics YOLO with custom modules (`RGG`, `RepConv`, `EMA`, `DySample`)
- **Frontend**: React 19 + TypeScript + Vite + Tailwind CSS (Impeccable Operate design standard, zero emojis)

---

## Quickstart

### 1. Start Backend Server
```bash
cd simulator/backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start Frontend Dev Server
In a separate terminal:
```bash
cd simulator/frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Model Checkpoint Weights (Folder-Based Discovery)

You do **not** need to rename your `.pt` files. Simply drop any `.pt` file (such as `best.pt`, `last.pt`, `epoch300.pt`, etc.) into its corresponding folder:

1. **Original Paddy YOLO** (Baseline):
   - Drop your `.pt` file into: `simulator/backend/weights/original/`

2. **Paddy YOLO EMA + Dysample** (Improved):
   - Drop your `.pt` file into: `simulator/backend/weights/improved/`

The backend automatically searches each folder, prioritizing `best.pt` first, or selects the most recent `.pt` file found.

*Note: If weights are not yet present in either folder, the system operates in an automated calibration fallback mode with synthetic detections and measured latencies so the UI is immediately interactive.*
