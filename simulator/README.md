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

## Model Checkpoint Weights

Place your trained `.pt` weights in `simulator/backend/weights/`:
- `paddy_yolo_baseline.pt` (Baseline Paddy-YOLO model)
- `paddy_yolo_ema_dysample.pt` (Improved Paddy-YOLO EMA + DySample model)

*Note: If weights are not yet present, the system runs in an automated calibration fallback mode with synthetic detections and measured latencies so the UI is immediately interactive.*
