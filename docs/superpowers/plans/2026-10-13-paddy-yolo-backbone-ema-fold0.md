# Paddy-YOLO with Backbone EMA Fold 0 Training Implementation Plan

> **Goal:** Configure and prepare Paddy-YOLO with Efficient Multi-Scale Attention (EMA) integrated into the backbone (Configuration B of the ablation study), ensure robust topological pre-trained weight transfer from Ultralytics `yolov8n.pt`, set up Fold 0 training on a dedicated branch `Paddy-Yolo-EMA`, and include paper PDFs and baseline replication CSV benchmark results for post-training comparison.

**Architecture:**
- **Model Config (`configs/paddy-yolo-ema.yaml`)**:
  - Backbone: 4 RGG stages (P2, P3, P4, P5).
  - Attention: EMA modules inserted after selected RGG stages in the backbone (P3, P4, P5) to refine cross-spatial pest features and suppress background foliage before neck fusion.
  - Neck: Original feature fusion neck with conventional upsample operations (`nearest`).
  - Heads: 3 Detection Heads at P3, P4, P5 with DIMIoU Strategy H loss.
- **Pre-trained Weight Integration (`train.py`)**:
  - Full topological weight mapping from `yolov8n.pt` into the shifted backbone and neck layers.
- **Branch Management**:
  - Branch `Paddy-Yolo-EMA` isolating EMA experimentation while preserving baseline comparison CSVs and PDFs.

---

### Task 1: Create Branch `Paddy-Yolo-EMA` and Bring Baseline Results

**Files:**
- Create and checkout branch: `Paddy-Yolo-EMA`
- Copy baseline results from `Paddy-Yolo-Baseline`:
  - `data/kaggle_baseline_5fold_results/paddy_yolo_baseline_5fold_master_summary.csv`
  - `data/kaggle_baseline_5fold_results/paddy_yolo_baseline_individual_folds.csv`

**Verification:**
- `git branch` confirms on `Paddy-Yolo-EMA`.
- Both CSV files exist and contain baseline 5-fold replication numbers.

---

### Task 2: Define Backbone-EMA Architecture (`configs/paddy-yolo-ema.yaml`)

**Files:**
- Modify: `configs/paddy-yolo-ema.yaml`

**Layer Specification:**
- Backbone:
  - 0: Conv [64, 3, 2] (P1/2)
  - 1: Conv [128, 3, 2] (P2/4)
  - 2: RGG [128, True] (Backbone P2)
  - 3: Conv [256, 3, 2] (P3/8)
  - 4: RGG [256, True] (Backbone P3)
  - 5: EMA [] (Backbone P3 EMA Attention)
  - 6: Conv [512, 3, 2] (P4/16)
  - 7: RGG [512, True] (Backbone P4)
  - 8: EMA [] (Backbone P4 EMA Attention)
  - 9: Conv [1024, 3, 2] (P5/32)
  - 10: RGG [1024, True] (Backbone P5)
  - 11: EMA [] (Backbone P5 EMA Attention)
  - 12: SPPF [1024, 5] (P5)
- Neck:
  - 13: nn.Upsample [None, 2, "nearest"]
  - 14: Concat [[-1, 8]] (cat backbone P4 EMA)
  - 15: C2f [512]
  - 16: nn.Upsample [None, 2, "nearest"]
  - 17: Concat [[-1, 5]] (cat backbone P3 EMA)
  - 18: C2f [256] -> Detection Head P3
  - 19: Conv [256, 3, 2]
  - 20: Concat [[-1, 15]]
  - 21: C2f [512] -> Detection Head P4
  - 22: Conv [512, 3, 2]
  - 23: Concat [[-1, 12]] (cat SPPF)
  - 24: C2f [1024] -> Detection Head P5
  - 25: Detect [nc, [18, 21, 24]]

**Verification:**
- Load model with `YOLO("configs/paddy-yolo-ema.yaml")`.
- Verify forward pass shape and parameter count (~3.34M - 3.38M).

---

### Task 3: Topological Pre-trained Weight Integration in `train.py`

**Files:**
- Modify: `train.py`
- Modify: `tests/test_paddy_yolo_ema.py`

**Mapping Logic (`transfer_pretrained_weights`):**
Map `yolov8n.pt` layers:
- Backbone:
  - `model.0` -> `model.0` (Conv P1)
  - `model.1` -> `model.1` (Conv P2)
  - `model.2` (C2f cv1, cv2) -> `model.2` (RGG cv1, cv2)
  - `model.3` -> `model.3` (Conv P3)
  - `model.4` (C2f cv1, cv2) -> `model.4` (RGG cv1, cv2)
  - `model.5` -> `model.6` (Conv P4)
  - `model.6` (C2f cv1, cv2) -> `model.7` (RGG cv1, cv2)
  - `model.7` -> `model.9` (Conv P5)
  - `model.8` (C2f cv1, cv2) -> `model.10` (RGG cv1, cv2)
  - `model.9` -> `model.12` (SPPF)
- Neck:
  - `model.12` -> `model.15` (C2f P4)
  - `model.15` -> `model.18` (C2f P3)
  - `model.16` -> `model.19` (Conv P3->P4)
  - `model.18` -> `model.21` (C2f P4)
  - `model.19` -> `model.22` (Conv P4->P5)
  - `model.21` -> `model.24` (C2f P5)
- Box Regression:
  - `model.22.cv2` -> `model.25.cv2` (Bounding box regression branches)

**Verification:**
- Unit test in `tests/test_paddy_yolo_ema.py` verifying that > 90% of non-attention tensors transfer with zero shape mismatches.

---

### Task 4: Fold 0 Training Kernel & Verification

**Files:**
- Modify: `build_fold0_ema_kernel.py`
- Generate: `kaggle_fold0_ema_kernel/paddy_yolo_ema_fold0.ipynb`
- Update: `verify_paddy_yolo.py`

**Kernel Features:**
- Automated extraction of project code.
- Checks `yolov8n.pt` pre-trained weights and initializes `paddy-yolo-ema_init.pt`.
- Trains Fold 0 (train on folds 1, 2, 3; val on fold 4; test on fold 0).
- Post-training validation generates comparison table against:
  1. Paper reported metrics (Table 8 / 10).
  2. Our baseline replication metrics (`paddy_yolo_baseline_5fold_master_summary.csv`).
- Exports results to CSV and saves checkpoint.

**Verification:**
- Run test suite: `pytest`.
- Run verification: `python verify_paddy_yolo.py --smoke-test`.
- Build kernel: `python build_fold0_ema_kernel.py`.
