# Enhanced Model Architecture & Runtime Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authoritatively verify that the enhanced Paddy-YOLO (EMA + DySample, Fold 0) implementation is mathematically identical to its training code, fully integrated with all 3,528,752 weights, free of NaNs/mismatches, and actively serving real predictions.

**Architecture:** The Enhanced model embeds Efficient Multi-Scale Attention (EMA, ICASSP 2023) after stages P3, P4, and P5 in the backbone, and replaces nearest-neighbor upsamplers with DySample dynamic upsamplers (ICCV 2023) in the PAFPN neck. Weights are loaded from `PaddyYolo_Enhanced.pt` (trained on Fold 0 at Kaggle, 300 epochs) using Ultralytics YOLO with custom module registrations.

**Tech Stack:** Python 3.13, PyTorch 2.6 (CPU/CUDA), Ultralytics YOLOv8 engine, FastAPI, Pytest, Pillow.

## Global Constraints
- Target Checkpoint: `simulator/backend/weights/improved/PaddyYolo_Enhanced.pt` (Fold 0).
- Baseline Checkpoint: `simulator/backend/weights/original/PaddyYOLO_Baseline.pt`.
- No mock fallbacks allowed: both models must report `is_mock == False`.
- Zero emojis in all code, test files, artifacts, and documentation.
- Maintain backwards compatibility: `ema.py` must support both Fold 0 registered pooling operators and legacy mean-context checkpoints.

---

### Task 1: Checkpoint State Dictionary & Architectural Alignment Audit

**Files:**
- Create: `simulator/backend/tests/test_model_weights_integrity.py`
- Reference: `simulator/backend/paddy_yolo/modules/ema.py`
- Reference: `simulator/backend/paddy_yolo/modules/dysample.py`
- Reference: `simulator/backend/paddy_yolo/modules/rgg.py`
- Reference: `simulator/backend/weights/improved/PaddyYolo_Enhanced.pt`

**Interfaces:**
- Consumes: `PaddyYolo_Enhanced.pt` PyTorch checkpoint dictionary.
- Produces: Quantitative verification that every parameter tensor in `PaddyYolo_Enhanced.pt` matches the registered architecture, has 0 NaN/Inf values, and matches expected shapes.

- [ ] **Step 1: Write the failing integrity test**

```python
# simulator/backend/tests/test_model_weights_integrity.py
"""Integrity test suite verifying weight alignment and mathematical parameters."""

import sys
from pathlib import Path
import torch

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent.parent
for p in (ROOT_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paddy_yolo.modules.register import register_paddy_modules

register_paddy_modules()


def test_enhanced_checkpoint_weights_and_layers():
    """Verify enhanced checkpoint has exactly 26 layers, matching YAML and zero NaNs."""
    weights_path = BACKEND_DIR / "weights" / "improved" / "PaddyYolo_Enhanced.pt"
    assert weights_path.exists(), f"Missing weights: {weights_path}"

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model = ckpt["model"]

    # Parameter count verification
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3528752, f"Expected 3,528,752 parameters, got {total_params}"

    # Numerical validity
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"
        assert not torch.isinf(param).any(), f"Inf detected in parameter {name}"

    # Architectural layer sequence verification (26 layers: 0 to 25)
    layers = list(model.model)
    assert len(layers) == 26, f"Expected 26 layers, got {len(layers)}"

    # Layers 5, 8, 11 must be EMA
    for ema_idx in (5, 8, 11):
        ema_layer = layers[ema_idx]
        assert ema_layer.__class__.__name__ == "EMA"
        assert hasattr(ema_layer, "pool_h"), f"Layer {ema_idx} missing pool_h"
        assert hasattr(ema_layer, "pool_w"), f"Layer {ema_idx} missing pool_w"
        assert hasattr(ema_layer, "agp"), f"Layer {ema_idx} missing agp"
        assert hasattr(ema_layer, "conv1x1"), f"Layer {ema_idx} missing conv1x1"
        assert hasattr(ema_layer, "conv3x3"), f"Layer {ema_idx} missing conv3x3"
        assert hasattr(ema_layer, "gn"), f"Layer {ema_idx} missing gn"

    # Layers 13, 16 must be DySample
    for dy_idx in (13, 16):
        dy_layer = layers[dy_idx]
        assert dy_layer.__class__.__name__ == "DySample"
        assert hasattr(dy_layer, "offset"), f"Layer {dy_idx} missing offset conv"
        assert dy_layer.scale == 2
        assert dy_layer.style == "lp"
        assert dy_layer.groups == 4


def test_baseline_checkpoint_weights_and_layers():
    """Verify baseline checkpoint has 23 layers and zero NaNs."""
    weights_path = BACKEND_DIR / "weights" / "original" / "PaddyYOLO_Baseline.pt"
    assert weights_path.exists(), f"Missing weights: {weights_path}"

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model = ckpt["model"]

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3501712, f"Expected 3,501,712 parameters, got {total_params}"

    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"
        assert not torch.isinf(param).any(), f"Inf detected in parameter {name}"

    layers = list(model.model)
    assert len(layers) == 23, f"Expected 23 layers, got {len(layers)}"
```

- [ ] **Step 2: Run test to verify it executes and passes or identifies any discrepancy**

Run: `pytest simulator/backend/tests/test_model_weights_integrity.py -v`
Expected: PASS with 2 passed tests confirming 3,528,752 parameters for Enhanced and 3,501,712 for Baseline.

- [ ] **Step 3: Commit the test**

```bash
git add simulator/backend/tests/test_model_weights_integrity.py
git commit -m "test(weights): add parameter integrity and architecture layout tests"
```

---

### Task 2: PyTorch Tensor Forward Trace & FP16 Precision Verification

**Files:**
- Create: `simulator/backend/tests/test_enhanced_forward_trace.py`
- Reference: `simulator/backend/model_loader.py`

**Interfaces:**
- Consumes: `model_loader.get_detectors()`.
- Produces: Verified end-to-end tensor flow through all 26 layers yielding exactly 8,400 anchor bounding box candidates.

- [ ] **Step 1: Write the forward trace test**

```python
# simulator/backend/tests/test_enhanced_forward_trace.py
"""Test forward tensor pass and anchor head outputs for baseline and enhanced models."""

import sys
from pathlib import Path
import torch

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent.parent
for p in (ROOT_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_loader import get_detectors


def test_enhanced_forward_pass_tensor_output():
    """Verify enhanced model forward pass produces valid anchor predictions."""
    _, improved_detector = get_detectors()
    assert not improved_detector.is_mock, "Enhanced detector is unexpectedly running in mock mode"

    model = improved_detector.model.model
    model.eval()

    # Input tensor matching standard 640x640 YOLO resolution
    # Note: weights are float16 or float32 depending on device; cast input to match model weights
    first_param = next(model.parameters())
    x = torch.zeros((1, 3, 640, 640), dtype=first_param.dtype, device=first_param.device)

    with torch.no_grad():
        preds = model(x)

    # In eval mode, Ultralytics Detect head returns tuple (inference_output, train_outputs)
    # or a tensor of shape (batch, 4 + nc, num_anchors)
    # num_anchors for 640x640: (80x80) + (40x40) + (20x20) = 6400 + 1600 + 400 = 8400
    if isinstance(preds, (list, tuple)):
        raw_out = preds[0]
    else:
        raw_out = preds

    assert raw_out.shape[0] == 1, f"Batch dimension mismatch: {raw_out.shape}"
    assert raw_out.shape[1] == 20, f"Expected 20 channels (4 box coords + 16 classes), got {raw_out.shape[1]}"
    assert raw_out.shape[2] == 8400, f"Expected 8400 anchors across 3 scales, got {raw_out.shape[2]}"
    assert not torch.isnan(raw_out).any(), "NaN found in anchor output tensor"


def test_baseline_forward_pass_tensor_output():
    """Verify baseline model forward pass produces valid anchor predictions."""
    baseline_detector, _ = get_detectors()
    assert not baseline_detector.is_mock, "Baseline detector is unexpectedly running in mock mode"

    model = baseline_detector.model.model
    model.eval()

    first_param = next(model.parameters())
    x = torch.zeros((1, 3, 640, 640), dtype=first_param.dtype, device=first_param.device)

    with torch.no_grad():
        preds = model(x)

    if isinstance(preds, (list, tuple)):
        raw_out = preds[0]
    else:
        raw_out = preds

    assert raw_out.shape[0] == 1
    assert raw_out.shape[1] == 20
    assert raw_out.shape[2] == 8400
    assert not torch.isnan(raw_out).any()
```

- [ ] **Step 2: Run test to verify tensor trace passes**

Run: `pytest simulator/backend/tests/test_enhanced_forward_trace.py -v`
Expected: PASS with 2 passed tests confirming exact `(1, 20, 8400)` anchor tensor shape for both models.

- [ ] **Step 3: Commit the test**

```bash
git add simulator/backend/tests/test_enhanced_forward_trace.py
git commit -m "test(trace): add end-to-end tensor trace and anchor dimension verification"
```

---

### Task 3: Empirical Specimen Inference & Real-World Detection Verification

**Files:**
- Create: `simulator/backend/tests/test_specimen_detection.py`
- Reference: `simulator/backend/model_loader.py`
- Reference: `simulator/backend/samples/`
- Reference: User uploaded specimen `media_1794140636482_883be2cb.jpg`

**Interfaces:**
- Consumes: PIL Image objects from test specimen files.
- Produces: High-confidence rice pest detections with coordinates, confidence scores, and species names.

- [ ] **Step 1: Write the specimen detection test**

```python
# simulator/backend/tests/test_specimen_detection.py
"""Test object detection on real rice pest field specimens."""

import sys
from pathlib import Path
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent.parent
for p in (ROOT_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_loader import get_detectors


def test_user_uploaded_specimen_detection():
    """Verify both baseline and enhanced models detect rice leaf roller on user specimen."""
    specimen_path = Path("C:/Users/PC/.gemini/antigravity/brain/5890b6aa-a989-439d-b45b-05fd6e18d8cc/.user_uploaded/media_1794140636482_883be2cb.jpg")
    if not specimen_path.exists():
        # Fallback to local sample image if uploaded image path not available
        specimen_path = BACKEND_DIR / "samples" / "sample_rice_leaf_roller.jpg"

    img = Image.open(str(specimen_path)).convert("RGB")
    baseline_detector, improved_detector = get_detectors()

    # Verify real models, not mocks
    assert not baseline_detector.is_mock
    assert not improved_detector.is_mock

    # Run predictions
    res_base = baseline_detector.predict(img, conf=0.05)
    res_impr = improved_detector.predict(img, conf=0.05)

    assert res_base["detection_count"] > 0, "Baseline failed to detect pest on user specimen"
    assert res_impr["detection_count"] > 0, "Enhanced model failed to detect pest on user specimen"

    # Enhanced model must detect rice leaf roller with >= 0.70 confidence
    impr_species = [d["species_name"] for d in res_impr["detections"]]
    impr_confs = [d["confidence"] for d in res_impr["detections"]]
    assert "rice leaf roller" in impr_species, f"Expected rice leaf roller in detections, got {impr_species}"
    assert max(impr_confs) >= 0.70, f"Expected highest confidence >= 0.70, got {max(impr_confs)}"


def test_samples_directory_detection():
    """Verify enhanced model executes cleanly on all packaged sample images."""
    samples_dir = BACKEND_DIR / "samples"
    sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
    assert len(sample_files) > 0, "No sample images found"

    _, improved_detector = get_detectors()
    for sample_file in sample_files:
        img = Image.open(str(sample_file)).convert("RGB")
        res = improved_detector.predict(img, conf=0.05)
        assert res["model_name"] == "Paddy YOLO EMA + Dysample"
        assert not res["is_mock"]
        assert res["timing"]["inference_ms"] > 0
        assert "detections" in res
```

- [ ] **Step 2: Run test to verify real detections**

Run: `pytest simulator/backend/tests/test_specimen_detection.py -v`
Expected: PASS with 2 passed tests validating live detection.

- [ ] **Step 3: Commit the test**

```bash
git add simulator/backend/tests/test_specimen_detection.py
git commit -m "test(detection): verify real pest specimen detection with baseline and enhanced models"
```

---

### Task 4: API Health Status & Mock-Free Assertion

**Files:**
- Modify: `simulator/backend/tests/test_api.py`

**Interfaces:**
- Consumes: FastAPI TestClient hitting `/api/health` and `/api/predict`.
- Produces: Hard assertion that neither model is mock in production or test runs.

- [ ] **Step 1: Update `test_api.py` with hard `is_mock == False` assertions**

Modify `test_api.py` to add:
```python
def test_health_endpoint_not_mock():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["models"]["baseline"]["is_mock"] is False, "Baseline model is running in mock fallback mode"
    assert data["models"]["improved"]["is_mock"] is False, "Enhanced model is running in mock fallback mode"
    assert data["models"]["improved"]["weights_file"] == "PaddyYolo_Enhanced.pt"
```

- [ ] **Step 2: Run full backend test suite**

Run: `pytest simulator/backend/tests -v`
Expected: All tests pass across `test_api.py`, `test_model_weights_integrity.py`, `test_enhanced_forward_trace.py`, and `test_specimen_detection.py`.

- [ ] **Step 3: Commit updated API tests**

```bash
git add simulator/backend/tests/test_api.py
git commit -m "test(api): assert is_mock is false for both baseline and enhanced models"
```

---

### Task 5: Live Execution Verification & User Walkthrough Update

**Files:**
- Modify: `C:\Users\PC\.gemini\antigravity\brain\5890b6aa-a989-439d-b45b-05fd6e18d8cc\walkthrough.md`

**Interfaces:**
- Consumes: Test results and live port checks.
- Produces: Documented evidence and exact step-by-step startup procedure for the user.

- [ ] **Step 1: Check port 8000 and 5173 status**

Run PowerShell command: `Get-NetTCPConnection -LocalPort 8000, 5173 -ErrorAction SilentlyContinue`
Expected: Clear report of whether backend or frontend is currently bound.

- [ ] **Step 2: Update walkthrough.md with verification proof**

Record all parameter counts, weight shapes, detection confidences, and the root cause explanation of commit `4a7f989` vs commit `223f5b4`.
