# Paddy-YOLO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recreate Paddy-YOLO (YOLOv8-N with RGG feature extraction module and Dynamic Inner-MPDIoU loss) based on the 2025 research paper, integrated cleanly with Ultralytics YOLOv8.

**Architecture:** A modular Python package `paddy_yolo` containing custom neural modules (`RepConv`, `GhostConv` with 5x5 DWConv, `RGGConv`, `RGGBottleneck`, `RGG`), dynamic registry into Ultralytics model parser, a custom `DIMIoULoss` (Strategy H) inside `PaddyDetectionLoss`, `PaddyTrainer` subclassing `DetectionTrainer`, and dataset utilities for Kaggle R2000 with 5-fold cross validation.

**Tech Stack:** Python 3.10+, PyTorch, Ultralytics (YOLOv8), PyYAML, NumPy.

## Global Constraints
- Must match the paper's RGG structure: RepConv + 2 GhostConvs (with 5x5 DWConv) + Concat + 1x1 Conv.
- Must match DIMIoU Strategy H: $\text{ratio} = \frac{1}{2}\cos((1 - \text{IoU})\pi) + 1$.
- Must match paper hyperparameters: 300 epochs, AdamW, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005.
- Must support the 16 rice pest classes defined in Table 4 of the paper.
- Must integrate non-invasively with Ultralytics without mutating installed package files.

---

### Task 1: Package Structure, Environment Setup, and Base Configs

**Files:**
- Create: `requirements.txt`
- Create: `paddy_yolo/__init__.py`
- Create: `paddy_yolo/modules/__init__.py`
- Create: `paddy_yolo/loss/__init__.py`
- Create: `paddy_yolo/engine/__init__.py`
- Create: `paddy_yolo/dataset/__init__.py`
- Create: `configs/r2000.yaml`
- Create: `configs/hyp_paper.yaml`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: Nothing
- Produces: Package namespace `paddy_yolo`, dataset YAML configuration for 16 rice pest classes, hyperparameter YAML for paper configuration.

- [ ] **Step 1: Write the failing test**
Create `tests/test_env.py`:
```python
import importlib

def test_imports():
    assert importlib.util.find_spec("torch") is not None
    assert importlib.util.find_spec("ultralytics") is not None
    assert importlib.util.find_spec("paddy_yolo") is not None

def test_configs_exist():
    import os
    assert os.path.exists("configs/r2000.yaml")
    assert os.path.exists("configs/hyp_paper.yaml")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_env.py -v`
Expected: FAIL (module `paddy_yolo` not found or configs do not exist)

- [ ] **Step 3: Implement package structure, requirements, and configs**
Create `requirements.txt`, `paddy_yolo/__init__.py`, subpackage `__init__.py` files, `configs/r2000.yaml`, and `configs/hyp_paper.yaml`.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add requirements.txt configs/ paddy_yolo/ tests/test_env.py
git commit -m "feat: scaffold paddy_yolo package structure and base configs"
```

---

### Task 2: RepConv with Structural Re-parameterization

**Files:**
- Create: `paddy_yolo/modules/repconv.py`
- Test: `tests/test_repconv.py`

**Interfaces:**
- Consumes: `torch.nn.Module`, `torch.Tensor`
- Produces: `class RepConv(nn.Module)` with `switch_to_deploy()` and forward pass equivalent in training & eval mode.

- [ ] **Step 1: Write the failing test**
Create `tests/test_repconv.py`:
```python
import torch
from paddy_yolo.modules.repconv import RepConv

def test_repconv_forward():
    x = torch.randn(2, 32, 64, 64)
    conv = RepConv(32, 32)
    conv.train()
    out_train = conv(x)
    assert out_train.shape == (2, 32, 64, 64)

def test_repconv_deploy_equivalence():
    x = torch.randn(1, 16, 32, 32)
    conv = RepConv(16, 32)
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.switch_to_deploy()
        out_after = conv(x)
    assert torch.allclose(out_before, out_after, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_repconv.py -v`
Expected: FAIL (`cannot import name 'RepConv'`)

- [ ] **Step 3: Implement RepConv**
Create `paddy_yolo/modules/repconv.py` with multi-branch 3x3, 1x1, identity convolutions + BN in training mode, and fused single 3x3 conv weight/bias calculation in `switch_to_deploy()`.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_repconv.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/modules/repconv.py tests/test_repconv.py
git commit -m "feat: implement RepConv with structural re-parameterization"
```

---

### Task 3: GhostConv with 5x5 Depthwise Convolution

**Files:**
- Create: `paddy_yolo/modules/ghostconv.py`
- Test: `tests/test_ghostconv.py`

**Interfaces:**
- Consumes: `torch.nn.Module`
- Produces: `class GhostConv(nn.Module)` where primary conv is standard 3x3 (or 1x1) and cheap operation is 5x5 DWConv.

- [ ] **Step 1: Write the failing test**
Create `tests/test_ghostconv.py`:
```python
import torch
from paddy_yolo.modules.ghostconv import GhostConv

def test_ghostconv():
    x = torch.randn(2, 32, 40, 40)
    conv = GhostConv(32, 64, k=3, dw_k=5)
    out = conv(x)
    assert out.shape == (2, 64, 40, 40)
    assert conv.dw_conv.kernel_size == (5, 5)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_ghostconv.py -v`
Expected: FAIL (`cannot import name 'GhostConv'`)

- [ ] **Step 3: Implement GhostConv**
Create `paddy_yolo/modules/ghostconv.py` splitting output channels into primary conv and 5x5 depthwise conv, followed by concatenation along channel dimension.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_ghostconv.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/modules/ghostconv.py tests/test_ghostconv.py
git commit -m "feat: implement GhostConv with 5x5 DWConv"
```

---

### Task 4: RGGConv, RGGBottleneck, and RGG Block

**Files:**
- Create: `paddy_yolo/modules/rgg.py`
- Test: `tests/test_rgg.py`

**Interfaces:**
- Consumes: `RepConv` from `paddy_yolo.modules.repconv`, `GhostConv` from `paddy_yolo.modules.ghostconv`
- Produces: `RGGConv`, `RGGBottleneck`, and `RGG` (C2f-style bottleneck aggregator with RGG bottlenecks)

- [ ] **Step 1: Write the failing test**
Create `tests/test_rgg.py`:
```python
import torch
from paddy_yolo.modules.rgg import RGGConv, RGGBottleneck, RGG

def test_rgg_conv():
    x = torch.randn(2, 32, 20, 20)
    conv = RGGConv(32, 32)
    out = conv(x)
    assert out.shape == (2, 32, 20, 20)

def test_rgg_bottleneck():
    x = torch.randn(2, 32, 20, 20)
    block = RGGBottleneck(32, 32, shortcut=True)
    out = block(x)
    assert out.shape == (2, 32, 20, 20)

def test_rgg_module():
    x = torch.randn(2, 64, 40, 40)
    rgg = RGG(64, 64, n=2, shortcut=True)
    out = rgg(x)
    assert out.shape == (2, 64, 40, 40)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_rgg.py -v`
Expected: FAIL (`cannot import name 'RGGConv'`)

- [ ] **Step 3: Implement RGGConv, RGGBottleneck, and RGG**
Create `paddy_yolo/modules/rgg.py`:
- `RGGConv`: executes RepConv, GhostConv1, GhostConv2, concatenates 3C channels, passes through Conv2d(3C, C, 1).
- `RGGBottleneck`: Conv(c1, c2, 1) -> RGGConv(c2, c2) with residual addition if shortcut=True.
- `RGG`: C2f-style block where input is split, passed through `n` RGGBottlenecks, concatenated, and fused via Conv.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_rgg.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/modules/rgg.py tests/test_rgg.py
git commit -m "feat: implement RGGConv, RGGBottleneck, and RGG module"
```

---

### Task 5: Dynamic Registration and Paddy-YOLO Architecture YAML

**Files:**
- Create: `paddy_yolo/modules/register.py`
- Create: `configs/paddy-yolo.yaml`
- Test: `tests/test_model_build.py`

**Interfaces:**
- Consumes: `RGG`, `RepConv`, `GhostConv` from `paddy_yolo.modules`
- Produces: `register_paddy_modules()` function that registers custom modules into `ultralytics.nn.modules` and parses `configs/paddy-yolo.yaml` seamlessly via `YOLO("configs/paddy-yolo.yaml")`.

- [ ] **Step 1: Write the failing test**
Create `tests/test_model_build.py`:
```python
import torch
from ultralytics import YOLO
from paddy_yolo.modules.register import register_paddy_modules

def test_paddy_yolo_model_build():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo.yaml")
    x = torch.randn(1, 3, 640, 640)
    results = model.model(x)
    assert results is not None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_model_build.py -v`
Expected: FAIL

- [ ] **Step 3: Implement register_paddy_modules and configs/paddy-yolo.yaml**
Implement `paddy_yolo/modules/register.py` ensuring `RGG`, `RGGBottleneck`, `RGGConv`, `RepConv`, `GhostConv` are added to `ultralytics.nn.modules` and `parse_model`.
Create `configs/paddy-yolo.yaml` with the exact 4 RGG modules in the backbone replacing C2f.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_model_build.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/modules/register.py configs/paddy-yolo.yaml tests/test_model_build.py
git commit -m "feat: register RGG modules and define paddy-yolo.yaml"
```

---

### Task 6: DIMIoU Loss Function (Strategy H)

**Files:**
- Create: `paddy_yolo/loss/dimiou.py`
- Test: `tests/test_dimiou.py`

**Interfaces:**
- Consumes: Bounding box tensors $(B, 4)$ in xyxy or xywh format.
- Produces: `dimiou_loss(pred_boxes, target_boxes, eps=1e-7)` returning tensor of DIMIoU loss values.

- [ ] **Step 1: Write the failing test**
Create `tests/test_dimiou.py`:
```python
import torch
from paddy_yolo.loss.dimiou import dimiou_loss

def test_dimiou_identical_boxes():
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    b2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    loss = dimiou_loss(b1, b2)
    assert torch.isclose(loss, torch.tensor([0.0]), atol=1e-3)

def test_dimiou_disjoint_boxes():
    b1 = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    b2 = torch.tensor([[100.0, 100.0, 120.0, 120.0]])
    loss = dimiou_loss(b1, b2)
    assert loss.item() > 1.0  # MPDIoU corner distance penalty adds to loss
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_dimiou.py -v`
Expected: FAIL (`cannot import name 'dimiou_loss'`)

- [ ] **Step 3: Implement dimiou_loss with Strategy H**
Create `paddy_yolo/loss/dimiou.py`:
- Calculate base IoU.
- Calculate Strategy H dynamic ratio: `ratio = 0.5 * torch.cos((1.0 - iou) * torch.pi) + 1.0`.
- Scale boxes by ratio to compute `(A ∩ B)_ratio` and `(A ∪ B)_ratio`.
- Calculate MPDIoU corner distance penalties: `d1^2 / (w^2 + h^2) + d2^2 / (w^2 + h^2)`.
- Return `1.0 - inner_iou + corner_penalty`.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_dimiou.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/loss/dimiou.py tests/test_dimiou.py
git commit -m "feat: implement DIMIoU bounding box regression loss with Strategy H"
```

---

### Task 7: PaddyDetectionLoss and PaddyTrainer Integration

**Files:**
- Create: `paddy_yolo/loss/paddy_loss.py`
- Create: `paddy_yolo/engine/trainer.py`
- Test: `tests/test_loss_integration.py`

**Interfaces:**
- Consumes: `dimiou_loss` from `paddy_yolo.loss.dimiou`, `DetectionTrainer` from `ultralytics.models.yolo.detect`
- Produces: `PaddyDetectionLoss` and `PaddyTrainer` subclass.

- [ ] **Step 1: Write the failing test**
Create `tests/test_loss_integration.py`:
```python
import torch
from paddy_yolo.loss.paddy_loss import PaddyDetectionLoss
from paddy_yolo.engine.trainer import PaddyTrainer

def test_paddy_trainer_instantiation():
    trainer = PaddyTrainer(overrides={"model": "configs/paddy-yolo.yaml", "data": "configs/r2000.yaml", "epochs": 1})
    assert trainer is not None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_loss_integration.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PaddyDetectionLoss and PaddyTrainer**
Create `paddy_yolo/loss/paddy_loss.py`: Subclasses `v8DetectionLoss` and overrides bounding box loss calculation to use `dimiou_loss`.
Create `paddy_yolo/engine/trainer.py`: Subclasses `DetectionTrainer` and overrides `get_loss` or `init_criterion` to return `PaddyDetectionLoss`.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_loss_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/loss/paddy_loss.py paddy_yolo/engine/trainer.py tests/test_loss_integration.py
git commit -m "feat: implement PaddyDetectionLoss and PaddyTrainer"
```

---

### Task 8: Dataset Preparation and 5-Fold Cross Validation Generator

**Files:**
- Create: `paddy_yolo/dataset/prepare_r2000.py`
- Test: `tests/test_dataset_prep.py`

**Interfaces:**
- Consumes: Raw Kaggle dataset files or synthetic test generation flag `--generate-dummy`
- Produces: YOLO formatted images & labels, 5-fold cross validation split files `data_fold0.yaml` to `data_fold4.yaml`.

- [ ] **Step 1: Write the failing test**
Create `tests/test_dataset_prep.py`:
```python
import os
import shutil
from paddy_yolo.dataset.prepare_r2000 import generate_dummy_dataset, create_5fold_splits

def test_dummy_dataset_and_splits(tmp_path):
    data_dir = tmp_path / "dummy_data"
    generate_dummy_dataset(str(data_dir), num_images=20)
    assert os.path.exists(data_dir / "images")
    assert os.path.exists(data_dir / "labels")
    yaml_paths = create_5fold_splits(str(data_dir))
    assert len(yaml_paths) == 5
    assert os.path.exists(yaml_paths[0])
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_dataset_prep.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dataset preparation and 5-fold splitter**
Create `paddy_yolo/dataset/prepare_r2000.py` with:
- `generate_dummy_dataset(output_dir, num_images)` for instant local testing.
- `prepare_kaggle_r2000(input_dir, output_dir)` to parse Kaggle directory structure.
- `create_5fold_splits(dataset_dir)` to create 5-fold train/val/test splits and YAML configuration files.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_dataset_prep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add paddy_yolo/dataset/prepare_r2000.py tests/test_dataset_prep.py
git commit -m "feat: implement R2000 dataset prep and 5-fold cross validation splitter"
```

---

### Task 9: Training, Validation, Prediction Scripts, and Verification Suite

**Files:**
- Create: `train.py`
- Create: `val.py`
- Create: `predict.py`
- Create: `verify_paddy_yolo.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `PaddyTrainer`, `paddy-yolo.yaml`, `hyp_paper.yaml`, `data_fold*.yaml`
- Produces: Executable scripts for training, evaluation, inference, and comprehensive self-test.

- [ ] **Step 1: Write the failing test**
Create `tests/test_scripts.py`:
```python
import subprocess
import sys

def test_verify_script():
    cmd = [sys.executable, "verify_paddy_yolo.py", "--smoke-test"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_scripts.py -v`
Expected: FAIL (`verify_paddy_yolo.py does not exist`)

- [ ] **Step 3: Implement train.py, val.py, predict.py, and verify_paddy_yolo.py**
- `train.py`: CLI supporting `--data`, `--hyp`, `--epochs`, `--batch`, `--fold`, and launching `PaddyTrainer`.
- `val.py`: CLI supporting `--weights`, `--data`, computing mAP@0.5 and mAP@0.5:0.95.
- `predict.py`: CLI running inference on images with bounding box drawing.
- `verify_paddy_yolo.py`: Comprehensive test checking RepConv deploy equivalence, GhostConv receptive field, RGG bottleneck gradients, DIMIoU loss, model forward pass, parameter counts, and 1 mini-epoch train step on synthetic data.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_scripts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add train.py val.py predict.py verify_paddy_yolo.py tests/test_scripts.py
git commit -m "feat: add train, val, predict scripts and verification suite"
```

---

### Task 10: Full End-to-End Test Suite Execution and Documentation

**Files:**
- Create: `README.md`
- Test: Run `pytest` across all test suites and run `python verify_paddy_yolo.py`

- [ ] **Step 1: Create README.md**
Detailed usage guide including:
- Architecture summary and paper attribution
- Installation instructions
- How to download Kaggle R2000 dataset
- How to run 5-fold cross validation training with paper hyperparameters
- How to run evaluation and inference
- How to verify installation

- [ ] **Step 2: Run all tests**
Run: `pytest -v`
Expected: All tests PASS.

- [ ] **Step 3: Run full verification suite**
Run: `python verify_paddy_yolo.py`
Expected: All verification steps PASS with clean output.

- [ ] **Step 4: Commit and finalize**
```bash
git add README.md
git commit -m "docs: add Paddy-YOLO documentation and verify all tests pass"
```
