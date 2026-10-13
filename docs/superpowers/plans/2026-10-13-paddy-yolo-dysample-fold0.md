# Paddy-YOLO with DySample (Configuration C) Fold 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Configuration C of the rice pest detection ablation study (Paddy-YOLO with DySample dynamic upsampling in the feature fusion neck), validate unit tests and model architectures on a dedicated `Paddy-Yolo-DySample` branch, and generate an automated Kaggle kernel for training Fold 0 on GPU.

**Architecture:** Replace the two standard `nn.Upsample` (nearest-neighbor) operations in the Paddy-YOLO neck with ultra-lightweight `DySample` blocks (`scale=2`, `style='lp'`, `groups=4`). Maintain identical layer indexing to allow 100% direct pre-trained weight transfer from `yolov8n.pt`. Automate Fold 0 training packaging via a standalone builder script for Kaggle execution.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, Ultralytics (YOLOv8), NumPy, PyYAML, Pytest.

## Global Constraints
- Target branch is `Paddy-Yolo-DySample` branched from `Paddy-Yolo-Baseline`.
- Maintain exact 16 rice pest class ordering defined in Table 4 of the paper.
- Preserve DIMIoU Loss (Strategy H) and AdamW constant learning rate schedule (`configs/hyp_paper.yaml`).
- Zero placeholders in implementation tasks; all commands, code blocks, and test assertions must be complete.

---

### Task 1: Branch Management & Workspace Setup

**Files:**
- Create and checkout branch: `Paddy-Yolo-DySample` from `Paddy-Yolo-Baseline`

**Interfaces:**
- Consumes: Existing local git repository state.
- Produces: Active git branch `Paddy-Yolo-DySample` branched from `origin/Paddy-Yolo-Baseline`.

- [ ] **Step 1: Check git status and fetch remotes**

```bash
git fetch origin
git status
```

- [ ] **Step 2: Create and checkout the dedicated branch**

```bash
git checkout -b Paddy-Yolo-DySample origin/Paddy-Yolo-Baseline
```

- [ ] **Step 3: Verify the active branch and base commit**

```bash
git branch --show-current
git log -n 1 --oneline
```
Expected: Active branch is `Paddy-Yolo-DySample`, based on `Paddy-Yolo-Baseline`.

---

### Task 2: DySample Module Implementation & Unit Tests

**Files:**
- Create: `paddy_yolo/modules/dysample.py`
- Create: `tests/test_dysample.py`

**Interfaces:**
- Consumes: PyTorch `nn.Module`, `torch.nn.functional.grid_sample`, `torch.nn.functional.pixel_shuffle`.
- Produces: `class DySample(nn.Module)` accepting `(in_channels, scale=2, style='lp', groups=4, dyscope=False)`.

- [ ] **Step 1: Write the failing unit tests in `tests/test_dysample.py`**

```python
"""Unit tests for DySample dynamic upsampling module."""

import pytest
import torch
from paddy_yolo.modules.dysample import DySample


def test_dysample_shape_and_scale():
    """Verify DySample 2x spatial upsampling preserves channels and doubles H, W."""
    b, c, h, w = 2, 64, 20, 20
    x = torch.randn(b, c, h, w)
    dysample = DySample(in_channels=c, scale=2, style="lp", groups=4)
    out = dysample(x)
    assert out.shape == (b, c, 40, 40), f"Expected (2, 64, 40, 40), got {out.shape}"


def test_dysample_gradient_flow():
    """Verify backpropagation through grid_sample and offset convolution."""
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    dysample = DySample(in_channels=32, scale=2, style="lp", groups=4)
    out = dysample(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input tensor gradient must not be None"
    assert x.grad.norm().item() > 0.0, "Input tensor gradient must be non-zero"
    assert dysample.offset.weight.grad is not None, "Offset conv gradient must not be None"


def test_dysample_channel_group_fallback():
    """Verify DySample gracefully handles cases where in_channels is smaller than groups."""
    x = torch.randn(1, 2, 10, 10)
    dysample = DySample(in_channels=2, scale=2, style="lp", groups=4)
    out = dysample(x)
    assert out.shape == (1, 2, 20, 20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dysample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paddy_yolo.modules.dysample'`

- [ ] **Step 3: Implement `paddy_yolo/modules/dysample.py`**

```python
"""DySample: Ultra-lightweight Dynamic Upsampler for Object Detection.

Reference:
    Liu et al., "DySample: A Lightweight Dynamic Upsampler for Object Detection and Segmentation",
    ICCV 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Dynamic Point-based Upsampler.

    Args:
        in_channels: Number of input feature channels.
        scale: Spatial upsampling factor (typically 2).
        style: 'lp' (linear projection) or 'pl' (point-wise convolution). Default: 'lp'.
        groups: Number of channel groups for independent offset sampling. Default: 4.
        dyscope: Whether to use dynamic scope adjustment. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        assert style in ["lp", "pl"], f"Unsupported style '{style}'. Choose 'lp' or 'pl'."
        assert scale in [2, 4], f"Unsupported scale factor {scale}. Choose 2 or 4."

        self.scale = scale
        self.style = style

        # Ensure groups divide in_channels; fallback to 1 if not divisible or in_channels < groups
        if groups > in_channels or in_channels % groups != 0:
            groups = 1
        self.groups = groups
        self.dyscope = dyscope

        out_channels = 2 * groups * scale * scale if style == "lp" else 2 * scale * scale
        self.offset = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)

        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=True)
            nn.init.constant_(self.scope.weight, 0)
            nn.init.constant_(self.scope.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        g = self.groups

        offset = self.offset(x)
        if self.dyscope:
            scope = torch.sigmoid(self.scope(x)) * 0.5
            offset = offset * scope

        # Normalized base meshgrid coordinates in [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(
                -1.0 + 1.0 / (s * h),
                1.0 - 1.0 / (s * h),
                s * h,
                device=x.device,
                dtype=x.dtype,
            ),
            torch.linspace(
                -1.0 + 1.0 / (s * w),
                1.0 - 1.0 / (s * w),
                s * w,
                device=x.device,
                dtype=x.dtype,
            ),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, s*h, s*w)

        if self.style == "lp":
            offset = F.pixel_shuffle(offset, s)  # (B, 2*g, s*h, s*w)
            offset_x = offset[:, 0::2] / w
            offset_y = offset[:, 1::2] / h
            offset = torch.cat([offset_x, offset_y], dim=1)  # (B, 2*g, s*h, s*w)

            x_g = x.view(b * g, c // g, h, w)
            bg = base_grid.repeat(b * g, 1, 1, 1)  # (b*g, 2, s*h, s*w)
            off = offset.view(b * g, 2, s * h, s * w)
            sample_grid = (bg + off).permute(0, 2, 3, 1)  # (b*g, s*h, s*w, 2)

            out = F.grid_sample(
                x_g,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            return out.view(b, c, s * h, s * w)
        else:
            offset = F.pixel_shuffle(offset, s)
            offset[:, 0:1] /= w
            offset[:, 1:2] /= h
            sample_grid = (base_grid + offset).permute(0, 2, 3, 1)
            return F.grid_sample(
                x,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dysample.py -v`
Expected: 3 passed in < 1s.

- [ ] **Step 5: Commit**

```bash
git add paddy_yolo/modules/dysample.py tests/test_dysample.py
git commit -m "feat: implement DySample dynamic upsampler module with unit tests"
```

---

### Task 3: Module Registration & Ultralytics Dynamic Injection

**Files:**
- Modify: `paddy_yolo/modules/__init__.py`
- Modify: `paddy_yolo/modules/register.py`

**Interfaces:**
- Consumes: `DySample` from `paddy_yolo.modules.dysample`.
- Produces: Exported `DySample` in `paddy_yolo.modules`, registered in `CUSTOM_MODULES` and supported in `parse_model`.

- [ ] **Step 1: Export DySample in `paddy_yolo/modules/__init__.py`**

Add `DySample` to `__all__` in `paddy_yolo/modules/__init__.py`:
```python
from paddy_yolo.modules.dysample import DySample

__all__ = [
    "RepConv",
    "GhostConv",
    "RGGConv",
    "RGGBottleneck",
    "RGG",
    "EMA",
    "DySample",
    "register_paddy_modules",
]
```

- [ ] **Step 2: Update `paddy_yolo/modules/register.py`**

Import `DySample` and register it in `CUSTOM_MODULES` and `fallback_parse_model`:
```python
from paddy_yolo.modules.dysample import DySample

CUSTOM_MODULES: Dict[str, Type[nn.Module]] = {
    "RGG": RGG,
    "RGGBottleneck": RGGBottleneck,
    "RGGConv": RGGConv,
    "RepConv": RepConv,
    "GhostConv": GhostConv,
    "EMA": EMA,
    "DySample": DySample,
}
```
In `fallback_parse_model`:
```python
elif m in {
    getattr(ultralytics_tasks, "AIFI", None),
    EMA,
    getattr(ultralytics_tasks, "EMA", None),
    DySample,
    getattr(ultralytics_tasks, "DySample", None),
}:
    args = [ch[f], *args]
    c2 = ch[f]
```

- [ ] **Step 3: Test registration in `tests/test_register.py`**

Run: `python -c "from paddy_yolo.modules.register import register_paddy_modules; register_paddy_modules(); print('Registration OK')"`
Expected: Prints `Registration OK`.

- [ ] **Step 4: Commit**

```bash
git add paddy_yolo/modules/__init__.py paddy_yolo/modules/register.py
git commit -m "feat: register DySample in Paddy-YOLO Ultralytics runtime"
```

---

### Task 4: Architecture Config & Validation Test

**Files:**
- Create: `configs/paddy-yolo-dysample.yaml`
- Create: `tests/test_paddy_yolo_dysample.py`

**Interfaces:**
- Consumes: Ultralytics YAML parsing engine, `register_paddy_modules`.
- Produces: `configs/paddy-yolo-dysample.yaml` loading cleanly with parameter count ~3.16M params.

- [ ] **Step 1: Create `configs/paddy-yolo-dysample.yaml`**

```yaml
# Paddy-YOLO with DySample (Configuration C)
# Backbone: Standard RGG modules (Layers 0-9)
# Neck: DySample dynamic upsamplers replacing nn.Upsample (Layers 10 and 13)

nc: 16
scale: n
scales:
  n: [0.33, 0.25, 1024]

# Paddy-YOLO backbone
backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
  - [-1, 3, RGG, [128, True]]       # 2 (replaces C2f)
  - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
  - [-1, 6, RGG, [256, True]]       # 4 (replaces C2f)
  - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
  - [-1, 6, RGG, [512, True]]       # 6 (replaces C2f)
  - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
  - [-1, 3, RGG, [1024, True]]      # 8 (replaces C2f)
  - [-1, 1, SPPF, [1024, 5]]        # 9

# Paddy-YOLO neck with DySample
head:
  - [-1, 1, DySample, [2, "lp", 4]] # 10 (replaces nn.Upsample)
  - [[-1, 6], 1, Concat, [1]]       # 11: cat backbone P4
  - [-1, 3, C2f, [512]]             # 12

  - [-1, 1, DySample, [2, "lp", 4]] # 13 (replaces nn.Upsample)
  - [[-1, 4], 1, Concat, [1]]       # 14: cat backbone P3
  - [-1, 3, C2f, [256]]             # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]      # 16
  - [[-1, 12], 1, Concat, [1]]      # 17: cat head P4
  - [-1, 3, C2f, [512]]             # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]      # 19
  - [[-1, 9], 1, Concat, [1]]       # 20: cat head P5
  - [-1, 3, C2f, [1024]]            # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # 22: Detect(P3, P4, P5)
```

- [ ] **Step 2: Create validation test in `tests/test_paddy_yolo_dysample.py`**

```python
"""Tests for Paddy-YOLO with DySample architecture loading and forward pass."""

import torch
from paddy_yolo.modules.register import register_paddy_modules
from ultralytics import YOLO


def test_paddy_yolo_dysample_build():
    """Verify configs/paddy-yolo-dysample.yaml builds and runs forward pass."""
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-dysample.yaml")

    dummy_input = torch.randn(1, 3, 640, 640)
    preds = model(dummy_input, verbose=False)

    assert preds is not None
    # Check total parameter count is approximately 3.16M
    param_count = sum(p.numel() for p in model.model.parameters())
    assert 3_000_000 < param_count < 3_500_000, f"Unexpected param count: {param_count}"
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_paddy_yolo_dysample.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add configs/paddy-yolo-dysample.yaml tests/test_paddy_yolo_dysample.py
git commit -m "feat: define Paddy-YOLO DySample architecture and validation test"
```

---

### Task 5: Pretrained Weights Transfer & Verification

**Files:**
- Modify: `train.py`
- Test: `tests/test_paddy_yolo_dysample.py`

**Interfaces:**
- Consumes: `yolov8n.pt` COCO weights, `Paddy-YOLO-DySample` model.
- Produces: Correct initialization in `train.py` when `--weights yolov8n.pt` is passed.

- [ ] **Step 1: Verify direct 1-to-1 weight transfer logic in `train.py`**

Confirm that `paddy-yolo-dysample` uses standard direct layer mapping (identical to `paddy-yolo`) since layer indices 0..22 are unchanged. Add a test in `tests/test_paddy_yolo_dysample.py`:

```python
def test_pretrained_weights_transfer():
    """Verify transferring weights from yolov8n.pt into Paddy-YOLO-DySample."""
    from ultralytics import YOLO
    from pathlib import Path

    weights_path = Path("yolov8n.pt")
    if not weights_path.is_file():
        return

    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-dysample.yaml")
    model.load(str(weights_path))

    # Verify model forward pass succeeds after loading weights
    out = model(torch.randn(1, 3, 640, 640), verbose=False)
    assert out is not None
```

- [ ] **Step 2: Run test to verify transfer succeeds**

Run: `pytest tests/test_paddy_yolo_dysample.py -k test_pretrained_weights_transfer -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_paddy_yolo_dysample.py
git commit -m "test: verify pretrained weight transfer into Paddy-YOLO DySample"
```

---

### Task 6: Kaggle Fold 0 DySample Kernel Builder

**Files:**
- Create: `build_fold0_dysample_kernel.py`
- Generate: `kaggle_fold0_dysample_kernel/paddy_yolo_dysample_fold0.ipynb`

**Interfaces:**
- Consumes: Codebase files, base64 archiver, Jupyter notebook structure.
- Produces: Standalone `kaggle_fold0_dysample_kernel/paddy_yolo_dysample_fold0.ipynb` and `kernel-metadata.json`.

- [ ] **Step 1: Implement `build_fold0_dysample_kernel.py`**

Create `build_fold0_dysample_kernel.py` configured with:
- Model: `configs/paddy-yolo-dysample.yaml`
- Weights: `yolov8n.pt`
- Training command:
  ```bash
  python -u train.py --model configs/paddy-yolo-dysample.yaml --weights yolov8n.pt --data /kaggle/working/data/r2000/data_fold0.yaml --hyp configs/hyp_paper.yaml --epochs 300 --batch 32 --imgsz 640 --cache True --device 0 --project /kaggle/working/runs/train --name paddy_yolo_dysample_fold0
  ```
- Unseen test split evaluation:
  ```bash
  python -u val.py --weights /kaggle/working/runs/train/paddy_yolo_dysample_fold0/weights/best.pt --data /kaggle/working/data/r2000/data_fold0.yaml --split test --imgsz 640 --device 0
  ```
- Side-by-side comparison synthesis table against Baseline Fold 0 (67.55% mAP@50) and EMA Fold 0 (69.83% mAP@50).

- [ ] **Step 2: Execute script to build the Kaggle kernel bundle**

```bash
python build_fold0_dysample_kernel.py
```
Expected:
- Prints codebase archive byte count.
- Outputs `kaggle_fold0_dysample_kernel/paddy_yolo_dysample_fold0.ipynb` and `kernel-metadata.json`.

- [ ] **Step 3: Verify kernel notebook validity**

```bash
python -c "import json; nb = json.load(open('kaggle_fold0_dysample_kernel/paddy_yolo_dysample_fold0.ipynb', encoding='utf-8')); print(f'Valid notebook with {len(nb[\"cells\"])} cells.')"
```
Expected: `Valid notebook with 6 cells.`

- [ ] **Step 4: Commit**

```bash
git add build_fold0_dysample_kernel.py
git commit -m "feat: add Kaggle Fold 0 DySample kernel builder script"
```
