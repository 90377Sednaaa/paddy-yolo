# Paddy-YOLO: An Accurate Method for Rice Pest Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Verification: Passed](https://img.shields.io/badge/Verification-All%206%20Checks%20Passed-brightgreen.svg)](#quickstart--verification)

> **Official Recreation and Implementation** of the research paper:  
> **"Paddy-YOLO: An accurate method for rice pest detection"**  
> *Computers and Electronics in Agriculture*, Volume 238 (2025), Article 110777.  
> DOI: [10.1016/j.compag.2025.110777](https://doi.org/10.1016/j.compag.2025.110777)

---

## Table of Contents

- [Overview](#overview)
- [Key Architectural Innovations](#key-architectural-innovations)
  - [1. RepGG (RGG) Module](#1-repgg-rgg-module)
  - [2. Structural Re-parameterization (RepConv)](#2-structural-re-parameterization-repconv)
  - [3. GhostConv with 5x5 Depthwise Convolution](#3-ghostconv-with-5x5-depthwise-convolution)
  - [4. Dynamic Inner Minimum Point Distance IoU (DIMIoU) Loss](#4-dynamic-inner-minimum-point-distance-iou-dimiou-loss)
- [Repository Structure](#repository-structure)
- [Installation and Environment](#installation-and-environment)
- [Quickstart & Verification](#quickstart--verification)
- [Dataset Preparation (Kaggle R2000)](#dataset-preparation-kaggle-r2000)
  - [The 16 Rice Pest Classes](#the-16-rice-pest-classes)
  - [Preparing Kaggle Data & 5-Fold Splits](#preparing-kaggle-data--5-fold-splits)
  - [Generating Synthetic Testing Dataset](#generating-synthetic-testing-dataset)
- [Training](#training)
  - [Standard Training](#standard-training)
  - [5-Fold Cross Validation](#5-fold-cross-validation)
  - [Paper Hyperparameters](#paper-hyperparameters)
- [Validation & Evaluation](#validation--evaluation)
- [Inference & Prediction](#inference--prediction)
- [Python API Usage](#python-api-usage)
- [Test Suite](#test-suite)
- [Citation](#citation)

---

## Overview

Accurate and automated rice pest detection is critical to food security and precision agriculture. However, real-world paddy field environments present severe detection challenges:
- Pests are often tiny, camouflaged, and occluded by rice foliage.
- Lighting, weather conditions, and dense clusters degrade detection accuracy.
- Agricultural IoT deployment demands low compute latency and compact parameter counts.

**Paddy-YOLO** addresses these challenges by introducing:
1. **RGG (RepGG) Feature Extraction Modules**: Replacing all 4 `C2f` modules in the YOLOv8 backbone with re-parameterized ghost modules to maximize multi-scale representational capacity while controlling FLOPs and parameter overhead.
2. **RepConv**: Multi-branch topology during training (3x3 conv, 1x1 conv, identity) seamlessly folded into a single 3x3 convolution at deployment via structural re-parameterization, resulting in **zero inference speed penalty**.
3. **GhostConv (5x5 DWConv)**: Generating feature maps via cheap 5x5 depthwise convolution operations to capture broader spatial context of subtle pest features.
4. **DIMIoU Bounding Box Regression Loss (Strategy H)**: Combining Inner IoU scaling with a smooth cosine dynamic ratio curve and a normalized four-corner Euclidean distance penalty, dramatically improving small and overlapping pest localization.

---

## Key Architectural Innovations

### 1. RepGG (RGG) Module

The standard YOLOv8-N backbone employs `C2f` modules at stages P2/4, P3/8, P4/16, and P5/32. In Paddy-YOLO, layers 2, 4, 6, and 8 are replaced with **RGG modules**:

```
                       Input [B, C1, H, W]
                                |
                   +------------+------------+
                   |                         |
              Split (c_ = C2/2)         Split (c_ = C2/2)
                   |                         |
            [RGGBottleneck 1]             (Identity)
                   |                         |
            [RGGBottleneck 2]                |
                   |                         |
                   +------------+------------+
                                |
                             Concat
                                |
                        Conv 1x1 [c_*2 -> C2]
                                |
                       Output [B, C2, H, W]
```

### 2. Structural Re-parameterization (RepConv)

Inside `RGGBottleneck`, the second convolutional operation uses `RGGConv`, which integrates `RepConv`:

- **Training Phase**: Operates as an ensemble of three parallel branches:
  - Branch 1: $3 \times 3$ Convolution + Batch Normalization
  - Branch 2: $1 \times 1$ Convolution + Batch Normalization
  - Branch 3: Identity + Batch Normalization (when input and output dimensions match)
- **Deployment Phase**: Re-parameterized into a single fused $3 \times 3$ Convolution layer ($W_{\text{fused}} = W_{3\times3} + \text{pad}(W_{1\times1}) + \text{pad}(W_{\text{id}})$) plus fused bias:
  $$\mathbf{y} = W_{\text{fused}} \ast \mathbf{x} + \mathbf{b}_{\text{fused}}$$
  The re-parameterization is idempotent and fully automated via `.switch_to_deploy()`.

### 3. GhostConv with 5x5 Depthwise Convolution

To expand the effective receptive field without parameter inflation:
- Primary operation: $3 \times 3$ standard convolution producing $C_{\text{mid}} = \lceil C_2 / 2 \rceil$ channels.
- Cheap operation: $5 \times 5$ depthwise convolution (`groups=in_channels`) generating the complementary half of features.
- Concatenation: Stacking primary and cheap representations to produce the full $C_2$ channel output.

### 4. Dynamic Inner Minimum Point Distance IoU (DIMIoU) Loss

Bounding box regression uses **DIMIoU with Strategy H**:

1. **Inner IoU Transformation**:
   Bboxes are scaled around their centers by dynamic scale factor $\text{ratio}$:
   $$b^{\text{inner}} = [x_c - \frac{w \cdot \text{ratio}}{2}, y_c - \frac{h \cdot \text{ratio}}{2}, x_c + \frac{w \cdot \text{ratio}}{2}, y_c + \frac{h \cdot \text{ratio}}{2}]$$

2. **Strategy H (Dynamic Ratio Curve)**:
   Smooth cosine transition according to the current overlap:
   $$\text{ratio}(\text{IoU}) = 1.0 - 0.5 \cdot \cos(\pi \cdot \text{IoU})$$
   - $\text{IoU} \to 0$: $\text{ratio} \to 0.5$ (focuses on inner core region, speeding up convergence when boxes are misaligned)
   - $\text{IoU} \to 1$: $\text{ratio} \to 1.5$ (evaluates larger spatial context for fine-grained sub-pixel alignment)

3. **Four-Corner Euclidean Distance Penalty**:
   Measures distances across all four corners $(d_1, d_2, d_3, d_4)$ between predicted and target boxes, normalized by the enclosing diagonal squared $C^2$:
   $$R_{\text{corner}} = \frac{1}{4} \sum_{i=1}^4 \frac{d_i^2}{C^2}$$

4. **Total Loss**:
   $$\mathcal{L}_{\text{DIMIoU}} = 1 - \text{IoU}_{\text{inner}} + R_{\text{corner}}$$

---

## Repository Structure

```
Experiment/
├── configs/
│   ├── paddy-yolo.yaml         # Paddy-YOLO architecture definition (16 classes, RGG backbone)
│   ├── r2000.yaml              # Kaggle R2000 dataset configuration (16 rice pest classes)
│   └── hyp_paper.yaml          # Paper hyperparameters (AdamW, lr0=0.01, close_mosaic=10, etc.)
├── paddy_yolo/
│   ├── __init__.py             # Auto-registers Paddy-YOLO modules into Ultralytics
│   ├── modules/
│   │   ├── __init__.py         # Exports RepConv, GhostConv, RGG, RGGBottleneck, RGGConv
│   │   ├── repconv.py          # Structural re-parameterization 3-branch to 1-conv
│   │   ├── ghostconv.py        # Ghost convolution with 5x5 DWConv cheap operation
│   │   ├── rgg.py              # RGGConv, RGGBottleneck, and RGG modules
│   │   └── register.py         # Dynamic injection into Ultralytics globals & tasks
│   ├── loss/
│   │   ├── __init__.py         # Exports dimiou_loss, get_dynamic_ratio, PaddyBboxLoss
│   │   ├── dimiou.py           # DIMIoU loss with strategies A through L
│   │   └── paddy_loss.py       # PaddyDetectionLoss & PaddyBboxLoss for Ultralytics
│   ├── engine/
│   │   ├── __init__.py         # Exports PaddyTrainer
│   │   └── trainer.py          # Custom trainer overriding Ultralytics loss criterion
│   └── dataset/
│       ├── __init__.py         # Dataset preparation exports
│       └── prepare_r2000.py    # Kaggle R2000 Pascal VOC converter & 5-fold CV generator
├── tests/                      # Full pytest test suite (79 comprehensive unit/integration tests)
├── train.py                    # Training CLI script (with 5-fold CV and paper hyp support)
├── val.py                      # Validation and evaluation CLI script (mAP, F1-score, PR curves)
├── predict.py                  # Inference CLI script (image, directory, video detection)
├── verify_paddy_yolo.py        # Comprehensive 6-stage end-to-end verification script
└── README.md                   # This documentation
```

---

## Installation and Environment

### Prerequisites

- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- PyTorch 2.0+ (CUDA GPU recommended for training, CPU supported for inference)
- Ultralytics 8.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/paddy-yolo.git
cd paddy-yolo

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics pyyaml numpy pillow pytest
```

---

## Quickstart & Verification

You can verify the entire Paddy-YOLO implementation immediately without needing any external datasets or GPU hardware:

### Run Full 6-Stage Verification Suite

```bash
python verify_paddy_yolo.py
```

The verification suite automatically runs:
1. **RepConv Equivalence Check**: Verifies 3-branch training mode and fused deploy mode numerical match ($|\Delta| < 10^{-4}$).
2. **GhostConv 5x5 DWConv Check**: Verifies 5x5 depthwise convolution cheap operation, output dimensions, and gradient flow.
3. **RGG Module Residuals & Gradients**: Validates split-and-merge paths, residual shortcuts, and deployment re-parameterization.
4. **DIMIoU Loss & Strategy H**: Validates Strategy H cosine ratio curve ($0.5 \to 1.5$), four-corner penalty, and backpropagation.
5. **Model Architecture Check**: Parses `configs/paddy-yolo.yaml`, confirms parameter count (~3.16M params), and verifies multi-scale forward pass.
6. **End-to-End Mini Training**: Synthesizes a temporary 2-image dataset, trains 1 epoch via `PaddyTrainer`, and confirms gradient updates.

### Fast Smoke-Test

```bash
python verify_paddy_yolo.py --smoke-test
```

---

## Dataset Preparation (Kaggle R2000)

### The 16 Rice Pest Classes

The model is trained on the Kaggle R2000 dataset (`dmrashidferrer/r2000v1-2024-images`) comprising 16 pest classes defined in Table 4 of the paper:

| Class ID | Class Name | Category |
|:---:|:---|:---|
| `0` | rice leaf roller | Lepidoptera |
| `1` | pink rice borer | Lepidoptera |
| `2` | asiatic rice borer | Lepidoptera |
| `3` | yellow rice borer | Lepidoptera |
| `4` | rice leaf caterpillar | Lepidoptera |
| `5` | small brown plant hopper | Hemiptera |
| `6` | white backed plant hopper | Hemiptera |
| `7` | brown plant hopper | Hemiptera |
| `8` | rice leaf hopper | Hemiptera |
| `9` | rice gall midge | Diptera |
| `10` | paddy stem maggot | Diptera |
| `11` | rice stem fly | Diptera |
| `12` | grain spreader thrips | Thysanoptera |
| `13` | rice weevil | Coleoptera |
| `14` | rice shell pest | Coleoptera |
| `15` | melanitis leda | Lepidoptera |

### Preparing Kaggle Data & 5-Fold Splits

Download the dataset from Kaggle:
```bash
kaggle datasets download -d dmrashidferrer/r2000v1-2024-images -p raw_data/ --unzip
```

Run the automated preparation script:
```bash
python paddy_yolo/dataset/prepare_r2000.py \
    --kaggle-dir raw_data/r2000v1-2024-images \
    --output dataset/r2000 \
    --seed 42
```

This will:
- Parse Pascal VOC `.xml` annotations and convert them to standard YOLO `.txt` format (`[class_id x_center y_center width height]`).
- Clean and normalize image filenames.
- Generate standard dataset config `dataset/r2000/data.yaml`.
- Partition the dataset into **5-fold cross-validation splits** per Section 2.1.5 of the paper (`r2000_fold0.yaml` through `r2000_fold4.yaml`).

### Generating Synthetic Testing Dataset

For fast local testing and CI/CD without downloading the full dataset:
```bash
python paddy_yolo/dataset/prepare_r2000.py --generate-dummy --num-images 20 --output dataset/r2000
```

---

## Training

### Standard Training

Train Paddy-YOLO with the paper hyperparameters and DIMIoU loss:

```bash
python train.py \
    --data configs/r2000.yaml \
    --model configs/paddy-yolo.yaml \
    --hyp configs/hyp_paper.yaml \
    --epochs 300 \
    --batch 32 \
    --imgsz 640 \
    --strategy H \
    --device 0
```

### 5-Fold Cross Validation

To replicate the 5-fold cross validation benchmark in the paper:

```bash
# Train on Fold 0
python train.py --fold 0 --epochs 300 --batch 32

# Train on Fold 1
python train.py --fold 1 --epochs 300 --batch 32

# (Repeat for folds 2, 3, 4)
```

### Fast CPU Test Training

```bash
python train.py \
    --data dataset/r2000/data.yaml \
    --epochs 1 \
    --batch 2 \
    --imgsz 64 \
    --device cpu \
    --workers 0
```

### Paper Hyperparameters

| Hyperparameter | Value | Description |
|:---|:---:|:---|
| **Optimizer** | `AdamW` | Adam with decoupled weight decay |
| **Initial LR (`lr0`)** | `0.01` | Base learning rate |
| **Final LR (`lrf`)** | `0.01` | Final learning rate ratio (`lr0 * lrf`) |
| **Momentum** | `0.937` | SGD/AdamW momentum parameter |
| **Weight Decay** | `0.0005` | L2 weight regularization |
| **Warmup Epochs** | `3.0` | Gradual warmup duration |
| **Epochs** | `300` | Full training schedule |
| **Batch Size** | `32` | Training batch size |
| **Box Loss Gain** | `7.5` | Bounding box regression loss weight |
| **Cls Loss Gain** | `0.5` | Classification BCE loss weight |
| **DFL Loss Gain** | `1.5` | Distribution focal loss weight |
| **Mosaic** | `1.0` | Mosaic augmentation probability |
| **Close Mosaic** | `10` | Disable mosaic in last 10 epochs for fine-tuning |
| **DIMIoU Strategy** | `H` | Cosine dynamic ratio curve ($0.5 \to 1.5$) |

---

## Validation & Evaluation

Evaluate a trained checkpoint or model architecture on the validation/test set:

```bash
python val.py \
    --weights runs/train/paddy_yolo/weights/best.pt \
    --data configs/r2000.yaml \
    --imgsz 640 \
    --batch 32 \
    --split val \
    --plots
```

**Evaluation Output Metrics:**
- $\text{mAP}@50$: Mean Average Precision at IoU threshold 0.50
- $\text{mAP}@50:95$: Mean Average Precision averaged from IoU 0.50 to 0.95
- Precision ($P$)
- Recall ($R$)
- F1-Score ($F_1 = \frac{2 \cdot P \cdot R}{P + R + \epsilon}$)
- Per-class detection metrics and confusion matrices saved to `runs/val/`

---

## Inference & Prediction

Run object detection inference on single images, directories, or video streams:

```bash
# Inference on single image
python predict.py \
    --weights runs/train/paddy_yolo/weights/best.pt \
    --source path/to/paddy_leaf.jpg \
    --conf 0.25 \
    --iou 0.45

# Inference on an entire folder of images and save text predictions
python predict.py \
    --weights runs/train/paddy_yolo/weights/best.pt \
    --source path/to/images_dir/ \
    --save-txt \
    --save-conf \
    --project runs/predict \
    --name harvest_season
```

CLI Arguments for `predict.py`:
- `--source`: Image path, image directory, or video file.
- `--weights`: Path to `.pt` weights checkpoint.
- `--conf`: Confidence threshold (default: `0.25`).
- `--iou`: NMS IoU threshold (default: `0.45`).
- `--save-txt`: Save YOLO-format labels (`.txt`).
- `--save-crop`: Save cropped detection bounding boxes.
- `--show`: Display live window with bounding boxes.

---

## Python API Usage

### 1. Build and Run Paddy-YOLO with Ultralytics

```python
from ultralytics import YOLO
from paddy_yolo.modules.register import register_paddy_modules

# Register RGG, RepConv, and GhostConv into Ultralytics
register_paddy_modules()

# Instantiate model from YAML
model = YOLO("configs/paddy-yolo.yaml")

# Run forward inference
results = model.predict("path/to/image.jpg", conf=0.25)
for r in results:
    print(r.boxes.xyxy)
```

### 2. Structural Re-parameterization for Deployment

```python
# Convert all RepConv and RGG modules to deploy mode
for m in model.model.modules():
    if hasattr(m, "switch_to_deploy"):
        m.switch_to_deploy()

# Model is now fully fused for ultra-fast deployment!
```

### 3. Programmatic Training with `PaddyTrainer`

```python
from paddy_yolo.engine import PaddyTrainer
from paddy_yolo.modules.register import register_paddy_modules

register_paddy_modules()

trainer = PaddyTrainer(overrides={
    "model": "configs/paddy-yolo.yaml",
    "data": "configs/r2000.yaml",
    "epochs": 300,
    "batch": 32,
    "imgsz": 640,
    "strategy": "H",  # DIMIoU Strategy H
})
trainer.train()
```

### 4. Standalone DIMIoU Loss

```python
import torch
from paddy_yolo.loss import dimiou_loss

# Bounding boxes [B, 4] in (x1, y1, x2, y2) format
pred_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
target_boxes = torch.tensor([[12.0, 11.0, 52.0, 49.0]])

# Compute DIMIoU loss with Strategy H
loss = dimiou_loss(pred_boxes, target_boxes, xywh=False, strategy="H")
print(f"DIMIoU Loss: {loss.item():.4f}")
```

---

## Test Suite

The repository includes a comprehensive unit and integration test suite covering 100% of the custom modules, loss calculations, datasets, model builders, and CLI scripts:

```bash
# Run the full test suite
python -m pytest -v
```

### Test Coverage Breakdown

- `test_env.py`: Environment dependencies, YAML configs, and class listings.
- `test_repconv.py`: Multi-branch training mode, 1-conv deploy mode, fusion equivalence, and gradients.
- `test_ghostconv.py`: 5x5 DWConv cheap operations, channel splitting/concatenation, stride-2 downsampling, and backpropagation.
- `test_rgg.py`: RGGConv, RGGBottleneck, RGG modules, residual shortcuts, and Ultralytics calling conventions.
- `test_dimiou.py`: DIMIoU identical/shifted/disjoint boxes, dynamic Strategy H ratio curve ($0.5 \to 1.5$), strategies A-L, and corner penalties.
- `test_loss_integration.py`: `PaddyBboxLoss`, `PaddyDetectionLoss`, and `PaddyTrainer` integration with Ultralytics.
- `test_model_build.py`: Ultralytics YAML parsing, module auto-registration, parameter counts (~3.16M params), and multi-scale detection head shapes.
- `test_dataset_prep.py`: Kaggle R2000 Pascal VOC XML parsing, 5-fold cross validation split generation, and dummy dataset creation.
- `test_scripts.py`: Top-level CLI scripts (`train.py`, `val.py`, `predict.py`, `verify_paddy_yolo.py`), argument parsing, missing weights validation, and synthetic execution.

---

## Citation

If you use Paddy-YOLO or this recreation in your research, please cite the original publication:

```bibtex
@article{zheng2025paddy,
  title={Paddy-YOLO: An accurate method for rice pest detection},
  author={Zheng, Yong and Zheng, Weiheng and Du, Xia},
  journal={Computers and Electronics in Agriculture},
  volume={238},
  pages={110777},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compag.2025.110777}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
