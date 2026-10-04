# Paddy-YOLO Recreation Design Specification

- **Date**: 2026-10-04
- **Target**: Recreation of Paddy-YOLO (YOLOv8-N with RGG Feature Extraction & Dynamic Inner-MPDIoU Loss)
- **Reference Paper**: Zheng et al., *Paddy-YOLO: An accurate method for rice pest detection*, Computers and Electronics in Agriculture 238 (2025) 110777.

---

## 1. Executive Summary

This project reproduces **Paddy-YOLO**, an object detection model optimized for rice pest detection in complex natural field environments. Paddy-YOLO is built upon the Ultralytics YOLOv8 architecture (specifically YOLOv8-N) and introduces two core innovations:
1. **RGG Feature Extraction Module**: Replaces standard C2f blocks in the YOLOv8 backbone with an RGG module combining structural re-parameterization (RepConv), lightweight dual GhostConvs with 5x5 depthwise convolutions (DWConv), and multi-level feature concatenation.
2. **Dynamic Inner-MPDIoU (DIMIoU) Loss**: Replaces standard CIoU bounding box regression loss with an enhanced loss integrating MPDIoU corner distance constraints, InnerIoU auxiliary ratio scaling, and dynamic cosine ratio scheduling (Strategy H) conditioned on IoU.

The codebase is organized as a modular Python package (`paddy_yolo`) that integrates seamlessly with the official `ultralytics` package without altering site-packages.

---

## 2. Model Architecture

### 2.1 Base Network
- **Base Model**: YOLOv8-N (`depth_multiple: 0.33`, `width_multiple: 0.25`, `max_channels: 1024`).
- **Input Resolution**: 640x640x3.
- **Backbone Modifications**: The 4 C2f modules at layers 2, 4, 6, and 8 of YOLOv8-N are replaced by `RGG` modules.
- **Neck & Detection Heads**: Retains YOLOv8 PANet feature fusion neck and decoupled detection heads (P3, P4, P5) predicting 16 rice pest classes.

### 2.2 RGG Structural Components

#### A. RepConv
- **Structure during training**: Three parallel branches:
  - Branch 1: $3 \times 3$ Conv2d + BatchNorm2d
  - Branch 2: $1 \times 1$ Conv2d + BatchNorm2d
  - Branch 3: Identity + BatchNorm2d (when in_channels == out_channels)
- **Deployment fusion (`switch_to_deploy`)**: Fuses batch normalization weights and 1x1/identity branches into a single equivalent $3 \times 3$ Conv2d kernel and bias, eliminating multi-branch latency during inference.

#### B. GhostConv (with 5x5 DWConv)
- **Primary Convolution**: $1 \times 1$ or $3 \times 3$ standard Conv2d to produce $C/2$ intrinsic feature maps.
- **Cheap Operation**: $5 \times 5$ Depthwise Convolution (`groups = C/2`) with padding 2, expanding the receptive field to capture fine appendages (antennae, legs).
- **Concatenation**: Concatenates intrinsic and transformed feature maps along the channel axis.

#### C. RGGConv
- Input: $C$ channels.
- Branch 1: `RepConv(C, C)`
- Branch 2: `GhostConv(C, C)`
- Branch 3: `GhostConv(C, C)`
- Output: Concatenation of all three branches ($3C$ channels) projected back to $C$ channels via a $1 \times 1$ Conv2d.

#### D. RGGBottleneck
- Residual block with shortcut connection:
  $$\text{Output} = x + \text{RGGConv}(\text{Conv}_{1\times 1}(x))$$ (when shortcut is enabled and in_channels == out_channels).

#### E. RGG Module (C2f-RGG)
- Follows the C2f split-and-merge topology:
  - Input $C$ channels $\to 1\times 1$ Conv $\to$ split into two branches of $C/2$ channels.
  - One branch bypasses directly.
  - The second branch passes through $n$ sequential `RGGBottleneck` blocks.
  - All intermediate feature maps and the bypass are concatenated $\to 1\times 1$ Conv $\to C$ channels output.

---

## 3. Loss Function: Dynamic Inner-MPDIoU (DIMIoU)

### 3.1 Mathematical Definition
For ground truth box $A = (x_{A1}, y_{A1}, x_{A2}, y_{A2})$ and predicted box $B = (x_{B1}, y_{B1}, x_{B2}, y_{B2})$:

1. **Normalized Corner Distances (MPDIoU component)**:
   $$d_1^2 = (x_{A1} - x_{B1})^2 + (y_{A1} - y_{B1})^2$$
   $$d_2^2 = (x_{A2} - x_{B2})^2 + (y_{A2} - y_{B2})^2$$
   Normalized by $w^2 + h^2$ where $w, h$ are the image dimensions (or bounding box union dimensions).

2. **InnerIoU Auxiliary Scaling**:
   Given center coordinates and dimensions $(x_c, y_c, w, h)$, scaled boxes are computed with factor `ratio`:
   $$x_1^{\text{ratio}} = x_c - \frac{w \cdot \text{ratio}}{2}, \quad x_2^{\text{ratio}} = x_c + \frac{w \cdot \text{ratio}}{2}$$
   $$y_1^{\text{ratio}} = y_c - \frac{h \cdot \text{ratio}}{2}, \quad y_2^{\text{ratio}} = y_c + \frac{h \cdot \text{ratio}}{2}$$
   $$(A \cap B)_{\text{ratio}} = \max(0, \min(x_{A2}^{\text{ratio}}, x_{B2}^{\text{ratio}}) - \max(x_{A1}^{\text{ratio}}, x_{B1}^{\text{ratio}})) \times \max(0, \min(y_{A2}^{\text{ratio}}, y_{B2}^{\text{ratio}}) - \max(y_{A1}^{\text{ratio}}, y_{B1}^{\text{ratio}}))$$
   $$(A \cup B)_{\text{ratio}} = (w_A \cdot \text{ratio}) \times (h_A \cdot \text{ratio}) + (w_B \cdot \text{ratio}) \times (h_B \cdot \text{ratio}) - (A \cap B)_{\text{ratio}}$$

3. **Dynamic Cosine Ratio Scheduling (Strategy H)**:
   $$\text{ratio} = \frac{1}{2} \cos((1 - \text{IoU}) \cdot \pi) + 1$$
   - Maps IoU $\in [0, 1]$ smoothly to ratio $\in [0.5, 1.5]$.
   - Emphasizes high-IoU samples and stabilizes convergence on challenging samples.

4. **Loss**:
   $$\mathcal{L}_{\text{DIMIoU}} = 1 - \frac{(A \cap B)_{\text{ratio}}}{(A \cup B)_{\text{ratio}}} + \frac{d_1^2}{w^2 + h^2} + \frac{d_2^2}{w^2 + h^2}$$

---

## 4. Dataset & 5-Fold Cross Validation

### 4.1 Classes (16 rice pests)
0: rice leaf roller, 1: pink rice borer, 2: asiatic rice borer, 3: yellow rice borer, 4: rice leaf caterpillar, 5: small brown plant hopper, 6: white backed plant hopper, 7: brown plant hopper, 8: rice leaf hopper, 9: rice gall midge, 10: paddy stem maggot, 11: rice stem fly, 12: grain spreader thrips, 13: rice weevil, 14: rice shell pest, 15: melanitis leda.

### 4.2 Split & Pipeline
- **Kaggle Source**: `dmrashidferrer/r2000v1-2024-images` (2,024 images, 2,281 instances).
- **Split Strategy**: 5-fold cross validation. For each fold index $k \in \{0, 1, 2, 3, 4\}$:
  - Train: 3 folds
  - Val: 1 fold
  - Test: 1 fold
- **Mock/Synthetic Generator**: Provides `--generate-dummy` in `paddy_yolo/dataset/prepare_r2000.py` to synthesize a lightweight testing set with images and labels, enabling instant smoke-tests on CPU.

---

## 5. Software Architecture & File Layout

```text
d:\Codes\Experiment\
├── configs/
│   ├── paddy-yolo.yaml             # Model architecture config
│   ├── r2000.yaml                  # Dataset config (16 classes)
│   └── hyp_paper.yaml              # Hyperparameters (Paper Table 6)
├── paddy_yolo/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── repconv.py              # RepConv with re-param fuse
│   │   ├── ghostconv.py            # GhostConv (5x5 DWConv)
│   │   ├── rgg.py                  # RGGConv, RGGBottleneck, RGG
│   │   └── register.py             # Injects RGG into Ultralytics parser
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── dimiou.py               # DIMIoU loss computation
│   │   └── paddy_loss.py           # v8DetectionLoss replacement
│   ├── engine/
│   │   ├── __init__.py
│   │   └── trainer.py              # PaddyTrainer subclass
│   └── dataset/
│       ├── __init__.py
│       └── prepare_r2000.py        # Dataset parser, Kaggle helper, 5-fold CV
├── train.py                        # Training script
├── val.py                          # Validation script
├── predict.py                      # Detection script
├── verify_paddy_yolo.py            # Complete self-verification script
└── requirements.txt
```

---

## 6. Verification and Testing Plan

1. **Unit Tests**:
   - `RepConv`: Verify training multi-branch vs eval fused single-conv equivalence.
   - `GhostConv`: Verify shape and receptive field.
   - `RGG`: Verify forward pass, channel dimensions, and gradient backprop.
   - `DIMIoU`: Verify corner distances, ratio scaling, and non-negative loss values.
2. **Model Integration Test**:
   - Parse `paddy-yolo.yaml` with Ultralytics parser.
   - Check parameter count and FLOPs match expected range (~3.3M params, ~9.1 GFLOPs).
   - Verify forward pass on batch $(B=2, C=3, H=640, W=640)$ produces expected detection tensors.
3. **Pipeline End-to-End Test**:
   - Generate dummy dataset via `prepare_r2000.py --generate-dummy`.
   - Run 1 epoch of training via `PaddyTrainer`.
   - Run validation and ensure predictions and metrics are produced without error.
