# Technical Design Specification: Paddy-YOLO with DySample (Configuration C) Fold 0

> **Date:** 2026-10-13  
> **Status:** Draft (Pending User Review)  
> **Target Branch:** `Paddy-Yolo-DySample` (branched from `Paddy-Yolo-Baseline`)  
> **Goal:** Implement Configuration C of the ablation study (Paddy-YOLO with Dynamic Upsampling replacing nearest-neighbor upsamplers in the feature fusion neck) and build a dedicated Kaggle Fold 0 benchmark pipeline.

---

## 1. Background & Objectives

In the rice pest detection ablation study ([`docs/Ablation_Study_Summary_Paddy_YOLO_EMA_DySample.md`](file:///d:/Codes/Experiment/docs/Ablation_Study_Summary_Paddy_YOLO_EMA_DySample.md)), four configurations are defined to isolate architectural contributions:
* **Configuration A**: Baseline Paddy-YOLO (RGG backbone + nearest upsampling neck)
* **Configuration B**: Paddy-YOLO + EMA (RGG + EMA backbone + nearest upsampling neck)
* **Configuration C**: **Paddy-YOLO + DySample** (RGG backbone + DySample dynamic upsampling neck)
* **Configuration D**: Paddy-YOLO + EMA + DySample (Joint attention & dynamic upsampling)

This specification defines the complete end-to-end implementation for **Configuration C** evaluated on **Fold 0** of the R2000 benchmark dataset on Kaggle GPU.

---

## 2. Architecture & Mathematical Formulation

### 2.1 DySample (Dynamic Upsampler) Formulation
DySample (*Liu et al., ICCV 2023*) formulates dynamic upsampling from a point sampling perspective rather than dynamic convolution kernels (e.g., CARAFE), achieving extreme parameter efficiency and high speed without custom CUDA kernels.

Given input feature map $\mathcal{X} \in \mathbb{R}^{C \times H \times W}$ and upsampling scale factor $s = 2$:
1. **Sampling Point Offset Generation**:
   A lightweight linear projection / $1 \times 1$ conv predicts point offsets $\Delta \mathcal{O} \in \mathbb{R}^{2g \times sH \times sW}$ where $g$ is the number of groups (default $g = 4$):
   $$\Delta \mathcal{O} = \text{Conv}_{1 \times 1}(\mathcal{X})$$
2. **Dynamic Grid Construction**:
   A uniform sampling grid $\mathcal{G} \in \mathbb{R}^{2 \times sH \times sW}$ is perturbed by the predicted offsets:
   $$\mathcal{G}' = \mathcal{G} + \Delta \mathcal{O}$$
3. **Resampling**:
   Output features $\mathcal{Y} \in \mathbb{R}^{C \times sH \times sW}$ are sampled using bilinear interpolation:
   $$\mathcal{Y} = \text{grid\_sample}(\mathcal{X}, \mathcal{G}', \text{mode}='bilinear', \text{padding\_mode}='zeros', \text{align\_corners}=False)$$

### 2.2 Model Architecture: `configs/paddy-yolo-dysample.yaml`
In the baseline Paddy-YOLO neck, two `nn.Upsample` layers perform spatial scaling:
* **Layer 10**: Upsamples P5 ($32\times \to 16\times$) before concatenation with Backbone P4.
* **Layer 13**: Upsamples P4 ($16\times \to 8\times$) before concatenation with Backbone P3.

In `configs/paddy-yolo-dysample.yaml`:
```yaml
# Paddy-YOLO head with DySample
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

**Key Advantage**: Zero layer-shift. All layer indices remain identical to baseline `paddy-yolo.yaml` (0 to 22), preserving 100% direct pre-trained weight transfer from `yolov8n.pt`.

---

## 3. Module Registration & Integration

1. **`paddy_yolo/modules/dysample.py`**:
   Implements `DySample(in_channels, scale=2, style='lp', groups=4, dyscope=False)`.
2. **`paddy_yolo/modules/register.py`**:
   * Add `DySample` to `CUSTOM_MODULES`.
   * Add parsing clause in `fallback_parse_model`:
     ```python
     elif m is DySample or (isinstance(m, type) and m.__name__ == "DySample"):
         c1 = ch[f]
         args = [c1, *args]
         c2 = c1
     ```
3. **`paddy_yolo/modules/__init__.py`**:
   Export `DySample`.

---

## 4. Kaggle Fold 0 Training & Packaging

1. **Kernel Builder (`build_fold0_dysample_kernel.py`)**:
   * Bundles codebase into base64 archive.
   * Generates `kaggle_fold0_dysample_kernel/paddy_yolo_dysample_fold0.ipynb` and `kernel-metadata.json`.
   * Unpacks project in `/kaggle/working`.
   * Verifies GPU / PyTorch environment.
   * Downloads or loads R2000 dataset, creating 5-fold splits.
   * Trains Fold 0 for 300 epochs (`configs/hyp_paper.yaml`, AdamW, batch 32, imgsz 640).
   * Evaluates `best.pt` on the unseen Fold 0 test split (`--split test`).
   * Synthesizes and exports results to CSV comparing Fold 0 DySample vs Baseline Fold 0 (67.55% mAP@50) and EMA Fold 0 (69.83% mAP@50).

---

## 5. Verification & Testing

* **Unit Tests (`tests/test_dysample.py`)**:
  * Output shape verification: $(B, C, H, W) \to (B, C, 2H, 2W)$ across varying batch sizes ($B=1, 4$) and channel dimensions ($C=64, 128, 256$).
  * Gradient flow and backpropagation: $\frac{\partial \mathcal{L}}{\partial \mathcal{X}}$ and $\frac{\partial \mathcal{L}}{\partial \mathcal{W}_{\text{proj}}}$ are non-zero.
  * Model architecture loading: `YOLO("configs/paddy-yolo-dysample.yaml")` parses cleanly.
  * Pre-trained weight transfer: `transfer_pretrained_weights` runs without shape mismatches.
* **Kernel Verification**:
  * Run `python build_fold0_dysample_kernel.py` to confirm notebook generation without errors.
