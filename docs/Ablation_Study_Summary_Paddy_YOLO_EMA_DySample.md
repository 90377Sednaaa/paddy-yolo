# Ablation Study Summary: Modified Paddy-YOLO with EMA and DySample

This ablation study evaluates the individual and combined contributions of Efficient Multi-Scale Attention (EMA) and Dynamic Upsampling (DySample) to the original Paddy-YOLO architecture. All configurations retain the original RGG feature extraction modules, detection heads, and DIMIoU bounding box regression loss.

---

## 1. Ablation Configurations

| Configuration | Backbone | Feature Fusion Neck | Purpose |
| :--- | :--- | :--- | :--- |
| **A. Baseline Paddy-YOLO** | Original RGG modules | Original Upsample (nearest) | Establish reference accuracy and computational complexity. |
| **B. Paddy-YOLO + EMA** | RGG + EMA | Original Upsample (nearest) | Assess attention-based feature refinement and background suppression. |
| **C. Paddy-YOLO + DySample** | Original RGG modules | DySample replaces two Upsample operations | Assess adaptive upsampling and multi-scale feature reconstruction. |
| **D. Paddy-YOLO + EMA + DySample** | RGG + EMA | DySample replaces two Upsample operations | Assess joint effects of feature attention and dynamic upsampling. |

---

## 2. Description of Each Configuration

### A. Baseline Paddy-YOLO
The original Paddy-YOLO architecture serves as the control model. It retains the RGG modules for feature extraction, conventional upsampling operations in the feature fusion neck, and the original detection heads. Its performance provides the reference for evaluating the proposed modifications.

### B. Paddy-YOLO + EMA
EMA modules are inserted after selected RGG stages in the backbone while retaining the original upsampling operations in the neck. This configuration investigates whether cross-spatial attention improves feature discrimination by emphasizing pest-related information and suppressing irrelevant background features.

### C. Paddy-YOLO + DySample
The original backbone is preserved, while the two conventional upsampling operations in the feature fusion neck are replaced with DySample. This configuration evaluates whether adaptive sampling improves spatial detail preservation and multi-scale feature reconstruction, particularly for small pests.

### D. Paddy-YOLO + EMA + DySample
Both modifications are integrated into the original architecture, with EMA refining the backbone features and DySample improving upsampling during neck fusion. This configuration investigates whether combining both mechanisms provides complementary improvements in pest detection accuracy while maintaining computational efficiency.

---

## 3. Evaluation Metrics

All four configurations are evaluated using identical training conditions on the **IP102** and **R2000** datasets:

- **Precision ($P$)**: Proportion of correct pest predictions.
- **Recall ($R$)**: Ability to detect actual pest instances.
- **mAP@0.5**: Overall detection accuracy at an IoU threshold of 0.5.
- **mAP@0.5:0.95**: Performance across stricter localization thresholds.
- **Small-object AP**: Detection performance for small pests using a consistent size definition.
- **Parameters (M)**: Model size and parameter efficiency.
- **GFLOPs**: Computational complexity.
- **Inference latency (ms)**: Inference speed under a consistent hardware setup.

---

## 4. Expected Research Contribution

The ablation study aims to determine whether EMA and DySample independently improve Paddy-YOLO and whether their combination produces additional benefits. EMA is expected to enhance feature discrimination, while DySample is expected to improve multi-scale spatial feature reconstruction.
The key comparison is Configuration D against Configurations B and C. If the combined model achieves higher detection accuracy without a substantial increase in computational cost, the findings would support the effectiveness of the hybrid architecture.
