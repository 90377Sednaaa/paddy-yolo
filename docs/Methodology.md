# Chapter 4: Methodology

This chapter presents the methodology employed to develop and evaluate the modified Paddy-YOLO model for rice pest detection in complex field environments. The proposed architecture builds upon the lightweight Paddy-YOLO framework by integrating Efficient Multi-Scale Attention (EMA) into the feature extraction backbone and Dynamic Upsampling (DySample) into the multi-scale feature fusion neck. The methodology is structured into five core sections: data preprocessing and splitting, proposed model architecture, training configuration, ablation plan, and evaluation metrics.

---

## 4.1 Data Preprocessing and Splitting

The primary dataset used in this study is the Kaggle R2000 rice pest benchmark (`dmrashidferrer/r2000v1-2024-images`), which consists of 2,024 natural field images containing 2,281 annotated pest instances. The dataset covers 16 taxonomically critical rice pest species spanning five insect orders: *Lepidoptera* (rice leaf roller, pink rice borer, asiatic rice borer, yellow rice borer, rice leaf caterpillar, and melanitis leda), *Hemiptera* (small brown plant hopper, white backed plant hopper, brown plant hopper, and rice leaf hopper), *Diptera* (rice gall midge, paddy stem maggot, and rice stem fly), *Thysanoptera* (grain spreader thrips), and *Coleoptera* (rice weevil and rice shell pest). These species present realistic field challenges, including tiny physical sizes, background camouflage, foliage occlusion, and dense clustering. In addition, the large-scale IP102 benchmark dataset is utilized to evaluate model generalization across diverse agricultural environments and insect categories.

Data preprocessing standardizes heterogeneous field annotations into a uniform format for model training. Raw Pascal VOC XML annotations containing absolute bounding box coordinates are converted into normalized YOLO format $(c_x, c_y, w, h) \in [0.0, 1.0]$. Input images are resized to a standard resolution of $640 \times 640$ pixels using letterboxing, which preserves native aspect ratios by padding borders rather than distorting insect morphology. All pixel values are normalized from the discrete range $[0, 255]$ into $[0.0, 1.0]$ in standard 3-channel RGB color space.

To improve generalization and prevent overfitting under variable outdoor lighting, an online data augmentation pipeline is applied during training. Mosaic augmentation ($p = 1.0$) stitches four randomly cropped and scaled images into a single training sample, exposing the model to varied contextual scales and improving miniature pest detection. In accordance with standard practice, mosaic augmentation is automatically disabled during the final 10 epochs (`close_mosaic = 10`) to allow fine-tuning on natural, uncorrupted image distributions. Further augmentations include random horizontal flipping ($p = 0.5$), spatial translation ($\pm 10\%$), scale scaling ($\pm 50\%$), and HSV color space adjustments (hue $\pm 0.015$, saturation $\pm 0.7$, value $\pm 0.4$) to simulate daylight fluctuations and foliage shadows.

A 5-fold cross-validation scheme is implemented to ensure robust evaluation and eliminate data leakage. The dataset of 2,024 images is partitioned into five mutually exclusive folds using a fixed pseudo-random seed ($\text{seed} = 42$). For each fold iteration $k \in \{0, 1, 2, 3, 4\}$, the data is allocated in a strict 3:1:1 ratio: three folds (60%) are combined for training, one fold (20%) is reserved for validation and model checkpoint selection, and the remaining fold (20%) serves as an independent holdout test set. This ensures every image is tested exactly once while maintaining complete isolation between training and testing splits.

---

## 4.2 Proposed Model Architecture

The proposed model is an anchor-free single-stage object detector designed for high accuracy and low latency in agricultural settings. It builds on the baseline Paddy-YOLO detector, which utilizes a YOLOv8-N scale profile, and introduces two primary enhancements: EMA attention modules inserted after deep backbone stages to suppress leaf background noise, and DySample point-based dynamic upsampling operators placed in the neck to preserve spatial details of tiny pests during feature fusion.

```
+-----------------------------------------------------------------------------------------------+
|                                   Input Image [3, 640, 640]                                   |
+-----------------------------------------------------------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                                Backbone Feature Extraction                                    |
|  P1/2  : Conv [64, 3x3, s=2]                                                                  |
|  P2/4  : Conv [128, 3x3, s=2] -> RGG Module [128]                                             |
|  P3/8  : Conv [256, 3x3, s=2] -> RGG Module [256] -> EMA Attention Module                     |
|  P4/16 : Conv [512, 3x3, s=2] -> RGG Module [512] -> EMA Attention Module                     |
|  P5/32 : Conv [1024, 3x3, s=2] -> RGG Module [1024] -> EMA Attention Module -> SPPF [1024]    |
+-----------------------------------------------------------------------------------------------+
                             |                   |                   |
                   (P3 Feature Map)    (P4 Feature Map)    (P5 Feature Map)
                             |                   |                   |
                             v                   v                   v
+-----------------------------------------------------------------------------------------------+
|                             DySample Feature Fusion Neck                                      |
|  Top-Down Path:                                                                               |
|    - P5 -> DySample (scale=2) -> Concat with P4 -> C2f [512]                                  |
|    - P4 -> DySample (scale=2) -> Concat with P3 -> C2f [256] -----------------> Head P3 (Small)|
|                                                                                     |         |
|  Bottom-Up Path:                                                                    |         |
|    - Head P3 -> Conv [256, 3x3, s=2] -> Concat with P4 -> C2f [512] --------------> Head P4   |
|    - Head P4 -> Conv [512, 3x3, s=2] -> Concat with P5 -> C2f [1024] -------------> Head P5   |
+-----------------------------------------------------------------------------------------------+
                                                                                      |
                                                                                      v
+-----------------------------------------------------------------------------------------------+
|                                   Decoupled Detection Heads                                   |
|  Classification Head: BCE Loss                                                                |
|  Regression Head    : Dynamic Inner-MPDIoU (DIMIoU Strategy H) + DFL Loss                     |
+-----------------------------------------------------------------------------------------------+
```

The backbone features the RepGG (RGG) module, which replaces all four conventional $C2f$ blocks in the YOLOv8 backbone. Inside each RGG block, feature processing is handled by RepConv and lightweight GhostConvs. During training, RepConv employs a multi-branch topology consisting of parallel $3 \times 3$ convolution, $1 \times 1$ convolution, and identity branches to enhance gradient representation. Prior to inference, these parallel branches are fused via structural re-parameterization into a single equivalent $3 \times 3$ convolution layer ($W_{\text{fused}} = W_{3\times3} + \text{pad}(W_{1\times1}) + \text{pad}(W_{\text{id}})$), incurring zero runtime latency penalty. The GhostConv layers generate intrinsic features through a standard $3 \times 3$ convolution and complement them using cheap $5 \times 5$ depthwise convolutions, expanding the effective receptive field to capture subtle appendages (such as legs and antennae) without substantial parameter cost.

The Efficient Multi-Scale Attention (EMA) module is integrated into the backbone after selected RGG stages (P3, P4, and P5). EMA divides input channels into sub-groups to learn parallel multi-scale features and applies 1D directional coordinate pooling along the horizontal and vertical axes to encode precise spatial coordinates. By combining a $1 \times 1$ coordinate branch with a $3 \times 3$ spatial branch, EMA calculates cross-spatial pixel interactions without channel dimensionality reduction. This allows the network to highlight pest-related visual cues and suppress repetitive rice leaf background textures before features are forwarded into the neck.

The Dynamic Upsampling (DySample) module replaces the two conventional nearest-neighbor upsampling layers in the feature fusion neck (Layers 10 and 13). Unlike nearest-neighbor interpolation which uses fixed kernels, DySample formulates upsampling from a point-sampling perspective. A lightweight $1 \times 1$ convolution generates dynamic coordinate offsets $\Delta \mathcal{O}$, creating a perturbed sampling grid $\mathcal{G}' = \mathcal{G} + \Delta \mathcal{O}$. Output features are resampled using bilinear grid interpolation (`grid_sample`). This point-based formulation reconstructs sharper spatial boundaries for small pests without using heavy dynamic convolution kernels, maintaining low computational complexity and full compatibility with standard inference runtimes.

Bounding box regression is optimized using the Dynamic Inner-MPDIoU (DIMIoU) loss with Strategy H dynamic ratio scheduling. DIMIoU scales predicted and target bounding boxes around their centers by an auxiliary scale ratio $\text{ratio}(\text{IoU}) = 0.5 \cos((1 - \text{IoU})\pi) + 1.0$. When IoU is low, the ratio scales down toward 0.5 to concentrate on the central core and accelerate convergence; when IoU is high, the ratio smoothly expands to 1.5 to emphasize fine-grained edge alignment. Combined with a normalized four-corner Euclidean distance penalty ($R_{\text{corner}} = \frac{d_1^2 + d_2^2}{C^2}$), the total regression loss accelerates convergence and improves localization accuracy for overlapping and small rice pests.

---

## 4.3 Training Configuration

Model training uses the AdamW optimizer with decoupled weight decay ($\lambda = 0.0005$) and momentum parameter $\beta_1 = 0.937$. The learning rate is initialized at $\eta_0 = 0.01$ and follows a cosine annealing decay schedule down to a final ratio of 0.01 ($\eta_{\min} = 0.0001$) across 300 epochs. A warmup phase of 3.0 epochs linearly increases momentum from 0.8 to 0.937 while initializing bias learning rate to 0.1, preventing early gradient instability.

To leverage semantic representations from large-scale pre-training, all models are initialized with weights from `yolov8n.pt` pre-trained on the MS COCO dataset. A topological weight transfer routine maps weights into the modified network: backbone convolutions and SPPF layers transfer directly, RGG projections receive weights from corresponding $C2f$ layers, and neck layers are mapped topologically to preserve pre-trained knowledge across shifted layer indices.

Training is performed at a batch size of 32 with input resolution $640 \times 640$ pixels using Automatic Mixed Precision (AMP FP16) to maximize memory efficiency and throughput. The loss balance weights are set to $\lambda_{\text{box}} = 7.5$ for DIMIoU regression, $\lambda_{\text{cls}} = 0.5$ for classification, and $\lambda_{\text{dfl}} = 1.5$ for distribution focal loss. Experiments are conducted using PyTorch 2.x and CUDA 12.x on an NVIDIA GPU workstation.

| Category | Hyperparameter | Setting | Description |
| :--- | :--- | :---: | :--- |
| **Schedule** | Epochs | `300` | Full training duration |
| | Batch Size | `32` | Training batch size |
| | Image Size | `640` | Input resolution ($640 \times 640$) |
| **Optimizer** | Optimizer | `AdamW` | Adam with decoupled weight decay |
| | Initial LR (`lr0`) | `0.01` | Peak base learning rate |
| | Final LR (`lrf`) | `0.01` | Final learning rate multiplier ($\eta_{\min} = 10^{-4}$) |
| | Momentum | `0.937` | First-moment decay factor |
| | Weight Decay | `0.0005` | $L_2$ weight regularization |
| | Warmup Epochs | `3.0` | Initial warmup duration |
| **Loss** | Box Loss Gain | `7.5` | Multiplier for DIMIoU loss |
| | Class Loss Gain | `0.5` | Multiplier for BCE loss |
| | DFL Loss Gain | `1.5` | Multiplier for distribution focal loss |
| | DIMIoU Strategy | `H` | Cosine ratio schedule ($0.5 \to 1.5$) |
| **Augmentation** | Mosaic | `1.0` | 4-image mosaic probability |
| | Close Mosaic | `10` | Final epochs with mosaic disabled |
| | Flip Left-Right | `0.5` | Random horizontal flip probability |
| | Translation | `0.1` | Random translation range fraction |
| | Scaling | `0.5` | Random scale jitter range fraction |

---

## 4.4 Ablation Plan

A $2 \times 2$ factorial ablation study is conducted to evaluate the individual and combined contributions of Efficient Multi-Scale Attention (EMA) and Dynamic Upsampling (DySample) to the original Paddy-YOLO architecture. All configurations retain the original RGG feature extraction modules, decoupled detection heads, and DIMIoU bounding box regression loss under identical training conditions, isolating the architectural impact of each enhancement.

| Configuration | Backbone | Feature Fusion Neck | Purpose |
| :--- | :--- | :--- | :--- |
| **A. Baseline Paddy-YOLO** | Original RGG modules | Original Upsample (nearest) | Establish reference accuracy and computational complexity. |
| **B. Paddy-YOLO + EMA** | RGG + EMA | Original Upsample (nearest) | Assess attention-based feature refinement and background suppression. |
| **C. Paddy-YOLO + DySample** | Original RGG modules | DySample replaces two Upsample operations | Assess adaptive upsampling and multi-scale feature reconstruction. |
| **D. Paddy-YOLO + EMA + DySample** | RGG + EMA | DySample replaces two Upsample operations | Assess joint effects of feature attention and dynamic upsampling. |

Configuration A serves as the control model, replicating the baseline Paddy-YOLO architecture with standard nearest-neighbor upsampling in the neck and no attention blocks in the backbone. Configuration B inserts EMA modules after selected RGG stages in the backbone while retaining conventional upsampling in the neck, evaluating whether cross-spatial attention improves feature discrimination by emphasizing pest characteristics and suppressing foliage clutter. Configuration C maintains the original backbone while replacing the two neck upsampling operations with DySample, investigating whether adaptive point sampling improves spatial detail preservation, particularly for miniature pests. Configuration D combines both enhancements, examining whether the joint integration of backbone attention and dynamic neck upsampling provides complementary accuracy gains while maintaining real-time computational efficiency.

---

## 4.5 Evaluation

Model performance is evaluated across all four configurations using identical training conditions on both the R2000 and IP102 datasets. Detection accuracy is measured using Precision ($P = \frac{TP}{TP + FP}$), Recall ($R = \frac{TP}{TP + FN}$), and $F_1$-score ($F_1 = \frac{2PR}{P + R}$). Overall detection efficacy is reported using Mean Average Precision at an IoU threshold of 0.50 ($\text{mAP}@0.5$), which serves as the primary evaluation metric, and $\text{mAP}@0.5:0.95$, which averages precision across ten stricter IoU thresholds from 0.50 to 0.95 with a step of 0.05 to evaluate precise boundary localization.

To evaluate performance across different pest body sizes, scale-specific detection is assessed using small-object average precision ($\text{AP}_S$, bounding box area $< 32^2$ pixels), medium-object average precision ($\text{AP}_M$, area between $32^2$ and $96^2$ pixels), and large-object average precision ($\text{AP}_L$, area $> 96^2$ pixels). Computational efficiency is quantified through total trainable parameters (M), computational complexity in GFLOPs, model checkpoint size (MB), and inference latency (ms per frame decomposed into preprocessing, model forward inference, and NMS post-processing). Inference throughput in Frames Per Second (FPS) is reported under a standardized hardware setup to assess real-time viability on agricultural edge hardware.

All metrics are aggregated across the 5-fold cross-validation runs and reported as macro-averaged mean and standard deviation ($\mu \pm \sigma$). Paired statistical testing (two-tailed paired Student's $t$-test and Wilcoxon signed-rank test at significance level $\alpha = 0.05$) is conducted between Configuration D and the baseline and single-module configurations to confirm that observed performance gains represent statistically significant architectural improvements rather than random data partitioning artifacts.
