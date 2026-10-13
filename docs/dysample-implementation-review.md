# DySample implementation review

Reviewed on 2026-10-13 against the supplied architecture diagram, the official source, the code embedded in the previous Kaggle notebook, the downloaded five-fold checkpoints and reports, and the user's 15-page `E:/College/PADDY YOLO/RRL/DySample.pdf`. The PDF-specific review is complete. No model configuration, training hyperparameter or running experiment was changed during this review; BiFPN has not been added.

The checks found no mismatch in the DySample sampling implementation or its placement. Our Python module and the version packaged in the previous run have exactly the same parsed Python syntax tree as the [official DySample source](https://github.com/tiny-smart/dysample/blob/main/dysample.py), ignoring attribution comments and formatting. Offset prediction, normal initialization with standard deviation 0.001 and zero bias, initial sampling positions, pixel shuffle, offset scaling and bilinear grid sampling with `align_corners=False` and border padding are preserved.

The image places two DySample modules on the top-down path: after SPPF, then after the first fused C2f. Our model uses the same placement, followed by the original concatenation-based fusion:

| Stage | Our path | Input and output at 640px |
| --- | --- | --- |
| P5 to P4 | SPPF 9 → DySample 10 → Concat 11 → C2f 12 | 256 × 20 × 20 → 256 × 40 × 40 |
| P4 to P3 | C2f 12 → DySample 13 → Concat 14 → C2f 15 | 128 × 40 × 40 → 128 × 80 × 80 |

Both modules use `scale=2`, `style='lp'`, `groups=4`, `dyscope=False`. Channels are inferred from the input feature map; they are not the unscaled channel values printed in YAML arguments. Detection remains at P3/P4/P5 with 64/128/256 input channels. The diagram alone cannot identify its implementation's exact style, grouping factor or scope selection. The supplied PDF depicts an order compatible with `lp`, but leaves the exact constructor settings unspecified; code or a model configuration would be needed to verify them. Its fusion blocks differ from ours, so performance attributed to the combined illustrated architecture cannot automatically be attributed to DySample alone.

Fresh tests compared the local and official modules with identical weights on non-square feature maps at both actual input channel counts. Outputs, input gradients and offset-weight gradients matched exactly. Setting learned offsets to zero reproduced ordinary bilinear interpolation with maximum absolute error below 3.6e-7. This confirms DySample's starting sampling grid is bilinear; it does not preserve the nearest-neighbor baseline's values. Any gain or loss can therefore reflect both interpolation changes and learned sampling. Six existing focused integration tests passed, including full-model forward/backward and detection-shape checks.

All ten DySample modules across the five saved best checkpoints contain the intended settings and finite weights. Their offset weights differ from the first-epoch checkpoints, showing that the offset branches were updated during training. Learned weight standard deviations are approximately 0.379–0.471, compared with the original 0.001 initialization. This is a reason to inspect predicted sampling displacements on real held-out images if diagnosing localization, but weight magnitude alone does not establish excessive displacement or a bug. No representative displacement measurement was performed in this review.

Two confirmed experimental differences limit the earlier accuracy comparison:

1. Both archived notebooks construct the custom initialized model before `PaddyTrainer` seeds the RNG. Although the recorded training seed is 42, it did not control the initial unmatched RGG/classification weights or DySample offset weights. This is a reproducibility defect in the initialization workflow, not a modification of the official DySample module. It was corrected during preparation of the subsequent EMA run by seeding before model construction. The earlier completed runs retain this limitation.
2. Pretrained RGG + DIMIoU used one GPU with batch 32; DySample used two-worker DDP with batch 16 per worker and global batch 32. The installed Ultralytics trainer wraps the model in ordinary DDP without converting BatchNorm to synchronized BatchNorm. Consequently, BatchNorm training statistics are computed over different local batch sizes. PyTorch documents the explicit conversion required for [synchronized BatchNorm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.SyncBatchNorm.html). Equal global batch sizes do not make these runs identical training conditions.

The channel-count correction introduced for the subsequent EMA branches does not explain the earlier DySample scores. DySample's two layers both consume the immediately preceding feature, and the earlier inferred channels were already 256 and 128, as verified in the saved checkpoints and preflight.

The prior held-out test results remain valid observations of their respective runs:

| Fold | DySample minus baseline mAP@50 | DySample minus baseline mAP@50–95 |
| ---: | ---: | ---: |
| 0 | +0.09 pp | -1.83 pp |
| 1 | -2.93 pp | -4.64 pp |
| 2 | -3.46 pp | -2.67 pp |
| 3 | -0.19 pp | -1.04 pp |
| 4 | -0.76 pp | -0.75 pp |
| Mean | -1.45 pp | -2.19 pp |

The loss at stricter IoU thresholds appears in all five folds. This warrants inspecting localization and predicted offsets, but it does not establish the cause or statistical significance. The exported overall and per-class mAP tables do not isolate small-object AP, so they cannot directly answer whether small-object performance improved. Dataset splits and configured hyperparameters match between experiments, but initialization and GPU execution were not fully controlled.

For a future investigation, keep the existing fusion neck and first compare nearest-neighbor, ordinary bilinear and official DySample under the same GPU count and correctly seeded initialization. Control the common initial weights and data splits, and retain validation-selected checkpoints for held-out evaluation. If placement is then investigated, separate P5→P4-only and P4→P3-only replacements can isolate which stage helps. These are proposed ablations, not changes made or training jobs launched here. Changes to grouping, style or learning rates should follow evidence from the PDF and sampling diagnostics rather than assume a guaranteed improvement.

PyTorch also notes that CUDA `grid_sample` backward can be [nondeterministic](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html). Correctly setting a seed improves control of initialization, but does not guarantee bitwise-identical DySample GPU training.

Review evidence is saved in `data/dysample_implementation_review/verification.json`, with the downloaded official source and extracted source from the previous notebook in that directory. The existing [five-fold comparison CSV](../reports/rgg-dimiou-vs-dysample-fivefold-comparison.csv) contains exact aggregate and per-class results. PDF text, inspected page renders and independently extracted Table 3 are preserved in `data/dysample_implementation_review/pdf/`.

The supplied PDF is Gao et al. (2026), [Lightweight YOLOv8s-BD for small and dense agricultural pest detection using dynamic upsampling and bidirectional feature fusion](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1913220/full), Frontiers in Plant Science 17:1913220, DOI 10.3389/fpls.2026.1913220. Its Figure 4 on PDF page 6 is the supplied diagram. Figure 5 on page 7 depicts linear offset prediction before pixel shuffle, matching the order of our `lp` variant. The diagram does not establish the exact author's constructor arguments. Section 2.3 explains both static and dynamic scope, but supplies no numeric group count, explicit `lp`/`pl` setting or unambiguous selection of one scope mode for the reported experiments. No reproducible model YAML or repository URL is supplied in the PDF or linked article text. These settings remain unspecified rather than verified identical to ours.

Table 3 on PDF page 8 contains a DySample-only ablation, so its contribution can be considered separately from the combined model. Table cells were checked visually and independently extracted with PyMuPDF:

| Paper configuration | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | 73.3% | 67.4% | 70.6% | 45.3% |
| YOLOv8s + DySample only | 70.8% | 69.7% | 71.3% | 45.8% |
| DySample-only change | -2.5 pp | +2.3 pp | +0.7 pp | +0.5 pp |

The paper's combined architecture reports 72.3%/46.9% mAP, but those values include a different fusion architecture. They are not a target accuracy or expected gain for our DySample-only experiment. The isolated gain is modest and includes a precision/recall tradeoff. It does not imply DySample improves every dataset or configuration.

Several experimental conditions differ materially:

| Condition | Supplied paper | Previous Paddy-YOLO experiment |
| --- | --- | --- |
| Model | YOLOv8s, native C2f backbone | Nano-scaled Paddy-YOLO, RGG backbone and DIMIoU |
| Dataset | 24-class Pest24 with GAN augmentation in all Table 3 rows | 16-class R2000, existing preparation without GAN augmentation |
| Optimizer | SGD, initial LR 0.01 | AdamW, initial LR 0.01 |
| Hardware | One A40, batch 32 | DySample: two T4 workers, global batch 32 |
| Evaluation repeats | Five repeated training runs on a 6:2:2 image split | Five different held-out folds with a 3:1:1 split |
| Epochs / image size | 300 / 640px | 300 / 640px |

Section 2.5 on PDF page 5 reports the optimizer and hardware settings. Section 2.6 on page 7 and Table 3 describe repeated runs; their standard deviations measure a different source of variation from our across-fold standard deviations. Equal numeric learning rates do not make SGD and AdamW updates equivalent. The optimizer difference is a plausible factor in offset learning, not demonstrated evidence that it caused our lower mAP. The paper supplies no offset-specific learning rate, regularizer or clamp that we omitted.

Section 2.3's wording should not be implemented literally where it conflicts with the original mechanism. Multiplying by 0.25 changes offset amplitude; it does not downsample a feature map. Figure 5's `G` represents a regular coordinate grid, and `δ` represents sampling coordinates rather than a fused feature tensor. The official module then samples the input using that grid. Its static-scope option still predicts content-dependent offsets; `dyscope=False` does not turn off dynamic upsampling. The dynamic-scope alternative uses a sigmoid-controlled multiplier. Our earlier DySample+ experiment already enabled that alternative at both stages and scored lower on fold 0; that single-fold result does not settle all configurations, but the PDF does not justify assuming scope activation would fix performance.

The paper's complexity table also needs implementation details before serving as an engineering reference. Its DySample-only row lists parameters decreasing from 11.2M to 11.1M. Replacing only parameter-free nearest-neighbor upsampling with our official learnable DySample necessarily adds parameters (+12,352 here). The reported decrease cannot be explained by that substitution alone; counting conventions, a different implementation, unreported structural changes or a reporting error would need clarification. Its latency and rounded FLOP values should likewise not replace direct profiling of our implementation.

The PDF therefore supports our placement and general sampling design, but does not identify a verified implementation defect explaining the previous run. The next useful investigation remains a controlled nearest/bilinear/DySample comparison with matched initial common weights, correctly timed seeding and the same GPU count. Inspecting actual predicted offset magnitudes and border sampling on real held-out images can then test the localization hypothesis before adjusting scope, groups or stage selection. These are proposed checks only; no new training was launched and the existing EMA run was not changed.
