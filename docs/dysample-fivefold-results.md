# Plain Paddy-YOLO-DySample: five-fold results

Retrieved on 2026-10-13 (Asia/Manila) from the completed private [dmrashidferrer notebook](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-dysample-r2000-fivefold). All five folds completed 300 epochs and held-out test evaluation.

Each fold independently initialized from compatible `yolov8n.pt` weights; the training logs record 283 transferred tensors per fresh model. Both neck upsamplers, at layers 10 and 13, use plain DySample with `scale=2`, `style='lp'`, `groups=4`, `dyscope=False`. RGG remains at layers 2, 4, 6 and 8, with DIMIoU Strategy H and the original P3/P4/P5 detection head. Training used two Tesla T4 GPUs, AdamW, global batch 32, image size 640, seed 42, `configs/hyp_paper.yaml` and checkpoint period 50. The logs show approximately 1.12–1.19 hours of training per fold, totaling 5.80 hours of training before test evaluation and notebook finalization.

## Held-out test results

Values are unweighted means and sample standard deviations over five folds. Standard deviations are in percentage points. Each best checkpoint was selected using its validation split, then evaluated on its held-out test split in FP32.

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| Precision | 65.10% | 5.88 pp |
| Recall | 58.66% | 3.25 pp |
| F1, derived from P/R | 61.59% | 3.46 pp |
| mAP@50 | 62.86% | 3.67 pp |
| mAP@50-95 | 44.16% | 4.01 pp |

F1 in the exported reports is calculated as `2 * precision * recall / (precision + recall)` separately for each fold and class, then averaged across folds. Overall F1 is the harmonic mean of overall precision and recall within each fold; it is not an independently measured mean of per-class F1 values. This uses the same definition as the pretrained RGG + DIMIoU report.

| Fold | Test images | Test instances | Precision | Recall | mAP@50 | mAP@50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 405 | 446 | 56.60% | 56.98% | 59.13% | 39.06% |
| 1 | 405 | 456 | 72.32% | 55.86% | 65.17% | 45.64% |
| 2 | 405 | 463 | 62.82% | 56.13% | 58.58% | 40.77% |
| 3 | 405 | 437 | 65.75% | 61.60% | 65.85% | 47.68% |
| 4 | 404 | 479 | 67.99% | 62.72% | 65.55% | 47.63% |

All 16 classes have test metrics in all five folds. The five held-out splits contain 2,024 images and 2,281 annotated instances in total. These scores are distinct from the earlier single-fold validation reports.

## Comparison with pretrained RGG + DIMIoU

The retrieved [pretrained baseline](pretrained-rgg-dimiou-results.md) has matching split-membership and hyperparameter SHA-256 hashes. Both experiments independently start each fold from compatible `yolov8n.pt` weights.

| Metric | RGG + DIMIoU mean | DySample mean | DySample minus baseline |
| --- | ---: | ---: | ---: |
| Precision | 64.05% | 65.10% | +1.04 pp |
| Recall | 61.24% | 58.66% | -2.58 pp |
| F1, derived from P/R | 62.50% | 61.59% | -0.90 pp |
| mAP@50 | 64.31% | 62.86% | -1.45 pp |
| mAP@50-95 | 46.34% | 44.16% | -2.19 pp |

The pretrained RGG + DIMIoU run achieved higher mean recall and both mAP measures in these experiments; DySample achieved higher mean precision. The baseline trained on one T4, while DySample used two-worker DDP with the same total batch size. That training execution difference means these results do not isolate architecture as the only experimental variable. No statistical significance claim is made from these fold averages.

A subsequent [implementation review](dysample-implementation-review.md) confirmed that both archived notebooks constructed custom weights before setting the trainer's seed. The recorded seed therefore did not control initial unmatched model tensors. This was corrected for subsequent training, but remains a reproducibility limitation of these completed runs. The review verified the official DySample code, placement, feature compatibility and trained offset branches; the difference in GPU counts also changes local BatchNorm batch sizes.

## Files and verification

- [Per-class and overall summary](../reports/dysample-fivefold-summary.csv): 17 rows, with five-fold means in percent and sample standard deviations in percentage points.
- [Per-fold results](../reports/dysample-fivefold-per-fold-results.csv): 85 rows, including overall and all 16 classes in each fold, exact source metrics converted to percent, and image/instance counts.
- [Baseline comparison](../reports/rgg-dimiou-vs-dysample-fivefold-comparison.csv): 17 rows, with both models' means, sample standard deviations and differences in percentage points.
- Downloaded metrics, training logs, curves, confusion matrices, histories and checkpoints: `data/kaggle_dysample_fivefold_results/dysample_fivefold/`.
- Checkpoints for each fold: `proposed/foldN/weights/` under that directory, containing `best.pt`, `last.pt` and periodic `epoch0.pt`, `epoch50.pt`, `epoch100.pt`, `epoch150.pt`, `epoch200.pt`, `epoch250.pt`.
- Held-out test curves and confusion matrices: `test/foldN/`.
- Downloaded architecture and hyperparameter configurations: `data/kaggle_dysample_fivefold_results/configs/`.
- Kaggle console log response: `data/kaggle_dysample_fivefold_complete.log`; its returned events cover the later portion of execution. Complete per-fold training logs are preserved separately as `foldN_training.log`.
- Local verification evidence: `data/kaggle_dysample_fivefold_results/dysample_fivefold/retrieval_verification.json`.

All five training histories contain exactly epochs 1 through 300, with finite numeric values. Overall and per-class means and sample standard deviations were independently recomputed and match the downloaded summaries. All 40 checkpoints passed ZIP integrity checks and loaded successfully on CPU. Each checkpoint has the expected four RGG stages, two plain DySample modules, 16-class head with strides 8/16/32, and serialized DIMIoU H initializer without a fixed ratio.

Completed `best.pt` and `last.pt` files have optimizer state stripped and epoch -1. All 30 periodic checkpoints retain optimizer and scaler state at their named zero-based epochs. Continuing an interrupted run requires the corresponding output tree and history, as described in [the recovery instructions](dysample-fivefold-run.md).
