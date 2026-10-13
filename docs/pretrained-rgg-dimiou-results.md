# Pretrained RGG + DIMIoU: five-fold results

Retrieved on 2026-10-13 (Asia/Manila) from the completed private [dmrashidpferrer notebook](https://www.kaggle.com/code/dmrashidpferrer/paddy-yolo-pretrained-rgg-dimiou-20261012). This is the pretrained experiment, separate from the random-initialization RGG + DIMIoU run under `dmrashidferrer`.

All five folds completed 300 epochs. The manifest confirms four sequential RGG stages at layers 2, 4, 6 and 8, nearest-neighbor neck upsampling, DIMIoU Strategy H without a fixed ratio, and independent initialization from `yolov8n.pt` for every fold. Training used AdamW, global batch 32, image size 640, seed 42, AMP training and FP32 validation. This notebook trained on device 0 (one Tesla T4), unlike the two-worker DySample runs.

## Held-out test results

The table reports an unweighted mean and sample standard deviation over all five test folds. Standard deviations are in percentage points.

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| Precision | 64.05% | 6.10 pp |
| Recall | 61.24% | 2.22 pp |
| F1 | 62.50% | 3.26 pp |
| mAP@50 | 64.31% | 3.68 pp |
| mAP@50-95 | 46.34% | 3.98 pp |

Overall F1 follows the existing runner's definition: the harmonic mean of overall precision and recall within each fold, followed by averaging across folds. It is not an independently calculated average of per-class F1 scores.

| Fold | Best validation epoch | Test mAP@50 | Test mAP@50-95 |
| ---: | ---: | ---: | ---: |
| 0 | 278 | 59.04% | 40.89% |
| 1 | 282 | 68.10% | 50.28% |
| 2 | 239 | 62.05% | 43.44% |
| 3 | 270 | 66.04% | 48.73% |
| 4 | 278 | 66.31% | 48.38% |

These are five-fold **test** results. The earlier plain DySample and DySample+ comparisons reported fold 0 validation results, so those numbers are not a direct comparison with this table. The new plain DySample five-fold job is now complete: see its [held-out test results and comparison](dysample-fivefold-results.md), with matching split and hyperparameter hashes and the GPU training difference documented.

## Retrieved artifacts and CSVs

- [Five-fold summary CSV](../reports/pretrained-rgg-dimiou-fivefold-summary.csv): overall plus all 16 classes, with means and sample standard deviations. Metric means are percentages; standard deviations are percentage points.
- [Per-fold result CSV](../reports/pretrained-rgg-dimiou-per-fold-results.csv): 85 exact-result rows, covering overall plus 16 classes for every fold. Per-class F1 is computed from that class's precision and recall.
- Complete downloaded outputs: `data/pretrained_rgg_dimiou_results_download/rgg_dimiou_results/`.
- Each fold's best and last checkpoints: `train/rgg_dimiou_foldN/weights/{best,last}.pt` under that output directory.
- Training CSVs, arguments, plots and previews: `train/rgg_dimiou_foldN/`.
- Held-out test plots and previews: `test/foldN/`.
- Exact source reports: `foldN_test_metrics.json`, `individual_folds.csv`, `fivefold_summary.json` and `run_manifest.json`.
- Complete execution log: `data/kaggle_pretrained_rgg_dimiou_complete.log`.

All five histories contain exactly epochs 1 through 300 with finite values. Both best and last checkpoints for every fold passed ZIP integrity checks and loaded successfully; their RGG indices, 16-class detection head, nearest-neighbor upsamplers and serialized DIMIoU H initializer were verified. The completed checkpoints have optimizer state stripped and epoch -1. Summary means and sample standard deviations independently recomputed from the per-fold metrics match the downloaded summary.
