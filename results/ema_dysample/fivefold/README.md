# EMA + DySample: five-fold results

All five folds completed **300 epochs each**. Kaggle ended with `CANCEL_ACKNOWLEDGED` during notebook export. All selected result artifacts were recovered; cancellation did not truncate the recorded training histories.

The run uses corrected RGG, three backbone EMA blocks and two DySample neck blocks, nano scale, `yolov8n.pt`, 640-pixel input, and batch **32 across two T4 GPUs (16 each)**. AdamW uses `lr0=0.001` and zero bias warmup.

The 3:1:1 training/validation/test rotation matches the paper Table 5 image and instance counts exactly. These are reconstructed memberships; the authors’ original image lists were unavailable. The downloaded remote split manifest passes content-leakage and coverage checks.

Scores below are mean ± sample standard deviation, with standard deviations in percentage points. Each fold evaluates its own best checkpoint selected using validation.

| Split | mAP50 | mAP50–95 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 79.30% ± 1.96 pp | 63.48% ± 2.63 pp | 79.46% ± 4.31 pp | 72.03% ± 1.04 pp | 75.51% ± 1.99 pp |
| Test | 78.17% ± 2.73 pp | 62.12% ± 2.61 pp | 79.13% ± 5.71 pp | 70.89% ± 3.08 pp | 74.62% ± 2.16 pp |

| Fold directory | Test mAP50 | Test mAP50–95 |
| --- | ---: | ---: |
| [fold0](fold0/metrics.json) | 79.34% | 60.93% |
| [fold1](fold1/metrics.json) | 75.78% | 60.86% |
| [fold2](fold2/metrics.json) | 82.07% | 66.34% |
| [fold3](fold3/metrics.json) | 78.22% | 62.81% |
| [fold4](fold4/metrics.json) | 75.45% | 59.67% |

Artifacts:

- [fivefold_results.png](fivefold_results.png): five individual test results and their arithmetic mean; regenerate with [plot_results.py](plot_results.py).
- `fold0/` through `fold4/`: metrics, training history, best/last/periodic checkpoints, plots and logs.
- [summary.json](summary.json) and [fold_metrics.csv](fold_metrics.csv): aggregate and individual fold scores.
- `splits/`: actual Kaggle fold memberships and data YAMLs. Absolute paths inside these files refer to Kaggle.
- `configuration/` and `logs/`: architecture, hyperparameters, run record, preflight and cancellation details.
- [download_manifest.json](download_manifest.json): artifact byte counts and SHA-256 checksums.
- [verification.json](verification.json) and [checkpoint_verification.json](checkpoint_verification.json): recovery and checkpoint checks.

Recovered 355 remote artifacts; all checksums verified. All five best checkpoints load locally and produce finite CPU predictions with 16 classes and strides 8/16/32.

Notebook: https://www.kaggle.com/code/dmrashidpferrer/paddy-yolo-ema-dysample-paper-5fold
