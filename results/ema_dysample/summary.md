# EMA + DySample: single-fold Kaggle results

The run completed all **300 epochs** on `dmrashidpferrer/paddy-yolo-ema-dysample-fold0`, notebook version 2.

The nano model uses three EMA blocks in the backbone and two DySample blocks in the neck, initialized from `yolov8n.pt`. Training used 640-pixel images, seed 42, and a total batch size of **32 across two GPUs (16 per GPU)**.

The `dmrashidpferrer/r2000-pestnet` supplied split contains 1,214 training images, 304 validation images, and 506 test images across 16 classes. This run uses one supplied split; the scores below are not averages across multiple folds.

## Best checkpoint evaluation

| Split | mAP50 | mAP50-95 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 64.07% | 45.36% | 63.64% | 56.73% | 59.99% |
| Test | 59.03% | 41.75% | 56.17% | 57.26% | 56.71% |

These are the best checkpoint's evaluation scores from [metrics.json](metrics.json). The final epoch's CSV values can differ because they describe the model at epoch 300.

## Saved artifacts

- [Best checkpoint](train/ema_dysample_fold0/weights/best.pt) and [last checkpoint](train/ema_dysample_fold0/weights/last.pt), plus checkpoints saved every 25 epochs.
- [Training history](results.csv) and [training curves](train/ema_dysample_fold0/results.png).
- Validation and test precision/recall curves, confusion matrices, and example predictions in `evaluate/val/` and `evaluate/test/`.
- `training.log`, `kaggle_run.log`, environment and preflight logs, TensorBoard events, and training arguments.
- `dataset/` contains the dataset audit and original Kaggle data configuration; its absolute dataset paths refer to the Kaggle environment.
- `run_config.json`, `launch.json`, `verification.json`, and `weight_transfer.json` preserve the run configuration and initialization records.
- [Download manifest](download_manifest.json) records artifact paths, byte counts, SHA-256 checksums, and any failed downloads.

The downloaded best checkpoint was loaded locally with the custom Paddy-YOLO modules and completed a CPU inference check with finite predictions. [checkpoint_verification.json](checkpoint_verification.json) confirms three EMA modules, two DySample modules, 16 classes, and strides 8/16/32. Kaggle preflight checks passed all 100 tests.

The [pretraining audit](pretraining_audit.md) verifies the original Kaggle initialization against its actual `yolov8n.pt`: all 283 transferred tensors match exactly, covering 70.80% of model parameter elements. It also records unstable early validation behavior and training settings that merit investigation.

Kaggle notebook: https://www.kaggle.com/code/dmrashidpferrer/paddy-yolo-ema-dysample-fold0
