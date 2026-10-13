# Paddy-YOLO-DySample+: R2000 fold 0 results

Retrieved on 2026-10-13 (Asia/Manila) from the private [Kaggle run](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-dysample-plus-r2000-fold0), version 1. The run completed all 300 epochs, with all numeric entries in its epoch history finite. Training took 1.143 hours (about 68.6 minutes); total recorded notebook execution was about 70.5 minutes.

The best saved checkpoint matches epoch 288, selected by mAP@50-95. Both neck upsamplers use `scale=2`, `style='lp'`, `groups=4`, `dyscope=True`. All downloaded checkpoints contain both scope branches.

| Saved best-checkpoint validation metric | Plain DySample, epoch 287 | DySample+, epoch 288 | Plus minus plain, percentage points |
| --- | ---: | ---: | ---: |
| Precision | 70.619% | 66.858% | -3.761 |
| Recall | 59.542% | 60.178% | +0.636 |
| mAP@50 | 68.115% | 65.314% | -2.801 |
| mAP@50-95 | 51.066% | 46.667% | -4.399 |

These values come from the saved training histories and were checked against the checkpoint's stored metrics. At epoch 300, DySample+ had mAP@50 64.550% and mAP@50-95 45.997%, below its selected checkpoint.

The final revalidation log for the stripped best checkpoint reports precision 66.8%, recall 60.7%, mAP@50 65.3%, and mAP@50-95 46.7%. The per-class CSV uses this final revalidation table, which is rounded in the console log; it is a separate evaluation from the more precise saved epoch metrics above.

## Downloaded artifacts

- [Per-class and overall CSV](../reports/dysample-plus-fold0-per-class-overall.csv): 16 class rows plus overall; metrics expressed as percentages.
- [Plain DySample versus Plus comparison CSV](../reports/dysample-vs-plus-fold0-comparison.csv): matched class rows and percentage-point differences from both final revalidation logs. Positive deltas mean Plus is higher.
- Training artifacts: `data/kaggle_dysample_plus_results/dysample_training/proposed_plus/fold0/`, containing `results.csv`, `args.yaml`, training/precision-recall/F1 plots, confusion matrices and training/validation image previews.
- Checkpoints: that directory's `weights/best.pt`, `weights/last.pt`, and `epoch0.pt`, `epoch50.pt`, `epoch100.pt`, `epoch150.pt`, `epoch200.pt`, `epoch250.pt`.
- Complete execution log: `data/kaggle_dysample_plus_complete_v1.log`.
- Run configurations and fold lists: `data/kaggle_dysample_plus_results/configs/` and `data/kaggle_dysample_plus_results/data/r2000/`.

All eight checkpoints loaded successfully. The six periodic checkpoints retain optimizer/scaler state and their zero-based epochs, permitting continuation from those epochs. Completed `best.pt` and `last.pt` have optimizer state stripped and epoch set to -1; they can initialize a new training stage but cannot resume the original optimizer state.

## Comparison checks and interpretation

The downloaded fold lists match exactly between runs: 1,214 training image paths, 405 validation paths, and 405 test paths. Saved training arguments match apart from model initialization filenames and output directories. Both runs used `yolov8n.pt`, 300 epochs, batch 32, image size 640, two GPUs, AdamW, seed 42, and checkpoint period 50.

Validation covers 405 images and 456 labeled instances. In the final rounded per-class results, Plus improves AP@50 for 6 classes and AP@50-95 for 5 classes, but declines for the remaining classes. Rice leaf roller has the largest AP@50-95 decline, from 58.5% to 33.9%; asiatic rice borer improves from 23.8% to 25.0%.

This run provides no overall validation improvement from enabling dynamic scope at both layers. Plain DySample has higher aggregate AP on this fold. These are one-fold, one-seed validation results; the held-out test split was not evaluated, and broader conclusions require repeated comparisons.
